from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
import json
import os
from pathlib import Path
from pprint import pprint
import shutil
import textwrap
from bs4 import BeautifulSoup
import click
import pandas as pd

from models.llm import GPT_4_Turbo_Preview, ModelFactoryLLM
from rdflib import ConjunctiveGraph, URIRef
from utils import logger, _html2txt, collect_json, embed, get_expected_types, is_json_disjoint, jaccard_similarity, jsonld_search_property, lookup_schema_type, md5hex, schema_simplify

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from tqdm import tqdm
tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

RANDOM_SEED = 42

@click.group
def cli():
    pass

@cli.command()
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--limit", type=click.INT)
@click.option("--skip", type=click.INT)
def create_dataset(outfile, limit, skip):
    g = ConjunctiveGraph()
    g.parse("schemaorg/shacl/schemaorg_datashapes.shacl")
    g.parse("schemaorg/examples/schemaorg-all-examples.ttl", format="ttl")

    query = """
    SELECT ?prop ?ref ?jsonld WHERE {
        ?node   a <http://www.w3.org/ns/shacl#PropertyShape> ;
                <http://www.w3.org/ns/shacl#path> ?prop .
        ?prop <http://example.org/hasExample> ?example .
        ?example    <http://example.org/json> ?jsonld ;
                    <http://example.org/pre-markup> ?ref .
    }
    """
    df = None

    def load_json(json_str):
        try:   
            soup = BeautifulSoup(json_str, "html.parser")
            q = soup.find("script")
            json_str = q.get_text() if q else soup.get_text()
            return json.loads(json_str)
        except: 
            return None
        
    def clean_json(stub):
        repl = deepcopy(stub)
        if isinstance(repl, dict):
            for k, v in stub.items():
                if k == "type":
                    repl.pop(k)
                    repl["@type"] = schema_simplify(lookup_schema_type(v, default=v))
                
                if k == "@type":
                    repl[k] = schema_simplify(lookup_schema_type(v, default=v))
        elif isinstance(repl, list):
            for i, s in enumerate(stub):
                repl[i] = clean_json(s)
        return repl

    iter = 0
    records = []
    for qres in tqdm(g.query(query)):        
        ref = _html2txt(str(qres.get("ref")), force=True)         
        prop = qres.get("prop")
        prop_simple = schema_simplify(prop)
        example = load_json(qres.get("jsonld").toPython())   
        example = clean_json(example)     
        example_snippets = jsonld_search_property(example, prop_simple, keep_parent_class=True)

        if len(example_snippets) == 0:
            logger.warning(f"Cannot find {prop_simple} in {example}")
            continue
        
        for example_snippet in example_snippets:
            if len(collect_json(example_snippet)) == 0: 
                logger.warning(f"There is no workable property-value pair in {example_snippet}")
                continue
            
            # If the overlap between example and ref is less than 20% generate ref from example
            jaccard_sim = jaccard_similarity(ref, json.dumps(example, ensure_ascii=False))
            if jaccard_sim <= 0.217: 
                # Ask GPT to generate a document 
                
                prompt = OrderedDict({
                    "context1": textwrap.dedent(f"""
                    - Given the JSON-LD markup below:
                    ```json
                    {json.dumps(example, ensure_ascii=False)} 
                    ```
                    """),
                    "task": textwrap.dedent(f"""
                    Task: Write a document with the information provided by the markup.
                    Constraints:
                    - the output can contain many paragraphs.
                    - the output must include all information
                    """)
                })
                
                llm = GPT_4_Turbo_Preview()
                ref = llm.query(prompt)
                
                # Append class, property, value to ref
                # infos = collect_json(example, value_transformer=lambda k,v,e: f"{e} {k} {v}")
                # ref = "\n".join([ref] + infos)
            
            records.append({ "ref": ref,  "prop": prop.toPython(), "example": json.dumps(example, ensure_ascii=False), "example_snippet": json.dumps(example_snippet, ensure_ascii=False), "jaccard_sim": jaccard_sim })
        iter += 1
            
    df = pd.DataFrame.from_records(records)
    df.replace("null", None, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_parquet(outfile)
    

@cli.command()
@click.argument("model", type=click.STRING)
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option("--expert", is_flag=True, default=False)
@click.option("--cot", is_flag=True, default=False)
@click.option("--chain", is_flag=True, default=False)
@click.option("--icl", is_flag=True, default=False)
@click.option("--limit", type=click.INT)
@click.option("--skip", type=click.INT)
@click.option("--template", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--clear", is_flag=True, default=False)
def evaluate_prop_checker_zs(model, infile, outdir, expert, cot, chain, icl, limit, skip, template, clear):
    
    if clear:
        shutil.rmtree(outdir, ignore_errors=True)
    
    llm = None
    test_df = pd.read_parquet(infile)
    
    y_pred = []
    y_true = []
    count = 0
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        if skip is not None and i < skip: 
            continue
        
        if count == limit: break
        
        ref, prop, example_snippet = row["ref"], row["prop"], row["example_snippet"]
                
        id = md5hex(ref + prop + example_snippet)
        jsonld_fn = f"{outdir}/{id}.json"
        Path(jsonld_fn).parent.mkdir(parents=True, exist_ok=True)
        
        logfile = f"{Path(jsonld_fn).parent}/{Path(jsonld_fn).stem}_semantic_pred.json"
        result = None
        if os.path.exists(logfile) and os.stat(logfile).st_size > 0:
            with open(logfile, "r") as f:
                try:
                    log = json.load(f)
                    result = log["aggregation"]["score"] if "aggregation" in log.keys() else log["chunk_0"]["score"]
                    force_redo = False
                except KeyError:
                    logger.warning(f"Could not find 'score' in {logfile}")
                    force_redo = True
        else:
            force_redo = True

        if force_redo:

            if llm is None:
                llm = ModelFactoryLLM.create_model(model)

            jsonld = None
            with open(jsonld_fn, "w") as f:
                jsonld = json.loads(example_snippet)
                if jsonld is None:
                    logger.warning(f"{example_snippet} could not be parsed as JSON")
                    continue
                if len(jsonld) == 1 and isinstance(list(jsonld.values())[0], str):
                    jsonld = { f"http://schema.org/{k}": v for k, v in jsonld.items() }
                else:
                    jsonld["@context"] = "http://schema.org/"
                json.dump(jsonld, f, ensure_ascii=False)
            result = int(llm._evaluate_semantic_conformance(
                jsonld_fn, in_context_learning=icl, chain_of_thought=cot, 
                chain_prompt=chain, expert=expert, prompt_template=template
            )["pred"])
        
        decision_threshold = 0.5 # < 0.5: bias towards positive, > 0.5: bias towards negative
        pred_label = int(result >= decision_threshold)
        true_label = 0 if row["label"] == "negative" else 1
        y_pred.append(pred_label)
        y_true.append(true_label)
        
        count += 1
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    cost = 0.0
    if llm is None:
        stats = llm.get_stats_df()
        cost = stats["estimated_cost"].sum() if not stats.empty else None

    results = { "precision": precision, "recall": recall, "f1-score": f1, "accuracy": accuracy, "avg_cost": cost }
    
    result_fn = f"{Path(outdir).parent}/{Path(outdir).stem}.json"
    with open(result_fn, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(results)

@cli.command()
@click.argument("model", type=click.STRING)
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option("--limit", type=click.INT)
@click.option("--expert", is_flag=True, default=False)
@click.option("--cot", is_flag=True, default=False)
@click.option("--chain", is_flag=True, default=False)
@click.option("--icl", is_flag=True, default=False)
@click.option("--skip", type=click.INT)
@click.option("--template", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--clear", is_flag=True, default=False)
def evaluate_halu_checker_zs(model, infile, outdir, limit, expert, cot, chain, icl, skip, template, clear):

    if clear:
        shutil.rmtree(outdir, ignore_errors=True)

    llm = None
    test_df = pd.read_parquet(infile)

    y_pred = []
    y_true = []

    count = 0

    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        if skip is not None and i < skip: 
            continue
        
        if count == limit:
            break

        ref, example, example_snippet = row["ref"], row["example"], row["example_snippet"]
        
        id = md5hex(ref + example + example_snippet)
                
        jsonld_fn = f"{outdir}/{id}.json"
        Path(jsonld_fn).parent.mkdir(parents=True, exist_ok=True)
        logfile = f"{Path(jsonld_fn).parent}/{Path(jsonld_fn).stem}_factual_pred.json"
        # logger.info(f"Checking {jsonld_fn}...")
        
        result = None
        force_redo = True
        if os.path.exists(logfile) and os.stat(logfile).st_size > 0:

            with open(logfile, "r") as f:
                try:
                    log = json.load(f)
                    result = log["aggregation"] if "aggregation" in log.keys() else log["chunk_0"]["score"]
                    force_redo = False
                except KeyError:
                    logger.warning(f"Could not find 'score' in {logfile}")
                    force_redo = True
        if force_redo:

            if llm is None:
                llm = ModelFactoryLLM.create_model(model)

            document_fn = f"{Path(jsonld_fn).parent}/{Path(jsonld_fn).stem}_doc.txt"
            with open(document_fn,"w") as f:
                f.write(ref)

            jsonld = None
            with open(jsonld_fn, "w") as f:
                jsonld = json.loads(example_snippet)
                
                if jsonld is None:
                    logger.warning(f"{example_snippet} could not be parsed as JSON")
                    continue
                
                json.dump(jsonld, f, ensure_ascii=False)
            result = llm._evaluate_factual_consistency(
                jsonld_fn, document=document_fn, in_context_learning=icl, 
                chain_of_thought=cot, chain_prompt=chain, expert=expert,
                prompt_template=template
            )["pred"]
        
        if result is None:
            print(jsonld_fn)
        
        # Interpolate score to label
        decision_threshold = 0.5 # < 0.5: bias towards positive, > 0.5: bias towards negative
        pred_label = int(result >= decision_threshold)
        true_label = 0 if row["label"] == "negative" else 1
        y_pred.append(pred_label)
        y_true.append(true_label)
        
        if pred_label != true_label:
            if pred_label == 1 and true_label == 0:
                print("False positive", logfile)
            elif pred_label == 0 and true_label == 1:
                print("False negative", logfile)
        
        count += 1

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    cost = 0.0
    if llm:
        stats = llm.get_stats_df()
        cost = stats["estimated_cost"].sum() if not stats.empty else None

    results = { "precision": precision, "recall": recall, "f1-score": f1, "accuracy": accuracy, "avg_cost": cost }
    result_fn = f"{Path(outdir).parent}/{Path(outdir).stem}.json"
    with open(result_fn, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(results)
    
def get_candidates(k,v,e,**kwargs):

    prop_check = kwargs["prop_check"]
    ref_key = kwargs["ref_key"]
    ref_value = kwargs["ref_value"]

    key = f"http://schema.org/{k}"
    expected_types = get_expected_types(key, simplify=True)

    if expected_types is None:
        logger.warning(f"Could not get definition for {key}")
        return None

    # In case of prop_check, skip if non-text
    if prop_check and ("Text" not in expected_types or isinstance(v, dict)):
        return None

    if isinstance(v, dict):
        results = []
        ent_type = e
        for dk, dv in v.items():
            if dk == "@type":
               ent_type = dv 
               continue
            if dk in ["@id", "@context"]: 
                continue
            
            result = get_candidates(dk, dv, ent_type, prop_check=prop_check, ref_key=ref_key, ref_value=ref_value)
            if result is not None:
                results.extend(result)
        if len(results) > 0:
            return (k, results)
    elif isinstance(v, list):
        results = []
        for item in v:
            result = get_candidates(k, item, e, prop_check=prop_check, ref_key=ref_key, ref_value=ref_value)
            if result is not None:
                results.extend(result)
        if len(results) > 0:
            return (k, results)
    
    elif prop_check and get_type(v) == "string":
        return (k, v)

    elif not prop_check:
        ref_expected_type = get_expected_types(ref_key, simplify=True)
        if ref_expected_type and ("Text" in ref_expected_type and "Text" in expected_types):
            return (k, v)
        elif get_type(v) == get_type(ref_value):
            return (k, v)
    return None
    
def get_type(var):
    try: 
        float(var)
        return "number"
    except ValueError: pass
    except TypeError: 
        raise TypeError(f"{var} is of type dict, expecting str!")
                
    if var.lower() in ["true", "false"]:
        return "boolean"
        
    if isinstance(var, str):
        return "string"
    else:
        raise NotImplementedError(f"Unrecognized type for {var}")
    
def generate(prop, example, pv_pair, prop_check ):
    #TODO Why only 20 number props?
    
    prop_canon = f"http://schema.org/{prop}"
    expected_types = get_expected_types(prop_canon, simplify=True)
    if prop_check and not ( "Text" in expected_types ):
        # logger.warning(f"{prop} is not a text property! Skipping...")
        return None, None
    
    key = schema_simplify(URIRef(prop)) if prop.startswith("http://schema.org") else prop
    value = pv_pair[prop]      
    result = None
    explanations = []
            
    if isinstance(value, dict):
        tmp = {}
        for k, v in value.items():
            if k in ["@type", "@id"]: continue
            sub, explanation = generate(k, example, {k: v}, prop_check)
            if sub is not None:
                tmp[k] = sub              
                explanations.extend(explanation)
        result = None if len(tmp) == 0 else tmp
    elif isinstance(value, list):
        tmp = []
        for item in value:
            sub, explanation = generate(key, example, {key: item},prop_check)
            if sub is not None:
                tmp.append(sub)
                explanations.extend(explanation)
        result = None if len(tmp) == 0 else tmp
    else:
        # If prop_check and not text, skip
        if prop_check and get_type(value) != "string": 
            # logger.warning(f"{value} is not Text")
            return None, None
            
        candidates = dict([
            t for t in
            collect_json(
                example, 
                key_filter=lambda k,e: k != key, 
                value_transformer=lambda k,v,e,**kwargs:get_candidates(k,v,e,prop_check=prop_check, ref_key=key, ref_value=value)
            )
            if t is not None
        ])
            
        if len(candidates) == 0:
            # logger.warning(f"Could not find suitable candidates for {prop}")
            return None, None
                
        # Choose candidate with furthest semantic distance
        key_embedding = embed(key)
        candidates_props = list(candidates.keys())
        similarities = {candidate_prop: key_embedding.similarity(embed(candidate_prop)) for candidate_prop in candidates_props}
        best_candidate = min(similarities, key=similarities.get)  
        result = candidates[best_candidate]
    
        explanation = f"expect {prop}, got {best_candidate}"
        explanations.append(explanation)
    return result, explanations

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--explain", is_flag=True, default=False)
@click.option("--limit", type=click.INT)
@click.option("--skip", type=click.INT)
def generate_negative_examples_halu_simple(infile, outfile, explain, limit, skip):     
    in_df = pd.read_parquet(infile)
    index_pairs = list(combinations(in_df.index.values, 2))
    
    records = []
    
    for idx1, idx2 in tqdm(index_pairs):
        row1 = in_df.iloc[idx1, :]
        row2 = in_df.iloc[idx2, :]
        
        ref1, prop1, example1, example_snippet1 = row1["ref"], row1["prop"], row1["example"], row1["example_snippet"]
        ref2, prop2, example2, example_snippet2 = row2["ref"], row2["prop"], row2["example"], row2["example_snippet"]
        
        if ref1 == ref2: continue
        
        example1 = json.loads(example1)
        example2 = json.loads(example2)
        
        if not is_json_disjoint(example1, example2): continue
                
        # Create a negative example for every properties pair in both sample
        records.append({
            "ref": ref2, "prop": prop1, "example": json.dumps(example1, ensure_ascii=False), "example_snippet": example_snippet1
        })
        
        records.append({
            "ref": ref1, "prop": prop2, "example": json.dumps(example2, ensure_ascii=False), "example_snippet": example_snippet2
        })
    
    out_df = pd.DataFrame.from_records(records).drop_duplicates().reset_index(drop=True)
    out_df = out_df.groupby(by="example").sample(random_state=RANDOM_SEED).reset_index(drop=True)
    out_df.to_parquet(outfile)

import mapply

mapply.init(
    n_workers=-1,
    chunk_size=1,
    max_chunks_per_worker=8,
    progressbar=True
)

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--explain", is_flag=True, default=False)
@click.option("--limit", type=click.INT)
@click.option("--skip", type=click.INT)
@click.option("--prop-check", is_flag = True,default = False)
def generate_negative_examples(infile, outfile, explain, limit, skip, prop_check):
    in_df = pd.read_parquet(infile)
    
    def process_row(row):     
        ref, prop, example, example_snippet = row["ref"], row["prop"], row["example"], row["example_snippet"]
        
        json_ex = json.loads(example)

        json_pv_pair = json.loads(example_snippet)    
        key = schema_simplify(URIRef(prop))  
        replacement, explanation = generate(key, json_ex, json_pv_pair, prop_check)
        if replacement is None: return {}
        neg_example = {key: replacement}
        return { "ref": ref,  "prop": prop, "example": json.dumps(example, ensure_ascii=False), "example_snippet": json.dumps(neg_example, ensure_ascii=False), "explain": explanation }

    records = in_df.mapply(process_row, axis=1).to_list()
    # records = in_df.parallel_apply(process_row, axis=1).to_list()
    # records = in_df.progress_apply(process_row, axis=1).to_list()
    out_df = pd.DataFrame.from_records(records).dropna().reset_index(drop=True)
    out_df.to_parquet(outfile)

def train_test_split(infile):
    df = pd.read_parquet(infile)
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    traindir = "data/SchemaExamples/train"

    for i, row in X_train.iterrows():
        ref, examples = row["ref"], row["examples"]
        id = md5hex(f"{ref}{i}")
        
        corpus_fn = f"{traindir}/corpus/{id}.txt"
        Path(corpus_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_fn, "w") as f:
            f.write(ref)
            
        example_fn = f"{traindir}/corpus/baseline/{id}.json"
        Path(example_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(example_fn, "w") as f:
            f.write(examples)

    testdir = "data/SchemaExamples/train" 
    for i, row in X_test.iterrows():
        ref, examples = row["ref"], row["examples"]
        id = md5hex(f"{ref}{i}")
        
        corpus_fn = f"{testdir}/corpus/{id}.txt"
        Path(corpus_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_fn, "w") as f:
            f.write(ref)
            
        example_fn = f"{testdir}/corpus/baseline/{id}.json"
        Path(example_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(example_fn, "w") as f:
            f.write(examples)

if __name__ == "__main__":
    cli()
