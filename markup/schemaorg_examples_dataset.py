from collections import OrderedDict
from copy import deepcopy
from io import StringIO
from itertools import chain
import json
import os
from pathlib import Path
from pprint import pprint
import random
import shutil
import textwrap
from bs4 import BeautifulSoup
import click
import pandas as pd

from models.llm import GPT
from rdflib import ConjunctiveGraph, URIRef
from sklearn.model_selection import train_test_split
from utils import _html2txt, collect_json, get_expected_types, jsonld_search_property, md5hex, schema_simplify

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from tqdm import tqdm
tqdm.pandas()

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

import spacy
nlp = spacy.load("en_core_web_md")

RANDOM_SEED = 42

@click.group
def cli():
    pass

@cli.command()
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def create_dataset(outfile):
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
        except: return None

    records = []
    for qres in tqdm(g.query(query)):
        ref = qres.get("ref").toPython()
        prop = qres.get("prop").toPython()
        prop_simple = schema_simplify(URIRef(prop))
        example = load_json(qres.get("jsonld").toPython())        
        example_snippets = jsonld_search_property(example, prop_simple)

        if len(example_snippets) == 0:
            print(f"Cannot find {prop_simple} in {example}")
            continue
        
        for example_snippet in example_snippets:
            if len(collect_json(example_snippet)) == 0: 
                print(f"There is no workable property-value pair in {example_snippet}")
                continue
        
            records.append({ "ref": ref,  "prop": prop, "example": json.dumps(example), "example_snippet": json.dumps(example_snippet) })
        
    df = pd.DataFrame.from_records(records)
    df["ref"] = df["ref"].apply(lambda x: _html2txt(x, force=True))
    df.replace("null", None, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_parquet(outfile)
    

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--expert", is_flag=True, default=False)
@click.option("--cot", is_flag=True, default=False)
@click.option("--icl", is_flag=True, default=False)
@click.option("--limit", type=click.INT)
@click.option("--skip", type=click.INT)
@click.option("--clear", is_flag=True, default=False)
def evaluate_prop_checker_zs(infile, outfile, expert, cot, icl, limit, skip, clear):
    
    tmpdir = ".tmp/prop_checks_zs"
    if clear:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    llm = GPT(model="gpt-3.5-turbo-16k")
    test_df = pd.read_parquet(infile)
    
    y_pred = []
    y_true = test_df["label"].apply(lambda x: 0 if x == "negative" else 1).to_list()
    count = 0
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        
        if skip is not None and i < skip: 
            continue
        
        if count == limit: break
        
        ref, prop, example_snippet = row["ref"], row["prop"], row["example_snippet"]
                
        prop_simple = schema_simplify(URIRef(prop))
        id = md5hex(ref + str(i))
        jsonld_fn = f"{tmpdir}/{id}.json"
        Path(jsonld_fn).parent.mkdir(parents=True, exist_ok=True)
        
        logfile = f"{Path(jsonld_fn).parent}/{Path(jsonld_fn).stem}_semantic_pred.json"
        result = None
        if os.path.exists(logfile) and os.stat(logfile).st_size > 0: 
            with open(logfile, "r") as f:
                try:
                    log = json.load(f)
                    result = int(log["score"])
                except KeyError:
                    raise ValueError(f"Could not find 'score' in {logfile}")
        else:
            jsonld = None
            with open(jsonld_fn, "w") as f:
                jsonld = json.loads(example_snippet)
                if jsonld is None:
                    print(f"{example_snippet} could not be parsed as JSON")
                    continue
                if len(jsonld) == 1 and isinstance(list(jsonld.values())[0], str):
                    jsonld = { f"http://schema.org/{k}": v for k, v in jsonld.items() }
                else:
                    jsonld["@context"] = "http://schema.org/"
                json.dump(jsonld, f)
            result = int(llm._evaluate_semantic_conformance(jsonld_fn, in_context_learning=icl, chain_of_thought=cot, expert=expert)["pred"])
        y_pred.append(result)
        
        count += 1
    
    print(list(set(y_pred)), list(set(y_true)))

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    results = { "precision": precision, "recall": recall, "f1-score": f1, "accuracy": accuracy }
    with open(outfile, "w") as f:
        json.dump(results, f)
    print(results)

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--limit", type=click.INT)
@click.option("--expert", is_flag=True, default=False)
@click.option("--cot", is_flag=True, default=False)
@click.option("--icl", is_flag=True, default=False)
@click.option("--breakdown", is_flag=True, default=False)
@click.option("--clear", is_flag=True, default=False)
def evaluate_halu_checker_zs(infile, outfile, limit, expert, cot, icl, breakdown, clear):

    tmpdir = ".tmp/halu_checks_zs"
    if clear:
        shutil.rmtree(tmpdir, ignore_errors=True)

    llm = GPT(model="gpt-3.5-turbo-16k")
    test_df = pd.read_parquet(infile)

    y_pred = []
    y_true = test_df["label"].apply(lambda x: 0 if x == "negative" else 1).to_list()
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if i == limit:
            break

        ref, prop, example_snippet = row["ref"], row["prop"], row["example_snippet"]

        prop_simple = schema_simplify(URIRef(prop))
        id = md5hex(ref + str(i))
        jsonld_fn = f"{tmpdir}/{id}.json"
        Path(jsonld_fn).parent.mkdir(parents=True, exist_ok=True)

        logfile = f"{Path(jsonld_fn).parent}/{Path(jsonld_fn).stem}_semantic_pred.json"
        result = None
        if os.path.exists(logfile) and os.stat(logfile).st_size > 0:
            with open(logfile, "r") as f:
                try:
                    log = json.load(f)
                    result = int(log["valids"])
                except KeyError:
                    raise ValueError(f"Could not find 'valids' in {logfile}")
        else:
            document_fn = f"{Path(jsonld_fn).parent}/{Path(jsonld_fn).stem}_doc.txt"
            with open(document_fn,"w") as f:
                f.write(ref)

            jsonld = None
            with open(jsonld_fn, "w") as f:
                jsonld = json.loads(example_snippet)
                if jsonld is None:
                    print(f"{example_snippet} could not be parsed as JSON")
                    continue
                if len(jsonld) == 1 and isinstance(list(jsonld.values())[0], str):
                    jsonld = { f"http://schema.org/{k}": v for k, v in jsonld.items() }
                else:
                    jsonld["@context"] = "http://schema.org/"
                json.dump(jsonld, f)
            result = int(llm._evaluate_factual_consistency(jsonld_fn, document=document_fn, in_context_learning=icl, breakdown=breakdown, chain_of_thought=cot, expert=expert)["pred"])
        y_pred.append(result)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    results = { "precision": precision, "recall": recall, "f1-score": f1, "accuracy": accuracy }
    with open(outfile, "w") as f:
        json.dump(results, f)
    print(results)

def camel_case_split(s):
    words = [[s[0]]]
 
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
 
    return [''.join(word) for word in words]

def embed(word):
    return nlp(" ".join(camel_case_split(word)))
    
def get_candidates(k,v,e,**kwargs):

    prop_check = kwargs["prop_check"]
    ref_key = kwargs["ref_key"]
    ref_value = kwargs["ref_value"]

    key = f"http://schema.org/{k}"
    expected_types = get_expected_types(key, simplify=True)

    if expected_types is None:
        print(f"Could not get definition for {key}")
        return None

    # In case of prop_check, skip if non-text
    if prop_check and ("Text" not in expected_types or isinstance(v, dict)):
        return None

    if isinstance(v, list):
        results = []
        for item in v:
            result = get_candidates(k, item, e, prop_check=prop_check, ref_key=ref_key, ref_value=ref_value)
            if result is not None:
                results.extend(result)
        if len(results) > 0:
            return (k, results)
    
    elif prop_check and is_text(v):
        return (k, v)

    elif not prop_check:
        ref_expected_type = get_expected_types(ref_key, simplify=True)
        if ref_expected_type and ("Text" in ref_expected_type and "Text" in expected_types):
            return (k, v)
        elif not is_text(v) and not is_text(ref_value):
            return (k, v)
    return None
    
def is_text(var):
    try: 
        float(var)
        return False
    except ValueError: pass
                
    if var.lower() in ["true", "false"]:
        return False
        
    return isinstance(var, str)
    
def generate(prop, example, pv_pair, prop_check ):
    
    prop_canon = f"http://schema.org/{prop}"
    expected_types = get_expected_types(prop_canon, simplify=True)
    if prop_check and not ( "Text" in expected_types ):
        # print(f"{prop} is not a text property! Skipping...")
        return None, None
    
    key = schema_simplify(URIRef(prop)) if prop.startswith("http://schema.org") else prop
    value = pv_pair[prop]      
    result = deepcopy(value)
    explanations = []
            
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ["@type", "@id"]: continue
            sub, explanation = generate(k, example, {k: v}, prop_check)
            if sub is not None:
                result[k] = sub              
                explanations.extend(explanation)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            sub, explanation = generate(key, example, {key: item},prop_check)
            if sub is not None:
                result[i] = sub
                explanations.extend(explanation)
    else:
        # If prop_check and not text, skip
        if prop_check and not is_text(value): 
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
            # print(f"Could not find suitable candidates for {prop}")
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
        return { "ref": ref,  "prop": prop, "example": json.dumps(example), "example_snippet": json.dumps(neg_example), "explain": explanation }

    records = in_df.parallel_apply(process_row, axis=1).to_list()
    print(records)    
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
