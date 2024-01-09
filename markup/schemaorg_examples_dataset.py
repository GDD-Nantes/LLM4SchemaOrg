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
from models.llm import ChatGPT
from rdflib import ConjunctiveGraph, URIRef
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import backoff
from openai.error import APIConnectionError, ServiceUnavailableError, Timeout, RateLimitError
from utils import _html2txt, collect_json, get_schema_example, get_type_definition, jsonld_search_property, md5hex, schema_simplify

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
        example = load_json(qres.get("jsonld").toPython())        
        example_snippet = jsonld_search_property(example, schema_simplify(URIRef(prop)))
        
        # if example_snippet is None:
        #     print(prop)
        #     pprint(example)
        #     raise RuntimeError()
        
        records.append({ "ref": ref,  "prop": prop, "example": json.dumps(example), "example_snippet": json.dumps(example_snippet) })
        
    df = pd.DataFrame.from_records(records)
    df["ref"] = df["ref"].apply(lambda x: _html2txt(x, force=True))
    # df.replace("null", None, inplace=True)
    # df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_feather(outfile)
    

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--limit", type=click.INT)
@click.option("--expert", is_flag=True, default=False)
@click.option("--cot", is_flag=True, default=False)
@click.option("--icl", is_flag=True, default=False)
@click.option("--breakdown", is_flag=True, default=False)
@click.option("--clear", is_flag=True, default=False)
def evaluate_prop_checker_zs(infile, outfile, limit, expert, cot, icl, breakdown, clear):
    
    tmpdir = ".tmp/prop_checks_zs"
    if clear:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    llm = ChatGPT(model="gpt-3.5-turbo-16k")
    test_df = pd.read_feather(infile)
    
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
            result = int(llm._evaluate_semantic_conformance(jsonld_fn, in_context_learning=icl, breakdown=breakdown, chain_of_thought=cot, expert=expert)["pred"])
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
    
def get_expected_types(k,v,e):
    key = f"http://schema.org/{k}"
    key_def = get_type_definition(prop=key, simplify=True, include_expected_types=True)
    if len(key_def) == 0:
        print(f"Could not get definition for {key}")
        return None
    expected_types = key_def.get(key)["expected_types"]
    if "Text" not in expected_types or isinstance(v, dict):
        return None
              
    if isinstance(v, list):
        results = []
        for item in v:
            result = get_expected_types(k, item, e)
            if result is not None:
                results.extend(result)
        return (k, results)
    elif isinstance(v, str):
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
        
    
def notChildOf(child, parents):
    if not isinstance(parents, list):
        parents = [parents]
    children = [] 
    for p_child in children:
        children.extend(get_type_definition(schema_type_url=p_child, parents=False, simplify=True))
    return child not in children
    
def generate(prop, example, pv_pair):
    
    prop_canon = f"http://schema.org/{prop}"
    expected_types = get_type_definition(prop=prop_canon, simplify=True, include_expected_types=True).get(prop_canon)["expected_types"]
    if not ( "Text" in expected_types ):
        print(f"{prop} is not a text property! Skipping...")
        return None, None
    
    key = schema_simplify(URIRef(prop)) if prop.startswith("http://schema.org") else prop
    value = pv_pair[prop]      
    result = deepcopy(value)
    explanations = []
            
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ["@type", "@id"]: continue
            sub, explanation = generate(k, example, {k: v})
            if sub is not None:
                #TODO make a combination of all possible subs
                result[k] = sub              
                explanations.extend(explanation)
        return result, explanations
    elif isinstance(value, list):
        for i, item in enumerate(value):
            sub, explanation = generate(key, example, {key: item})
            if sub is not None:
                #TODO make a combination of all possible subs
                result[i] = sub
                explanations.extend(explanation)
        return result, explanations
    elif is_text(value):                         
        # context = jsonld_search_property(example, key, parent=True)
        # prop_type = f"http://schema.org/{context['@type']}"
            
        candidates = dict([
            t for t in
            # collect_json(example, key_filter=lambda k,e: notChildOf(k, prop_type), value_transformer=get_expected_types)
            collect_json(example, key_filter=lambda k,e: k != key, value_transformer=get_expected_types)
            if t is not None
        ])
            
        if len(candidates) == 0:
            print(f"Could not find suitable candidates for {prop}")
            return None, None
            
        #TODO random.seed(a=?)
        reason, replacement = random.choice(list(candidates.items()))
        explanation = f"expect {prop}, got {reason}"
        return replacement, [explanation]
                
    return None, None

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--explain", is_flag=True, default=False)
@click.option("--limit", type=click.INT)
@click.option("--skip", type=click.INT)
# @backoff.on_exception(backoff.expo, (APIConnectionError, ServiceUnavailableError, Timeout, RateLimitError))
def generate_negative_examples(infile, outfile, explain, limit, skip):
    llm = ChatGPT(model="gpt-3.5-turbo-16k")
    in_df = pd.read_feather(infile)
    records = []     
    count = 0
        
    for i, row in tqdm(in_df.iterrows(), total=len(in_df)):
        
        if skip is not None and i < skip: 
            continue
                      
        if count == limit: break
                
        ref, prop, example, example_snippet = row["ref"], row["prop"], row["example"], row["example_snippet"]
        
        json_ex = json.loads(example) 
        json_pv_pair = json.loads(example_snippet)    
        key = schema_simplify(URIRef(prop))  
     
        replacement, explanation = generate(key, json_ex, json_pv_pair)      
        
        if replacement is None: continue
        
        neg_example = json.dumps({key: replacement})
                        
        explanation = "\n".join(explanation)
        records.append({ "ref": ref, "prop": prop, "example": example, "example_snippet": neg_example, "explain": explanation })
        
        count += 1
    
    out_df = pd.DataFrame.from_records(records)
    out_df.to_feather(outfile)
    out_df.to_excel(f"{Path(outfile).parent}/train.xlsx", "Negatives", engine="xlsxwriter")

                

def train_test_split(infile):
    df = pd.read_feather(infile)
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
