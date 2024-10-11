#TODO 
"""
- Add a function to calculate the mean of document size in bytes.
- Add a function to calculate the number of properties per type across baseline markups.
"""

from collections import Counter
import glob
from itertools import chain
import json
import os
from pathlib import Path
from pprint import pprint
import re
import click

import numpy as np
import pandas as pd
from rdflib import ConjunctiveGraph
from tqdm import tqdm

#from ftlangdetect import detect
from langdetect import detect

from utils import collect_json, get_infos, get_type_definition, to_jsonld

@click.group
def cli():
    pass


data_dir = "data/WDC/Pset"
documents = glob.glob(f"{data_dir}/**/*.txt", recursive=True)

def collect_keys(stub):
    results = []
    if isinstance(stub, dict):
        for k, v in stub.items():
            if k in ["@context", "@type", "@id"]:
                continue

            if k is None:
                raise RuntimeError()
            results.append(k)
            results.extend(collect_keys(v))
    elif isinstance(stub, list):
        for v in stub:
            results.extend(collect_keys(v))
    return results

def collect_types(stub):
    results = []
    if isinstance(stub, dict):
        ent_type = stub.get("@type")
        if ent_type:
            if isinstance(ent_type, str):
                ent_type = [ent_type]
            results.append(ent_type)
    elif isinstance(stub, list):
        for v in stub:
            results.extend(collect_types(v))
    return results

@cli.command()
def collect_checker_errors_stats():
    records = []
    for document in tqdm(documents):
        info = {}
        document_id = Path(document).stem
        base_dir = Path(document).parent
    
        if re.search(r"^[a-z0-9]{32}$", document_id) is None:
            continue       

        # Get document main types
        document_main_types = None
        with open(f"{base_dir}/{document_id}_class.json") as f:
            document_main_types = json.load(f)["markup_classes"]

        if document_main_types is None:
            raise RuntimeError(f"Could not find main types for {document_id}")

        for main_types in document_main_types:
            for model in ["baseline", "gpt3", "gpt4"]:
                instance = ""
                for metric in ["factual", "semantic"]:
                    if metric != "":
                        metric = f"_{metric}"

                    info = {}
                    info["document_main_types"] = document_main_types
                    info["document_size_bytes"] = os.stat(document).st_size
                    info["document_id"] = document_id
                    info["document_path"] = document
                    info["model"] = model
                    info["metric"] = metric

                    instance = "_expected" if model == "baseline" else "_pred"

                    log_stem = f"{document_id}_{'_'.join(main_types)}{metric}{instance}.json"
                    log_fn = None

                    if model == "baseline":
                        log_fn = f"{base_dir}/baseline/{log_stem}"
                    elif model == "gpt3":
                        log_fn = f"{base_dir}/GPT_3_Turbo_16K/text2kg_prompt3/{log_stem}"
                    elif model == "gpt4":
                        log_fn = f"{base_dir}/GPT_4_Turbo_Preview/text2kg_prompt3/{log_stem}"
                    
                    with open(log_fn) as f:
                        log = json.load(f)

                        if metric in ["_factual", "_semantic"]:
                            report = log["chunk_0"]
                            if "aggregation" in log.keys():
                                report = log["aggregation"]
                            
                            report.pop("status", None)
                            report.pop("score")

                            props = []
                            responses = []
                            for k, v in report.items():
                                prop = k.split("[TOK_Q_DELIM]")
                                if len(prop) == 1: continue
                                prop = prop[0]
                                if prop is None:
                                    raise RuntimeError()
                                prop_def = get_type_definition(prop=f"http://schema.org/{prop}", exit_on_first=True)
                                if len(prop_def) == 0:
                                    continue

                                response = int(
                                    v["response"] == "TOKPOS"
                                    if isinstance(v, dict)
                                    else v
                                )
                                props.append(prop)
                                responses.append(response)

                            info["property"] = props
                            info["response"] = responses
        
                    records.append(info)
    df = pd.DataFrame.from_records(records)
    df.to_parquet("data/WDC/Pset/checker_errors_stats.parquet")

@cli.command()
def collect_languages():
    results = []
    for document in tqdm(documents):
        with open(document, "r") as f:
            text = f.read()
            lang = detect(text)
            results.append(lang)
    series = pd.Series(results).to_frame("language")
    series.to_csv("languages.csv", index=False)
    print(series.value_counts(normalize=True))

@cli.command()
def collect_shacl_stats():
    msgs = []
    shacl_report_files = glob.glob("data/WDC/Pset/**/baseline/**/*_shacl_*.json", recursive=True)
    for shacl_report_file in shacl_report_files:
        with open(shacl_report_file) as f:
            shacl_report = json.load(f)
            for msg in shacl_report["msgs"].values():
                for m in msg:
                    if "does not conform to one or more shapes" in m:
                        msgs.append("value shape error")
                    elif "is not a property of" in m:
                        print(shacl_report_file)
                        msgs.append("invalid property error")
                    elif "is not a type defined by the schema" in m:
                        msgs.append("invalid type error")
                    else:
                        raise NotImplementedError(f"Unknown error {repr(m)}")
    
    print(Counter(msgs))

if __name__ == "__main__":
    cli()