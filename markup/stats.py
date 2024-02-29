#TODO 
"""
- Add a function to calculate the mean of document size in bytes.
- Add a function to calculate the number of properties per type across baseline markups.
"""

import glob
from itertools import chain
import json
import os
from pathlib import Path
from pprint import pprint
import re
from collections_extended import setlist

import pandas as pd
from tqdm import tqdm

from utils import collect_json, to_jsonld


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

metrics = ["", "shacl", "factual", "semantic"]
metrics_names = ["input", "shacl", "factual", "semantic"]

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
        for metric, metric_name in zip(metrics, metrics_names):
            if metric != "":
                metric = f"_{metric}"

            instance = ""
            
                        
            for model in ["baseline", "gpt3", "gpt4"]:
                
                info = {}
                info["document_main_types"] = document_main_types
                info["document_size_bytes"] = os.stat(document).st_size
                info["document_id"] = document_id
                info["document_path"] = document
                info["model"] = model
                info["metric"] = metric_name

                if metric_name != "input":
                    instance = "_expected_filtered" if model == "baseline" else "_pred_filtered"

                markup_fn = None
                if model == "baseline":
                    markup_fn = f"{base_dir}/baseline/{document_id}_{'_'.join(main_types)}{metric}{instance}.jsonld"
                elif model == "gpt3":
                    markup_fn = f"{base_dir}/GPT_3_Turbo_16K/text2kg_prompt3/{document_id}_{'_'.join(main_types)}{metric}{instance}.jsonld"
                elif model == "gpt4":
                    markup_fn = f"{base_dir}/GPT_4_Turbo_Preview/text2kg_prompt3/{document_id}_{'_'.join(main_types)}{metric}{instance}.jsonld"

                markup = to_jsonld(markup_fn, simplify=True)
                types = collect_types(markup)

                if "document_sub_types" not in info:
                    info["document_sub_types"] = []
                info["document_sub_types"].extend(types)

                properties = collect_keys(markup)
                if "document_props" not in info:
                    info["document_props"] = []
                info["document_props"].extend(properties)
        
                records.append(info)

df = pd.DataFrame.from_records(records)
print(df)
# df.astype(str).to_csv("test.csv")
df.to_parquet("stats.parquet")