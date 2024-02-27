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
import re
from collections_extended import setlist

import pandas as pd
from tqdm import tqdm

from utils import collect_json, to_jsonld


data_dir = "data/WDC/Pset"
documents = glob.glob(f"{data_dir}/**/*.txt", recursive=True)

records = []
for document in tqdm(documents):
    info = {}
    info["document_size_bytes"] = os.stat(document).st_size
    document_id = Path(document).stem
    print(document_id)

    if re.search(r"^[a-z0-9]{32}$", document_id) is None:
        continue    

    base_dir = Path(document).parent
    
    # Get document main types
    with open(f"{base_dir}/{document_id}_class.json") as f:
        info["document_main_types"] = json.load(f)["markup_classes"]
    
    # Get document markup
    if len(info["document_main_types"]) == 0:
        continue

    for main_types in info["document_main_types"]:
        baseline_markup_fn = f"{base_dir}/baseline/{document_id}_{'_'.join(main_types)}.jsonld"
        document_markup = to_jsonld(baseline_markup_fn)
        document_markup_triplets = collect_json(document_markup, value_transformer=lambda k,v,e: (k,v,e))
        for k,v,e in document_markup_triplets:
            if e:
                if "document_sub_types" not in info:
                    info["document_sub_types"] = []
                info["document_sub_types"].append(e)
            
            if "document_props" not in info:
                info["document_props"] = []
            info["document_props"].append(k)
    
    records.append(info)

df = pd.DataFrame.from_records(records)
df.to_parquet("stats.parquet")
        
     