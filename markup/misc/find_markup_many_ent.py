from glob import glob
import json
import re

import pandas as pd
from tqdm import tqdm

records = []

for logfn in tqdm(glob("data/WDC/Pset/**/*.jsonld", recursive=True)):
    if re.search(r"(shacl|factual|semantic|compression|jaccardms|chunk)", logfn) is not None:
        continue

    with open(logfn, "r") as f:

        info = {"markup_count": 0}
        markup = json.load(f)
    
        if isinstance(markup, dict):
            info["markup_count"] += 1
        elif isinstance(markup, list):
            info["markup_count"] += len(markup)
        else:
            raise RuntimeError(f"Unknown type {type(markup)}")
        
        if info["markup_count"] > 1:
            print(f"{logfn} has {info['markup_count']} markups")

        records.append(info)

df = pd.DataFrame.from_records(records)
print(df["markup_count"].value_counts())

