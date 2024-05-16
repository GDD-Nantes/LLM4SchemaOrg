import glob
import pandas as pd
import os
from itertools import product
import json

import sys
sys.path.append(os.path.join(os.getcwd(), "markup"))

from utils import filter_json

print(config)

DATA_DIR = config["data_dir"] # data/WDC/Pset or data/WDC/GenPromptXP
SAMPLE_FEATURE = config.get("sample_feature")
SAMPLE_FEATURE = ["pset_length", "count_sum"] if SAMPLE_FEATURE is None else SAMPLE_FEATURE.split(",")

DOCUMENT = config.get("document_id")
if DOCUMENT is not None:
    DOCUMENT = DOCUMENT.split(",")

TARGET_STRATA = config.get("stratum")
if TARGET_STRATA is not None:
    TARGET_STRATA = TARGET_STRATA.split(",")

# SAMPLING
N_STRATA = 3 # Number of strata for stratified sampling
STRATUM_SAMPLE_SIZE = 30
MARGIN_OF_ERROR = 0.05

# LLM
MODELS = config.get("models")
MODELS = ["GPT_3_Turbo_16K", "GPT_4_Turbo_Preview"] if MODELS is None else MODELS.split(",")

METRICS = config.get("metrics")
METRICS = ["shacl", "factual", "semantic", "jaccardms"] if METRICS is None else METRICS.split(",")
# METRICS = METRICS + [ 
#     "raw_compression" if metric == "jaccardms" else "compression"
#     if metric == "semantic" else metric + "_compression"
#     for metric in METRICS
# ]

# PROMPT TEMPLATES
PROMPT_TEMPLATE_DIR = "prompts/generation"
PROMPT_VERSIONS = config.get("prompt_template")
PROMPT_VERSIONS = [ Path(template_file).stem for template_file in os.listdir(PROMPT_TEMPLATE_DIR) ] if PROMPT_VERSIONS is None else PROMPT_VERSIONS.split(",")

def get_model_results(wildcards):
    gw = glob_wildcards("{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,([A-Z]+[a-z]+)+(_([A-Z]+[a-z]+)+)*}_{metric,[a-z]+(_[a-z]+)?}.csv")
    # print(gw)
    def combinator(data_dir, sample_feature, stratum, model, prompt_ver, document_id, document_classes, metric):
        for data_dir_u, sample_feature_u, stratum_u, model_u, prompt_ver_u, document_id_u, document_classes_u, metric_u in zip(data_dir, sample_feature, stratum, model, prompt_ver, document_id, document_classes, metric):

            if model_u[1] not in MODELS: continue
            if data_dir_u[1] != DATA_DIR: continue
            if sample_feature_u[1] not in SAMPLE_FEATURE: continue
            if int(stratum_u[1]) not in range(N_STRATA): continue
            if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
            if metric_u[1] not in METRICS: continue
            if prompt_ver_u[1] not in PROMPT_VERSIONS: continue

            yield (data_dir_u, sample_feature_u, stratum_u, model_u, prompt_ver_u, document_id_u, document_classes_u, metric_u)

    result = expand(
        "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id}_{document_classes}_{metric}.csv",
        combinator,
        data_dir=gw.data_dir,
        sample_feature=gw.sample_feature,
        stratum=gw.stratum,
        model=gw.model,
        prompt_ver=gw.prompt_ver,
        document_id=gw.document_id,
        document_classes=gw.document_classes,
        metric=gw.metric,
    )

    return result

def merge_results(fns):
    
        
    return result

rule all:
    input: 
        expand(
            "{data_dir}/results.csv",
            data_dir=[DATA_DIR]
        )

rule assemble_model:
    input:
        get_model_results

    output: "{data_dir}/results.csv"
    run:
        dfs = []
        for fn in input:
            df = pd.read_csv(fn)
            match = re.search(rf"{DATA_DIR}/(\w+)/(stratum_\d+)/corpus/(\w+)/(\w+)/([a-z0-9]+)_(([A-Z]+[a-z]+)+(_([A-Z]+[a-z]+)+)*)_([a-z]+(_[a-z]+)?)\.csv", fn)
            df["sample_feature"] = match.group(1)
            df["stratum"] = match.group(2)
            df["approach"] = match.group(3)
            df["prompt_ver"] = match.group(4)
            df["document_id"] = match.group(5)
            df["document_classes"] = match.group(6)
            df["metric"] = match.group(10)
            dfs.append(df)
    
        result = pd.concat(dfs)
        result.to_csv(str(output), index=False)
        