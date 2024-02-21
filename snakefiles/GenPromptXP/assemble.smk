import glob
import pandas as pd
import os
from itertools import product
import json

import sys
sys.path.append(os.path.join(os.getcwd(), "markup"))

from utils import filter_json

print(config)

DATA_DIR = "data/WDC/GenPromptXP"
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
MODELS = ["GPT", "Mistral_7B_Instruct", "Mixtral_8x7B_Instruct"] if MODELS is None else MODELS.split(",")

METRICS = config.get("metrics")
METRICS = ["shacl", "factual", "semantic", "jaccardms"] if METRICS is None else METRICS.split(",")

# PROMPT TEMPLATES
PROMPT_TEMPLATE_DIR = "prompts/generation"
PROMPT_VERSIONS = config.get("prompt_template")
PROMPT_VERSIONS = [ Path(template_file).stem for template_file in os.listdir(PROMPT_TEMPLATE_DIR) ] if PROMPT_VERSIONS is None else PROMPT_VERSIONS.split(",")

def get_feature_results(wildcards):
    gw = glob_wildcards(f"{DATA_DIR}/{{sample_feature}}/stratum_{{stratum}}/corpus/baseline/{{document_id,[a-z0-9]+}}_{{document_classes,([A-Z][a-z]+)(_[A-Z][a-z]+)*}}.jsonld")
    def combinator(data_dir, sample_feature, model):
        for data_dir_u, model_u in product(data_dir, model):
            for sample_feature_u in sample_feature:
                if SAMPLE_FEATURE and sample_feature_u[1] not in SAMPLE_FEATURE: continue
                if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
                yield (data_dir_u, sample_feature_u, model_u)

    results = expand(
        "{data_dir}/{sample_feature}/{model}.csv",
        combinator,
        data_dir=DATA_DIR,
        sample_feature=gw.sample_feature,
        model=MODELS
    )

    return results

def get_strata_results(wildcards):
    pattern = f"{wildcards.data_dir}/{wildcards.sample_feature}/stratum_{{stratum,[0-9]+}}/corpus/baseline/{{document_id,[a-z0-9]+}}_{{document_classes,([A-Z][a-z]+)(_[A-Z][a-z]+)*}}.jsonld"
    gw = glob_wildcards(pattern)

    def combinator(stratum, model):
        for model_u in model:
            for stratum_u in stratum:
                if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
                yield (stratum_u, model_u)

    results = expand(
        "{{data_dir}}/{{sample_feature}}/stratum_{stratum}/corpus/{model}.csv",
        combinator,
        stratum=range(N_STRATA),
        model=MODELS
    )

    return results

def get_model_results(wildcards):
    pattern = f"{wildcards.data_dir}/{wildcards.sample_feature}/stratum_{wildcards.stratum}/corpus/baseline/{{document_id,[a-z0-9]+}}_{{document_classes,([A-Z][a-z]+)(_[A-Z][a-z]+)*}}.jsonld"
    gw = glob_wildcards(pattern)

    def combinator(document_id, document_classes, metric, prompt_ver):
        for metric_u, prompt_ver_u in product(metric, prompt_ver):
            for document_id_u, document_classes_u in zip(document_id, document_classes):
                if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
                yield (document_id_u, document_classes_u, metric_u, prompt_ver_u)

    return expand(
        f"{wildcards.data_dir}/{wildcards.sample_feature}/stratum_{wildcards.stratum}/corpus/{wildcards.model}/{{prompt_ver}}/{{document_id}}_{{document_classes}}_{{metric}}.csv",
        combinator,
        document_id=gw.document_id,
        document_classes=gw.document_classes,
        metric=METRICS,
        prompt_ver=PROMPT_VERSIONS
    )

def merge_results(fns, add_column: dict = {}, add_filename=False):
    dfs = []
    for fn in fns:
        df = pd.read_csv(fn)
        if add_filename:
            df["id"] = fn
        dfs.append(df)
    result = pd.concat(dfs)
    
    for col, value in add_column.items():
        result[col] = value
    
    return result

rule all:
    input: "results.csv"

rule assemble_model:
    input:
        expand(
            "{data_dir}/{sample_feature}/{model}.csv",
            data_dir=DATA_DIR,
            sample_feature=SAMPLE_FEATURE,
            model=MODELS
        )

    output: "results.csv"
    run:
        df = merge_results(input)
        df.to_csv(str(output), index=False)

rule assemble_feature:
    input: 
        expand(
            "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}.csv",
            data_dir=DATA_DIR,
            sample_feature=SAMPLE_FEATURE,
            stratum=range(N_STRATA),
            model=MODELS
        )
    
    output: "{data_dir}/{sample_feature}/{model}.csv"
    run:
        df = merge_results(input, add_column={"feature": wildcards.sample_feature})
        df.to_csv(str(output), index=False)

rule assemble_stratum:
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}.csv"
    run:
        model_results = get_model_results(wildcards)
        print(model_results)
        df = merge_results(model_results, add_column={"stratum": wildcards.stratum}, add_filename=True)
        df.to_csv(str(output), index=False)
        