import glob
import pandas as pd
import os
from itertools import product
import json

import sys
sys.path.append(os.path.join(os.getcwd(), "markup"))

from utils import filter_json

print(config)

DATA_DIR = "data/WDC/Pset"
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
MODELS = ["GPT"] if MODELS is None else MODELS.split(",")

print(MODELS)

METRICS = config.get("metrics")
METRICS = ["shacl", "factual", "semantic", "coverage"] if METRICS is None else METRICS.split(",")
# ruleorder: generate_baseline > generate_markup > evaluate_markup > assemble

def get_eval_results(wildcards):
    gw = glob_wildcards("{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld")
    
    def combinator(data_dir, sample_feature, stratum, model, document_id, document_classes):
        for model_u in model:
            for data_dir_u, sample_feature_u, stratum_u, document_id_u, document_classes_u in zip(data_dir, sample_feature, stratum, document_id, document_classes):
                if SAMPLE_FEATURE and sample_feature_u[1] not in SAMPLE_FEATURE: continue
                if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
                yield (data_dir_u, sample_feature_u, stratum_u, model_u, document_id_u, document_classes_u)

    return expand(
        "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_coverage.csv",
        combinator,
        data_dir=gw.data_dir,
        sample_feature=gw.sample_feature,
        stratum=gw.stratum,
        model=MODELS,
        document_id=gw.document_id,
        document_classes=gw.document_classes
    )

def do_filter_json_shacl(infile, logfile, outfile):
    with open(infile, "r") as in_fs, open(logfile, "r") as log_fs, open(outfile, "w") as out_fs:
        markup = json.load(in_fs)
        log = json.load(log_fs)

        for prop, msg in log["msgs"].items():
            if prop.startswith("schema1:"):
                prop = prop.replace("schema1:", "")
                markup = filter_json(markup, prop)
            elif prop.startswith("http://schema.org/"):
                prop = prop.replace("http://schema.org/", "")
                markup = filter_json(markup, "@type", value=prop)

        json.dump(markup, out_fs, ensure_ascii=False)

def do_filter_json_factual(infile, logfile, outfile):
    with open(infile, "r") as in_fs, open(logfile, "r") as log_fs, open(outfile, "w") as out_fs:
        markup = json.load(in_fs)
        log = json.load(log_fs)

        ran_with_map_reduce = "aggregation" in log.keys() and len(log) > 1

        info = log["aggregation"] if ran_with_map_reduce else log["chunk_0"]

        for prop, res in info.items():
            if prop in ["status", "score"]: continue
            is_res_negative = res == False if ran_with_map_reduce else "TOKNEG" in res.get("response") 
            if isinstance(res, dict) and "TOKNEG" in res.get("response"):
                markup = filter_json(markup, prop)
        json.dump(markup, out_fs, ensure_ascii=False)

rule all:
    input: 
        get_eval_results

rule evaluate_coverage:
    input: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_coverage.csv"
    params:
        # predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld",
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_pred_filtered.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"
        
        shell(f"python markup/markup.py validate-one {params.predicted} {wildcards.model} coverage --expected {params.baseline} --document {params.document} --outfile {output} --basename {basename} {target_classes_args}")

rule evaluate_semantic:
    input: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_pred_filtered.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_expected_filtered.jsonld", 

        predicted_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_pred.json",
        baseline_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_expected.json",

        predicted_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_pred_filtered.jsonld",
        baseline_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"
        
        shell(f"python markup/markup.py validate-one {params.predicted} {wildcards.model} semantic --expected {params.baseline} --document {params.document} --outfile {output} --basename {basename} {target_classes_args}")
        shell(f"cp {params.predicted} {params.predicted_filtered}")
        shell(f"cp {params.baseline} {params.baseline_filtered}")
        
        do_filter_json_factual(params.predicted, params.predicted_log, params.predicted_filtered)
        do_filter_json_factual(params.baseline, params.baseline_log, params.baseline_filtered)

rule evaluate_factual:
    input: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl.csv",
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_pred_filtered.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_expected_filtered.jsonld", 

        predicted_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_pred.json",
        baseline_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_expected.json",

        predicted_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_pred_filtered.jsonld",
        baseline_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"

        shell(f"python markup/markup.py validate-one {params.predicted} {wildcards.model} factual --expected {params.baseline} --document {params.document} --outfile {output} --basename {basename} {target_classes_args}")
        shell(f"cp {params.predicted} {params.predicted_filtered}")
        shell(f"cp {params.baseline} {params.baseline_filtered}")
        
        do_filter_json_factual(params.predicted, params.predicted_log, params.predicted_filtered)
        do_filter_json_factual(params.baseline, params.baseline_log, params.baseline_filtered)

rule evaluate_shacl: 
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld", 

        predicted_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_pred.json",
        baseline_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_expected.json",

        predicted_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_pred_filtered.jsonld",
        baseline_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"

        shell(f"python markup/markup.py validate-one {params.predicted} {wildcards.model} shacl --expected {params.baseline} --document {params.document} --outfile {output} {target_classes_args}")
        shell(f"cp {params.predicted} {params.predicted_filtered}")
        shell(f"cp {params.baseline} {params.baseline_filtered}")
        
        do_filter_json_shacl(params.predicted, params.predicted_log, params.predicted_filtered)
        do_filter_json_shacl(params.baseline, params.baseline_log, params.baseline_filtered)
            

