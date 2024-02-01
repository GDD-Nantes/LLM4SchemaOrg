import glob
import pandas as pd
import os
from itertools import product

DATA_DIR = "data/WDC/Pset"
SAMPLE_FEATURE = ["count_sum", "pset_length"]

# SAMPLING
N_STRATA = 3 # Number of strata for stratified sampling
STRATUM_SAMPLE_SIZE = 30
MARGIN_OF_ERROR = 0.05

# LLM
MODELS = ["GPT"]
METRICS = ["shacl", "factual", "semantic", "coverage"]

# ruleorder: generate_baseline > generate_markup > evaluate_markup > assemble

def get_eval_results(wildcards):
    gw = glob_wildcards("{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z_]+}.jsonld")
    
    def combinator(data_dir, sample_feature, stratum, model, document_id, document_classes):
        # for model_u in product(model): 
        for model_u in model:
            for data_dir_u, sample_feature_u, stratum_u, document_id_u, document_classes_u in zip(data_dir, sample_feature, stratum, document_id, document_classes):
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

rule all:
    input: 
        get_eval_results

rule evaluate_coverage:
    input:  
        factual="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_factual.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_coverage.csv"
    params:
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id}_{document_classes}.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        shell(f"python markup/markup.py validate-one {params.predicted} {wildcards.model} coverage {params.baseline} {params.document} --outfile {output} {target_classes_args}")

rule evaluate_factual:
    input:   
        semantic="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_semantic.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_factual.csv"
    params:
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id}_{document_classes}.jsonld",
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    shell: "python markup/markup.py validate-one {params.predicted} {wildcards.model} factual {params.baseline} {params.document} --outfile {output}"

rule evaluate_semantic:
    input: 
        shacl="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_shacl.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_semantic.csv"
    params:
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id}_{document_classes}.jsonld",  
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    shell: "python markup/markup.py validate-one {params.predicted} {wildcards.model} semantic {params.baseline} {params.document} --outfile {output}"

rule evaluate_shacl: 
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}_shacl.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id}_{document_classes}.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt"
    shell: "python markup/markup.py validate-one {params.predicted} {wildcards.model} shacl {params.baseline} {params.document} --outfile {output}"
