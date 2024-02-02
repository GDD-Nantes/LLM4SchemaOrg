import glob
import pandas as pd
import os
from itertools import product
import json

DATA_DIR = "data/WDC/Pset"
SAMPLE_FEATURE = ["count_sum", "pset_length"]

# SAMPLING
N_STRATA = 3 # Number of strata for stratified sampling
STRATUM_SAMPLE_SIZE = 30
MARGIN_OF_ERROR = 0.05

# LLM
MODELS = ["Mistral_7B_Instruct"]
METRICS = ["shacl", "factual", "semantic", "coverage"]

# ruleorder: generate_baseline > generate_markup > evaluate_markup > assemble
def get_generated_markups(wildcards):
    gw = glob_wildcards("{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld")
    
    def combinator(data_dir, sample_feature, stratum, model, document_id, document_classes):
        # for model_u in product(model): 
        for model_u in model:
            for data_dir_u, sample_feature_u, stratum_u, document_id_u, document_classes_u in zip(data_dir, sample_feature, stratum, document_id, document_classes):
                yield (data_dir_u, sample_feature_u, stratum_u, model_u, document_id_u, document_classes_u)

    return expand(
        "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}.jsonld",
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
        get_generated_markups

rule generate_markup:
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id}_{document_classes}.jsonld"
    params:
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt",
        target_class_fn = "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}_class.json"
    run: 
        print(wildcards.document_id)
        target_classes = str(wildcards.document_classes).split("_")
        subtarget_classes = None
        with open(params.target_class_fn, "r") as f:
            target_class_infos = json.load(f)
            subtarget_classes = [ f"http://schema.org/{u}" for u in target_class_infos["pset_classes"] ]
                            
            if sorted(target_classes) == sorted(subtarget_classes):
                subtarget_classes = None

        target_classes = [ f"http://schema.org/{u}" for u in target_classes ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        subtarget_classes_args = " ".join([ f"--subtarget-class {tc}" for tc in subtarget_classes ]) if subtarget_classes else ""
        # infile, outfile, model, explain, target_class, subtarget_class
        shell(f"python markup/markup.py generate-markup-one {params.document} {output} {wildcards.model} {target_classes_args} {subtarget_classes_args}")
