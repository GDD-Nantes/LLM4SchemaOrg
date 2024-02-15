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
MODELS = ["GPT_3_Turbo_16K", "GPT_4_32K", "Mixtral_8x7B_Instruct"] if MODELS is None else MODELS.split(",")

print(MODELS)

METRICS = config.get("metrics")
METRICS = ["shacl", "factual", "semantic", "jaccardms"] if METRICS is None else METRICS.split(",")

# PROMPT TEMPLATES
PROMPT_TEMPLATE_DIR = "prompts/generation"
PROMPT_VERSIONS = config.get("prompt_template")
PROMPT_VERSIONS = [ Path(template_file).stem for template_file in os.listdir(PROMPT_TEMPLATE_DIR) ] if PROMPT_VERSIONS is None else PROMPT_VERSIONS.split(",")

def add_column_and_export(file, add_columns):
    df = pd.read_csv(file)
    for k, v in add_columns.items():
        df[k] = v
    
    df.to_csv(file, index=False)

def get_eval_results(wildcards):
    gw = glob_wildcards(f"{DATA_DIR}/{{sample_feature}}/stratum_{{stratum}}/corpus/baseline/{{document_id,[a-z0-9]+}}_{{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}}.jsonld")
    
    def combinator(data_dir, sample_feature, stratum, model, prompt_ver, document_id, document_classes):
        for data_dir_u, model_u, prompt_ver_u in product(data_dir, model, prompt_ver):
            if PROMPT_VERSIONS and prompt_ver_u[1] not in PROMPT_VERSIONS: continue
            for sample_feature_u, stratum_u, document_id_u, document_classes_u in zip(sample_feature, stratum, document_id, document_classes):
                if SAMPLE_FEATURE and sample_feature_u[1] not in SAMPLE_FEATURE: continue
                if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
                yield (data_dir_u, sample_feature_u, stratum_u, model_u, prompt_ver_u, document_id_u, document_classes_u)

    return expand(
        "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id}_{document_classes}_jaccardms.csv",
        combinator,
        data_dir=DATA_DIR,
        sample_feature=gw.sample_feature,
        stratum=gw.stratum,
        model=MODELS,
        prompt_ver=PROMPT_VERSIONS,
        document_id=gw.document_id,
        document_classes=gw.document_classes
    )

rule all:
    input: 
        get_eval_results

rule evaluate_jaccardms:
    input: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_jaccardms.csv"
    params:
        # predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld",
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_pred_filtered.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt",
        semantic_compression = "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_compression.csv"
    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"
        
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct jaccardms --expected {params.baseline} --document {params.document} --outfile {output} --basename {basename} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct compression --expected {params.baseline} --document {params.document} --outfile {params.semantic_compression} --basename {basename} {target_classes_args}")

        add_column_and_export(str(output), add_columns={"prompt_ver": wildcards.prompt_ver})

rule evaluate_semantic:
    input: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual.csv"
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_pred_filtered.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_expected_filtered.jsonld", 

        predicted_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_pred.json",
        baseline_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_expected.json",

        predicted_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_pred_filtered.jsonld",
        baseline_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_semantic_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt",
        
        factual_jaccardms="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_jaccardms.csv",
        factual_compression="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_compression.csv"

    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"
                
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct semantic --expected {params.baseline} --document {params.document} --outfile {output} --basename {basename} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct jaccardms --expected {params.baseline} --document {params.document} --outfile {params.factual_jaccardms} --basename {basename} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct compression --expected {params.baseline} --document {params.document} --outfile {params._factual_compression} --basename {basename} {target_classes_args}")

        add_column_and_export(str(output), add_columns={"prompt_ver": wildcards.prompt_ver})
        add_column_and_export(str(params.semantic_jaccardms), add_columns={"prompt_ver": wildcards.prompt_ver})

        shell(f"cp {params.predicted} {params.predicted_filtered}")
        shell(f"cp {params.baseline} {params.baseline_filtered}")
        
        shell(f"python markup/markup.py do-filter-json-factual {params.predicted} {params.predicted_log} {params.predicted_filtered}")
        shell(f"python markup/markup.py do-filter-json-factual {params.baseline} {params.baseline_log} {params.baseline_filtered}")

rule evaluate_factual:
    input: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl.csv",
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_pred_filtered.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_expected_filtered.jsonld", 

        predicted_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_pred.json",
        baseline_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_expected.json",

        predicted_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_pred_filtered.jsonld",
        baseline_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_factual_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt",
        
        shacl_jaccardms="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_jaccardms.csv",
        shacl_compression="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_compression.csv"

    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"

        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct factual --expected {params.baseline} --document {params.document} --outfile {output} --basename {basename} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct jaccardms --expected {params.baseline} --document {params.document} --outfile {params.shacl_jaccardms} --basename {basename} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct compression --expected {params.baseline} --document {params.document} --outfile {params.shacl_compression} --basename {basename} {target_classes_args}")

        add_column_and_export(str(output), add_columns={"prompt_ver": wildcards.prompt_ver})
        add_column_and_export(str(params.factual_jaccardms), add_columns={"prompt_ver": wildcards.prompt_ver})

        shell(f"cp {params.predicted} {params.predicted_filtered}")
        shell(f"cp {params.baseline} {params.baseline_filtered}")
        
        shell(f"python markup/markup.py do-filter-json-factual {params.predicted} {params.predicted_log} {params.predicted_filtered}")
        shell(f"python markup/markup.py do-filter-json-factual {params.baseline} {params.baseline_log} {params.baseline_filtered}")

rule evaluate_shacl: 
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl.csv"
    params: 
        predicted="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld",
        baseline="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/baseline/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld", 

        predicted_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_pred.json",
        baseline_log="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_expected.json",

        predicted_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_pred_filtered.jsonld",
        baseline_filtered="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_shacl_expected_filtered.jsonld", 
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt",

        raw_jaccardms="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_raw_jaccardms.csv",
        raw_compression= "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}_raw_compression.csv"

    run: 
        target_classes = [ f"http://schema.org/{u}" for u in str(wildcards.document_classes).split("_") ]
        target_classes_args = " ".join([ f"--target-class {tc}" for tc in target_classes ])
        basename = f"{wildcards.document_id}_{wildcards.document_classes}"

        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct shacl --expected {params.baseline} --document {params.document} --outfile {output} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct jaccardms --expected {params.baseline} --document {params.document} --outfile {params.raw_jaccardms} --basename {basename} {target_classes_args}")
        shell(f"python markup/markup.py validate-one {params.predicted} Mixtral_8x7B_Instruct compression --expected {params.baseline} --document {params.document} --outfile {params.raw_compression} --basename {basename} {target_classes_args}")


        add_column_and_export(str(output), add_columns={"prompt_ver": wildcards.prompt_ver})
        add_column_and_export(str(params.shacl_jaccardms), add_columns={"prompt_ver": wildcards.prompt_ver})

        shell(f"cp {params.predicted} {params.predicted_filtered}")
        shell(f"cp {params.baseline} {params.baseline_filtered}")
        
        shell(f"python markup/markup.py do-filter-json-shacl {params.predicted} {params.predicted_log} {params.predicted_filtered}")
        shell(f"python markup/markup.py do-filter-json-shacl {params.baseline} {params.baseline_log} {params.baseline_filtered}")
            

