import glob
import pandas as pd
import os
from itertools import product
import json

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

METRICS = ["shacl", "factual", "semantic", "coverage"]

# PROMPT TEMPLATES
PROMPT_TEMPLATE_DIR = "prompts/generation"
PROMPT_VERSIONS = config.get("prompt_template")
PROMPT_VERSIONS = [ Path(template_file).stem for template_file in os.listdir(PROMPT_TEMPLATE_DIR) ] if PROMPT_VERSIONS is None else PROMPT_VERSIONS.split(",")

# ruleorder: generate_baseline > generate_markup > evaluate_markup > assemble
def get_generated_markups(wildcards):
    gw = glob_wildcards(f"{DATA_DIR}/{{sample_feature}}/stratum_{{stratum}}/corpus/baseline/{{document_id,[a-z0-9]+}}_{{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}}.jsonld")
    
    def combinator(data_dir, sample_feature, stratum, model, prompt_ver, document_id, document_classes):
        for data_dir_u, model_u, prompt_ver_u in product(data_dir, model, prompt_ver):
            if PROMPT_VERSIONS and prompt_ver_u[1] not in PROMPT_VERSIONS: continue
            for sample_feature_u, stratum_u, document_id_u, document_classes_u in zip(sample_feature, stratum, document_id, document_classes):
                if SAMPLE_FEATURE and sample_feature_u[1] not in SAMPLE_FEATURE: continue
                if DOCUMENT and document_id_u[1] not in DOCUMENT: continue
                yield (data_dir_u, sample_feature_u, stratum_u, model_u, prompt_ver_u, document_id_u, document_classes_u)

    return expand(
        "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id}_{document_classes}.jsonld",
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
        get_generated_markups

rule generate_markup:
    output: "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{model}/{prompt_ver}/{document_id,[a-z0-9]+}_{document_classes,[a-zA-Z]+(_[a-zA-Z]+)*}.jsonld"
    params:
        document="{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}.txt",
        target_class_fn = "{data_dir}/{sample_feature}/stratum_{stratum}/corpus/{document_id}_class.json"
    run: 
        print(wildcards.document_id)

        template_file=f"{PROMPT_TEMPLATE_DIR}/{wildcards.prompt_ver}.json"

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
        shell(f"python markup/markup.py generate-markup-one {params.document} {output} {wildcards.model} {target_classes_args} {subtarget_classes_args} --template {template_file}")
