from collections import Counter
import glob
from hashlib import md5
import json
import os
from pathlib import Path
from pprint import pprint
import re
import click
import openai
import pandas as pd
from rdflib import BNode, ConjunctiveGraph, Literal, URIRef
from tqdm import tqdm
from models.validator import ValidatorFactory
from models.llm import ModelFactoryLLM

from utils import close_ontology, filter_graph, get_page_content, get_schema_example, get_type_definition, html_to_rdf_extruct, jsonld_search_property, lookup_schema_type, schema_simplify, scrape_webpage, to_jsonld, transform_json

from itertools import chain, islice
import extruct

@click.group
def cli():
    pass

@cli.command()
@click.argument("jsonld", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("key", type=click.STRING)
@click.option("--value", type=click.STRING)
@click.option("--parent", is_flag=True)
def search_jsonld(jsonld, key, value, parent):
    markup = to_jsonld(jsonld, simplify=True, clean=True)
    pprint(markup)
    pprint(jsonld_search_property(markup, key, value=value, parent=parent))

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def extract_markup(infile):
    kg_extruct = html_to_rdf_extruct(infile)
    ref_markups = to_jsonld(kg_extruct, simplify=True, keep_root=True)

    for ref_markup in ref_markups.values():
        sub_markups = jsonld_search_property(ref_markup, key="@type", value=['House', 'Product'])
        print(sub_markups)

@cli.command()
@click.argument("rdf", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def convert_to_jsonld(rdf):
    data = to_jsonld(rdf, simplify=True, clean=True)
    print(json.dumps(data, ensure_ascii=False))

@cli.command()
@click.argument("url", type=click.STRING)
@click.option("--focus", is_flag=True, default=False)
def get_examples(url, focus):
    print(get_schema_example(url, focus=focus))

@cli.command()
@click.argument("graph", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("schema-type", type=click.STRING)
def filter_graph_by_type(graph, schema_type):
    g = ConjunctiveGraph()
    g.parse(graph)
    result = filter_graph(g, pred=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), obj=URIRef(lookup_schema_type(schema_type)))
    print(result.serialize())

@cli.command()
@click.argument("infile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--query", type=click.STRING)
@click.option("--topk", type=click.INT, default=20)
@click.option("--sort", type=click.STRING)
def extract_content(infile, outdir, query, topk, sort):
    df = pd.read_csv(infile)

    if query is not None:
        df = df.query(query)

    if sort is not None:
        m = re.search("by=(\w+),asc=(\w+)", sort)
        by = m.group(1)
        asc = eval(m.group(2))
        df = df.sort_values(by=by, ascending=asc).reset_index(drop=True)

    if topk is None:
        topk = len(df)

    print(df)

    nb_success = 0
    cursor = 0

    with tqdm(total=topk) as pbar:
        while nb_success < topk:
            source = df.iloc[cursor]["source"]
            id = md5(str(source).encode()).hexdigest()
            print(id, source)
            outfile = os.path.join(outdir, f"{id}.txt")
            
            if os.path.exists(outfile) and os.stat(outfile).st_size > 0:
                print(f"{outfile} already exists...")
                nb_success += 1
                cursor += 1
                pbar.update(1)
                continue
            
            
            # Scrape the page content
            try:
                content = get_page_content(source)
                if content is not None and len(content) > 0:
                    # Write the content
                    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
                    with open(outfile, mode="w") as ofs:
                        ofs.write(content)
            except RuntimeError as e:
                print(f"Could not scrape {id}.", e)
                if isinstance(e, RuntimeError):
                    cursor += 1
                continue
            
            cursor += 1
            nb_success += 1
            pbar.update(1)

@cli.command()
@click.argument("url", type=click.STRING)
def generate_corpus(url):
    if os.path.isfile(url):
        print(scrape_webpage(url))
    elif os.path.isdir(url):
        for file in glob.glob(f"{url}/*.txt"):
            id = str(Path(file).stem)
            cache_file = f".cache/{id}_raw.html"
            new_text = scrape_webpage(cache_file)
            with open(file, "w") as f:
                f.write(new_text)
                
    elif url.startswith("http"):
        print(get_page_content(url))

@cli.command()
@click.option("--url", type=click.STRING)
@click.option("--prop", type=click.STRING)
@click.option("--parents", is_flag=True, default=False)
@click.option("--simple", is_flag=True, default=False)
@click.option("--expected-types", is_flag=True, default=False)
@click.option("--comment", is_flag=True, default=False)
def get_schema_properties(url, prop, parents, simple, expected_types, comment):
    result = get_type_definition(schema_type_url=url, prop=prop, parents=parents, simplify=simple, include_expected_types=expected_types, include_comment=comment)
    print(result)
    
@cli.command()
@click.argument("path_to_jsonld", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("target_type", type=click.STRING)
@click.argument("method", type=click.Choice(["shacl", "factual", "semantic", "sameas"]))
def validate_one(path_to_jsonld, target_type, method):
    
    if method == "shacl":
        llm_validator = ValidatorFactory.create_validator("ShaclValidator", shape_graph="schemaorg/shacl/schemaorg_datashapes.shacl")
        llm_validator.validate(path_to_jsonld)
    elif method == "factual":
        llm = ModelFactoryLLM.create_model("GPT", target_type=target_type)
        validator = ValidatorFactory.create_validator("FactualConsistencyValidator", retriever=llm)
        document = f"{Path(path_to_jsonld).parent.parent}/{Path(path_to_jsonld).stem.split('_')[0]}.txt"
        id = Path(path_to_jsonld).stem
        validator.validate(path_to_jsonld, document=document, outfile=f"data/WDC/Painting/corpus/GPT/{id}_expected.json")
    elif method == "semantic":
        llm = ModelFactoryLLM.create_model("GPT", target_type=target_type)
        validator = ValidatorFactory.create_validator("SemanticConformanceValidator", retriever=llm)
        document = f"{Path(path_to_jsonld).parent.parent}/{Path(path_to_jsonld).stem.split('_')[0]}.txt"
        validator.validate(path_to_jsonld, document=document)
    elif method == "sameas":
        llm = ModelFactoryLLM.create_model("GPT", target_type=target_type)
        validator = ValidatorFactory.create_validator("SameAsLLMValidator", retriever=llm)
        expected_file = f"{Path(path_to_jsonld).parent.parent}/baseline/{Path(path_to_jsonld).stem}.nq"
        validator.validate(path_to_jsonld, expected_file=expected_file)
           
@cli.command()
@click.argument("nq_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("csv_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("target", type=click.STRING)
def generate_baseline(nq_file, csv_file, infile, target):
    
    ids = []
    if os.path.isdir(infile):
        ids = [ Path(fn).stem for fn in os.listdir(infile) if fn.endswith(".txt") ]
    elif os.path.isfile(infile):
        ids = [ Path(infile).stem ]
    
    sample = pd.read_csv(csv_file)
    for id in tqdm(ids):
        results = sample.query("`id` == @id")
        offset = results["offset"].item()
        length = results["length"].item()

        outfile = f"{infile}/baseline/{id}.nq"
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        
        print(outfile)
        
        if not os.path.exists(outfile) or os.stat(outfile).st_size == 0:
            with open(nq_file, 'r') as nq_fs, open(outfile, "w") as ofs:
                for line in islice(nq_fs, offset, offset + length):
                    ofs.write(line)
        
        g = ConjunctiveGraph()
        g.parse(outfile)
        
        g = filter_graph(g, pred=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), obj=URIRef(lookup_schema_type(target)))
        g.serialize(outfile, format="nquads")
        
                    
@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("target", type=click.STRING)
def generate_baseline_scrape(infile, target):
    
    ids = []
    if os.path.isdir(infile):
        ids = [ Path(fn).stem for fn in os.listdir(infile) if fn.endswith(".txt") ]
    elif os.path.isfile(infile):
        ids = [ Path(infile).stem ]
    
    for id in tqdm(ids):
        html_source = f".cache/{id}_raw.html"
        markup = html_to_rdf_extruct(html_source)
        markup = filter_graph(markup, pred=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), obj=URIRef(lookup_schema_type(target)))
        markup = to_jsonld(markup, simplify=True, clean=True)
        
        outfile = f"{infile}/baseline/{id}.json"
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(markup, f, ensure_ascii=False)    
    
@cli.command()
@click.argument("indata", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("model", type=click.Choice(["Llama2_7B", "Llama2_70B", "Llama2_13B", "GPT", "Mistral_7B_Instruct", "HuggingChatLLM"]))
@click.option("--hf-model", type=click.STRING)
@click.option("--outdir", type=click.Path(exists=False, file_okay=False, dir_okay=True))
@click.option("--validate", type=click.STRING) #default="shacl,coverage,factual,semantic,sameas"
@click.option("--explain", is_flag=True, default=False)
@click.option("--force-rewrite", is_flag=True, default=False)
@click.option("--force-validate", is_flag=True, default=False)
@click.option("--target-classes", type=click.STRING)
@click.pass_context
def run_markup_llm(ctx: click.Context, indata, model, hf_model, outdir, validate, explain, force_rewrite, force_validate, target_classes):

    if validate is not None:
        validate = [ v.strip() for v in validate.split(",") ]
    
    if outdir is None:
        outdir = indata
        if os.path.isfile(indata):
            outdir = str(Path(indata).parent)

    if target_classes is not None:
        target_classes = [tc.strip() for tc in target_classes.split(",")]
    
    documents = []
    if os.path.isdir(indata):
        documents = glob.glob(f"{indata}/*.txt")
    elif os.path.isfile(indata):
        documents = [ indata ]
        
    final_eval_df = pd.DataFrame()
    
    for document in documents:                
        model_dirname = model
        if hf_model is not None:
            model_dirname += "-" + hf_model.split("/")[-1]
            
        document_id = Path(document).stem

        target_class_fn = f"{Path(document).parent}/{Path(document).stem}_class.json"
        if os.path.exists(target_class_fn):
            with open(target_class_fn, "r") as f:
                target_classes = json.load(f)["pset_classes"]
                    
        llm_model = None
        if hf_model is not None:
            llm_model = ModelFactoryLLM.create_model(model, hf_model=hf_model)
        else:
            llm_model = ModelFactoryLLM.create_model(model)

        class_suffix = "_".join(target_classes)        
        predicted_fn = os.path.join(outdir, model_dirname, f"{document_id}_{class_suffix}.json")
        Path(predicted_fn).parent.mkdir(parents=True, exist_ok=True)
        
        print(predicted_fn)

        jsonld = None

        # Prediction
        if os.path.exists(predicted_fn) and os.stat(predicted_fn).st_size > 0 and not force_rewrite:
            print(f"{predicted_fn} already exists, skipping...")
            with open(predicted_fn, "r") as f:
                jsonld = json.load(f)
        else:
            try:
                with open(document, "r") as dfs, open(predicted_fn, "w") as f:
                    page = dfs.read()
                    if explain:
                        print(llm_model.predict(target_classes, page, explain=True))
                        continue
                    else:
                        jsonld = llm_model.predict(target_classes, page)
                        json.dump(jsonld, f, ensure_ascii=False) 
            except json.decoder.JSONDecodeError:
                # with open(predicted_fn, "w") as f:
                #     f.write(str(jsonld))
                continue
            except AttributeError:
                continue
            # except TypeError as e:
            #     print(jsonld)
            #     raise e
        
        # Evaluation
        if validate:
            outdir = indata
            if os.path.isfile(indata):
                outdir = str(Path(indata).parent)
            
            expected_fn = glob.glob(f"{outdir}/baseline/{document_id}*")[0]
            eval_df = pd.DataFrame()
            
            result_df = None

            for metric in validate:
                result_fn = f"{Path(predicted_fn).parent}/{Path(predicted_fn).stem}_{metric}.csv"
                require_update = force_validate
                if os.path.exists(result_fn) and os.stat(result_fn).st_size > 0:
                    result_df = pd.read_csv(result_fn)
                    require_update = result_df.empty or force_validate
                else:
                    require_update = True
                if require_update:
                    print(f"Updating {result_fn}...")
                    records = []
                    for target_class in target_classes:
                        eval_result = llm_model.evaluate(target_class, metric, predicted_fn, expected_fn, document=document)
                        eval_result["approach"] = model_dirname
                        eval_result["metric"] = metric
                        records.append(eval_result)
                    
                    result_df = pd.DataFrame.from_records(records)
                    # Function to extract dictionary values and concatenate keys to column name
                    def extract_and_concat(row, col_name):
                        dictionary = row[col_name]
                        if isinstance(dictionary, dict):
                            for key, value in dictionary.items():
                                new_col_name = col_name + '-' + key
                                row[new_col_name] = value
                        return row

                    # Iterate through columns and apply the function
                    for col in result_df.columns:
                        if col not in ['metric', 'approach']:
                            result_df = result_df.apply(lambda row: extract_and_concat(row, col), axis=1)
                            if isinstance(result_df[col][0], dict):
                                result_df.drop(col, axis=1, inplace=True)
                            
                    id_vars = ['metric', 'approach', "class"] if metric == "coverage" else ['metric', 'approach']
                    result_df = pd.melt(result_df, id_vars=id_vars, var_name='instance', value_name='value')
                    result_df.to_csv(result_fn, index=False)
                
                eval_df = pd.concat([eval_df, result_df])
                
            eval_df.to_csv(f"{Path(predicted_fn).parent}/{Path(predicted_fn).stem}.csv", index=False)
            final_eval_df = pd.concat([final_eval_df, eval_df])
    final_eval_df.to_csv(f"{outdir}/{model_dirname}.csv", index=False)

if __name__ == "__main__":
    cli()