from collections import Counter
import glob
from hashlib import md5
import json
import os
from pathlib import Path
from pprint import pprint
import re
from SPARQLWrapper import JSON, SPARQLWrapper
import click
import openai
import pandas as pd
from rdflib import BNode, ConjunctiveGraph, Literal, URIRef
from tqdm import tqdm
from models.validator import ValidatorFactory
from models.llm import ModelFactoryLLM

from utils import close_ontology, collect_json, filter_graph, get_page_content, get_schema_example, get_type_definition, html_to_rdf_extruct, lookup_schema_type, schema_simplify, scrape_webpage, to_jsonld, transform_json

from itertools import islice
import extruct

@click.group
def cli():
    pass

#TODO: extract markup using extruct

@cli.command()
@click.argument("rdf", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def convert_to_jsonld(rdf):
    def __write_prompt(key, values, ent_type):
        return f"{ent_type} {schema_simplify(key)} {schema_simplify(values)}" 
    data = to_jsonld(rdf, simplify=True)
    # pprint(data)
    prompts = collect_json(data, lambda k, v, e: v)
    print(len(prompts))
    
@cli.command()
@click.argument("rdf", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def convert_and_simplify(rdf):
    data = to_jsonld(rdf)
    data = transform_json(data, schema_simplify, schema_simplify)

@cli.command()
@click.argument("url", type=click.STRING)
def get_examples(url):
    print(get_schema_example(url))

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
@click.argument("url", type=click.STRING)
@click.option("--parents", is_flag=True, default=True)
@click.option("--simple", is_flag=True, default=False)
@click.option("--expected-types", is_flag=True, default=False)
@click.option("--comment", is_flag=True, default=False)
def get_schema_properties(url, parents, simple, expected_types, comment):
    result = get_type_definition(url, parents=True, simplify=simple, include_expected_types=expected_types, include_comment=comment)
    print(result)
    
@cli.command()
@click.argument("path_to_jsonld", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("method", type=click.Choice(["shacl", "factual", "semantic", "sameas"]))
def validate_one(path_to_jsonld, method):
    
    if method == "shacl":
        llm_validator = ValidatorFactory.create_validator("ShaclValidator", shape_graph="shacl/schemaorg/test.shacl")
        llm_validator.validate(path_to_jsonld)
    elif method == "factual":
        llm = ModelFactoryLLM.create_model("ChatGPT", target_type="Painting")
        validator = ValidatorFactory.create_validator("FactualConsistencyValidator", retriever=llm)
        document = f"{Path(path_to_jsonld).parent.parent}/{Path(path_to_jsonld).stem.split('_')[0]}.txt"
        id = Path(path_to_jsonld).stem
        validator.validate(path_to_jsonld, document=document, outfile=f"data/WDC/Painting/corpus/ChatGPT/{id}_expected.json")
    elif method == "semantic":
        llm = ModelFactoryLLM.create_model("ChatGPT", target_type="Painting")
        validator = ValidatorFactory.create_validator("SemanticConformanceValidator", retriever=llm)
        document = f"{Path(path_to_jsonld).parent.parent}/{Path(path_to_jsonld).stem.split('_')[0]}.txt"
        validator.validate(path_to_jsonld, document=document)
    elif method == "sameas":
        llm = ModelFactoryLLM.create_model("ChatGPT", target_type="Painting")
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
            json.dump(markup, f)

@cli.command()
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def close_schemaorg_ontology(outfile):
    g = ConjunctiveGraph()
    # g.parse("https://schema.org/version/latest/schemaorg-shapes.shacl")
    # g.parse("https://schema.org/version/latest/schemaorg-subclasses.shacl")
    
    g.parse("shacl/schemaorg/schemaorg_datashapes.shacl")
    g = close_ontology(g)
    g.serialize(outfile, format="turtle")
    
    
@cli.command()
@click.argument("target-type", type=click.STRING)
@click.argument("indata", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.argument("model", type=click.Choice(["Llama2_7B", "Llama2_70B", "Llama2_13B", "ChatGPT", "Mistral_7B_Instruct", "HuggingChatLLM"]))
@click.option("--hf-model", type=click.STRING)
#@click.option("--validate", type=click.Choice(["shacl", "factual", "semantic", "sameas"]))
@click.option("--validate", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
@click.pass_context
def run_markup_llm(ctx: click.Context, target_type, indata, model, hf_model, validate, overwrite):
    
    llm_model = None
    if hf_model is not None:
        llm_model = ModelFactoryLLM.create_model(model, target_type=target_type, hf_model=hf_model)
    else:
        llm_model = ModelFactoryLLM.create_model(model, target_type=target_type)
    
    documents = []
    if os.path.isdir(indata):
        documents = glob.glob(f"{indata}/*.txt")
    elif os.path.isfile(indata):
        documents = [ indata ]
    
    for document in documents:
                
        model_dirname = model
        if hf_model is not None:
            model_dirname += "-" + hf_model.split("/")[-1]
            
        document_id = Path(document).stem
        
        parent_dir = indata
        if os.path.isfile(indata):
            parent_dir = str(Path(indata).parent)
            
        predicted_fn = os.path.join(parent_dir, model_dirname, f"{document_id}.json")
        Path(predicted_fn).parent.mkdir(parents=True, exist_ok=True)
        
        print(predicted_fn)

        jsonld = None
        
        # Prediction
        if os.path.exists(predicted_fn) and os.stat(predicted_fn).st_size > 0 and not overwrite:
            print(f"{predicted_fn} already exists, skipping...")
            with open(predicted_fn, "r") as jfs:
                jsonld = json.load(jfs)
        else:
            try:
                with open(document, "r") as dfs, open(predicted_fn, "w") as jfs:
                    page = dfs.read()
                    jsonld = llm_model.predict(page)
                    json.dump(jsonld, jfs)
            except openai.error.Timeout:
                continue   
            except json.decoder.JSONDecodeError:
                # with open(predicted_fn, "w") as f:
                #     f.write(str(jsonld))
                continue
            except AttributeError:
                continue
            except TypeError as e:
                print(jsonld)
                raise e
        
        # Evaluation
        if validate:
            parent_dir = indata
            if os.path.isfile(indata):
                parent_dir = str(Path(indata).parent)
            
            expected_fn = glob.glob(f"{parent_dir}/baseline/{document_id}.*")[0]
            eval_df = pd.DataFrame()

            #for metric in ["ngrams", "graph-emb", "shacl", "text-emb", "coverage"]:
            for metric in ["shacl", "coverage", "factual", "semantic", "sameas"]:
                result_fn = f"{Path(predicted_fn).parent}/{Path(predicted_fn).stem}_{metric}.csv"
                require_update = True
                if os.path.exists(result_fn):
                    result_df = pd.read_csv(result_fn)
                    require_update = result_df.empty
                if require_update:
                    print(f"Updating {result_fn}...")
                    eval_result = llm_model.evaluate(metric, predicted_fn, expected_fn)
                    eval_result["approach"] = model_dirname
                    eval_result["metric"] = metric
                    
                    result_df = pd.DataFrame.from_records([eval_result])
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
                            
                    result_df = pd.melt(result_df, id_vars=['metric', 'approach'], var_name='instance', value_name='value')
                    result_df.to_csv(result_fn, index=False)
                
                eval_df = pd.concat([eval_df, result_df])
                
            eval_df.to_csv(f"{Path(predicted_fn).parent}/{Path(predicted_fn).stem}.csv", index=False)


if __name__ == "__main__":
    cli()