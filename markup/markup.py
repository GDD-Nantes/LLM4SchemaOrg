import glob
from hashlib import md5
import json
import os
from pathlib import Path
import re
from SPARQLWrapper import JSON, N3, SPARQLWrapper
import click
import openai
import pandas as pd
from rdflib import RDF, SH, BNode, Graph, Literal, Namespace, URIRef
from tqdm import tqdm
from models.validator import ValidatorFactory
from models.llm import ModelFactoryLLM

from utils import get_page_content, get_ref_attrs

from itertools import islice

@click.group
def cli():
    pass

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
            except Exception as e:
                print(f"Could not scrape {id}.", e)
                if isinstance(e, RuntimeError):
                    cursor += 1
                continue
            
            cursor += 1
            nb_success += 1
            pbar.update(1)

@cli.command()
@click.argument("url", type=click.STRING)
def scrape_webpage(url):
    print(get_page_content(url))

@cli.command()
@click.argument("url", type=click.STRING)
def get_schema_properties(url):
    print(get_ref_attrs(url))
    
@cli.command()
@click.argument("shape_graph", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("path_to_jsonld", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def validate_one(shape_graph, path_to_jsonld):
    llm_validator = ValidatorFactory.create_validator("ShexValidator", shape_graph=shape_graph)
    llm_validator.validate(path_to_jsonld)
    llm_validator.get_messages()

@cli.command()
@click.argument("path_to_jsonld", type=click.Path(exists=True, file_okay=True, dir_okay=False))  
@click.option("--format", type=click.Choice(["ttl", "json-ld", "n3"]), default="ttl")  
def convert(path_to_jsonld, format):
    g = Graph()
    g.parse(path_to_jsonld)
    print(g.serialize(format=format))

@cli.command()
@click.argument("predicted", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("expected", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--method", type=click.Choice(["ngrams", "graph-emb", "shacl", "text-emb"]), default="ngrams")
def evaluate_one(predicted, expected, method):
    ModelFactoryLLM.create_model("AbstractModelLLM").evaluate(method, predicted, expected)
       
@cli.command()
@click.argument("nq_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("csv_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def generate_baseline(nq_file, csv_file, corpus_dir):
    
    sample = pd.read_csv(csv_file)
    ids = [ Path(id).stem for id in os.listdir(corpus_dir) if id.endswith(".txt") ]
    for id in tqdm(ids):
        results = sample.query("`id` == @id")
        offset = results["offset"].item()
        length = results["length"].item()

        outfile = f"{corpus_dir}/baseline/{id}.nq"
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)

        with open(nq_file, 'r') as nq_fs, open(outfile, "w") as ofs:

            for line in islice(nq_fs, offset, offset + length):
                ofs.write(line)

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--merge", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def close_shacl_shapes(infile, merge):
    """Load an input SHACL shape graph and close each shape
    """        
    shape_graph = Graph()
    shape_graph.parse(infile, format="turtle")
    
    if merge is not None:
        shape_graph.parse(merge, format="turtle")
    
    out_graph = Graph()
    
    # Define namespaces
    # SHACL = Namespace("http://www.w3.org/ns/shacl1#") # push the sh:closed to the end of each shape
    SHACL = Namespace("http://datashapes.org/dash#") # dash:closedByTypes
        
    for s, p, o in tqdm(shape_graph):
        
        out_graph.add((s, p, o))
        if p == RDF.type and o in [SH.NodeShape, SH.PropertyShape]:
            # out_graph.add((s, SHACL.closed, Literal(True)))
            out_graph.add((s, SHACL.closedByTypes, Literal(True)))
        
    outfile =f"{Path(infile).parent}/{Path(infile).stem}-closed.shacl"
    out_graph.serialize(outfile, format="turtle")
    
    txt = ""
    with open(outfile, "r") as r:
        txt = r.read()
        prefix = re.search(rf"@prefix (\w+): \<{re.escape(SHACL)}\>", txt).group(1)
    with open(outfile, "w") as w:
        # txt = txt.replace(f"{prefix}:closed", "sh:closed")
        txt = txt.replace(f"@prefix {prefix}", "@prefix dash")
        txt = txt.replace(f"{prefix}:closedByTypes", "dash:closedByTypes")
        w.write(txt)
 
@cli.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("model", type=click.Choice(["Llama2_7B", "Llama2_70B", "Llama2_13B", "ChatGPT", "Mistral_7B_Instruct", "HuggingChatLLM"]), default="Llama2_7B")
@click.option("--hf-model", type=click.STRING)
@click.pass_context
def run_markup_llm(ctx: click.Context, datadir, model, hf_model):
    
    llm_model = None
    if hf_model is not None:
        llm_model = ModelFactoryLLM.create_model(model, hf_model=hf_model)
    else:
        llm_model = ModelFactoryLLM.create_model(model)
    
    for document in glob.glob(f"{datadir}/*.txt"):
                
        model_dirname = model
        if hf_model is not None:
            model_dirname += "-" + hf_model.split("/")[-1]
            
        document_id = Path(document).stem
            
        predicted_fn = os.path.join(datadir, model_dirname, f"{document_id}.json")
        Path(predicted_fn).parent.mkdir(parents=True, exist_ok=True)
        
        print(predicted_fn)

        jsonld = None
        
        # Prediction
        if os.path.exists(predicted_fn) and os.stat(predicted_fn).st_size > 0:
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
        expected_fn = os.path.join(datadir, "baseline", f"{document_id}.ttl")
        
        eval_df = pd.DataFrame()

        for metric in ["ngrams", "graph-emb", "shacl", "text-emb"]:
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
                        
                result_df = pd.melt(result_df, id_vars=['metric', 'approach'], var_name='name', value_name='value')
                result_df.to_csv(result_fn, index=False)
            
            eval_df = pd.concat([eval_df, result_df])
            
        eval_df.to_csv(f"{Path(predicted_fn).parent}/{Path(predicted_fn).stem}.csv", index=False)


if __name__ == "__main__":
    cli()