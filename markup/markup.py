from collections import Counter
import glob
from hashlib import md5
import json
import os
from pathlib import Path
from pprint import pprint
import re
from SPARQLWrapper import JSON, N3, SPARQLWrapper
import click
import openai
import pandas as pd
from rdflib import RDF, SH, BNode, ConjunctiveGraph, Literal, Namespace, URIRef
from tqdm import tqdm
from models.validator import ValidatorFactory
from models.llm import ModelFactoryLLM

from owlready2 import get_ontology, close_world
from utils import close_ontology, filter_graph_by_type, get_n_grams, get_page_content, get_schema_example, get_type_definition, lookup_schema_type, scrape_webpage

from itertools import islice
import extruct

@click.group
def cli():
    pass

#TODO: extract markup using extruct

@cli.command()
@click.argument("url", type=click.STRING)
def get_examples(url):
    print(get_schema_example(url))

@cli.command()
@click.argument("graph", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("schema-type", type=click.STRING)
def filter_graph(graph, schema_type):
    g = ConjunctiveGraph()
    g.parse(graph)
    result = filter_graph_by_type(g, schema_type)
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
@click.argument("indirfile", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def clean_corpus(indirfile):
    
    documents = None
    if os.path.isfile(indirfile):
        documents = [ indirfile ]
    elif os.path.isdir(indirfile):
        documents = glob.glob(f"{indirfile}/*.txt")
    
    # Using GPT-3 ngrams deduplication
    ngrams_file = ".cache/ngrams.json"
    if os.path.exists(ngrams_file):
        with open(ngrams_file, "r") as f:
            ngrams = json.load(f).keys()
            
            for document in documents:
                content = None
                with open(document, "r") as f:
                    content = f.read()
                    for ngram in ngrams:
                        pattern = "\s+".join([re.escape(term) for term in ngram.split()])
                        pattern = re.compile(pattern)
                        content = re.sub(pattern, "", content)
                
                print(content)
                # with open(document, "w") as f:
                #     f.write(content)

@cli.command()
@click.argument("indir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--ngrams", type=click.INT, default=13)
@click.option("--threshold", type=click.INT, default=10)
def generate_ngrams(indir, ngrams, threshold):
    """Generate ngrams for a collection of documents

    Args:
        infile (_type_): The input directory that contains text documents.
        ngrams (_type_): the length of gram sequence
    """
    all_ngrams = dict()
    for document in glob.glob(f"{indir}/**/*.txt", recursive=True):
        id = Path(document).stem
        content = None
        with open(document, "r") as f:
            content = f.read()
            for ngram in get_n_grams(content, ngrams):
                if ngram not in all_ngrams.keys():
                    all_ngrams[ngram] = []
                all_ngrams[ngram].append(id)
    
    all_ngrams = { k:len(set(v)) for k, v in all_ngrams.items() }
    filtered_ngrams = { k:v for k, v in all_ngrams.items() if v > threshold }
    
    with open(".cache/ngrams.json", "w") as f:
        json.dump(filtered_ngrams, f)

@cli.command()
@click.argument("url", type=click.STRING)
@click.option("--parents", is_flag=True, default=True)
@click.option("--simple", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def get_schema_properties(url, parents, simple, verbose):
    result = get_type_definition(url, parents=True, simplify=simple, verbose=verbose)
    print(result)
    
@cli.command()
@click.argument("path_to_jsonld", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def validate_one(path_to_jsonld):
    # llm_validator = ValidatorFactory.create_validator("ShaclValidator", shape_graph="shacl/schemaorg/test.shacl")
    # llm_validator.validate(path_to_jsonld)
    
    llm = ModelFactoryLLM.create_model("ChatGPT", target_type="Painting")
    validator = ValidatorFactory.create_validator("FactualConsistencyValidator", retriever=llm)
    document = f"{Path(path_to_jsonld).parent.parent}/{Path(path_to_jsonld).stem.split('_')[0]}.txt"
    id = Path(path_to_jsonld).stem
    validator.validate(path_to_jsonld, document=document, outfile=f"data/WDC/Painting/corpus/ChatGPT/{id}_expected.json")
    
    # llm = ModelFactoryLLM.create_model("ChatGPT", target_type="Painting")
    # validator = ValidatorFactory.create_validator("TypeConformanceLLMValidator", retriever=llm)
    # document = f"{Path(path_to_jsonld).parent.parent}/{Path(path_to_jsonld).stem.split('_')[0]}.txt"
    # validator.validate(path_to_jsonld, document=document)

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=True))  
# @click.option("--in-format", type=click.Choice(["nquads", "turtle", "json-ld", "n3"]), default="nquads")  
# @click.option("--out-format", type=click.Choice(["nquads", "turtle", "json-ld", "n3"]), default="turtle")  
def convert(infile):
    
    sources = []
    if os.path.isdir(infile):
        sources = [ infile + fn for fn in os.listdir(infile) ]
    elif os.path.isfile(infile):
        sources = [infile]
    
    for source in tqdm(sources):
        print(source)
        g = ConjunctiveGraph()
        g.parse(location=source, format="nquads")
        outfile = f"{Path(source).parent}/{Path(source).stem}.ttl"
        g.serialize(outfile, format="turtle")

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument("target-type", type=click.STRING)
@click.argument("predicted", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("expected", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--method", type=click.Choice(["ngrams", "graph-emb", "shacl", "text-emb", "coverage"]), default="ngrams")
@click.pass_context
def evaluate_one(ctx: click.Context, target_type, predicted, expected, method):
    abstract_model = ModelFactoryLLM.create_model("AbstractModelLLM", target_type=target_type)
    results = abstract_model.evaluate(method, predicted, expected)
    print(results)
    
@cli.command()
@click.argument("csv_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def generate_baseline_db(csv_file, infile):
    
    ids = []
    if os.path.isdir(infile):
        ids = [ Path(fn).stem for fn in os.listdir(infile) if fn.endswith(".txt") ]
    elif os.path.isfile(infile):
        ids = [ Path(infile).stem ]
    
    def process(node):
        if node["type"] == "bnode":
            return BNode(node["value"])
        elif node["type"] == "uri":
            return URIRef(node["value"])
        elif node["type"] == "literal":
            return Literal(node["value"])
        elif node["type"] == "typed-literal":
            return Literal(node["value"], datatype=node["datatype"])
        else:
            raise NotImplementedError(f"{node} not yet implemented!")
    
    virtuoso = SPARQLWrapper("http://localhost:32772/sparql") 
    sample = pd.read_csv(csv_file)
    for id in tqdm(ids):
        source = sample.query("`id` == @id")["source"].item()
        query = f"""
        SELECT ?s ?p ?o WHERE {{
            GRAPH <{source}> {{
                ?s a <http://schema.org/Recipe> .
                ?s ?p ?o .
            }}
        }}
        """
        virtuoso.setQuery(query)
        virtuoso.setReturnFormat(JSON)

        # Execute the query and parse the results
        results = virtuoso.query().convert()
        
        g = ConjunctiveGraph()
        
        # Process and load the results into the graph
        for result in results["results"]["bindings"]:
            subject = process(result["s"])
            predicate = process(result["p"])
            obj = process(result["o"])
            
            # TODO somehow without this line, an error is raised
            print(result)

            # Add the triple to the rdflib Graph
            g.add((subject, predicate, obj))
        
        outfile = f"{Path(infile).parent}/baseline/{id}.ttl" if os.path.isfile(infile) else f"{Path(infile)}/baseline/{id}.ttl"
    
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        g.serialize(outfile, format="ttl")
       
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
        
        g = filter_graph_by_type(g, target)
        g.serialize(outfile, format="nquads")
        
                    
@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def generate_baseline_scrape(infile):
    
    ids = []
    if os.path.isdir(infile):
        ids = [ Path(fn).stem for fn in os.listdir(infile) if fn.endswith(".txt") ]
    elif os.path.isfile(infile):
        ids = [ Path(infile).stem ]
    
    for id in tqdm(ids):
        with open(f".cache/{id}_raw.html", "r") as f:
            webpage = f.read()
            data = extruct.extract(webpage)
            microdata = data["microdata"]
            g = ConjunctiveGraph()
            g.parse(data=str(microdata), format="microdata")
            print(g.serialize(format="json-ld"))

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
            for metric in ["coverage", "factual", "type-llm"]:
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