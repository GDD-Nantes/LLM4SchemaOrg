import json
import os
from pathlib import Path
import textwrap
import click
import pandas as pd
from rdflib import ConjunctiveGraph
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import _html2txt, get_schema_example, get_type_definition, md5hex

@click.group
def cli():
    pass

@cli.command()
@click.argument("outfile", type=click.Path(file_okay=True, dir_okay=False))
def create_dataset(outfile):
    g = ConjunctiveGraph()
    g.parse("schemaorg/shacl/schemaorg_datashapes.shacl")
    g.parse("schemaorg/examples/schemaorg-all-examples.ttl", format="ttl")

    query = """
    SELECT ?ref ?prop WHERE {
        ?node   a <http://www.w3.org/ns/shacl#PropertyShape> ;
                <http://www.w3.org/ns/shacl#path> ?prop .
        ?prop <http://example.org/hasExample> ?example .
        ?example    <http://example.org/json> ?jsonld ;
                    <http://example.org/pre-markup> ?ref .
    }
    """
    df = None

    def load_json(json_str):
        try: return json.dumps(json.loads(json_str))
        except: return None

    records = []
    for qres in tqdm(g.query(query)):
        ref = qres.get("ref").toPython()
        prop = qres.get("prop").toPython()
        examples = get_schema_example(prop, focus=True)
        records.append({ "ref": ref,  "prop": prop, "examples": examples })
        
    df = pd.DataFrame.from_records(records)
    df = df.explode("examples")

    df["ref"] = df["ref"].apply(lambda x: _html2txt(x, force=True))
    df["examples"] = df["examples"].apply(load_json)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_feather(outfile)

@cli.command()
@click.argument("infile", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def generate_negative_examples(infile):
    df = pd.read_feather(infile)
    records = []
    for i, row in df.iterrows():
        ref, prop, examples = row["ref"], row["prop"], row["examples"]
        pos_definition = get_type_definition(prop=prop, simplify=True, include_comment=True, include_expected_types=True).get(prop)
        pos_definition_verb = f"expected type is {pos_definition['expected_types']} and description aligns with {pos_definition['comment']}"
        neg_definition_verb = f"expected type is {pos_definition['expected_types']} but description does not aligns with {pos_definition['comment']}"
        prompt = textwrap.dedent(f"""
        - Given the text below:
        ```
        {ref}
        ```

        - Give the "positive" description for the property "name":

        ```text
        {pos_definition_verb}
        ```

        - Given the schema.org markup below for the property "name":

        ```json
        
        ```

        The markup is a positive example, i.e., the property-value pair aligns with the description.

        
        Task: Fill  [MASK] in the following template so that the outcome is a negative sample. Give several samples and explain.

        ```json
        {
        "name":  [MASK]
        }
        ```

        Constraints: 
        -  The filled value is of the expected type "Text" but does not align with the expected description "The name of the item."
        - the filled value must use elements provided in the input text.
        - the output must conform to JSON-LD format.
        """)

def train_test_split(infile):
    df = pd.read_feather(infile)
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    traindir = "data/SchemaExamples/train"

    for i, row in X_train.iterrows():
        ref, examples = row["ref"], row["examples"]
        id = md5hex(f"{ref}{i}")
        
        corpus_fn = f"{traindir}/corpus/{id}.txt"
        Path(corpus_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_fn, "w") as f:
            f.write(ref)
            
        example_fn = f"{traindir}/corpus/baseline/{id}.json"
        Path(example_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(example_fn, "w") as f:
            f.write(examples)

    testdir = "data/SchemaExamples/train" 
    for i, row in X_test.iterrows():
        ref, examples = row["ref"], row["examples"]
        id = md5hex(f"{ref}{i}")
        
        corpus_fn = f"{testdir}/corpus/{id}.txt"
        Path(corpus_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_fn, "w") as f:
            f.write(ref)
            
        example_fn = f"{testdir}/corpus/baseline/{id}.json"
        Path(example_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(example_fn, "w") as f:
            f.write(examples)

if __name__ == "__main__":
    cli()
