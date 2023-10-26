import glob
from hashlib import md5
import json
import os
from pathlib import Path
import click
import pandas as pd
from models import LLaMA_70B_LLM

from utils import get_archive_url, get_page_content

@click.group
def cli():
    pass

@cli.command()
@click.argument("infile", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument("outdir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def extract_content(infile, outdir):
    df = pd.read_csv(infile)
    for source in df["source"]:
        id = md5(str(source).encode()).hexdigest()
        outfile = os.path.join(outdir, f"{id}.txt")
        
        if os.path.exists(outfile):
            print(f"{outfile} already exists...")
            continue
        
        # Scrape the page content
        try:
            target_url = get_archive_url(source)
            content, references = get_page_content(target_url)
        
            # Write the content
            Path(outfile).parent.mkdir(parents=True, exist_ok=True)
            with open(outfile, mode="w") as ofs:
                ofs.write(content)
        except Exception as e:
            print(f"Could not scape {id}.", e)

@cli.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def run_markup_llm(datadir):
    
    llama_model = LLaMA_70B_LLM()
    
    for document in glob.glob(f"{datadir}/*.txt"):
        outfile = os.path.join(datadir, Path(document).stem, ".json")
        with open(document, "r") as dfs, open(outfile, "w") as jfs:
            page = dfs.read()
            jsonld = llama_model.predict(page)
            json.dump(jsonld, jfs)
            

if __name__ == "__main__":
    cli()