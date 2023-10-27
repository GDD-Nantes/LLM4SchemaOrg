import glob
from hashlib import md5
import json
import os
from pathlib import Path
from bs4 import BeautifulSoup
import click
import pandas as pd
import requests
from models.llm import *

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
@click.argument("url", type=click.STRING)
def scrape_webpage(url):
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html5lib')
        
        # Define elements to exclude (e.g., header, footer, navbar, etc.)
        elements_to_exclude = ['header', 'footer', 'nav', 'script', 'style']
        
        # Remove specified elements from the parsed HTML
        for element in elements_to_exclude:
            for tag in soup.find_all(element):
                tag.extract()  # Remove the tag and its content
        
        # Extract and return the text content
        text_content = soup.get_text().strip()
        print(text_content)
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None

@cli.command()
@click.argument("url", type=click.STRING)
def get_schema_properties(url):
    print(get_ref_attrs(url))

@cli.command()
@click.argument("datadir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("model", type=click.Choice(["Llama2_7B", "Llama2_70B", "Llama2_13B", "ChatGPT"]), default="Llama2_7B")
def run_markup_llm(datadir, model):
    
    llm_model = ModelFactoryLLM.create_model(model)
    
    for document in glob.glob(f"{datadir}/*.txt"):
        outfile = os.path.join(datadir, f"{Path(document).stem}.json")
        with open(document, "r") as dfs, open(outfile, "w") as jfs:
            page = dfs.read()
            jsonld = llm_model.predict(page)
            json.dump(jsonld, jfs)
            

if __name__ == "__main__":
    cli()