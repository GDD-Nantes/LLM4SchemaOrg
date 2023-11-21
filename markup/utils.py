from hashlib import md5
from io import BytesIO
import json
import os
from pathlib import Path
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import html2text
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from warcio.archiveiterator import ArchiveIterator

from rdflib import ConjunctiveGraph, URIRef

def ping(url):
    try:
        response = requests.get(url, allow_redirects=False)
        return (200 <= response.status_code < 300)
    except:
        return False

def get_ref_attrs(schema_type_url):
    schema_attrs = []
    soup = BeautifulSoup(requests.get(schema_type_url).text, "html.parser")
    table = soup.find(class_="definition-table")
    for tr in soup.find_all("th", class_="prop-nam"):
        schema_attrs.append(tr.get_text().strip())
        
    return schema_attrs

def md5hex(obj):
    return md5(str(obj).encode()).hexdigest()

def get_page_content(target_url):

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    # Please note: f-strings require Python 3.6+

    # The URL of the Common Crawl Index server
    CC_INDEX_SERVER = 'http://index.commoncrawl.org/'
    LANGUAGES_CACHE_FILE = ".cache/languages.cache"
    Path(LANGUAGES_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    LANGUAGES_CACHE = dict()
    if os.path.exists(LANGUAGES_CACHE_FILE):
        with open(LANGUAGES_CACHE_FILE, "r") as cache_file:
            LANGUAGES_CACHE = json.load(cache_file)

    md5ingest = md5hex(target_url)
    if md5ingest in LANGUAGES_CACHE:
        languages = LANGUAGES_CACHE[md5ingest]
        if not (len(languages) == 1 and "eng" in languages):
            raise RuntimeError("Skipping because the content is not in English!")

    # The Common Crawl index you want to query
    INDEX_NAME = 'CC-MAIN-2021-43'      # Replace with the latest index name

    # Function to search the Common Crawl Index
    def search_cc_index(url):
        encoded_url = quote_plus(url)
        index_url = f'{CC_INDEX_SERVER}{INDEX_NAME}-index?url={encoded_url}&output=json'
        print(index_url)
        response = http.get(index_url)
        print("Response from CCI:", response.text)  # Output the response from the server
        if response.status_code == 200:
            records = response.text.strip().split('\n')
            return [json.loads(record) for record in records]
        else:
            return None

    # Function to fetch the content from Common Crawl
    def fetch_page_from_cc(records):
        for record in records:
            offset, length = int(record['offset']), int(record['length'])
            prefix = record['filename'].split('/')[0]
            base_name = os.path.basename(record['filename'])
            languages = record["languages"].split(",") if "languages" in record else ["unknown"]
            LANGUAGES_CACHE[md5ingest] = languages

            with open(LANGUAGES_CACHE_FILE, "w") as cache_file:
                json.dump(LANGUAGES_CACHE, cache_file)
            dest_url = record["url"]

            # Filter page with English as the only language
            if not (len(languages) == 1 and "eng" in languages):
                raise RuntimeError("Skipping because the content is not in English!")

            s3_url = f'https://data.commoncrawl.org/{record["filename"]}'
            response = http.get(s3_url, headers={'Range': f'bytes={offset}-{offset+length-1}'})
            if response.status_code == 206:
                # Process the response content if necessary
                # For example, you can use warcio to parse the WARC record
                
                with BytesIO(response.content) as stream:
                    for r in ArchiveIterator(stream):
                        # Check if the current record is a response record and has the expected URL
                        if r.rec_type == 'response' and r.rec_headers.get_header('WARC-Target-URI') == dest_url:
                            # Extract the content of the response
                            page_content = r.content_stream().read().decode('utf-8')
                            return page_content
            else:
                raise ConnectionError(f"Failed to fetch data: status {response.status_code}, message: {response.content.decode()}")
    
    def skip_certain_tags(h2t, tag, attrs, start):
        if tag in ['header', 'footer', 'nav', 'script', 'style']:
            return False
    
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.tag_callback = skip_certain_tags
    
    cache_file = f".cache/{md5ingest}_raw.html"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            content = f.read()
            text = converter.handle(content)
            return text
    else:
        # Search the index for the target URL
        records = search_cc_index(target_url)
        if records:
            print(f"Found {len(records)} records for {target_url}")

            # Fetch the page content from the first record
            content = fetch_page_from_cc(records)
            if content:
                print(f"Successfully fetched content for {target_url}")
                # You can now process the 'content' variable as needed
                with open(cache_file, "w") as f:
                    f.write(content)
                text = converter.handle(content)
                return text
        else:
            print(f"No records found for {target_url}")
            
def filter_graph_by_type(graph: ConjunctiveGraph, schema_type, root=None):
    result = ConjunctiveGraph()
    target_type = URIRef(lookup_schema_type(schema_type))
    if root is None:
        for s in graph.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), target_type):
            subgraph = filter_graph_by_type(graph, schema_type, root=s)
            for s1, p, o in subgraph:
                result.add((s1, p, o))
    else:
        for p, o in graph.predicate_objects(root):
            result.add((root, p, o))
            
            subgraph = filter_graph_by_type(graph, schema_type, root=o)
            for s1, p1, o1 in subgraph:
                result.add((s1, p1, o1))
    return result
        
def lookup_schema_type(schema_type):
    g = ConjunctiveGraph()
    g.parse("https://schema.org/version/latest/schemaorg-all-http.nt")

    query = f"""
    SELECT ?class WHERE {{
        ?class <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .
        FILTER ( contains( lcase(str(?class)), {repr(str(schema_type).lower())}) )
    }}
    """

    results = g.query(query)
    candidates = [row.get("class") for row in results ]
    return str(candidates[0]).strip("<>")