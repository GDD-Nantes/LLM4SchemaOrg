from hashlib import md5
from io import BytesIO
import json
import os
from pathlib import Path
import re
import string
import time
from typing import Dict, List, Union
from urllib.parse import quote_plus, urlparse
from bs4 import BeautifulSoup
import html2text
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from warcio.archiveiterator import ArchiveIterator

from trafilatura import extract

from rdflib import BNode, ConjunctiveGraph, URIRef

import nltk
from nltk import ngrams
from collections import Counter

def ping(url):
    try:
        response = requests.get(url, allow_redirects=False)
        return (200 <= response.status_code < 300)
    except:
        return False
    
def to_jsonld(rdf):
    g = ConjunctiveGraph()
    g.parse(rdf)
                
    bnode_info = {}
    entities = g.subject_objects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
    for ent, ent_type in entities: 
        if ent not in bnode_info:
            bnode_info[ent] = dict()  
        bnode_info[ent]["@type"] = ent_type  
        for p, o in g.predicate_objects(ent):                
            # Ignore type assertions and links to blank nodes
            if p.n3() == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                continue
                                    
            bnode_info[ent][p] = o
    
    results = dict()
    
    for ent, info in bnode_info.items():
        for k, v in info.items():
            if isinstance(v, BNode):
                stub = bnode_info.pop(v)
                info[k] = stub
    
    return bnode_info

def get_type_definition(schema_type_url, prop=None, parents=True, simplify=False, verbose=False) -> Union[Dict, List]:
    """Get the definition for specific Schema.org class. 
    The result is a list of predicate or a dictionary with predicate as key, 
    expected types and comment as values.
    """
    
    def _simplify(url: str):
        return url.strip("<>").replace("http://schema.org/", "")
    
    g = ConjunctiveGraph()
    g.parse("https://schema.org/version/latest/schemaorg-all-http.nt")
    
    results = dict()
    
    # Get the attribute of class
    query = f"""
    SELECT ?prop ?range ?comment WHERE {{
        ?prop <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .
        ?prop <http://schema.org/domainIncludes> {URIRef(schema_type_url).n3()} .
        ?prop <http://schema.org/rangeIncludes> ?range .
        ?prop <http://www.w3.org/2000/01/rdf-schema#comment> ?comment .
    }}
    """
    
    if prop:
        prop_clean = URIRef(prop).n3()
        
        query = f"""
        SELECT ?range ?comment WHERE {{
            {prop_clean} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .
            {prop_clean} <http://schema.org/domainIncludes> {URIRef(schema_type_url).n3()} .
            {prop_clean} <http://schema.org/rangeIncludes> ?range .
            {prop_clean} <http://www.w3.org/2000/01/rdf-schema#comment> ?comment .
        }}
        """
            
    qresults = g.query(query)    
    for row in qresults:
        prop_clean = prop
        if prop is None:
            prop_clean = row.get("prop").n3()
        expected_type = row.get("range").n3()
        comment = row.get("comment").toPython().strip()
        if simplify:
            prop_clean = _simplify(prop_clean)
            expected_type = _simplify(expected_type)

        if prop_clean not in results:
            results[prop_clean] = dict()

        if results[prop_clean].get("expected_types") is None:
            results[prop_clean]["expected_types"] = []
        
        if expected_type not in results[prop_clean]["expected_types"]:
            results[prop_clean]["expected_types"].append(expected_type)
            
        results[prop_clean]["comment"] = comment
    
    if not verbose:
        results = list(results.keys())
            
    # Recursively get the attributes of parent classes
    if parents:
        parent_classes = g.objects(URIRef(schema_type_url), URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"))
        for parent_class in parent_classes:
            p_results = get_type_definition(parent_class, prop=prop, simplify=simplify, verbose=verbose)
            
            if verbose:
                results.update(p_results)
            else:
                results.extend(p_results)
    
    return results
    

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
        
    cache_file = f".cache/{md5ingest}_raw.html"
    if not os.path.exists(cache_file):        
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
        else:
            print(f"No records found for {target_url}")
    return scrape_webpage(cache_file)

def _html2txt(content):
    def skip_certain_tags(h2t, tag, attrs, start):
        for attr in attrs.values():
            if attr is None: continue
            if re.search(r"(navigation|footer|header)", attr.lower()) is not None:
                return False
                
        if tag in ['header', 'footer', 'nav', 'script', 'style']:
            return False
    
    def detect_main(tag):
        if tag.name == 'main':
            return True
       
        return False
    
    def chain_rules(site_rules, root):
        block = root
        if isinstance(site_rules, list):
            for site_rule in site_rules:
                block = chain_rules(site_rule, root=block)
                
        elif isinstance(site_rules, dict):
            tag = site_rules["tag"]
            attrs = site_rules["attrs"]
            block = root.find(tag, attrs=attrs)
        
        return block
    
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.tag_callback = skip_certain_tags
    
    # Retrieve the content of <main>
    soup = BeautifulSoup(content, 'html.parser')
        
    url = soup.find("link", rel="canonical").get("href")
    host = urlparse(url).netloc
        
    rules_file = ".cache/html_tags.json"
    # The content could be clearly in main tag
    if soup.find("main"):
        content = str(soup.find("main"))
    # or where the microdata a are
    elif soup.find_all(attrs={'itemscope': True, 'itemprop': True, 'itemtype': True}):
        tags = [ str(tag) for tag in soup.find_all(attrs={'itemscope': True, 'itemprop': True, 'itemtype': True}) ]
        content = "\n".join(tags)
    # or where we say they are
    elif os.path.exists(rules_file):
        with open(rules_file, "r") as f:
            rules = json.load(f)
            site_rules = rules.get(host)
            
            if site_rules:
                content = str(chain_rules(site_rules, root=soup))
            else:
                raise RuntimeError(f"Could not extract content for host {host}. Review the content manually and put the correct tag in {rules_file}!")
    # if not, create a rule                                 
    return converter.handle(content)

def _trafilatura(content):
    result = extract(content, include_links=True, include_images=True, include_tables=True, deduplicate=True)
    return result

def scrape_webpage(cache_file):
    
    with open(cache_file, "r") as f:
        content = f.read()
        try:
            text = _html2txt(content)
            #text = _trafilatura(content)
            return text
        except RuntimeError as e:
            raise RuntimeError(f"{str(e)}. HTML file: {cache_file}")
    
def get_n_grams(text, n):
    import nltk
    from nltk import ngrams
    from collections import Counter
    
    # Tokenize the text into words
    #words = nltk.word_tokenize(text)
    words = re.split(r"\s+", text) 
    
    # Remove punctuations
    #words = [word for word in words if word not in list(string.punctuation)]
    
    # Generate n-grams
    thirteen_grams = ngrams(words, n)
    
    return [' '.join(gram) for gram in thirteen_grams]
       
def filter_graph_by_type(graph: ConjunctiveGraph, schema_type, root=None, visited: list=[]):
    """Extract a subgraph where the root is an entity with certain type

    Args:
        graph (ConjunctiveGraph): _description_
        schema_type (_type_): _description_
        root (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    result = ConjunctiveGraph()
    
    if root is None:
        visited = []
        target_type = URIRef(lookup_schema_type(schema_type))
        for s in graph.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), target_type):
            subgraph = filter_graph_by_type(graph, schema_type, root=s, visited=visited)
            for s1, p, o in subgraph:
                result.add((s1, p, o))
                visited.append(s)
    else:
        for p, o in graph.predicate_objects(root):
            result.add((root, p, o))
                        
            if o not in visited:      
                visited.append(o)
                subgraph = filter_graph_by_type(graph, schema_type, root=o, visited=visited)
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