from copy import deepcopy
from hashlib import md5
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
from pprint import pprint
import re
from typing import Any, Dict, List, Union
from urllib.parse import quote_plus, urlparse
import warnings
import backoff
import html2text
from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from warcio.archiveiterator import ArchiveIterator

from rdflib import BNode, ConjunctiveGraph, Graph, Literal, URIRef

import extruct

CC_INDEX_SERVER = 'http://index.commoncrawl.org/'
LANGUAGES_CACHE_FILE = ".cache/languages.cache"  
INDEX_NAME = 'CC-MAIN-2021-43'    
    
def html_to_rdf_extruct(html_source) -> ConjunctiveGraph:
        id = Path(html_source).stem
    
        data = None
        
        with open(html_source, "r") as f:
            html_content = f.read()  
            data = extruct.extract(
                html_content, syntaxes=["microdata", "rdfa", "json-ld"], errors="ignore"
            )
        
        kg_jsonld = ConjunctiveGraph()
        if "json-ld" in data.keys():
            for md in data["json-ld"]:
                kg_jsonld += kg_jsonld.parse(
                    data=json.dumps(md, ensure_ascii=False),
                    format="json-ld",
                    publicID=id,
                )

        kg_rdfa = ConjunctiveGraph()
        if "rdfa" in data.keys():
            for md in data["rdfa"]:
                kg_rdfa += kg_rdfa.parse(
                    data=json.dumps(md, ensure_ascii=False),
                    format="json-ld",
                    publicID=id,
                )

        kg_microdata = ConjunctiveGraph()
        if "microdata" in data.keys():
            for md in data["microdata"]:
                kg_microdata += kg_microdata.parse(
                    data=json.dumps(md, ensure_ascii=False),
                    format="json-ld",
                    publicID=id,
                )

        kg_extruct = kg_jsonld + kg_rdfa + kg_microdata

        return kg_extruct
    
def jsonld_search_property(stub, key): 
    """Recursively search for a property in a JSONLD

    Args:
        stub (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(stub, dict):
        if key in stub.keys():
            return { key: stub[key] }
        for v in stub.values():    
            result = jsonld_search_property(v, key)
            if result: return result
    elif isinstance(stub, list):
        for item in stub:
            result = jsonld_search_property(item, key)
            if result: return result
    
    # raise ValueError(f"Could not find {key} in {stub}")
    return None
        
def get_schema_example(schema_url, focus=False):
        
    """Scrape the schema.org page for examples

    Args:
        schema_url (_type_): schema.org page for property or class
        focus (bool, optional): only retrieve the part where `schema_url` is a item key in jsonld example. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    # text = requests.get(str(schema_url)).text
    # soup = BeautifulSoup(text, "html.parser")
        
    # examples = soup.find_all("div", class_="jsonld")
    
    results = []
    
    g = ConjunctiveGraph()
    example_file = os.path.realpath("schemaorg/examples/schemaorg-all-examples.ttl")
    g.parse(example_file)
    
    query = f"""
    SELECT ?jsonld WHERE {{
        {URIRef(schema_url).n3()} <http://example.org/hasExample> ?example .
        ?example <http://example.org/json> ?jsonld .
    }}
    """
    
    examples = []
    qres = g.query(query)
    for qr in qres:
        examples.append(qr.get("jsonld").toPython())
        
    for example in examples:
        # jsonld_str = example.find("pre", class_="prettyprint").get_text().strip().split("\n")
        soup = BeautifulSoup(example, "html.parser")
        q = soup.find("script")
        jsonld_str = q.get_text() if q else soup.get_text()
        jsonld = json.loads(jsonld_str)
        if focus:
            jsonld = jsonld_search_property(jsonld, schema_simplify(URIRef(schema_url)))
        results.append(json.dumps(jsonld))
    
    return results
    
    
def schema_simplify(node):
    if isinstance(node, URIRef):
        return node.n3().strip("<>").replace("http://schema.org/", "")#.replace("file://", "")
    elif isinstance(node, Literal):
        return node.value or str(node)
    elif isinstance(node, list):
        return [ schema_simplify(n) for n in node ]
    elif isinstance(node, (str, float, int)): # Python primitives
        return node
    else:
        raise NotImplementedError(f"{type(node)} is not yet supported!")

def close_ontology(graph: ConjunctiveGraph):
    """Load an input SHACL shape graph and close each shape 
    by bringing all property from parent class to currend class shape 
    then add sh:closed at the end
    """             
    query = f"""
    SELECT DISTINCT ?shape ?parentShape ?parentProp WHERE {{
        ?shape  a <http://www.w3.org/ns/shacl#NodeShape> ;
                a <http://www.w3.org/2000/01/rdf-schema#Class> ;
                <http://www.w3.org/2000/01/rdf-schema#subClassOf>* ?parentShape .
                
        ?parentShape <http://www.w3.org/ns/shacl#property> ?parentProp .
        FILTER(?parentShape != ?shape)
    }}
    """ 
    
    results = graph.query(query)
    visited_shapes = set()
    for result in results:
        shape = result.get("shape")
        parent_prop = result.get("parentProp")
        graph.add((shape, URIRef("http://www.w3.org/ns/shacl#property"), parent_prop))
        graph.add((shape, URIRef("http://www.w3.org/ns/shacl#closed"), Literal(True)))
        
        # subj sh:ignoredProperties ( rdf:type owl:sameAs )
        # https://www.w3.org/TR/turtle/#collections
        if shape not in visited_shapes:
            ignored_props = graph.collection(BNode())
            ignored_props += [URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef("http://www.w3.org/2002/07/owl#sameAs")]
            
            graph.add((shape, URIRef("http://www.w3.org/ns/shacl#ignoredProperties"), ignored_props.uri))
            visited_shapes.add(shape)
    
    # Replace xsd:float with xsd:double
    for prop, value in graph.subject_objects(URIRef("http://www.w3.org/ns/shacl#datatype")):
        if value == URIRef("http://www.w3.org/2001/XMLSchema#float"):
            graph.set((prop, URIRef("http://www.w3.org/ns/shacl#datatype"), URIRef("http://www.w3.org/2001/XMLSchema#double")))
        elif value == URIRef("http://www.w3.org/2001/XMLSchema#date"):
            graph.set((prop, URIRef("http://www.w3.org/ns/shacl#datatype"), URIRef("http://schema.org/Date")))
        elif value == URIRef("http://www.w3.org/2001/XMLSchema#dateTime"):
            graph.set((prop, URIRef("http://www.w3.org/ns/shacl#datatype"), URIRef("http://schema.org/DateTime")))
    
    return graph

def extract_preds(graph: ConjunctiveGraph, ref_type, root=None, visited: list=[], depth_limit=1, depth=0, simplify=False):
    results = set()
    if root is None:
        for s in graph.subjects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef(ref_type)):
            visited.append(s)
            results.update(
                extract_preds(
                    graph, ref_type, root=s, simplify=simplify,
                    visited=visited, depth_limit=depth_limit, depth=depth
                )
            )
    else:
        for p, o in graph.predicate_objects(root):
            results.add(p)
            if o not in visited and depth < depth_limit:
                visited.append(o)
                results.update(
                    extract_preds(
                        graph, ref_type, root=o, simplify=simplify,
                        visited=visited, depth_limit=depth_limit, depth=depth+1
                    )
                )
                
    if simplify:
        results = set([ schema_simplify(p) for p in results ])
                
    return results
    
def to_jsonld(rdf, filter_by_type=None, simplify=False, clean=False):
    
    g = None
    if isinstance(rdf, Graph):
        g = rdf
    else:
        g = ConjunctiveGraph()
        g.parse(rdf)
    
    # Suppose that the input jsonld is filtered by type
    # if filter_by_type:
    #     g = filter_graph_by_type(g, filter_by_type)
         
    # Build a basic dictionary with RDFlib objects       
    bnode_info = {}
    entities = g.subject_objects(URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")) # This only works if rdf is perfect
    for ent, ent_type in entities: 
        if ent not in bnode_info:
            bnode_info[ent] = dict()  
        bnode_info[ent]["@type"] = ent_type  
        for p, o in g.predicate_objects(ent):   
            # Ignore type assertions and links to blank nodes
            if p.n3() == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
                continue
            
            if p not in bnode_info[ent]:
                bnode_info[ent][p] = []
                
            bnode_info[ent][p].append(o)

    # Replace BNodes with their real values
    redundants = set()
    for ent, info in bnode_info.items():
        for key, values in list(info.items()):
            
            # TODO: There should not be any cycle in a markup, so add special cases here
            if key in [URIRef("http://schema.org/url")]:
                continue
            for idx, v in enumerate(values):
                if isinstance(v, (BNode, URIRef)):
                    # Key Error means that the type assertion triple was dropped silently by rdflib, 
                    # indicating type error that should be picked up by shacl
                    try:
                        stub = bnode_info.get(v)
                        # If item is an bnode with URI as id...
                        if isinstance(v, URIRef): 
                            # ... item is a simple URIRef, nothing to do
                            if stub is None:
                                continue
                            
                            # ... item is a BNode with other attributes, add url to BNode
                            stub["url"] = v.toPython()
                        else:    
                            redundants.add(v)
                        bnode_info[ent][key][idx] = stub
                    except KeyError as err:
                        bnode_info[ent][key] = None
                        warnings.warn(f"{err}. It means that the related type assertion triple was dropped silently by rdflib, hinting a type error that could be picked up by SHACL. Assinging None...")
            
            # If values is a list of length 1, replace with field with that item
            if len(bnode_info[ent][key]) == 1:
                bnode_info[ent][key] = bnode_info[ent][key][0]
    
    # Remove redundant BNodes
    for redundant in redundants:
        if redundant in bnode_info.keys():
            bnode_info.pop(redundant)
            
    
    # Remove root BNodes with the actual markup
    # There should be only 1 root
    # TODO: many roots
    if len(bnode_info) == 1:
        for root, markup in bnode_info.items():
            bnode_info = markup
                        
    # Simplify
    if simplify:        
        bnode_info = transform_json(bnode_info, schema_simplify, schema_simplify)
    
    # Clean
    if clean:
        bnode_info = transform_json(
            bnode_info,
            value_transformer=lambda v: v[0] if len(v) == 0 else v
        )
    
    bnode_info["@context"] = "http://schema.org"
    return bnode_info

def transform_json(stub, key_transformer=None, value_transformer=None):  
    key_transformer = key_transformer or (lambda k: k)
    value_transformer = key_transformer or (lambda v: v)
    
    if isinstance(stub, list):
        result = [ transform_json(item, key_transformer, value_transformer) for item in stub ]    
        return result[0] if len(result) == 1 else result
    elif isinstance(stub, dict):
        # visited = set()
        result = {}
        ent_type = None
        for k, values in stub.items():
            if k == "@type":
                ent_type = value_transformer(values)
                result["@type"] = ent_type
                continue
            
            if k == "@context": continue

            new_k = key_transformer(k)
            
            # Recursively add prompt for dependant entities             
            if values is None: continue
            result[new_k] = transform_json(values, key_transformer, value_transformer)
            # visited.add(k)
        
        return result
    else:
        return value_transformer(stub)

def collect_json(stub, value_transformer, *args) -> List[Any]:
    results = []
    if isinstance(stub, dict):
        ent_type = None
        for k, values in stub.items():
            if k == "@type":
                ent_type = values  
                continue
            
            if k == "@context": continue

            # Recursively add prompt for dependant entities             
            if values is None: continue
            args = [k, values, ent_type]
            results.extend(collect_json(values, value_transformer, *args))
                                
    elif isinstance(stub, list):
        for item in stub:
            results.extend(collect_json(item, value_transformer, *args))
    else:
        results.append(value_transformer(*args))
    return results

def get_type_definition(schema_type_url=None, prop=None, parents=True, simplify=False, include_expected_types=False, include_comment=False) -> Union[Dict, List]:
    """Get the definition for specific Schema.org class. 
    The result is a list of predicate or a dictionary with predicate as key, 
    expected types and comment as values.
    """
    if schema_type_url is None: parents = False
    
    g = ConjunctiveGraph()
    g.parse("https://schema.org/version/latest/schemaorg-all-http.nt")
    
    results = dict()
    
    prop_var = URIRef(prop).n3() if prop else "?prop"
    domain_var = URIRef(schema_type_url).n3() if schema_type_url else "?domain"
    
    # Get the attribute of class
    query = f"""
    SELECT ?prop ?range ?comment WHERE {{
        {prop_var} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .
        {prop_var} <http://schema.org/domainIncludes> {domain_var} .
        {prop_var} <http://schema.org/rangeIncludes> ?range .
        {prop_var} <http://www.w3.org/2000/01/rdf-schema#comment> ?comment .
    }}
    """
    
    # if prop:
    #     prop_clean = URIRef(prop).n3()
            
    qresults = g.query(query)        
    for row in qresults:
        prop_clean = prop
        if prop is None:
            prop_clean = row.get("prop")
        expected_type = row.get("range")
        comment = row.get("comment").toPython().strip()
        if simplify:
            prop_clean = schema_simplify(prop_clean)
            expected_type = schema_simplify(expected_type)
        else:
            if prop is None:
                prop_clean = prop_clean.n3()
            expected_type = expected_type.n3()

        if prop_clean not in results:
            results[prop_clean] = dict()

        if include_expected_types:
            if results[prop_clean].get("expected_types") is None:
                results[prop_clean]["expected_types"] = []
            
            if expected_type not in results[prop_clean]["expected_types"]:
                results[prop_clean]["expected_types"].append(expected_type)
        
        if include_comment:
            results[prop_clean]["comment"] = comment
    
    if not include_expected_types and not include_comment:
        results = list(results.keys())
            
    # Recursively get the attributes of parent classes
    if parents:
        parent_classes = g.objects(URIRef(schema_type_url), URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"))
        for parent_class in parent_classes:
            p_results = get_type_definition(parent_class, prop=prop, simplify=simplify, include_expected_types=include_expected_types, include_comment=include_comment)
            
            if include_expected_types or include_comment:
                results.update(p_results)
            else:
                results.extend(p_results)

    return results
    

def md5hex(obj):
    return md5(str(obj).encode()).hexdigest()

#TODO: backoff instead
@backoff.on_predicate(backoff.expo, predicate=lambda x: x is None)
def search_cc_index(url):    
    encoded_url = quote_plus(url)
    index_url = f'{CC_INDEX_SERVER}{INDEX_NAME}-index?url={encoded_url}&output=json'
    # print(index_url)
    response = requests.get(index_url)
    # print("Response from CCI:", response.text)  # Output the response from the server
    if response.status_code == 200:
        records = response.text.strip().split('\n')
        return [json.loads(record) for record in records]
    else:
        return None
        
def lang_detect(target_url):
    languages = None
    Path(LANGUAGES_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    LANGUAGES_CACHE = dict()
    if os.path.exists(LANGUAGES_CACHE_FILE):
        with open(LANGUAGES_CACHE_FILE, "r") as cache_file:
            LANGUAGES_CACHE = json.load(cache_file)

    md5ingest = md5hex(target_url)
    if md5ingest in LANGUAGES_CACHE:
        languages = LANGUAGES_CACHE[md5ingest]
        return languages
    else:   
        records = search_cc_index(target_url)
        for record in records:
            languages = record["languages"].split(",") if "languages" in record else ["unknown"]
            LANGUAGES_CACHE[md5ingest] = languages

        with open(LANGUAGES_CACHE_FILE, "w") as cache_file:
            json.dump(LANGUAGES_CACHE, cache_file)
        
        return languages

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

def _html2txt(content, force=False):
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
    
    def process_rules(site_rules: dict, root):
        block = root
        if "chain" in site_rules.keys():
            for site_rule in site_rules["chain"]:
                block = process_rules(site_rule, root=block)
        elif "list" in site_rules.keys():
            block = [process_rules(site_rule, root=root) for site_rule in site_rules["list"]]
        else:
            tag = site_rules["tag"]
            attrs = site_rules["attrs"]
            block = root.find(tag, attrs=attrs)
            if block is None:
                raise ValueError(f"Couldn't find element for tag {repr(tag)}, attrs {attrs} ")
        
        return block
    
    def stringify(block):
        if isinstance(block, list):
            return "\n".join([stringify(b) for b in block])
        return str(block)        
    
    def contains_url(tag):
        found = False
        if tag.name == "link":
            rel_attrs = tag.get("rel")
            if rel_attrs is not None:
                if isinstance(rel_attrs, list):
                    found = "canonical" in rel_attrs
                else:
                    found = (rel_attrs == "canonical")
        elif tag.name == "meta":
            property_attrs = tag.get("property")
            if property_attrs is not None:
                if isinstance(property_attrs, list):
                    found = "og:url" in property_attrs
                else:
                    found = ( property_attrs == "og:url" )
        return found            

    converter = html2text.HTML2Text()
    converter.ignore_links = False
    converter.tag_callback = skip_certain_tags
    
    # Retrieve the content of <main>
    soup = BeautifulSoup(content, 'html.parser')
    
    host = None
    # Get the URL of the page
    elements = soup.find_all(contains_url)
    if len(elements) > 0:
        all_urls = [element.get('href') if element.name == 'link' else element.get('content') for element in elements]
        url = all_urls[0]
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
                content = stringify(process_rules(site_rules, root=soup))
            elif not force:
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
       
def filter_graph(graph: ConjunctiveGraph, subj=None, pred=None, obj=None, root=None, visited: list=[]):
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
        for s, _, _ in graph.triples((subj, pred, obj)):
            subgraph = filter_graph(graph, subj=subj, pred=pred, obj=obj, root=s, visited=visited)
            for s1, p, o in subgraph:
                result.add((s1, p, o))
                visited.append(s)
    else:
        for p, o in graph.predicate_objects(root):
            result.add((root, p, o))
                        
            if o not in visited:      
                visited.append(o)
                subgraph = filter_graph(graph, obj, root=o, visited=visited)
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

def ping(url):
    try:
        response = requests.get(url, allow_redirects=False)
        return (200 <= response.status_code < 300)
    except:
        return False