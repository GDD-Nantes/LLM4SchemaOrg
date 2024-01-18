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
import urllib3

import trafilatura
from warcio.archiveiterator import ArchiveIterator

from rdflib import BNode, ConjunctiveGraph, Graph, Literal, URIRef
import extruct

CC_INDEX_SERVER = 'http://index.commoncrawl.org/'
LANGUAGES_CACHE_FILE = ".cache/languages.cache"  
INDEX_NAME = 'CC-MAIN-2022-40'    
    
def html_to_rdf_extruct(html_source) -> ConjunctiveGraph:
        id = Path(html_source).stem
    
        data = None
        
        with open(html_source, "r") as f:
            # null byte issue: https://github.com/scrapinghub/extruct/issues/112
            html_content = f.read().strip().replace('\x00', '').encode('utf8')  
            data = extruct.extract(
                html_content, syntaxes=["microdata", "rdfa", "json-ld"], errors="ignore"
                # html_content, syntaxes=["json-ld"], errors="ignore"
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

        kg_extruct = kg_jsonld #+ kg_rdfa + kg_microdata

        return kg_extruct
    
def jsonld_search_property(stub, key, value=None, parent=False): 
    """Recursively search for a property and value (optional) in a JSONLD

    Args:
        stub (_type_): _description_
        key (_type_): _description_
        parent (boolean): returns the parent entity if True, or just the property-value pair if False

    Returns:
        _type_: a list of stubs that correspond to search criteria
    """

    results = []

    def equals(a, b):
        if isinstance(a, list) and isinstance(b, list):
            return sorted(a) == sorted(b)
        return a == b

    if isinstance(stub, dict):

        for k, v in stub.items():

            if (
                (value is not None and k == key and equals(v, value)) or
                (value is None and k == key)
            ):
                results.append(stub if parent else { key: stub[key] })
            
            result = jsonld_search_property(v, key, value=value, parent=parent)
            if result: results.extend(result)
            
    elif isinstance(stub, list):
        for item in stub:
            result = jsonld_search_property(item, key, value=value, parent=parent)
            if result: results.extend(result)
    # raise ValueError(f"Could not find {key} in {stub}")
    return results
        
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
            jsonlds = jsonld_search_property(jsonld, schema_simplify(URIRef(schema_url)))
            for jsonld in jsonlds:
                results.append(json.dumps(jsonld))
        else:
            results.append(json.dumps(jsonld))
    
    return results
    
def schema_stringify(node):
    if isinstance(node, dict):
        return json.dumps(node)
    elif isinstance(node, list):
        return ", ".join([ schema_stringify(n) for n in node])
    else:
        return schema_simplify(node)

def schema_simplify(node):
    if isinstance(node, URIRef):
        result = node.n3().strip("<>")
        if result.startswith("http"):
            return result.replace("http://schema.org/", "")
        elif result.startswith("file://"):
            return Path(result).name
    elif isinstance(node, Literal):
        return node.value or str(node)
    elif isinstance(node, list):
        return [ schema_simplify(n) for n in node ]
    elif isinstance(node, (str, float, int)): # Python primitives
        return node
    elif isinstance(node, dict):
        result = {}
        for k,v in node.items():
            result[schema_simplify(k)] = schema_simplify(v)
        return result
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
    
def to_jsonld(rdf, filter_by_type=None, simplify=False, clean=False, keep_root=False):
    
    g = None
    if isinstance(rdf, Graph):
        g = rdf
    elif rdf.endswith(".json"):
        with open(rdf, "r") as f:
            jsonld = json.load(f)
            if isinstance(jsonld, dict) and "@context" not in jsonld.keys():
                return jsonld
            else:
                print("Parsing JSON-LD...")
                g = ConjunctiveGraph()
                g.parse(rdf, format="json-ld")
    else:
        g = ConjunctiveGraph()
        g.parse(rdf)
    
    # Suppose that the input jsonld is filtered by type
    # if filter_by_type:
    #     subgraph = ConjunctiveGraph()
    #     for target_class in filter_by_type:
    #         subgraph += filter_graph(g, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), URIRef(target_class))
        
    #     g = subgraph
         
    # Build a basic dictionary with RDFlib objects       
    bnode_info = {}
    for s, p, o in g:
        if isinstance(s, (BNode, URIRef)):
            if s not in bnode_info.keys():
                bnode_info[s] = {}
            
            if p == URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"):
                if bnode_info[s].get("@type") is None:
                    bnode_info[s]["@type"] = []
                bnode_info[s]["@type"].append(o)
                continue
            
            if p not in bnode_info[s].keys():
                bnode_info[s][p] = []
            bnode_info[s][p].append(o)
        if isinstance(o, (BNode, URIRef)):
            if o not in bnode_info.keys():
                bnode_info[o] = {}  
                
    # Replace BNodes with their real values
    def __replace_bnodes(D: dict, entity_value, r: set, entity_key=None):

        result = entity_value
        if isinstance(entity_value, dict):
            result = {}
            clone = deepcopy(entity_value)
            while len(clone) > 0:
                k, v = clone.popitem()  
                                               
                if k in [URIRef("http://schema.org/url"), "@type"]: 
                    result[k] = v
                    continue
                
                # When it is a empty dictionary, mark for remove
                if isinstance(v, dict) and len(v) == 0:
                    r.add(k)
                else:
                    stub, r = __replace_bnodes(D, v, r, entity_key=k)
                    result[k] = stub
                        
        elif isinstance(entity_value, list):
            result = [ __replace_bnodes(D, item, r, entity_key=entity_key)[0] for item in entity_value ]
            
        elif isinstance(entity_value, (BNode, URIRef)):
            # Key Error means that the type assertion triple was dropped silently by rdflib, 
            # indicating type error that should be picked up by shacl
            try:
                result = D.get(entity_value)
                # If item is an bnode with URI as id...
                if isinstance(entity_value, URIRef): 
                    if result is not None:       
                        # ... if the item type is not expected, then it's a simple URIRef
                        expected_types = get_expected_types(entity_key, simplify=False)
                        if expected_types is not None and result.get("@type") not in expected_types:
                            # print(f"{result['@type']} not in {expected_types}")
                            result = entity_value.toPython()  
                        # ... item is a bnode has information
                        elif isinstance(result, dict) and len(result) > 0:
                            result["@id"] = entity_value.toPython()
                        # .. item is a simple URIRef
                        else:
                            result = entity_value.toPython()   
                    # ... item is a simple URIRef, nothing to do
                    else:
                        result = entity_value.toPython()    
                
                r.add(entity_value)
                result, r = __replace_bnodes(D, result, r, entity_key=entity_key)
                
            except KeyError as err:
                warnings.warn(f"{err}. It means that the related type assertion triple was dropped silently by rdflib, hinting a type error that could be picked up by SHACL. Assinging None...")
        
        elif isinstance(entity_value, Literal):
            result = entity_value.toPython() 
            
        return result, r

    # Remove redundant BNodes
    bnode_info, redundants = __replace_bnodes(bnode_info, bnode_info, set())
    for redundant in redundants:
        if redundant in bnode_info.keys():
            bnode_info.pop(redundant)

    # Remove root BNodes with the actual markup
    if len(bnode_info) == 1 and not keep_root:
        bnode_info = list(bnode_info.values())[0]
        bnode_info["@context"] = "http://schema.org"
    else:
        for k in bnode_info.keys():
            bnode_info[k]["@context"] = "http://schema.org"
                        
    # Simplify
    if simplify:        
        bnode_info = transform_json(bnode_info, schema_simplify, schema_simplify)
    
    # Clean
    if clean:
        bnode_info = clean_json(bnode_info)
        
    return bnode_info

def clean_json(stub):
    result = transform_json(
        stub,
        value_transformer=lambda v: v[0] if isinstance(v, list) and len(v) == 1 else v
    )
    return result

def transform_json(stub, key_transformer=None, value_transformer=None):  
    key_transformer = key_transformer or (lambda k: k)
    value_transformer = value_transformer or (lambda v: v)
    
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
            
            if k in ["@context", "@id"]: 
                result[k] = values
                continue

            new_k = key_transformer(k)
            
            # Recursively add prompt for dependant entities             
            if values is None: continue
            result[new_k] = transform_json(values, key_transformer, value_transformer)
            # visited.add(k)
        
        return result
    else:
        return value_transformer(stub)

def collect_json(stub, *args, key_filter=lambda k,e: True, value_transformer=lambda k,v,e: v, **kwargs) -> List[Any]:
    results = []
    if isinstance(stub, dict):
        ent_type = None
        for k, values in stub.items():
            if k == "@type":
                ent_type = values  
                continue
            
            if k in ["@context", "@id"]: continue

            # Recursively add prompt for dependant entities             
            if values is None: continue
            args = [k, values, ent_type]
            
            if key_filter(k, ent_type):
                results.extend(collect_json(values, *args, key_filter=key_filter, value_transformer=value_transformer, **kwargs))
                                
    elif isinstance(stub, list):
        for item in stub:
            results.extend(collect_json(item, *args, key_filter=key_filter, value_transformer=value_transformer, **kwargs))
    else:
        results.append(value_transformer(*args, **kwargs))
    return results

def get_type_definition(schema_type_url=None, prop=None, parents=True, simplify=False, include_expected_types=False, include_comment=False) -> Union[Dict, List]:
    """Get the definition for specific Schema.org class. 
    The result is a list of predicate or a dictionary with predicate as key, 
    expected types and comment as values.
    """
    
    g = ConjunctiveGraph()
    # g.parse("https://schema.org/version/latest/schemaorg-all-http.nt")
    g.parse("schemaorg/schemaorg-all-http.nt")
    
    results = dict()    
    prop_var = URIRef(prop).n3() if prop else "?prop"
    domain_var = URIRef(schema_type_url).n3() if schema_type_url and prop is None else "?domain"
    
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
    if parents and schema_type_url:
        parent_classes = g.objects(URIRef(schema_type_url), URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"))
        for parent_class in parent_classes:
            p_results = get_type_definition(parent_class, prop=prop, simplify=simplify, include_expected_types=include_expected_types, include_comment=include_comment)
            
            if include_expected_types or include_comment:
                results.update(p_results)
            else:
                results.extend(p_results)

    return results

def get_expected_types(prop, **kwargs):
    prop_simplified = schema_simplify(URIRef(prop)) if kwargs.get("simplified") else prop
    definition = get_type_definition(prop=prop, **kwargs, include_expected_types=True)
    if len(definition) == 0: return None
    return definition.get(prop_simplified)["expected_types"]

def md5hex(obj):
    return md5(str(obj).encode()).hexdigest()

#TODO: backoff instead
@backoff.on_predicate(backoff.expo, predicate=lambda x: x['status_code'] not in [200, 404, 503])
@backoff.on_exception(backoff.expo, (urllib3.exceptions.MaxRetryError, requests.exceptions.ProxyError, requests.exceptions.RetryError))
def search_cc_index(url):  
    encoded_url = quote_plus(url)
    index_url = f'{CC_INDEX_SERVER}{INDEX_NAME}-index?url={encoded_url}&output=json'
        
    response = requests.get(index_url)
    # print("Response from CCI:", response.text)  # Output the response from the server
    
    data = None
    status_code = response.status_code
    
    if response.status_code == 200:
        reponse_text = response.text.strip()
        if len(reponse_text) == 0:
            raise RuntimeError("CommonCrawl returned empty response with code 2OO!")
        records = reponse_text.split('\n')
        data = [json.loads(record) for record in records]
    
    return { "status_code": status_code, "data": data }
    
        
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
    else:   
        records = search_cc_index(target_url)["data"]
        if records:
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
            # if not (len(languages) == 1 and "eng" in languages):
            #     raise RuntimeError("Skipping because the content is not in English!")

            s3_url = f'https://data.commoncrawl.org/{record["filename"]}'
            headers = {'Range': f'bytes={offset}-{offset+length-1}'}
            response = http.get(s3_url, headers=headers)
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
        records = search_cc_index(target_url)["data"]
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
            return None
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
    
    url = None
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
    # or where the article tag is
    # elif soup.find("article"):
    #     content = str(soup.find("article"))
    # or where we say they are
    elif os.path.exists(rules_file):
        with open(rules_file, "r") as f:
            rules = json.load(f)
            site_rules = rules.get(host)
            
            if site_rules:
                content = stringify(process_rules(site_rules, root=soup))
            elif not force:
                raise RuntimeError(f"Could not extract content for host {host} (url = {url}). Review the content manually and put the correct tag in {rules_file}!")
    # if not, create a rule                                 
    return converter.handle(content)

def _trafilatura(content, accept_threshold=0.3, verbose=False):
    """Extract content from a webpage. If the output contains less than 30% of the input tokens, reject (@cite CCNet: 70% of the page content is boilerplate).
    """
    html_ref = BeautifulSoup(content, "html.parser").get_text()
    html_clean = trafilatura.extract(content, include_links=True, include_images=True, include_tables=True, deduplicate=True)

    tok_ref = len(re.split(r"\s+", html_ref)) if html_ref else None
    tok_clean = len(re.split(r"\s+", html_clean)) if html_clean else None

    clean_ratio = tok_clean / tok_ref if tok_ref is not None and tok_clean is not None else None
    if clean_ratio is not None and clean_ratio > accept_threshold:
        return html_clean if not verbose else (html_clean, tok_ref, tok_clean, clean_ratio)
    
    print(f"Could not clean webpage (clean_ratio: {tok_clean}/{tok_ref} = {clean_ratio})")
    return None if not verbose else (None, tok_ref, tok_clean, clean_ratio)



def scrape_webpage(cache_file):
    
    with open(cache_file, "r") as f:
        content = f.read()
        try:
            # text = _html2txt(content)
            text = _trafilatura(content)
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
    # g.parse("https://schema.org/version/latest/schemaorg-all-http.nt")
    g.parse("schemaorg/schemaorg-all-http.nt")

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