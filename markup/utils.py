from hashlib import md5
from io import StringIO
import re
import time
from bs4 import BeautifulSoup
import requests

from rdflib import Graph

def ping(url):
    try:
        response = requests.get(url, allow_redirects=False)
        return (200 <= response.status_code < 300)
    except:
        return False
    
def get_archive_url(url, wait=False):
    """Use Wayback Machine to retrieve the last working version of the said page

    Args:
        url (_type_): The original URL

    Returns:
        _type_: The wayback machine URL if possible
    """
    
    if ping(url): 
        print(f"{url} is alive!")
        return url
    
    wayback_url = f"http://web.archive.org/cdx/search/cdx"
    params = {
        "url": url,
        "matchType": "prefix",
        "collapse": "urlkey",
        "output": "json",
        "fl": "timestamp",
        "filter": "statuscode:200",
        "limit": "1",
        "sort": "asc"
    }
    
    status_code = -1
    response = None
    while status_code < 200 or status_code >= 300:
        response = requests.get(wayback_url, params=params)
        if response.status_code == 429: # Too many requests, https://archive.org/details/toomanyrequests_20191110
            if wait:
                print(f"Could not scrape {url}. Maximum rate (15 requests/min) reached! Cooldown for 5 minutes...")
                time.sleep(5*60+1) # web.archive.org/save has a rate limit of 25 requests/minute, resetting each minute
                continue
            else: raise RuntimeError(f"Could not scrape {url}.")
        
    #response.raise_for_status()
    results = response.json()
            
    if len(results) > 1:
        oldest_timestamp = results[1][0]
        archived_url = f"http://web.archive.org/web/{oldest_timestamp}/{url}"
        return archived_url
    else:
        return None

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

    page_content = requests.get("https://www.w3.org/services/html2txt", params={
        "url": target_url,
        "noinlinerefs": "on",
        "nonums": "on",
        "endrefs": "on",
        "internalrefs": "on"
    }).text

    references = {}
    content = ""
    with StringIO(page_content) as fs:
        isRef = False
        isContent = False
        for line in fs.readlines():
            line_strip = line.strip()

            if line_strip == "References":
                isRef = True
                continue

            if isRef and len(line_strip) > 0:
                srch_res = re.search(r"(\d+)\.\s+(.*)", line_strip)
                if srch_res is None: continue
                ref_id = srch_res.group(1)
                ref_val = srch_res.group(2)
                if line not in references:
                    references[ref_id] = ref_val
            else:
                content += line

    for ref_id, ref_val in references.items():
        content = re.sub(rf"\[{ref_id}\]", f"[ {ref_val} ]", content)

    return content, list(references.values())

def lookup_schema_type(schema_type):
    g = Graph()
    g.parse("https://schema.org/version/latest/schemaorg-all-https.nt")

    query = f"""
    SELECT ?class WHERE {{
        ?class <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .
        FILTER ( contains( lcase(str(?class)), {repr(str(schema_type).lower())}) )
    }}
    """

    results = g.query(query)
    candidates = [row.get("class") for row in results ]
    return str(candidates[0]).strip("<>")

def html2text(url):
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html5lib')
        
        # Define elements to exclude (e.g., header, footer, navbar, etc.)
        elements_to_exclude = ['header', 'footer', 'nav']
        
        # Remove specified elements from the parsed HTML
        for element in elements_to_exclude:
            for tag in soup.find_all(element):
                tag.extract()  # Remove the tag and its content
        
        # Extract and return the text content
        text_content = soup.get_text()
        return text_content
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")
        return None