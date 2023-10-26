import time
import requests

def ping(url):
    try:
        response = requests.get(url, allow_redirects=False, timeout=1)
        return (200 <= response.status_code < 300)
    except:
        return False
    
def get_website_content(url):
    """Use Wayback Machine to retrieve the last working version of the said page

    Args:
        url (_type_): The original URL

    Returns:
        _type_: The wayback machine URL if possible
    """
    
    if ping(url): return url
    
    wayback_url = f"http://web.archive.org/cdx/search/cdx"
    params = {
        "url": url,
        "matchType": "prefix",
        "collapse": "urlkey",
        "output": "json",
        "fl": "timestamp",
        "filter": "statuscode:200",
        "limit": "1",
        "sort": "desc"
    }
    
    status_code = -1
    response = None
    while status_code < 200 or status_code >= 300:
        response = requests.get(wayback_url, params=params)
        if response.status_code == 429: # Too many requests, sleep for 5 seconds
            print("Maximum rate (25 requests /min) reached! Cooldown for 60s...")
            time.sleep(60) # web.archive.org/save has a rate limit of 25 requests/minute, resetting each minute
            continue
        
    #response.raise_for_status()
    results = response.json()
            
    if len(results) > 1:
        oldest_timestamp = results[1][0]
        archived_url = f"http://web.archive.org/web/{oldest_timestamp}/{url}"
        return archived_url
    else:
        return None