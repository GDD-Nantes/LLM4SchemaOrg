from glob import glob
import json
import os
import re

for logfn in glob("data/WDC/Pset/**/*.json", recursive=True):
    if re.search(r".*(factual|semantic).*", logfn) is None:
        continue

    logsize = os.stat(logfn).st_size
    if logsize > 1e6:
        print(f"Deleting {logfn} with size {logsize} bytes")
        #os.remove(logfn)
        continue

    with open(logfn, "r") as f:
        log = json.load(f)
        chunks = [ c for c in log.keys() if c.startswith("chunk_") ]
        if len(chunks) > 2:
            print(logfn)

        if len(chunks) > 10:
            print(f"Deleting {logfn} with {len(chunks)} chunks")
            #os.remove(logfn)