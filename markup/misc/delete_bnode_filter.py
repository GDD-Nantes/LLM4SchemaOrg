from glob import glob
import json
import os
from pathlib import Path
import re

for logfn in glob("data/WDC/Pset/**/*.json", recursive=True):
    if re.search(r".*(shacl|factual|semantic).*", logfn) is None:
        continue

    with open(logfn, "r") as f:
        log = json.load(f)

        # semantic, factual
        can_delete = False
        logitems = log.values() if any([ c.startswith("chunk_") for c in log.keys() ]) else log["msgs"]
        for logitem in logitems:
            if isinstance(logitem, dict):
                for query in logitem.keys():
                    if query in ["status", "score"]: continue
                    value = query.split("[TOK_Q_DELIM]")[1]
                    if re.search(r"^[0-9a-z]{30,}$", value) is not None:
                        can_delete = True
                        break
                if can_delete:
                    break
            elif isinstance(logitem, str):
                value = logitem.split("[TOK_Q_DELIM]")
                value = value[1] if len(value) > 1 else value[0]
                if re.search(r"^[0-9a-z]{30,}$", value) is not None:
                    can_delete = True
                    break
        
        if can_delete:
            #print(f"Deleting {logfn} with bnode {value}")
            base_fn = re.sub(r"_(pred|expected)", "",  f"{Path(logfn).parent}/{Path(logfn).stem}")
            for target in glob(f"{base_fn}*.*"):
                os.remove(target)
                print(f"Deleting {target}")

        