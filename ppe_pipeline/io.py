import json, os
from pathlib import Path

def load_json(path):
    with open(path,'r') as f:
        return json.load(f)

def save_json(obj, path, indent=2):
    path=str(path)
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f:
        json.dump(obj, f, indent=indent)
