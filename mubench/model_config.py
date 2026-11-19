from pathlib import Path
import json,hashlib
from .config import ROOT

def generate(model,paths,output):
    config=json.loads((ROOT/'configs/models'/f'{model}.json').read_text())
    def resolve(x):
        if isinstance(x,dict):return {k:resolve(v) for k,v in x.items()}
        if isinstance(x,list):return [resolve(v) for v in x]
        if isinstance(x,str) and x.startswith('@'):
            key,_,suffix=x[1:].partition('/')
            if key=='adapter':base=ROOT/'adapters/native'/model
            else:
                if key not in paths:raise ValueError('Missing model path '+key)
                base=Path(paths[key]).expanduser().resolve()
            return str(base/suffix) if suffix else str(base)
        return x
    config=resolve(config)
    if 'loader_module' in config:config['loader_sha256']=hashlib.sha256(Path(config['loader_module']).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'config':str(output),'model':model}))
