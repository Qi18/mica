"""Shared, explicit MiniMind sequence-distillation encoding (no KL/logit mapping)."""
import json,hashlib,unicodedata
from pathlib import Path
ROOT=Path('/data/projects/minimind-lab')
CONFIG=ROOT/'configs/distill/phase8-sequence-v1.json'
def cfg():return json.loads(CONFIG.read_text())
def json_default(x):
 import numpy as np
 import torch
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
 if isinstance(x,(torch.dtype,torch.device,Path)):return str(x)
 if callable(x):return {'__callable__':getattr(x,'__module__','')+'.'+getattr(x,'__qualname__',type(x).__qualname__)}
 if isinstance(x,set):return sorted(x)
 raise TypeError('Unsupported JSON value: '+type(x).__name__)

def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=json_default)+'\n');t.replace(p)
def rows(p):return [json.loads(l) for l in Path(p).open()]
def write_rows(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:
  for r in data:f.write(json.dumps(r,ensure_ascii=False,default=json_default)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def norm(s):return ''.join(c.lower() for c in unicodedata.normalize('NFKC',s) if c.isalnum())
def encode(tok,row,answer=None):
 answer=row['completion'] if answer is None else answer
 prompt=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,open_thinking=False)
 prefix=tok.encode(prompt,add_special_tokens=False)
 full=tok.encode(prompt+answer+tok.eos_token,add_special_tokens=False)
 assert full[:len(prefix)]==prefix,'assistant boundary retokenization'
 labels=[-100]*len(prefix)+full[len(prefix):]
 assert len(full)<=cfg()['max_seq_len'],'no truncation allowed'
 assert sum(x!=-100 for x in labels[1:])>0
 return full,labels
def collate(tok,enc,device):
 import torch
 n=max(len(x[0]) for x in enc)
 out={k:[] for k in ['input_ids','labels','attention_mask']}
 for ids,lab in enc:
  pad=n-len(ids);out['input_ids'].append(ids+[tok.pad_token_id]*pad);out['labels'].append(lab+[-100]*pad);out['attention_mask'].append([1]*len(ids)+[0]*pad)
 out={k:torch.tensor(v,device=device) for k,v in out.items()}
 mask=out['labels']!=-100
 assert (out['input_ids'][mask]==out['labels'][mask]).all() and (out['attention_mask'][mask]==1).all()
 return out
