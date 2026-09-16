#!/usr/bin/env python3
"""Phase8 K00: pin/download official teacher; CPU strict-load and tokenizer checks only."""
import json,hashlib,sys,traceback,subprocess,time
from pathlib import Path
import requests,torch
from transformers import AutoTokenizer
R=Path('/data/projects/minimind-lab');sys.path.insert(0,str(R/'minimind'))
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
OUT=Path('/data/artifacts/minimind-lab/phase8-k00-teacher-gate-20260909')
REV='24a40130901f5229e841b41f0c070ebd95f3631d'
EXPECTED='a050020ea6d1b9e824693d0db525b1a0a8b40f36a934ea8d89da161368f20cc1'
UPSTREAM='393e387e9ad99f0f04c296e4c5e7353f4444629f'
STUDENT=Path('/data/artifacts/minimind-lab/S10-ifeval-curriculum-v4-20260907/checkpoints/s08_best_val_768.pth')
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True);assert not (OUT/'compatibility.json').exists(),'already prepared'
 torch.set_num_threads(4)
 base='https://modelscope.cn/api/v1/models/gongjy/minimind-3-pytorch/repo/files'
 resp=requests.get(base,params={'Revision':REV,'Recursive':'true'},timeout=30);resp.raise_for_status()
 meta=resp.json();assert meta['Success']
 entry=next(x for x in meta['Data']['Files'] if x['Path']=='full_sft_768_moe.pth')
 assert entry['Sha256']==EXPECTED and entry['Size']==406719725
 dump(OUT/'source-metadata.json',{'repository':'gongjy/minimind-3-pytorch','revision':REV,'file':entry,'tokenizer_source_commit':UPSTREAM,'published_variant':'full_sft'})
 ck=OUT/'full_sft_768_moe.pth'
 if not ck.exists():
  url=f'https://modelscope.cn/models/gongjy/minimind-3-pytorch/resolve/{REV}/full_sft_768_moe.pth'
  partial=OUT/'full_sft_768_moe.pth.partial';assert not partial.exists(),'inspect previous partial before retry'
  with requests.get(url,stream=True,timeout=(20,60)) as res:
   res.raise_for_status()
   with partial.open('xb') as f:
    for data in res.iter_content(8388608):f.write(data)
  assert partial.stat().st_size==entry['Size'] and sha(partial)==EXPECTED
  partial.rename(ck)
 assert sha(ck)==EXPECTED
 print('teacher downloaded and SHA verified',flush=True)
 tokens=OUT/'upstream-tokenizer';tokens.mkdir(exist_ok=True)
 for name in ['tokenizer.json','tokenizer_config.json']:
  content=subprocess.check_output(['git','show',f'{UPSTREAM}:model/{name}'],cwd=R)
  p=tokens/name
  if p.exists():assert p.read_bytes()==content
  else:p.write_bytes(content)
 a=AutoTokenizer.from_pretrained(tokens,local_files_only=True)
 b=AutoTokenizer.from_pretrained(R/'minimind/model',local_files_only=True)
 c=AutoTokenizer.from_pretrained('/data/artifacts/minimind-lab/D01-s10-preference-baseline-20260908/exported-fp32',local_files_only=True)
 for t in [b,c]:
  assert a.get_vocab()==t.get_vocab(),'vocabulary mapping'
  assert a.chat_template==t.chat_template,'chat template'
  assert (a.bos_token_id,a.eos_token_id,a.pad_token_id)==(t.bos_token_id,t.eos_token_id,t.pad_token_id),'special tokens'
  assert json.loads(a.backend_tokenizer.to_str())==json.loads(t.backend_tokenizer.to_str()),'tokenizer pipeline'
 assert sha(STUDENT)=='46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131'
 info={}
 for arm,path,moe in [('student',STUDENT,False),('teacher',ck,True)]:
  model=MiniMindForCausalLM(MiniMindConfig(use_moe=moe))
  state=torch.load(path,map_location='cpu',weights_only=True)
  model.load_state_dict(state,strict=True);model.eval()
  ids=b('中国的首都是北京。',return_tensors='pt',add_special_tokens=False).input_ids
  with torch.no_grad():logits=model(ids).logits
  assert logits.shape[-1]==6400 and torch.isfinite(logits).all()
  info[arm]={'checkpoint':str(path),'sha256':sha(path),'strict_load':True,'finite_cpu_forward':True,'parameters':sum(p.numel() for p in model.parameters())}
  del state,model
 dump(OUT/'compatibility.json',{'status':'compatibility-passed-not-quality-qualified','arms':info,'tokenizer_semantics_equal':True,'tokenizer_source_commit':UPSTREAM,'tokenizer_files_sha256':{n:sha(tokens/n) for n in ['tokenizer.json','tokenizer_config.json']},'teacher_tokenizer_provenance':'official PyTorch checkpoint interpreted with pinned official MiniMind tokenizer; pth does not embed tokenizer','quality_gate':'pending fresh paired fixed instruction evaluation; no distillation authorized by compatibility alone'})
 print('COMPATIBILITY_PASSED',json.dumps(info),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  OUT.mkdir(exist_ok=True);dump(OUT/f'prepare-failure-{time.time_ns()}.json',{'traceback':traceback.format_exc()});raise
