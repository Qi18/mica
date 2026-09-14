import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_kvcache_gqa as b
out=b.ROOT/"experiments/08-inference/I01-I04-a10-20260914"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
state=torch.load(b.CKPT,map_location="cpu",weights_only=True)
result={"fixed_thresholds_unchanged":True,"tests":[]}
with torch.inference_mode():
    for d in ("float32","bfloat16"):
        refs={}
        for heads in (4,8):
            m=b.model_for(state,heads,getattr(torch,d),flash=False)
            for length in (16,128,512):
                ids=torch.randint(3,6400,(2,length+4),generator=torch.Generator().manual_seed(42+length)).cuda()
                full=m(ids,logits_to_keep=4).logits.cpu()
                p=m(ids[:,:length],use_cache=True,logits_to_keep=1).past_key_values
                steps=[]
                for i in range(4):
                    o=m(ids[:,length+i:length+i+1],past_key_values=p,use_cache=True,logits_to_keep=1)
                    steps.append(o.logits.cpu());p=o.past_key_values
                result["tests"].append({"dtype":d,"heads":heads,"length":length,"kind":"manual_attention_cache_vs_full",**b.compare(full,torch.cat(steps,1),d)})
                if heads==4: refs[length]=full
                else: result["tests"].append({"dtype":d,"length":length,"kind":"manual_attention_mha_vs_gqa",**b.compare(refs[length],full,d)})
                del p,o
            del m;b.clear()
    for length in (16,128,512):
        x=torch.randn(2,length,768,generator=torch.Generator().manual_seed(42)).cuda().bfloat16()
        w=state["model.layers.0.self_attn.k_proj.weight"].cuda().bfloat16()
        wm=w.reshape(4,96,768).repeat_interleave(2,0).reshape(768,768)
        a=torch.nn.functional.linear(x,w).reshape(2,length,4,96).repeat_interleave(2,2)
        c=torch.nn.functional.linear(x,wm).reshape(2,length,8,96)
        result.setdefault("projection_tests",[]).append({"length":length,**b.compare(a.cpu(),c.cpu(),"bfloat16")})
b.dump(out/"precision-diagnosis.json",result)
print(json.dumps(result,ensure_ascii=False))
