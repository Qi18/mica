#!/usr/bin/env python3
"""Check merge algebra in FP32 and separately measure saved FP16 deployment drift."""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--merged", required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    sys.path.insert(0,str(Path(".").resolve()))
    from model.model_mica import MicaConfig, MicaForCausalLM
    from model.model_lora import apply_lora, load_lora
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    base=MicaForCausalLM(MicaConfig(hidden_size=768,num_hidden_layers=8)).eval()
    base.load_state_dict(torch.load(a.base,map_location="cpu",weights_only=True),strict=True)
    apply_lora(base,rank=16)
    load_lora(base,a.adapter)
    state={k:v.clone() for k,v in base.state_dict().items() if ".lora." not in k}
    for name,module in base.named_modules():
        if isinstance(module,torch.nn.Linear) and hasattr(module,"lora"):
            state[name+".weight"]=module.weight.detach()+module.lora.B.weight.detach()@module.lora.A.weight.detach()
    merged=MicaForCausalLM(base.config).eval()
    merged.load_state_dict(state,strict=True)
    saved=MicaForCausalLM(base.config).eval()
    saved.load_state_dict(torch.load(a.merged,map_location="cpu",weights_only=True),strict=True)
    base,merged,saved=base.cuda(),merged.cuda(),saved.cuda()
    tok=AutoTokenizer.from_pretrained("tokenizer")
    prompts=["Write a Python function add(a, b).","Write a function that reverses a list.","解释什么是机器学习。","Return only JSON with key answer and value 42.","Translate hello into Chinese.","Write a Python function to check even numbers."]
    results=[]
    with torch.inference_mode():
        for prompt in prompts:
            text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,open_thinking=False)
            ids=tok(text,return_tensors="pt").input_ids.cuda()
            x,y=base(ids).logits,merged(ids).logits
            fp32_pass=torch.allclose(x,y,atol=1e-4,rtol=1e-4)
            with torch.autocast("cuda",dtype=torch.float16):
                b,s=base(ids).logits.float(),saved(ids).logits.float()
                bg=base.generate(ids,max_new_tokens=64,do_sample=False)
                sg=saved.generate(ids,max_new_tokens=64,do_sample=False)
            results.append({"prompt":prompt,"fp32_max_abs":float((x-y).abs().max()),"fp32_allclose":fp32_pass,
                            "fp16_max_abs":float((b-s).abs().max()),"fp16_argmax_match":float((b.argmax(-1)==s.argmax(-1)).float().mean()),
                            "fp16_generated_tokens_equal":bool(torch.equal(bg,sg))})
    report={"fp32_atol":1e-4,"fp32_rtol":1e-4,"fp32_algebra_passed":all(r["fp32_allclose"] for r in results),
            "fp16_generation_equal_count":sum(r["fp16_generated_tokens_equal"] for r in results),"prompts":results,
            "note":"Numerical algebra check is separate from rounded deployment equivalence; six prompts are a smoke test only."}
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))
    if not report["fp32_algebra_passed"]:
        raise SystemExit(1)

if __name__=="__main__":
    main()
