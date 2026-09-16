#!/usr/bin/env python3
"""Phase6 A01 assistant-decision SFT; validated explicit right-padding."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase6_agent_common import encode_row, collate
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from eval_phase6_agent import evaluate, configure_environment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    environment = configure_environment(c.get("environment", "phase6_agent_common"))
    encode_row = environment.encode_row
    a.output.mkdir(parents=True, exist_ok=False)
    random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.cuda.set_per_process_memory_fraction(0.15)
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(c["base_model"])
    encoded = {}
    for split in ("train", "val"):
        rows = [json.loads(l) for l in (Path(c["data"])/(split+"-decisions.jsonl")).read_text().splitlines()]
        encoded[split] = [encode_row(tok, r) for r in rows]
        assert all(len(x[0]) <= c["max_length"] for x in encoded[split])
    model = AutoModelForCausalLM.from_pretrained(c["base_model"], torch_dtype=torch.float32).cuda()
    # Actual causal-LM loss must be invariant to right padding of the shorter example.
    model.eval()
    short = min(encoded["val"], key=lambda r: len(r[0]))
    long = max(encoded["val"], key=lambda r: len(r[0]))
    with torch.inference_mode():
        single = collate(tok, [short], "cuda")
        pair = collate(tok, [short, long], "cuda")
        loss_single = model(**single).loss
        pair["labels"][1] = -100
        loss_padded = model(**pair).loss
    assert torch.allclose(loss_single, loss_padded, atol=1e-4, rtol=1e-4), (loss_single.item(), loss_padded.item())
    del single, pair
    (a.output/"alignment-check.json").write_text(json.dumps({"passed": True, "single_loss": loss_single.item(), "padded_loss": loss_padded.item()}))
    provenance = {"config": c, "head": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
                  "base_sha256": hashlib.sha256((Path(c["base_model"])/"model.safetensors").read_bytes()).hexdigest(),
                  "dataset_manifest": json.loads((Path(c["data"])/"manifest.json").read_text())}
    (a.output/"source-snapshot.json").write_text(json.dumps({str(p): p.read_text() for p in (
        Path("scripts/phase6_agent_common.py"), Path(environment.__file__), Path("scripts/launch/train_phase6_sft.py"), Path("scripts/eval/eval_phase6_agent.py"))}))
    (a.output/"provenance.json").write_text(json.dumps(provenance, indent=2))
    (a.output/"working-tree.patch").write_bytes(subprocess.check_output(["git","diff"]))
    import swanlab
    run = swanlab.init(project="MiniMind-Lab", experiment_name=a.run_name, group="Phase6-Agent",
                       job_type="training", config=provenance)
    (a.output/"swanlab-url.txt").write_text(run.url+"\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"])
    micro, accum = c["micro_batch"], c["gradient_accumulation"]
    total_updates = math.ceil(len(encoded["train"])/(micro*accum))*c["epochs"]
    best, step, trained_tokens = float("inf"), 0, 0
    started = time.monotonic()
    def validate():
        model.eval()
        total_loss, total_tokens = 0., 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(encoded["val"]), micro):
                b = collate(tok, encoded["val"][i:i+micro], "cuda")
                tokens = int((b["labels"][:,1:] != -100).sum())
                total_loss += model(**b).loss.item()*tokens
                total_tokens += tokens
        return total_loss/total_tokens
    initial = validate()
    swanlab.log({"validation/nll": initial}, step=0)
    print(json.dumps({"initial_validation_nll": initial}), flush=True)
    for epoch in range(c["epochs"]):
        order = list(encoded["train"])
        random.Random(c["seed"]+epoch).shuffle(order)
        for start in range(0,len(order),micro*accum):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            chunk = order[start:start+micro*accum]
            denominator = sum(sum(x != -100 for x in y[1:]) for _, y in chunk)
            loss_sum = 0.
            for i in range(0,len(chunk),micro):
                b = collate(tok, chunk[i:i+micro], "cuda")
                tokens = int((b["labels"][:,1:] != -100).sum())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**b).loss
                assert torch.isfinite(loss), loss
                (loss * tokens/denominator).backward()
                loss_sum += loss.item()*tokens/denominator
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(grad), grad
            optimizer.step()
            step += 1
            trained_tokens += denominator
            metrics = {"train/loss": loss_sum, "train/grad_norm": float(grad), "train/target_tokens": trained_tokens,
                       "system/peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                       "system/elapsed_seconds": time.monotonic()-started}
            if step % c["eval_every"] == 0 or step == total_updates:
                val = validate()
                metrics["validation/nll"] = val
                if val < best or (c.get("select_final", False) and step == total_updates):
                    best = val
                    target = a.output/("checkpoint-"+str(step))
                    model.save_pretrained(target, safe_serialization=True)
                    tok.save_pretrained(target)
                    (a.output/"best-checkpoint.json").write_text(json.dumps({"path": str(target), "step": step, "validation_nll": best}))
            with (a.output/"metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **metrics})+"\n")
            swanlab.log(metrics, step=step)
            if step % 10 == 0:
                print(json.dumps({"step":step,"total":total_updates,**metrics}), flush=True)
    best_path = json.loads((a.output/"best-checkpoint.json").read_text())["path"]
    del optimizer, model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(best_path, torch_dtype=torch.float32).cuda()
    tasks = [json.loads(l) for l in (Path(c["data"])/"val.jsonl").read_text().splitlines()]
    result = evaluate(model,tok,tasks,a.output/"validation-agent")
    swanlab.log({"validation_agent/"+k:v for k,v in result.items() if isinstance(v,(int,float))},step=step+1)
    swanlab.finish()
    (a.output/"DONE.json").write_text(json.dumps({"status":"pilot-complete-not-phase-accepted","best":best_path,"validation":result},indent=2))

if __name__ == "__main__":
    main()
