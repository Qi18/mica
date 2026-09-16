#!/usr/bin/env python3
"""Batched greedy rollouts with strict ordered tool/argument/execution verification."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase6_agent_common import TOOLS, MAX_TURNS, MAX_NEW_TOKENS, initial_messages, expected_calls, parse_call, execute, answer, dumps

MAX_CONTEXT = 1024

def configure_environment(name="phase6_agent_common"):
    if name not in {"phase6_agent_common", "phase6_agent_v2", "phase6_retention_env"}:
        raise ValueError(name)
    module = importlib.import_module(name)
    for key in ("TOOLS", "MAX_TURNS", "MAX_NEW_TOKENS", "initial_messages", "expected_calls", "parse_call", "execute", "answer", "dumps"):
        globals()[key] = getattr(module, key)
    globals()["MAX_CONTEXT"] = getattr(module, "MAX_CONTEXT", 1024)
    return module

def evaluate(model, tok, tasks, output, batch_size=8, autocast_enabled=True):
    model.eval()
    tok.padding_side = "left"
    output.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.monotonic()
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start+batch_size]
        states = [{"task": t, "messages": initial_messages(t), "env": {}, "calls": [], "trace": [],
                   "schema": 0, "attempts": 0, "executed": 0, "temporary_errors": 0, "tokens": 0, "done": False, "final": None}
                  for t in batch]
        batch_start = time.monotonic()
        for turn in range(MAX_TURNS):
            active = [s for s in states if not s["done"]]
            if not active:
                break
            texts = [tok.apply_chat_template(s["messages"], tools=s["task"].get("tools", TOOLS), tokenize=False, add_generation_prompt=True,
                                             open_thinking=False) for s in active]
            inputs = tok(texts, padding=True, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False).to(model.device)
            if inputs.input_ids.size(1) + MAX_NEW_TOKENS > MAX_CONTEXT:
                raise RuntimeError("context budget exceeded; no silent truncation")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                ids = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            for state, seq in zip(active, ids[:, inputs.input_ids.size(1):]):
                raw = seq.tolist()
                if tok.eos_token_id in raw:
                    raw = raw[:raw.index(tok.eos_token_id)+1]
                state["tokens"] += len(raw)
                text = tok.decode(raw, skip_special_tokens=True).strip()
                state["trace"].append({"role": "assistant", "text": text})
                call = parse_call(text)
                if call is None:
                    if "tool_call" in text:
                        state["attempts"] += 1
                    state["final"] = text
                    state["done"] = True
                    continue
                state["attempts"] += 1
                state["schema"] += 1
                state["calls"].append(call)
                obs, ok = execute(state["task"], call, state["env"])
                state["executed"] += int(ok)
                state["messages"].extend([{"role": "assistant", "content": text}, {"role": "tool", "content": dumps(obs)}])
                state["temporary_errors"] += int(obs.get("error") in {"temporary_error", "moved"})
                state["trace"].append({"role": "tool", "observation": obs})
        elapsed = time.monotonic() - batch_start
        for s in states:
            expected, actual = expected_calls(s["task"]), s["calls"]
            tool = sum(a["name"] == e["name"] for a, e in zip(actual, expected))
            arg = sum(a == e for a, e in zip(actual, expected))
            exact = s["final"] == str(answer(s["task"]))
            e2e = actual == expected and exact and s["done"]
            results.append({"id": s["task"]["id"], "family": s["task"]["family"], "schema_valid": s["schema"],
                            "call_attempts": s["attempts"], "expected_calls": len(expected),
                            "tool_correct": tool, "arguments_correct": arg, "successful_executions": s["executed"],
                            "expected_successful_executions": len(expected) - int(s["task"]["family"] in {"retry", "redirect"}),
                            "expected_transient_errors": s["temporary_errors"],
                            "ordered_tool_chain": actual == expected, "final_answer_accuracy": exact,
                            "end_to_end_success": e2e, "unfinished": not s["done"], "calls": len(actual),
                            "turns": sum(t["role"] == "assistant" for t in s["trace"]), "tokens": s["tokens"],
                            "batch_amortized_seconds": elapsed / len(batch), "trace": s["trace"]})
        print(json.dumps({"evaluated": len(results), "total": len(tasks),
                          "e2e": sum(r["end_to_end_success"] for r in results)/len(results)}), flush=True)
    n = len(results)
    expected = sum(r["expected_calls"] for r in results)
    attempts = sum(r["call_attempts"] for r in results)
    summary = {
        "tasks": n, "schema_validity": sum(r["schema_valid"] for r in results)/max(1, attempts),
        "expected_call_coverage": sum(min(r["calls"], r["expected_calls"]) for r in results)/expected,
        "tool_selection_accuracy": sum(r["tool_correct"] for r in results)/expected,
        "argument_semantic_accuracy": sum(r["arguments_correct"] for r in results)/expected,
        "execution_success": sum(r["successful_executions"] for r in results)/max(1, sum(r["calls"] - r["expected_transient_errors"] for r in results)),
        "end_to_end_success": sum(r["end_to_end_success"] for r in results)/n,
        "final_answer_accuracy": sum(r["final_answer_accuracy"] for r in results)/n,
        "unfinished_rate": sum(r["unfinished"] for r in results)/n,
        "invalid_call_rate": (attempts-sum(r["schema_valid"] for r in results))/max(1, attempts),
        "mean_calls": sum(r["calls"] for r in results)/n, "mean_turns": sum(r["turns"] for r in results)/n,
        "mean_generated_tokens": sum(r["tokens"] for r in results)/n, "wall_seconds": time.monotonic()-started,
        "observation_usage": "ordered chain and exact final checked; causal counterfactual evaluation pending",
        "latency_protocol": "batched throughput/amortized time, not single-request latency",
        "precision": "fp32+bf16-autocast" if autocast_enabled else "float32",
        "decoder": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS, "max_turns": MAX_TURNS, "batch_size": batch_size}}
    for family in sorted({r["family"] for r in results}):
        group = [r for r in results if r["family"] == family]
        summary[family + "_e2e"] = sum(r["end_to_end_success"] for r in group)/len(group)
    (output/"predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in results))
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--run-name", required=True)
    p.add_argument("--environment", default="phase6_agent_common")
    a = p.parse_args()
    configure_environment(a.environment)
    if a.output.exists():
        raise SystemExit("refuse overwrite")
    tasks = [json.loads(l) for l in a.tasks.read_text().splitlines()]
    if a.limit:
        tasks = tasks[:a.limit]
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(0.15)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda()
    tok = AutoTokenizer.from_pretrained(a.model)
    import swanlab
    run = swanlab.init(project="MiniMind-Lab", experiment_name=a.run_name, group="Phase6-Agent", job_type="evaluation",
                       config={"model": a.model, "tasks": str(a.tasks), "limit": a.limit, "seed": 42, "environment": a.environment})
    summary = evaluate(model, tok, tasks, a.output)
    (a.output/"swanlab-url.txt").write_text(run.url+"\n")
    swanlab.log({"agent/"+k:v for k,v in summary.items() if isinstance(v,(int,float))})
    swanlab.finish()
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
