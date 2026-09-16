#!/usr/bin/env python3
"""Create disjoint record tasks and canonical Agent SFT decisions, without test trajectories."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phase6_agent_common import trajectories, encode_row, collate, execute, parse_call, call_text, expected_calls, answer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit("refuse overwrite: " + str(a.output))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rng = random.Random(606)
    splits, encoded, manifest = {}, [], {"seed": 606, "scope": "synthetic in-family record-disjoint, not BFCL", "splits": {}}
    for split, count in (("train", 2400), ("val", 240), ("test", 300)):
        tasks = []
        for i in range(count):
            family = ("lookup", "chain", "retry")[i % 3]
            rid = hashlib.sha256((split + ":" + str(i) + ":phase6-v1").encode()).hexdigest()[:8]
            value, offset = rng.randrange(1, 1000), rng.randrange(1, 100)
            prompt = (f"查询记录{rid}的数值。" if family != "chain" else f"查询记录{rid}的数值，再用工具加上{offset}。")
            tasks.append({"id": split + "-" + str(i), "family": family, "record_id": rid, "prompt": prompt,
                          "value": value, "offset": offset})
        splits[split] = tasks
    all_ids = [t["record_id"] for rows in splits.values() for t in rows]
    all_prompts = [t["prompt"] for rows in splits.values() for t in rows]
    assert len(set(all_ids)) == len(all_ids)
    assert len(set(all_prompts)) == len(all_prompts)
    # Oracle execution and exact schema verifier checks, including wrong IDs and malformed calls.
    for tasks in splits.values():
        for task in tasks:
            state, obs = {}, None
            for call in expected_calls(task):
                assert parse_call(call_text(call["name"], call["arguments"])) == call
                obs, ok = execute(task, call, state)
            assert ok and list(obs.values())[0] == answer(task)
    assert parse_call('<tool_call>{"name":"add_numbers","arguments":{"a":true,"b":2}}</tool_call>') is None
    assert parse_call('<tool_call>{"name":"unknown","arguments":{}}</tool_call>') is None
    assert parse_call('1 <tool_call>{"name":"lookup_record","arguments":{"record_id":"x"}}</tool_call>') is None
    for split in ("train", "val"):
        encoded.extend(encode_row(tok, r) for task in splits[split] for r in trajectories(task))
    lengths = sorted(len(x[0]) for x in encoded)
    assert max(lengths) <= 1024, max(lengths)
    short, long = min(encoded, key=lambda r: len(r[0])), max(encoded, key=lambda r: len(r[0]))
    for side in ("left", "right"):
        tok.padding_side = side
        batch = collate(tok, [short, long])
        for i, row in enumerate((short, long)):
            n = len(row[0])
            assert batch["input_ids"][i, :n].tolist() == row[0]
            assert batch["labels"][i, :n].tolist() == row[1]
            assert row[1][-1] == tok.eos_token_id
    a.output.mkdir(parents=True)
    for split, tasks in splits.items():
        path = a.output / (split + ".jsonl")
        path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
        manifest["splits"][split] = {"tasks": len(tasks), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if split != "test":
            decisions = [r for t in tasks for r in trajectories(t)]
            path = a.output / (split + "-decisions.jsonl")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in decisions))
            manifest["splits"][split]["decisions"] = len(decisions)
            manifest["splits"][split]["decisions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update({"max_sequence_tokens": max(lengths), "p95_sequence_tokens": lengths[int(.95 * len(lengths))],
                     "truncated_rows": 0, "record_overlap": 0, "prompt_overlap": 0,
                     "oracle_and_padding_checks": "passed", "test_policy": "frozen, not used for checkpoint selection"})
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
