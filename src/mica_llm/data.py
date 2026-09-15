"""Build tokenized JSONL from text or single-turn prompt/response records."""
import json
from pathlib import Path
from transformers import AutoTokenizer


def prepare(source, tokenizer_path, output, max_length=768):
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    rows = []
    for number, line in enumerate(Path(source).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "text" in row:
            ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
            labels = ids.copy()
        else:
            prefix = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                   tokenize=True, add_generation_prompt=True)
            reply = tokenizer.encode(row["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
            ids = prefix + reply
            labels = [-100] * len(prefix) + reply
        ids, labels = ids[:max_length], labels[:max_length]
        if len(ids) < 2 or all(x == -100 for x in labels[1:]):
            raise ValueError(f"row {number} has no supervised targets after truncation")
        rows.append({"input_ids": ids, "labels": labels})
    if not rows:
        raise ValueError("empty input")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return {"output": str(output), "rows": len(rows)}
