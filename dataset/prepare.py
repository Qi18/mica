"""Stream text/single-turn SFT records into tokenized JSONL atomically."""
import json
import os
from pathlib import Path
import tempfile
from transformers import AutoTokenizer


def prepare(source, tokenizer_path, output, max_length=768):
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tokens-", dir=output.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle, Path(source).open(encoding="utf-8") as source_handle:
            for number, line in enumerate(source_handle, 1):
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
                handle.write(json.dumps({"input_ids": ids, "labels": labels}) + "\n")
                count += 1
            if not count:
                raise ValueError("empty input")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic, no-clobber publication on the same filesystem.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"output": str(output), "rows": count}
