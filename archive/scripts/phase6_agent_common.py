"""Deterministic bounded Phase6 tool environment; no arbitrary code execution."""
import json
import re

TOOLS = [
    {"type": "function", "function": {"name": "lookup_record", "description": "查询记录的整数数值。遇到 temporary_error 时可重试一次。",
     "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "add_numbers", "description": "计算两个整数的和。",
     "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}},
]
SYSTEM = "使用给定工具完成任务。每次只调用一个工具，格式为<tool_call>JSON</tool_call>。根据实际工具返回值继续；遇到temporary_error重试一次。完成后只输出最终整数。"
MAX_TURNS = 4
MAX_NEW_TOKENS = 96

def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def call_text(name, arguments):
    return "<tool_call>" + dumps({"name": name, "arguments": arguments}) + "</tool_call>"

def initial_messages(task):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]

def expected_calls(task):
    lookup = {"name": "lookup_record", "arguments": {"record_id": task["record_id"]}}
    return [lookup] * (2 if task["family"] == "retry" else 1) + (
        [{"name": "add_numbers", "arguments": {"a": task["value"], "b": task["offset"]}}]
        if task["family"] == "chain" else [])

def answer(task):
    return task["value"] + (task["offset"] if task["family"] == "chain" else 0)

def parse_call(text):
    matches = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*", text, re.S)
    if not matches:
        return None
    try:
        parsed = json.loads(matches.group(1))
        if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
            return None
        name, args = parsed["name"], parsed["arguments"]
        if not isinstance(args, dict):
            return None
        if name == "lookup_record" and set(args) == {"record_id"} and isinstance(args["record_id"], str):
            return parsed
        if name == "add_numbers" and set(args) == {"a", "b"} and all(type(v) is int and abs(v) <= 10000 for v in args.values()):
            return parsed
    except (ValueError, TypeError):
        pass
    return None

def execute(task, call, state):
    """All operations are constant-time bounded dictionary access or integer addition."""
    if call["name"] == "lookup_record":
        if call["arguments"]["record_id"] != task["record_id"]:
            return {"error": "not_found"}, False
        state["lookups"] = state.get("lookups", 0) + 1
        if task["family"] == "retry" and state["lookups"] == 1:
            return {"error": "temporary_error", "retryable": True}, False
        return {"value": task["value"]}, True
    return {"result": call["arguments"]["a"] + call["arguments"]["b"]}, True

def trajectories(task):
    messages, state, rows = initial_messages(task), {}, []
    for call in expected_calls(task):
        text = call_text(call["name"], call["arguments"])
        rows.append({"messages": list(messages), "completion": text, "task_id": task["id"]})
        messages.append({"role": "assistant", "content": text})
        observation, _ = execute(task, call, state)
        messages.append({"role": "tool", "content": dumps(observation)})
    rows.append({"messages": list(messages), "completion": str(answer(task)), "task_id": task["id"]})
    return rows

def encode_row(tokenizer, row):
    prompt = tokenizer.apply_chat_template(row["messages"], tools=TOOLS, tokenize=False,
                                          add_generation_prompt=True, open_thinking=False)
    prefix = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + row["completion"] + tokenizer.eos_token, add_special_tokens=False).input_ids
    if full[:len(prefix)] != prefix:
        raise ValueError("assistant boundary retokenization")
    return full, [-100] * len(prefix) + full[len(prefix):]

def collate(tokenizer, encoded, device="cpu"):
    """Explicit RIGHT padding, identical input/label offsets; EOS remains supervised."""
    import torch
    width = max(len(ids) for ids, _ in encoded)
    ids, mask, labels = [], [], []
    for seq, target in encoded:
        pad = width - len(seq)
        ids.append(seq + [tokenizer.pad_token_id] * pad)
        mask.append([1] * len(seq) + [0] * pad)
        labels.append(target + [-100] * pad)
    batch = {k: torch.tensor(v, device=device) for k, v in (
        ("input_ids", ids), ("attention_mask", mask), ("labels", labels))}
    active = batch["labels"] != -100
    assert (batch["input_ids"][active] == batch["labels"][active]).all()
    assert (batch["attention_mask"][active] == 1).all()
    return batch
