"""Phase6 v2: observation-dependent graph traversal and bounded error recovery."""
import json
import re
from phase6_agent_common import dumps, call_text, collate

TOOLS = [
    {"type":"function","function":{"name":"lookup_record","description":"查询记录。temporary_error重试一次，moved按new_id继续查询。",
      "parameters":{"type":"object","properties":{"record_id":{"type":"string"}},"required":["record_id"],"additionalProperties":False}}},
    *[{"type":"function","function":{"name":name,"description":desc,
       "parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":False}}}
      for name,desc in (("add_numbers","计算a+b。"),("subtract_numbers","计算a-b。"))]
]
SYSTEM = ("先查询用户指定根记录。根记录有value和operation；若有next_id就查询它，否则按route选择left_id或right_id查询。"
          "temporary_error重试一次，moved按new_id查询。拿到叶记录value后，按根operation用add_numbers或subtract_numbers计算："
          "a必须是根value，b必须是叶value。每轮只调用一个工具，格式<tool_call>JSON</tool_call>。根据实际返回值继续，最后只输出结果整数。")
MAX_TURNS = 6
MAX_NEW_TOKENS = 96
MAX_CONTEXT = 1536

def initial_messages(task):
    return [{"role":"system","content":SYSTEM},{"role":"user","content":task["prompt"]}]

def parse_call(text):
    m = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*",text,re.S)
    if not m:
        return None
    try:
        c=json.loads(m.group(1))
        if not isinstance(c,dict) or set(c)!={"name","arguments"} or not isinstance(c["arguments"],dict):
            return None
        n,a=c["name"],c["arguments"]
        if n=="lookup_record" and set(a)=={"record_id"} and isinstance(a["record_id"],str):
            return c
        if n in ("add_numbers","subtract_numbers") and set(a)=={"a","b"} and all(type(v) is int and abs(v)<=10000 for v in a.values()):
            return c
    except (ValueError,TypeError):
        pass
    return None

def expected_calls(t):
    def lookup(rid):
        return {"name":"lookup_record","arguments":{"record_id":rid}}
    chain=[lookup(t["root_id"])]
    if t["family"]=="redirect":
        chain.append(lookup(t["alias_id"]))
    chain.append(lookup(t["leaf_id"]))
    if t["family"]=="retry":
        chain.append(lookup(t["leaf_id"]))
    chain.append({"name":t["operation"]+"_numbers","arguments":{"a":t["root_value"],"b":t["leaf_value"]}})
    return chain

def answer(t):
    return t["root_value"]+t["leaf_value"] if t["operation"]=="add" else t["root_value"]-t["leaf_value"]

def execute(t,c,state):
    a=c["arguments"]
    if c["name"]!="lookup_record":
        return {"result":a["a"]+a["b"] if c["name"]=="add_numbers" else a["a"]-a["b"]},True
    rid=a["record_id"]
    counts=state.setdefault("counts",{})
    counts[rid]=counts.get(rid,0)+1
    if rid==t["root_id"]:
        root={"value":t["root_value"],"operation":t["operation"]}
        if t["family"]=="branch":
            root.update({"route":t["route"],"left_id":t["leaf_id"] if t["route"]=="left" else t["distractor_id"],
                         "right_id":t["leaf_id"] if t["route"]=="right" else t["distractor_id"]})
        else:
            root["next_id"]=t["alias_id"] if t["family"]=="redirect" else t["leaf_id"]
        return root,True
    if rid==t["alias_id"] and t["family"]=="redirect":
        return {"error":"moved","new_id":t["leaf_id"]},False
    if rid==t["leaf_id"]:
        if t["family"]=="retry" and counts[rid]==1:
            return {"error":"temporary_error","retryable":True},False
        return {"value":t["leaf_value"]},True
    if rid==t["distractor_id"]:
        return {"value":t["distractor_value"]},True
    return {"error":"not_found"},False

def trajectories(t):
    messages,state,rows=initial_messages(t),{},[]
    for call in expected_calls(t):
        text=call_text(call["name"],call["arguments"])
        rows.append({"messages":list(messages),"completion":text,"task_id":t["id"]})
        messages.append({"role":"assistant","content":text})
        obs,_=execute(t,call,state)
        messages.append({"role":"tool","content":dumps(obs)})
    rows.append({"messages":list(messages),"completion":str(answer(t)),"task_id":t["id"]})
    return rows

def encode_row(tok,row):
    prompt=tok.apply_chat_template(row["messages"],tools=row.get("tools",TOOLS),tokenize=False,add_generation_prompt=True,open_thinking=False)
    prefix=tok(prompt,add_special_tokens=False).input_ids
    full=tok(prompt+row["completion"]+tok.eos_token,add_special_tokens=False).input_ids
    assert full[:len(prefix)]==prefix
    return full,[-100]*len(prefix)+full[len(prefix):]
