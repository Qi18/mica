"""Controlled multi-schema retention tasks; observations are a finite local fixture."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"eval"))
from eval_sft_behavior import TOOLS as SCHEMAS
from phase6_agent_common import dumps,call_text,collate
TOOLS=list(SCHEMAS.values())
MAX_TURNS=4
MAX_NEW_TOKENS=128
MAX_CONTEXT=1536

def initial_messages(t):
    return [{"role":"system","content":"需要工具时使用提供的工具，每轮只调用一个。收到结果后按用户要求回答，不要重复已经成功的调用。工具格式为<tool_call>JSON</tool_call>。"},
            {"role":"user","content":t["prompt"]}]
def expected_calls(t):return t["calls"]
def answer(t):return t["answer"]
def parse_call(text):
    m=re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*",text,re.S)
    if not m:return None
    try:
        c=json.loads(m.group(1));n=c["name"];a=c["arguments"]
        if set(c)!={"name","arguments"} or n not in SCHEMAS or not isinstance(a,dict):return None
        schema=SCHEMAS[n]["function"]["parameters"]
        if not set(schema.get("required",[]))<=set(a) or not set(a)<=set(schema["properties"]):return None
        for k,v in a.items():
            typ=schema["properties"][k]["type"]
            if typ=="string" and not isinstance(v,str):return None
            if typ=="integer" and type(v) is not int:return None
            if typ=="number" and (type(v) not in (int,float) or abs(v)>100000):return None
        return c
    except (ValueError,KeyError,TypeError):return None
def execute(t,c,state):
    if c["name"] not in {x["function"]["name"] for x in t["tools"]}:return {"error":"tool_unavailable"},False
    for expected,observation in zip(t["calls"],t["observations"]):
        if c==expected:return observation,True
    return {"error":"arguments_not_in_fixture"},False
def trajectories(t):
    messages=initial_messages(t);rows=[]
    for call,obs in zip(t["calls"],t["observations"]):
        completion=call_text(call["name"],call["arguments"])
        rows.append({"messages":list(messages),"completion":completion,"tools":t["tools"],"task_id":t["id"],"domain":"tool"})
        messages.extend([{"role":"assistant","content":completion},{"role":"tool","content":dumps(obs)}])
    rows.append({"messages":list(messages),"completion":str(t["answer"]),"tools":t["tools"],"task_id":t["id"],"domain":"tool"})
    return rows
