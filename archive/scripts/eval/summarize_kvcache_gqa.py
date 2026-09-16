#!/usr/bin/env python3
"""Validate complete coverage and write A10 inference ablation artifacts."""
import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/"experiments/08-inference/I01-I04-a10-20260914"
def read(name):return json.loads((out/name).read_text())
def write(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
summaries=read("summary.json")
raw=[json.loads(x) for x in (out/"raw.jsonl").read_text().splitlines()]
arms=("I01","I02","I03","I04")
expected={(a,b,l,128) for a in arms for b in (1,4,8) for l in (128,512,1024)}
expected|={(a,1,512,512) for a in arms}
key=lambda x:(x["arm"],x["batch"],x["input_tokens"],x["output_tokens"])
assert len(summaries)==40 and {key(x) for x in summaries}==expected
assert len(raw)==800
for scene in expected:
    rows=[r for r in raw if key(r)==scene]
    assert len(rows)==20 and {r["repeat"] for r in rows}==set(range(20))
    assert len({r["output_sha256"] for r in rows})==1
index={key(s):s for s in summaries}
def med(a,b,l,n,k):return index[(a,b,l,n)]["metrics"][k]["median"]
comparisons=[]
for b,l,n in sorted({(s["batch"],s["input_tokens"],s["output_tokens"]) for s in summaries}):
    assert len({tuple(index[(a,b,l,n)]["output_hashes"]) for a in arms})==1
    assert med("I02",b,l,n,"kv_bytes")==2*med("I04",b,l,n,"kv_bytes")
    speed=lambda a,c:med(a,b,l,n,"decode_tokens_s")/med(c,b,l,n,"decode_tokens_s")
    comparisons.append({"batch":b,"input_tokens":l,"output_tokens":n,
        "cache_speedup_mha":speed("I02","I01"),"cache_speedup_gqa":speed("I04","I03"),
        "gqa_speedup_cache_on":speed("I04","I02"),"gqa_speedup_cache_off":speed("I03","I01"),
        "combined_speedup":speed("I04","I01"),"cache_bytes_reduction":0.5,
        "peak_allocated_reduction":1-med("I04",b,l,n,"peak_allocated_bytes")/med("I02",b,l,n,"peak_allocated_bytes")})
write("comparisons.json",comparisons)
flat=[]
for s in summaries:
    row={k:s[k] for k in ("arm","batch","input_tokens","output_tokens")}
    for k,v in s["metrics"].items():
        row[k+"_median"]=v["median"];row[k+"_p95"]=v["p95"]
    flat.append(row)
with (out/"metrics.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
for metric,name,ylabel,scale in [("decode_tokens_s","throughput","Batch aggregate decode tokens/s",1),("peak_allocated_bytes","memory","Peak allocated GPU memory (MiB)",2**20)]:
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,b in zip(axes,(1,4,8)):
        for a,label in zip(arms,("MHA cache off","MHA cache on","GQA cache off","GQA cache on")):
            ax.plot([128,512,1024],[med(a,b,l,128,metric)/scale for l in (128,512,1024)],marker="o",label=label)
        ax.set(title=f"Batch {b}",xlabel="Prompt tokens",ylabel=ylabel);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("A10 / S10 BF16 / fixed FP32 token trace replay / 128 output tokens")
    fig.tight_layout();fig.savefig(out/(name+".png"),dpi=160);fig.savefig(out/(name+".svg"));plt.close(fig)
correctness=read("correctness.json");run=read("run.json")
assert run["status"]=="completed-with-bf16-consistency-limitations"
assert run["checkpoint_sha256"]==run["checkpoint_sha256_after"]
fp=[x for x in correctness["tests"] if x["dtype"]=="float32"]
bf=[x for x in correctness["tests"] if x["dtype"]=="bfloat16"]
gc=[x for x in correctness["generation_comparisons"] if x["dtype"]=="bfloat16"]
text=["# A10：MiniMind S10 KV Cache / GQA 推理实验","",
"## 结论与验收边界","",
"已完成 36 个主场景及 4 个长生成场景，每场景预热 5 次、正式测量 20 次，共 800 条正式记录。吞吐是整个 batch 的 decode tokens/s，不含 prefill。","",
f"FP32 一致性 {sum(x['pass'] for x in fp)}/{len(fp)} 通过；BF16 一致性 {sum(x['pass'] for x in bf)}/{len(bf)} 通过，未满足预先冻结阈值。BF16 最大绝对误差 {max(x['max_abs'] for x in bf):.6f}，最大相对 RMSE {max(x['relative_rmse'] for x in bf):.4%}。阈值未放宽。","",
f"4 条真实提示词、每条 32 token 的 BF16 greedy 对照中，{sum(x['first_difference'] is None for x in gc)}/{len(gc)} 组与 GQA cache-off 完全相同；其余的 token 一致率和首次分歧见 correctness.json。不能声称 BF16 输出严格等价。","",
"因此主测量改为固定 FP32 GQA greedy token 轨迹重放：四臂执行前向和 argmax，但将同一参考 token 送入下一步。数据衡量固定工作负载的效率，不是自由生成的质量验收。原计划的 BF16 一致性门未通过。FP32 真实提示词 greedy 对照 12/12 完全一致；BF16 自由生成不保证逐 token 相同。","",
"## 配置与测量","",
"- I01：等价 MHA 8Q/8KV，cache off；I02：等价 MHA，cache on。",
"- I03：原生 GQA 8Q/4KV，cache off；I04：原生 GQA，cache on。",
"- MHA 仅按 head 顺序复制 K/V 投影，其他参数相同，strict load；不是独立训练的 MHA。",
"- S10 权重 SHA256："+run["checkpoint_sha256"]+"；运行前后保持一致。",
"- GPU：A10-Server GPU 0；模型权重与浮点 buffer 均转 BF16，不是 FP32 权重加 autocast；torch "+run["torch"]+"；Transformers "+run["transformers"]+"。",
"- 原生模型源码未修改：完整前缀走 SDPA，缓存单 token 走手写 attention；GQA 使用 repeat_kv。收益不能直接外推到融合 GQA 内核或 vLLM。",
"- logits_to_keep=1；无采样、EOS 提前结束和分词计时。TTFT 边界同步一次，decode 内不逐步同步。TTFT 包含同步开销，是模型调用测量，不是服务请求延迟。",
"- 固定中文文本重复截取 128/512/1024 tokens；batch 内使用相同序列。S10 训练长度为 768，长输入仅用于效率测量。",
"- 场景 seed42 打乱、实验臂按场景轮转。每臂按块预热和测量，不是逐次交错，不能完全消除时段漂移。原始 telemetry 保留温度、频率、功率及 GPU 占用。",
"- allocated/reserved 与模型加载基线分别记录。实际 KV 长度=input+output-1，末次生成 token 尚未进入缓存。",
"- raw.jsonl 中 output_sha256 是重放 token 的哈希，只用于证明四组输入轨迹一致；真实 greedy 一致性看 correctness.json。","",
"## 收益归因","",
"| Batch | 输入 | 输出 | Cache 加速 MHA | Cache 加速 GQA | GQA 加速 cache-on | 组合加速 | GQA 峰值显存降低 cache-on |",
"|---:|---:|---:|---:|---:|---:|---:|---:|"]
low=min(c["cache_speedup_gqa"] for c in comparisons)
high=max(c["cache_speedup_gqa"] for c in comparisons)
glo=min(c["gqa_speedup_cache_on"] for c in comparisons)
ghi=max(c["gqa_speedup_cache_on"] for c in comparisons)
text[4:4]=[
    f"- 当前 GQA 实现开启缓存的 decode 加速比为 {low:.2f}–{high:.2f}×；小于 1 的场景是负收益，不能声称缓存总会加速。",
    f"- 开启缓存时，GQA 相对等价 MHA 的 decode 加速比为 {glo:.2f}–{ghi:.2f}×；缓存字节数在所有场景精确减半。",
    "- 以上是 A10 单卡、小模型和当前原生实现的固定轨迹结果，不是模型能力提升。",
    ""]
for c in comparisons:
    text.append(f"| {c['batch']} | {c['input_tokens']} | {c['output_tokens']} | {c['cache_speedup_mha']:.2f}× | {c['cache_speedup_gqa']:.2f}× | {c['gqa_speedup_cache_on']:.2f}× | {c['combined_speedup']:.2f}× | {c['peak_allocated_reduction']:.1%} |")
text+=["","GQA 的实际 KV 字节数在所有缓存场景均为 MHA 的 50%，不等于整体显存减半。加速比由独立测量中位数相除；接近 1 的小差异不声明统计显著。","",
"## 中位数明细","",
"| Arm | Batch | 输入 | 输出 | TTFT ms | Decode tok/s | TPOT ms | E2E ms | Peak allocated MiB | KV MiB |",
"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s in sorted(summaries,key=lambda s:(s["batch"],s["input_tokens"],s["output_tokens"],s["arm"])):
    m=lambda k:s["metrics"][k]["median"]
    text.append(f"| {s['arm']} | {s['batch']} | {s['input_tokens']} | {s['output_tokens']} | {m('ttft_ms'):.2f} | {m('decode_tokens_s'):.2f} | {m('tpot_ms'):.2f} | {m('e2e_ms'):.2f} | {m('peak_allocated_bytes')/2**20:.2f} | {m('kv_bytes')/2**20:.2f} |")
text+=["","![吞吐](throughput.png)","","![显存](memory.png)","","## 数值诊断与产物","",
"统一手写 attention 后，FP32 仍通过，BF16 cache/full 差异仍存在；独立 K 投影检查观察到矩阵形状变化带来的 BF16 舍入差异。差异不只来自 attention 路径。未完成逐算子的全部误差归因，不认定某个内核是唯一原因。","",
"- config.json / initial-config.json：修订后的固定轨迹定义和原始阈值。",
"- command.sh、run.json、checkpoint-manifest.txt：运行命令、版本、硬件、源码与权重 SHA。",
"- correctness.json、precision-diagnosis.json：原始失败和统一路径诊断。",
"- input-tokens.json、reference-traces.json：固定输入与参考生成轨迹。",
"- raw.jsonl / summary.json / metrics.csv：800 条原始测量、中位数和 P95。",
"- comparisons.json、throughput.png、memory.png：归因数据与图。",
"- source.patch / status-before.txt（仅保留在服务器，不随本次发布）：运行前已有未提交改动；未修改 MiniMind 模型源码。",
"- 未上传 SwanLab；本报告、评测脚本与结果发布到 GitHub，未更新受发布门槛约束的项目 README。","",
"本实验只评价当前实现的效率。MHA/GQA 模型质量对比仍需要同数据、同预算、多 seed 的受控训练实验。"]
(out/"report.md").write_text("\n".join(text)+"\n")
write("eval.json",{"status":"performance-completed-bf16-consistency-failed","scenarios":40,"formal_records":800,
"fp32_passed":all(x["pass"] for x in fp),"bf16_passed":all(x["pass"] for x in bf),"replayed_inputs_equal":True,
"gqa_cache_half_mha":True,"checkpoint_unchanged":True,"comparisons":comparisons})
print(json.dumps({"status":"report-ready","scenarios":40,"measurements":800,"comparisons":comparisons},ensure_ascii=False))
