# Phase6 A01-v3：混合回放修复实验

日期：2026-09-08。状态：**completed-not-promoted**。Phase6未收口，S10仍为通用基线。
本轮是Agent SFT，不是Agentic RL；A02/A03尚未训练。

## 1. 问题与对照

A01-v2在v2独立test上217/240，但旧Chat8/10→7/10、旧Tool7/8→1/8。
两个v3候选都从S10原始FP32导出重新分叉，加入chat/native-tool回放和新多schema任务。
两组之间仅LR不同（3e-6、1e-6），数据、顺序、seed42、targets和末步选择一致。
相对v2同时改变起点、数据和LR，不能把改善单独归因于回放。

## 2. 数据与训练

构建脚本：scripts/data/agentic/build_phase6_v3.py。
实际训练数据：/data/datasets/minimind-lab/phase6/mixed-agent-v3-r1。
指纹见[data manifest](data-manifest-v3-r1.json)。

- 新graph train800 / val160 / frozen test240，四类chain/branch/retry/redirect。
- 新多schema工具train800 / val160 / frozen test240，8类；仍是同族合成任务。
- 历史broad replay训练2000条chat+600条native-tool对话，验证150+50条。
- 训练9115条assistant decision、406611 targets；验证1346条decision。
- tokens：graph107871、native-tool67835、chat191528、新tool39377。
- 最大训练长度956，不截断；normalized prompt跨split重叠0，不声称语义零污染。
- 不整包使用S09定向旧题改写课程；历史broad来自旧S10来源，仅作保持训练。
- 初始候选发现JSON/schema异常后未开训；增加JSON解析及Draft7 schema验证，
  隔离14条无效候选并补采到指定规模。初始目录保留，但实际训练只使用r1。
- 每条decision使用自己的tools；无tools聊天不注入graph schema，原生带tools对话保留自身schema。
- assistant-only labels/EOS监督，ids和labels同向右padding；实际模型padding不变性检查通过。
- 两组1epoch、microbatch4×accum4、seq上限1536、570更新、FP32参数+BF16 autocast。
- 两组固定checkpoint-570，不用最低NLL替代行为验收。
- 训练器计时均约120.9秒，峰值allocated3095.6MiB；不含完整后续评测，
  GPU与驻留进程共享，不能作为专卡速度基准。
- 最后一步loss为0.6551/0.6590，验证NLL为0.3788/0.3801；
  最后一步loss不是epoch均值，也不能据此宣称通用能力收敛。

## 3. 同协议结果

| 模型 | 新graph val | 新多工具 val | 旧Chat | 旧Tool | 格式 | 复读异常 |
|---|---:|---:|---:|---:|---:|---:|
| S10 | 0/160 | 5/160 | 8/10 | 7/8 | 5/6 | 1/10 |
| A01-v3 LR3e-6 | 157/160 (98.125%) | 153/160 (95.625%) | 8/10 | 5/8 | 5/6 | 1/10 |
| A01-v3 LR1e-6 | 152/160 (95.0%) | 146/160 (91.25%) | 8/10 | 5/8 | 5/6 | 1/10 |

S10新validation为本轮补跑。旧Chat/Tool沿用同一模型、同一现有脚本、FP16及max_new_tokens128的
/data/artifacts/minimind-lab/A00-v2-frozen-test-20260908/behavior结果，非本轮重跑。
新graph为6轮×96tokens，新tool为4轮×128tokens，不要跨任务直接比较耗时或难度。
新工具E2E要求完整有序调用、参数语义匹配和预定最终答案精确匹配，不是开放域Chat分数。

两组除旧Tool门禁（至少7/8）外，本轮其余预注册门禁均通过。selected=null，不能promote。

## 4. 剩余失败与证据边界

两组3个旧Tool失败题输出一致，证据见各实验的behavior/samples.jsonl。

| 类型 | 实际行为 | 判定 |
|---|---|---|
| 当前时间的工具选择 | 调用random_number，参数有字符串及重复max键 | 选错工具且参数错误 |
| 随机数后平方 | min/max变成1/1，仅返回42，没有第二步计算 | 参数抽取与多步完成失败 |
| 独立双意图参数保持 | 把天气观察值22送进单位转换，而非用户指定30 | 观察值覆盖用户参数 |

旧评测argument_validity只查结构/类型，第三题可能通过该项却语义错误；
不能把75%参数有效率写成75%参数语义正确率。

源码表明新工具训练措辞较固定，时间任务显式带timezone和编号，多步族仅随机数→平方，
缺少“观察结果与独立用户参数并存”的双意图族。这是覆盖缺口证据，不足以单独证明完整根因。
专项验证提升有效，但旧Tool仍退化，通用保持不足。

## 5. 验收与下一步

- 不promote、不替换S10、不打开新graph/test240或tools/test240。
- A02/A03未训练，之前零方差rollout probe不是RL收益。
- IFEval、七项基准、新独立test尚未运行，Phase6不能收口。
- 下一轮先扩展多措辞、缺省参数、多意图/观察干扰的独立训练和验证，
  区分引用观察与保持用户参数，不复制这3条固定评测题及答案。
- 预注册新attempt且保持门槛不变；如增加S10参考分布保持，需要另做同预算消融，尚未实施。
- 当前val与旧smoke已用于开发迭代，不作最终独立证据；test不参与调参。
- Agentic RL仍需包含正负反馈、组内奖励有差异的rollout池，不能直接使用全成功池。

## 6. 复现与归档

[预注册](../../docs/phases/phase6-v3-preregister.md)。
配置：configs/agentic_rl/phase6-a01-v3-lr3e6.json、phase6-a01-v3-lr1e6.json。
训练入口：scripts/launch/run_phase6_v3_candidate.sh；
门禁汇总：scripts/eval/summarize_phase6_v3.py。

代码基线234797d加未提交Phase6修改；训练目录source-snapshot.json与provenance.json保留实际证据，
不能仅靠HEAD声称完整复现。各checkpoint-manifest.json记录CPFS权重SHA，权重未加入Git。
所有实验轻量结果已归档，未commit/push。SwanLab均在MiniMind-Lab：

- A01-v3-mixed-lr3e6-20260908 (training + graph validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/52u6fhj4
- A01-v3-mixed-lr3e6-20260908 (multi-tool validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/hauawrx1
- A01-v3-mixed-lr1e6-20260908 (training + graph validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/5h3qys7d
- A01-v3-mixed-lr1e6-20260908 (multi-tool validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/7kagdh51
- A00-v3-s10-graph-validation-20260908 (matched validation baseline): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1a6xkwzx
- A00-v3-s10-tools-validation-20260908 (matched validation baseline): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wtets9k0
- A01-v3-validation-summary-20260908 (joint gates): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/p6sbux9p

## 后续状态（不修改上述SFT历史结论）

随后用户明确要求探索性Agentic RL，固定三臂权重后打开了v3 frozen test做一次性比较。
因此这些test当前已打开，不能再用于SFT/RL调参后声称独立测试；见[RL v2报告](phase6-rl-v2-report.md)。
