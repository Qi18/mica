# Phase6 v2实验报告：专项能力提升，通用工具回归未通过

状态：A01 v2已训练并完成冻结test；**不提升为release，不进入A02/A03**。Phase6未收口。
全部结果来自CPFS原始逐题记录和SwanLab，不覆盖v1，不把planned写成completed。

## 过程

1. 保留v1所有数据/产物及未使用的test300；v1 A01在v2 validation E2E为0/160。
2. v2包含chain、branch、retry、redirect：根查询→按observation选择后继→叶查询→加减工具→最终整数。
3. train800任务/3600decisions，validation160，test240；跨split记录和prompt重叠0；
   用户措辞按split不同，schema和任务家族相同，不称作广泛OOD。最长953tokens，无截断。
4. A01-v2从A01-v1 checkpoint300继续全参SFT，LR1e-5，batch4×accum4，seed42，1epoch/225updates/107859targets。
   FP32参数+BF16 autocast；约52.65秒（含训练期间validation NLL），峰值allocated3082MiB。
5. 按validation NLL最小选择checkpoint225（0.00025237），再冻结SHA256和测试协议。
   best SHA256：51130347e6beae1698186ff26ebdc3a099ce23dfa8505e4f1590238e6511933c。
6. 在模型测试前落地a01-v2-test-freeze.md；A00=S10，A01=冻结v2权重。
   两者greedy/batch8/96tokens每轮/最多6轮/context1536；独立test240未用于调参或选择checkpoint。

## 结果

| 指标 | A00 S10 | A01-v2 |
|---|---:|---:|
| 独立test E2E | 0/240 | 217/240（90.42%） |
| tool schema validity | 51.09% | 99.89% |
| tool selection accuracy | 13.33% | 97.26% |
| argument semantic accuracy | 10.95% | 91.07% |
| execution success（实际有效调用，排除预期可恢复错误） | 97.34% | 82.87% |
| 未完成率 | 5.42% | 9.17% |
| 平均工具调用数 | 0.78 | 3.75 |
| 平均生成tokens | 68.75 | 143.83 |
| 既有Chat回归 | 8/10 | 7/10 |
| 既有Tool E2E回归 | 7/8 | 1/8 |

test四类E2E：branch51/60、chain54/60、redirect54/60、retry58/60。
配对bootstrap10000次/seed42：E2E差值+90.42pp，95%CI[+86.67,+94.17]pp。
范围仅限这个合成任务家族；不能外推成通用Agent提升。
A00 execution success分母仅为实际有效调用，基线常常没有走完链；
因此其高execution success不能替代E2E/调用覆盖率，也不能直接解释为比A01更会执行。

validation159/160（99.38%）与test217/240（90.42%）有明显差距，说明验证近满分不等于泛化充分。
反事实validation保持同prompt/ID，将根值+19、叶值+7；159/160对任务在两个环境都完成且最终答案随隐藏值改变。
这支持本环境中的observation使用，不代表开放域因果推理。

## RL可训练性检查

按预注册的饱和停止条件，先只采样、不更新权重。
从train抽32prompts×4samples，temperature0.8/top_p1/top_k0，最多6轮，全部128条轨迹成功。
32/32组reward方差为0（reward=0.8E2E+0.2有序正确调用前缀比例），16924 action tokens。
这批样本无法产生GRPO/CISPO组内相对优势；不能推广为所有任务永远无方差。
assistant-only mask与raw旧策略重复前向一致性检查通过；这不是已完成RL训练。
**A02/A03优化器更新均未发生，不能写RL提升。** 完整多轮RL训练脚本尚未交付，仅完成采样可训练性探针。

## 失败分析与发布判定

A01既有工具回归7/8→1/8、未完成5/8。逐题证据包括：
- 时间/天气查询收到observation后仍重复调用，未正确结束；
- unit_converter把from_unit/to_unit填成数字字符串；
- get_exchange_rate缺失/改写参数schema；
- 多工具任务误选时间工具，未完成所需转换。
Chat中还有17+25回复17、JSON country字段回复北京等错误。

这些现象与窄任务全参继续训练的能力遗忘相符，但尚无消融证明数据比例、LR、
v1→v2连续训练起点各自的因果贡献。不能单凭低NLL认定训练成功。
**本轮为completed-not-promoted：保留研究权重，S10继续作为通用release。**
Chat/Tool回归采用同脚本FP16/greedy/max_new_tokens128；它仅是小规模门禁。
IFEval、七项通用benchmark未执行，不声称通用知识保持。

## 后续修复方向（尚未执行）

- 从S10重新分叉，与v1→v2连续微调做受控对照；
- 加入来源可审计的通用Chat与多schema工具轨迹回放，覆盖直接回答、正确停止和错误恢复；
- 不把本次test或既有固定回归题直接复制进训练，先扫描污染并冻结新版本评测；
- 同assistant-target预算比较学习率/混合比例，按新validation的Agent+通用门禁联合选择；
- 通过行为回归且存在可验证失败/组内reward差异后，再启动Agentic GRPO/CISPO。
v2冻结test已打开；其结果仅用于报告，不再用作本版本调参/选checkpoint依据。

## 证据

配置：configs/agentic_rl/phase6-a01-v2-pilot.json。
方案：docs/phases/phase6-v2-preregister.md。
汇总：A01-v2-evaluation-summary-20260908/summary.json。
完整逐题、训练/采样日志、checkpoint均位于/data/artifacts/minimind-lab各同名目录。
本轮未提交/推送；使用234797d基础上的远程working tree，运行源码快照保存在训练artifacts。

### SwanLab

- A01-v2-transfer-baseline-20260908 (transfer baseline): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1t7ndogk
- A01-agent-sft-v2-pilot-20260908 (SFT pilot): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/mabi301l
- A01-v2-counterfactual-20260908 (counterfactual evaluation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/miy5qid6
- A01-v2-rl-eligibility-probe-20260908 (sampling-only RL eligibility): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/hsn1xolt
- A00-v2-frozen-test-20260908 (S10 independent test): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/j3b8qcu0
- A01-v2-frozen-test-20260908 (A01 independent test): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/gfltoxnx
- A01-v2-evaluation-summary-20260908 (evaluation aggregation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/8yso53w9
