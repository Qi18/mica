# A01：Agent SFT 首轮 pilot

状态：completed（pilot），Phase6 未收口；A02/A03 尚未启动。
从 S10 直接分叉，使用 record-agent-v1 2400 条任务生成6400条 assistant decisions。
1 epoch、400 optimizer updates、159083 supervised tokens；LR1e-5、microbatch4、accum4、seed42。
FP32参数 + BF16 autocast，GPU0空闲显存；训练含validation NLL约54.87秒，峰值allocated2341MiB。
这是模板化小数据 pilot，不是大规模正式训练。权重未覆盖 S10 或其他阶段。

## 正确性检查

显式右padding且标签同位置；prompt/tool observation/padding不进入监督，EOS参与监督。
真实模型单样本loss=2.0134909153，变长补齐loss=2.0134906769，误差<1e-6。
模型Qwen3兼容导出（MiniMind权重），CausalLM内部正常shift labels。未复用Phase5错位collator。
数据不同split记录ID/prompt重叠0，最长590tokens，截断0；范围是同模板族，不是模板OOD。

## 验证结果（不是test）

| 指标 | A00 S10 | A01 checkpoint300 |
|---|---:|---:|
| schema validity | 74.25% | 100% |
| tool selection accuracy | 54.00% | 100% |
| argument semantic accuracy | 49.25% | 99.75% |
| execution success | 82.84% | 99.69% |
| E2E | 1/240（0.42%） | 239/240（99.58%） |
| lookup E2E | 1/80 | 80/80 |
| chain E2E | 0/80 | 80/80 |
| retry E2E | 0/80 | 79/80 |
| 平均生成tokens | 78.36 | 66.35 |

SwanLab：本目录 swanlab-url.txt。来源：本目录 validation-agent-summary.json / A00 metrics.json。
best由validation NLL选择：step300，NLL0.0001260；初始NLL1.43910，step400 NLL0.0003423。
A01 工具调用门槛通过，但该模板族已接近满分，不适合据此证明RL增益。
后续在新的预注册任务版本上增加依赖深度、环境错误和组合泛化，并重新验证A01；
不得针对冻结test调参或将模板记忆包装成开放域Agent能力。

## 尚未完成

- 冻结test300仍未进行模型评测。
- observation反事实验证、IFEval与七项通用能力回归未跑；不声称通用能力保持。
- Agentic GRPO/CISPO未训练，不存在RL提升结论。
- baseline严格最终格式会扣掉带说明的正确数值，应结合轨迹分析，而非只看E2E涨幅。

## 资产

- checkpoint：`/data/artifacts/minimind-lab/A01-agent-sft-pilot-20260908/checkpoint-300`
- 完整日志、4次NLL评估、checkpoint选择、源码快照、base hash、数据manifest均在该实验artifacts目录。
- 启动日志：`/data/artifacts/minimind-lab/phase6-launch-20260908/a01.log`
