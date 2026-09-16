# Phase6 A01-v3 修复预注册

2026-09-08。v2新任务test217/240，但旧Tool7/8→1/8，因此A01未通过。
本轮修复不训练RL，不使用v2 test选checkpoint或生成答案；旧结果全部保留。

## 两个对照

两组都从S10原始FP32导出分叉，不从已退化A01继续：
- A01-v3-lr3e6：AdamW LR3e-6。
- A01-v3-lr1e6：AdamW LR1e-6。
相同数据/顺序/seed42，batch4×accum4，1epoch，FP32参数+BF16 autocast，seq1536。
**只比较各组完整一轮的末步checkpoint**，中间NLL只作诊断，不以它替代行为验收。
实际targets/steps/各域token占比由构建manifest和训练日志确定；两组同一完整数据集预算一致。

## 数据与格式修复

- 新graph train800/val160/test240，沿用v2环境语义但更换记录ID/数值和用户措辞，四类均衡。
- 新8类多schema工具任务 train800/val160/test240：数学、单位、天气、汇率、时间、翻译、随机数、随机数后计算。
- 回放历史broad chat-repair-v2：train选2000chat+600native-tool conversations；
  validation选其原始validation150chat+50native-tool conversations。
- 不整包回放S09定向旧回归题改写课程；历史broad本身仍是能力族课程/旧S10来源，
  因此回放NLL与固定Chat/Tool只能作为保持门禁，不能当成独立新能力证明。
- 每条assistant decision显式携带自己的tools；无tools Chat使用None，不注入graph schema；原生带tools对话保留自身schema。
- assistant-only label、EOS正常监督、显式右padding；拒绝过长样本，不截断完整工具链。
- train/validation/test按normalized user prompt去重，并排除固定Chat/Tool及旧graph val/test prompt；
  这是exact/normalized去重，不声称语义级零污染。新工具schema与既有工具相同，参数任务是新生成。
- 任务环境是可重复的有限fixture，禁止运行模型生成代码或访问网络。

## 选择与验收

先只看新validation和已知保持门禁：
- graph E2E≥90%，schema≥95%，参数语义/执行≥90%，retry≥60%；
- 新多schema tool validation E2E≥85%；
- 旧Chat至少8/10，旧Tool至少7/8，格式至少5/6，复读异常至多1/10；
- 两组均通过时选graph/tool validation E2E均值更高者，平分选低LR；任一门禁失败不能promote。
通过后写checkpoint freeze并只对选定模型和S10打开新graph/test240与tools/test240。
IFeval和七项基准仍为Phase6收口所需回归，不因小样本门禁通过宣称完成。
若两组失败，保存失败证据，不直接反复用已打开test进行调参。

## 结论边界

本轮同时改变起点、回放数据和学习率，不能将相对v2改善单独归因于某一个因素。
两组之间只比较LR；后续如需数据混合的因果结论，另做同起点/LR/targets的数据消融。
旧S10基线继续保留。所有attempt记录到MiniMind-Lab，代码/配置/数据指纹/训练评测结果归档。

## 执行记录（结果后追加，不改变门槛）

实际数据mixed-agent-v3-r1，隔离14条原生JSON/schema无效候选并重采样。
两组570更新/406611 targets，固定末步；旧Tool均5/8，未选模型，test未打开。
见[结果报告](../../experiments/05-agentic-rl/phase6-v3-report.md)。
