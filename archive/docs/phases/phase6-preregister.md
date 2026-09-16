# Phase 6 预注册：Agent SFT → Agentic RL

状态：启动准备；日期：2026-09-08。按统一计划第 11、16.4 节从 Phase2 S10 独立分叉。
Phase5 R04 的 SFT padding 对照仍待复验，不将其结果作为本阶段前置能力证据。

## 实验序列

- A00：S10 未做 Agent 专项训练的基线。
- A01：同一 S10 上的 Agent SFT；先验证轨迹学习。
- A02：仅在 A01 基础 Tool Call 稳定后，开始 Agentic GRPO。
- A03：从与 A02 相同的 A01 checkpoint 开始 Agentic CISPO。
- A04：可选 rollout 系统对比，当前不启动。

基础模型：`/data/artifacts/minimind-lab/D01-s10-preference-baseline-20260908/exported-fp32`。
A00/A01 为受控合成环境，不能替代 BFCL，也不代表开放域 Agent 能力。

## 环境与数据 v1

两种有严格 JSON schema 的本地工具：lookup_record、add_numbers。禁止执行模型生成代码。
记录查询、查询后加法、首次查询临时失败后重试三类等量任务。
train 2400、validation 240、test 300；按 record_id / prompt 分离，同模板族，不声称模板 OOD。
记录值不放入用户 prompt，必须从 observation 获取；测试任务不用于训练、LR 或 checkpoint 选择。
每轮最多一次工具调用，最多 4 轮、96 generated tokens/轮；上下文最多 1024 tokens。
工具是常数时间字典查询/有界整数加法，无网络或任意 eval；错误最多重试一次。
执行器显式返回 not_found、temporary_error；不将错误默认为成功。

必须验证：oracle 轨迹全部可执行、跨 split ID/prompt 无重叠、无截断、监督 token 与 input 同位置一致、
工具 observation/system/user/padding 不参与 loss、EOS 正常监督。
本阶段独立 collator 显式右填充，不复用 Phase5 有错位问题的 sft_batch。

## A01 首轮预算（pilot，不是正式能力结论）

FP32 参数 + BF16 autocast；GPU0 现有空余显存，不停驻留进程。
AdamW，LR 1e-5，microbatch 4，accumulation 4，有效 decision batch 16，seed42。
最多 1 epoch 首轮 pilot；每 100 updates 与末尾记录 validation token-weighted NLL，保留最低 NLL checkpoint。
loss 下降不能视作阶段通过；只在 validation 做诊断，正式 test 等 A01 方案冻结后才打开。

## 验收和报告口径

A01 启动 RL 的 validation 门槛：schema validity ≥95%、tool selection ≥90%、
argument semantic accuracy ≥90%、execution success ≥90%（预期临时失败独立报告，不纳入可执行成功分母）。
端到端成功率至少 70%，retry 子集至少 60%；未达到时只修 A01，不启动 RL。
所有门槛在本文件落地后才运行模型基线。

正式报告按统一计划记录：schema validity、tool selection、参数语义、execution success、
observation usage（必要时反事实改变返回值验证）、final answer accuracy、E2E、
平均调用/轮数、未完成率、无效调用率、延迟、token 成本。
E2E 必须工具链、语义参数、最终答案全部正确；仅出现答案子串不得分。
RL 主要对比独立 test E2E、retry 成功率与 paired bootstrap 区间；训练 reward 不代替评测。
A02/A03 同源 checkpoint、同 prompt/token 预算、同 decoder/seed，保留 A01 no-RL 对照。
收口前做既有 Chat/Tool、IFEval 与七项通用能力回归；未跑不写保持。
所有训练和评测归入 SwanLab MiniMind-Lab。保留失败 attempts，不覆盖历史 checkpoint。
