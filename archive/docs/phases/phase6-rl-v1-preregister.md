# Phase6 A02/A03 探索性Agentic RL对照预注册

2026-09-08，用户明确要求“开始agent rl对比一下”后登记，训练与新test之前落地。
这是A01保持门禁未通过时的探索性例外，不代表A01被接受或允许替换S10。
A01旧Tool5/8仍低于7/8门槛；Phase6不能因此收口。

## 固定三臂

- A01 no-RL：A01-v3 LR3e-6固定checkpoint-570，不更新。
- A02：同一A01 checkpoint，GRPO clipped surrogate。
- A03：同一A01 checkpoint，CISPO stop-gradient clipped IS surrogate。
不使用v2已经退化的权重；不从A02接着训练A03。

## 预算、数据和奖励

- 先probe：仅32个训练prompt，每prompt4条，T1/top_p1/top_k0；至少4个有奖励方差的group且有成功/失败，
  否则不盲目执行优化器，保存probe并重新设计rollout池。
- train来自mixed-agent-v3-r1的train.jsonl/tools-train.jsonl；同seed42顺序，
  每outer2graph+2tools、每prompt4条；32outer、2inner，64updates、512episodes。
- 相同prompt与生成上限预算，不宣称实际token数完全相同；分别记录实际tokens和耗时。
- LR1e-6，AdamW无weight decay，grad clip1，FP32参数+BF16 autocast，GPU0/1各进程15%显存上限。
- 奖励0.8×严格E2E + 0.2×正确有序工具前缀比例；完整工具链+参数+最终答案正确才得1。
  partial reward单独记录，评测只看严格E2E，不能用reward替代能力。
- 每个prompt内部reward均值/std归一化；零方差组advantage=0，不偷偷删除或重新采样。
- 每个assistant action保留原始采样token；system/user/tool observation以及历史上下文只作为条件，mask掉loss。
  EOS计入action；环境是本地fixture，无任意代码或网络执行。

## 优化实现与审计

两组统一全batch assistant-token mean和beta0.01 reference k3 KL。
因此这里的GRPO是token-normalized变体；CISPO带共享KL，不是MiniMax-M1完整配方复现。
GRPO ratio clip[0.8,1.2]，CISPO仅上限2且对weight stop-gradient。
除surrogate外其余相同；没有额外SFT replay更新，避免混入另一个变化。
采样T1、top_p1、top_k0、无repetition penalty；记录真实generation logprobs，
核验teacher-force max差<0.20/mean差<0.02，对小BF16/cache差异施加detached exp(old-generation)修正。
每个新rollout第一次update前old/new ratio须为1；dropout关闭，显式右padding，动作位置一致。
通过toy梯度方向、clipping、零方差、mask/ratio审计后才训练；NaN/Inf或KL>0.2停止且保留失败。
代码不复用Phase5有错位问题的sft_batch。

算法参考：[DeepSeekMath](https://arxiv.org/abs/2402.03300)、
[MiniMax-M1 Eq4–5](https://arxiv.org/html/2506.13585v1)、
[NeMo CISPO](https://docs.nvidia.com/nemo/rl/nightly/guides/cispo.html)。

## 固定评测与结论

- 只保存并比较两组固定末步，不能根据val/test挑checkpoint。
- 先同协议graph val160、tools val160和旧Chat/Tool；保持失败也要如实报告。
- 两组训练代码审计通过并完成后，登记三臂checkpoint SHA，再一次性打开v3 frozen graph test240 + tools test240，
  用于本次探索性RL三臂对比，不用于A01选择或后续调参。此前这些test未被模型评测。
- 该test之后即是已打开测试，后续迭代须另留新的冻结集；不把本轮val变成test。
- 汇总严格E2E、各family、参数/工具链、未完成、tokens和paired bootstrap95%CI。
  只有单seed、小规模合成fixture，不能宣称GRPO/CISPO普遍优劣或开放域Agent能力。
- RL专项是否提升与通用模型是否可晋级分开。IFEval/七项保持仍为正式Phase6收口门禁，本轮不能绕过。
- 训练与评测统一SwanLab MiniMind-Lab，轻量证据入experiments，权重保留CPFS；未经请求不commit/push。
