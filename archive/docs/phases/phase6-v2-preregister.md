# Phase6 v2：多步依赖与可恢复错误（预注册）

日期2026-09-08，v1 pilot A01 validation239/240已接近上限，因此另建v2而不改旧数据。
所有v1结果、冻结test均保留。v2仍为受控合成环境，不称作BFCL或通用Agent能力。

## 数据和协议

train800 / validation160 / test240，各均分chain、branch、retry、redirect四类，根/叶/干扰/迁移ID跨split不重复。
训练与验证/测试使用不同用户措辞；工具schema和环境语义相同，不把轻微措辞变化当作广泛OOD。
流程：查询根→从observation选择后继→查询叶→按根operation执行加法/减法→最终整数。
branch必须根据route选择左/右记录；retry处理叶查询临时失败；redirect遵循moved.new_id。
工具仅本地字典/有界整数加减，不执行生成代码，不联网。
最多6轮，每轮最多96token，context1536，greedy正式评测。
validation另有同prompt、同ID、不同隐藏数值的counterfactual副本；它绝不加入训练。

## 执行序列

1. 先评估v1 A01 checkpoint300在v2 validation上的能力迁移（A01 attempt v2-baseline）。
2. A01-v2 SFT继续学习v2轨迹：从v1 A01 checkpoint300继续，AdamW LR1e-5，batch4×accum4，
   首轮1epoch，seed42，每50updates验证token-weighted NLL并保留best。
3. v2 validation需schema≥95%、工具选择/语义参数/执行≥90%、E2E≥70%，retry≥60%，方可启动A02/A03。
   未满足先诊断A01，不降低门槛。若E2E再次≥98%，记录环境饱和，不人为削弱SFT制造RL收益。
4. 同一A01 checkpoint固定后，GRPO/CISPO各先做24outersteps、4prompts×4rollouts、2innerupdates pilot。
   LR5e-7，KL beta0.02，clip0.2 / CISPO maxratio2，seed42，temperature0.8、top_p1。
   奖励0.8×严格E2E+0.2×有序正确工具调用前缀比例；分量分别记录，不把shape reward当成功率。
   每条完整episode的advantage分配到它的各轮assistant生成token；系统/user/工具返回均不计loss。
   不以文本包含答案作为成功；EOS计入action，保留实际采样token，不拼接重tokenize伪造轨迹。
   先做old-policy ratio≈1与mask自检，训练监控KL/clip/奖励方差/零方差组/实际token预算。
5. no-RL A01保留。对A02/A03使用相同prompt数和每条轨迹token上限，记录实际tokens；
   不宣称实际token相等，若资源预算差异大另做matched-token对照。
6. 正式test必须在训练方案和checkpoint选择规则冻结后只开一次；当前只用validation诊断。

## 结论边界

v2 pilot不是Phase6最终验收。正式收口还要≥200条独立test、paired bootstrap、
反事实observation一致性、Chat/Tool/IFEval/七项回归和成本报告。
SwanLab统一MiniMind-Lab，A01的新增attempt不重编号或覆盖旧记录，A02/A03才对应RL方法。
若环境饱和/奖励方差几乎为零，记录负结果/无训练空间，不把调用RL脚本算作能力提升。
