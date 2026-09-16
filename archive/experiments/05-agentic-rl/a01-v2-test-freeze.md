# A01-v2 checkpoint freeze：独立测试前登记

2026-09-08。训练和checkpoint选择结束，现冻结：
- 模型：/data/artifacts/minimind-lab/A01-agent-sft-v2-pilot-20260908/checkpoint-225
- SHA256：51130347e6beae1698186ff26ebdc3a099ce23dfa8505e4f1590238e6511933c
- 选择依据：v2 validation NLL最小；225 optimizer updates，107859 target tokens。
- 独立测试基线A00：原始S10导出；A01-v1仅为迁移诊断，不替代A00。
- test：record-agent-v2/test.jsonl，240条，sha256 b5b00fb2c3ec21405e22832a3cfa5368701a560374cf19ae1953dd74d7ba338c。
- 两模型相同greedy、batch8、96tokens/turn、6turns、1536context。
- A01-v2 validation159/160；反事实159/160。32train prompts×4sampled rollouts全部成功，奖励方差0。
- 因预注册饱和条件，A02/A03不启动。先完成冻结A01的独立test和固定Chat/Tool回归。
- 本次test不再用来改v2训练、LR或选择别的checkpoint。今后若改环境必须单列版本和新的未见测试。
- 既有Chat/Tool回归用eval_sft_behavior.py统一默认FP16、greedy、max_new_tokens128。
- 本轮不宣称IFEval/七项回归完成，不将Phase6写成accepted。
