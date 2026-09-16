# K00 Qwen3-8B non-thinking IFEval

状态：running。用户2026-09-09要求先评测Qwen3-8B能力；本轮范围是完整IFEval541题的指令遵循，尚无最终分数。
GPU6；FP16非量化；0-shot、batch1、seed42、greedy、max_new_tokens1280；官方chat template明确enable_thinking=False并对照实际渲染验证。
权重五分片及配置/tokenizer均记录SHA；本地下载revision未恢复，不编造来源commit，以model-assets.json内容哈希固定实际资产。
不同模型各用自身tokenizer/chat template，1280tokens不是相同字符预算；对照原始题目doc_hash，不要求跨tokenizer的渲染prompt_hash一致。
记录prompt/inst strict与loose四项指标、生成长度/触顶比例及全部逐题输出；原始输出只留CPFS。
对照S10与官方MoE资格评测；Qwen与MiniMind词表不同，通过IFEval也不等于可以直接做token KL蒸馏，更不代表知识/推理全部评测完成。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/fjfxv8yz
脚本：scripts/eval/run_phase8_qwen3_ifeval.py。
CPFS：/data/artifacts/minimind-lab/phase8-k00-qwen3-8b-ifeval-20260909。最终以DONE.json、完整样本计数和权重再校验为准。
未训练、未修改权重、未操作其他项目，当前未提交推送。

## 完成记录（覆盖上述启动状态）

完整IFEval541题已完成：prompt strict80.9612%、instruction strict87.0504%、
prompt loose84.6580%、instruction loose89.4484%。权重前后校验通过。
这不是七项通用评测，也不能证明教师在所有领域强于学生。原始证据见DONE.json与manifest.json。
