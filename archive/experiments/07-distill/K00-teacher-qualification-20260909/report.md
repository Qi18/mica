# K00 teacher qualification

状态：进行中，先下载校验官方Full SFT MoE并验证与S10兼容性；尚未判定teacher合格，未启动正式蒸馏。
证据根目录：/data/artifacts/minimind-lab/phase8-k00-teacher-gate-20260909。
完整预注册见[Phase8](../../../docs/phases/phase8-distill.md)。

## 兼容性通过

source-metadata.json锁定官方release；compatibility.json确认两臂strict load、CPU有限logits及tokenizer语义一致。
prepare-failure*.json保留准备期网络超时和文件格式SHA混用失败，修复后未改变任何模型权重。
下一步：同原生适配路径完整IFEval 541题对照，结果出齐前不宣称teacher合格。

## 实际启动

两臂已实际生成评测响应，非只提交任务；完整541题评测进行中。
- student: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/15gy76hx
- teacher: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3uavnnjv

启动命令/commit/PID见launch.json，协议及权重SHA见各臂manifest。进度快照不代表最终完成，最终以CPFS DONE.json及逐题检查为准。
