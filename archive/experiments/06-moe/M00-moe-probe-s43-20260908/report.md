# M00-moe-probe-s43-20260908

100步架构探针完成，不是全量训练或模型晋级。
有效targets=223021；64行短validation NLL=7.158418。
稳态有效targets/s=31446.88；峰值allocated MiB=4322.10。
checkpoint strict load及optimizer roundtrip通过；不代表独立进程resume验收。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nkxulnb9
权重和optimizer仅CPFS：/data/artifacts/minimind-lab/phase7-probe-20260908/M00-moe-probe-s43-20260908/checkpoint.pt；SHA=558e3e33d0b4aa7217d3aed9afe08dd8987a08de75ccd2767201fca689616b11。
