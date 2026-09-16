# M00-dense-probe-s43-20260908

100步架构探针完成，不是全量训练或模型晋级。
有效targets=223021；64行短validation NLL=7.176444。
稳态有效targets/s=61846.52；峰值allocated MiB=2375.41。
checkpoint strict load及optimizer roundtrip通过；不代表独立进程resume验收。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/lqpjih38
权重和optimizer仅CPFS：/data/artifacts/minimind-lab/phase7-probe-20260908/M00-dense-probe-s43-20260908/checkpoint.pt；SHA=b4570b0961724fa96f9a0fe498770bca03b8ea90b45df6b3369a8350363db232。
