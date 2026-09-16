# M00-moe-probe-s42-20260908

100步架构探针完成，不是全量训练或模型晋级。
有效targets=223021；64行短validation NLL=7.162615。
稳态有效targets/s=30681.12；峰值allocated MiB=4310.54。
checkpoint strict load及optimizer roundtrip通过；不代表独立进程resume验收。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/zn6tg5eb
权重和optimizer仅CPFS：/data/artifacts/minimind-lab/phase7-probe-20260908/M00-moe-probe-s42-20260908/checkpoint.pt；SHA=e11f54329b2292dc84e65771e3b7298a625027723b395646c5e7d95a34b49493。
