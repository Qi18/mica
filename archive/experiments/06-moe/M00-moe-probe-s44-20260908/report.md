# M00-moe-probe-s44-20260908

100步架构探针完成，不是全量训练或模型晋级。
有效targets=223021；64行短validation NLL=7.209452。
稳态有效targets/s=31433.13；峰值allocated MiB=4302.80。
checkpoint strict load及optimizer roundtrip通过；不代表独立进程resume验收。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3dv7s2og
权重和optimizer仅CPFS：/data/artifacts/minimind-lab/phase7-probe-20260908/M00-moe-probe-s44-20260908/checkpoint.pt；SHA=12a3e872c6f293779a82ddeff4b3242f0ead8067d5739f4f980f084b3802c191。
