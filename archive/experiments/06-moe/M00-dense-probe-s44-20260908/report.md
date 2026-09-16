# M00-dense-probe-s44-20260908

100步架构探针完成，不是全量训练或模型晋级。
有效targets=223021；64行短validation NLL=7.054892。
稳态有效targets/s=62602.24；峰值allocated MiB=2362.55。
checkpoint strict load及optimizer roundtrip通过；不代表独立进程resume验收。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/zc825k5w
权重和optimizer仅CPFS：/data/artifacts/minimind-lab/phase7-probe-20260908/M00-dense-probe-s44-20260908/checkpoint.pt；SHA=b520f573197fdf4c114ac2177f8fd023abf5b0bd24e55bae2daa1c36affbb65e。
