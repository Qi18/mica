# M00-dense-probe-s42-20260908

100步架构探针完成，不是全量训练或模型晋级。
有效targets=223021；64行短validation NLL=7.129863。
稳态有效targets/s=60781.84；峰值allocated MiB=2363.22。
checkpoint strict load及optimizer roundtrip通过；不代表独立进程resume验收。
SwanLab：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qtfry4f7
权重和optimizer仅CPFS：/data/artifacts/minimind-lab/phase7-probe-20260908/M00-dense-probe-s42-20260908/checkpoint.pt；SHA=9a94845700de3ed7bdba75f8ce7cca27103d9b37d2d6752d5b62614251b6b956。
