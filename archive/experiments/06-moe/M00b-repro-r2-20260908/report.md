# M00b-repro-r2-20260908

状态：invalidated（正式训练前恢复门禁未通过）。没有启动正式训练；原始权重/optimizer保留在/data/artifacts/minimind-lab/phase7-formal-r2-20260908。
确定性计算后仍有1.054e-6偏差，未放宽容差；后续固定DDP归约路径。
- dense-resumecheck: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/c78mrgp0
- dense-continuous: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qd02jh26
