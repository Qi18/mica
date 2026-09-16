# Stage 0 L20 环境与历史基线盘点

## 状态

Stage 0 已完成。环境、NCCL、源码运行时、实验模板和 SwanLab 登录均通过硬门禁；2026-08-23T05:18:03Z 再次执行强制 preflight，结果为 pass。

## 环境验收

| 检查项 | 结果 | 证据 |
|---|---|---|
| ACK API | 通过 | direct /readyz 返回 HTTP 200 |
| Pod | 通过 | 训练 Pod 1/1 Running，0 restart |
| SSH | 通过 | 真实 OpenSSH 握手进入 L20 Pod |
| GPU | 通过 | 8×NVIDIA L20，单卡 46068 MiB |
| GPU 空闲 | 通过 | 8 卡均 0 MiB、0% utilization，无 compute process |
| PyTorch/CUDA | 通过 | PyTorch 2.6.0+cu124、CUDA 12.4，CUDA matmul 成功 |
| BF16 | 通过 | torch.cuda.is_bf16_supported() 为 True |
| NCCL | 通过 | 8 rank all-reduce 均得到 36 |
| CPFS | 通过 | /data 为 NFS read-write，容量约 3.6T |
| /dev/shm | 通过 | 800 GiB |
| SwanLab | 通过 | cloud verify 通过；凭据未写入仓库、日志或 CPFS |

网络切换后 ACK API 白名单已恢复，但公开实验记录不保存公网 IP、ACL、集群 ID、账号或凭据。

SwanLab 登录有两个时间点：04:22:35Z 首次检查为未登录（`metrics.csv` 的 `swanlab_login_first_check`、`environment.json` 的 `smoke_tests.swanlab_login`），在 Pod 上完成登录后 05:18:03Z 强制复检为 pass（`environment.json` 的 `smoke_tests.swanlab_login_recheck`）。Stage 0 不训练模型，因此没有云端 run，`swanlab-url.txt` 只记录该时间线以及首批云端 run（E02 的 seed42/43/44）的位置。

## 独立环境

- 环境路径：/data/venvs/minimind-lab。
- Pod 镜像缺少 python3.10-venv，已在当前临时 Pod 安装后创建环境。
- MiniMind 官方 requirements 已安装。
- CUDA/PyTorch 使用镜像系统包，MiniMind 固定的 Python 包安装在 venv。
- pip check 唯一告警来自系统 litellm 与 MiniMind 固定 openai==1.59.6；MiniMind 训练入口不导入 litellm，因此先保留告警而不偏离官方 pin。

## 模型 smoke test

在 BF16 CUDA 上完成随机初始化前向：

| 模型 | 参数量 | logits | finite | peak allocated |
|---|---:|---|---|---:|
| Dense | 63,912,192 | [1, 16, 6400] | 是 | 149.35 MiB |
| MoE | 198,416,640 | [1, 16, 6400] | 是 | 406.41 MiB |

该结果只证明模型构造和前向可运行，不代表训练效果。

## Official Zero 历史资产

只读检查范围：

- 当前权威仓库 /data/projects/minimind-lab；
- /data/cache；
- /data/projects 中排除 .git、node_modules、.deps 等依赖目录后的候选文件；
- CPFS .snapshots。

未找到以下资产：

- pretrain_t2t_mini.jsonl 或 sft_t2t_mini.jsonl；
- Pretrain/SFT 权重或 resume checkpoint；
- Official Zero 日志、run.json、metrics.csv；
- 可关联的 SwanLab URL/run id；
- CPFS snapshot。

因此历史 Official Zero 仍只能写成“曾运行并恢复”，不能写成已完成。

## 运行时保护

新增的启动基础设施：

- preflight_l20.py：检查 GPU 空闲、CUDA/BF16、CPFS、/dev/shm 和 SwanLab 登录；
- run_guarded.py：单实例文件锁、最终命令回显、GPU 二次门禁、信号转发；
- nccl_smoke.py：8 卡 NCCL all-reduce；
- init_experiment.py：生成固定实验目录；
- validate_experiment.py：检查必需文件和 JSON/CSV 基础格式。

MiniMind 源码没有 SIGTERM 即时 checkpoint handler。启动器只保证把信号转发给整个子进程组并避免孤儿进程；恢复点仍是最近一次 save_interval checkpoint。

## 阶段门

| 阶段门 | 状态 |
|---|---|
| 环境 smoke test | 通过 |
| GPU 占用确认 | 通过 |
| 历史资产盘点 | 通过，结论为未找到 |
| 实验模板可验证 | 通过，生成、格式校验和清理 smoke test 成功 |
| SwanLab 登录 | 通过；账号、服务地址和 netrc 权限已验证 |
| run id → 源码/曲线/checkpoint 追溯 | Stage 1 首次训练 run 的验收项，不阻塞 Stage 0 |

## 下一步

1. 进入 Stage 1 Tokenizer/Dataset，冻结数据版本、tokenizer 配置和基线样本。
2. 初始化首个正式实验目录，并把配置、源码 commit 和 SwanLab run id 建立映射。
3. 训练启动前继续强制执行 preflight 和单任务锁；`swanlab sync` 仍需单独授权。
