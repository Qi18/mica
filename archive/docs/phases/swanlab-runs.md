# SwanLab run 索引

阶段报告按 [`../experiment_plan.md`](../experiment_plan.md) 第 16.5 节链接到对应训练 run。本文件只是跨阶段的汇总视图；每个实验的权威录入处仍是 `experiments/<stage>/<id>/swanlab-url.txt` 和 `experiments/registry.csv` 的 `swanlab_url`，两者不一致时以实验目录为准并当天修正本表。

Workspace 固定为 `@richliu0153`。完整 URL 形如 `https://swanlab.cn/@richliu0153/<project>/runs/<run_id>`。

| 实验 | run 角色 | project | run_id | URL 出处 |
|---|---|---|---|---|
| E00-l20-baseline-20260823 | 无云端 run（只验环境不训练） | - | `n/a-no-cloud-run` | `swanlab-url.txt`（含登录时间线） |
| E01-tokenizer-dataset-20260823 | 不适用（不训练模型） | - | `n/a-no-training` | `swanlab-url.txt` 说明 |
| E02-model-probe-20260823 | probe seed42（step1–50 与 step51–100 resume 共用同一 run） | MiniMind-Lab | `iq14wfm1nc1ca8iigdbop` | `swanlab-url.txt` |
| E02-model-probe-20260823 | probe seed43 | MiniMind-Lab | `is8yx09hw3341ar8qvvfa` | `swanlab-url.txt` |
| E02-model-probe-20260823 | probe seed44 | MiniMind-Lab | `kngkosspce6sfuzkoztzm` | `swanlab-url.txt` |
| P01-dense-pretrain-mini-20260824 | formal（`P01-Pretrain-Mini-64M-Seq768`，由 `MiniMind-Lab-Stage3/7iochx9kfe75qa2pt6d1u` 同步） | MiniMind-Lab | `nfax3tyg0j217j1cz8y0b` | `swanlab-url.txt` |
| P02-dense-pretrain-full-20260824 | formal（由 `MiniMind-Lab-Stage5/bs7n0qfcxykk13fammxis` 同步） | MiniMind-Lab | `3i1muwq039fpfv89fq4ru` | `experiments/01-pretrain/P02-dense-pretrain-full-20260824/swanlab-url.txt`（main） |
| P03-dense-pretrain-v1-1b28-20260901 | probe（100 step） | MiniMind-Lab | `d9is4iayxaw41ba95u92s` | `/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/probe-b32-a1-step100/attempts/20260901T080420Z-540642/driver.log` |
| P03-dense-pretrain-v1-1b28-20260901 | formal（1 epoch） | MiniMind-Lab | `qdpjh47fjt98184oos4bl` | `/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/formal-b32-a1-epoch1/attempts/20260901T080746Z-542788/driver.log` |
| P03-dense-pretrain-v1-1b28-20260901 | eval logging（`P03-Eval-V1-1B28-64M-Seq768`） | MiniMind-Lab | `k9st16wqu3i7ijy2d7q9h` | `swanlog/run-20260901_110827-k9st16wqu3i7ijy2d7q9h/backup.swanlab` |

| S01-dense-sft-official-mini-smoke-20260903 | 官方 mini SFT smoke（run 已删除，保留历史 receipt） | MiniMind-Lab | `e4php2e6` | `experiments/02-sft/S01-dense-sft-official-mini-smoke-20260903/swanlab-url.txt` |
| S02-dense-sft-official-full-1epoch-20260904 | 官方完整 SFT | MiniMind-Lab | `7ngous7e` | CPFS driver log |
| S03-dense-sft-official-pilot-8m-20260904 | 官方 pilot 训练 | MiniMind-Lab | `7duhsb00` | `swanlab-url.txt` |
| S03-dense-sft-official-pilot-8m-20260904 | 官方 pilot 评测 | MiniMind-Lab | `50vjh86o` | `swanlab-url.txt` |
| S04-dense-sft-custom-pilot-8m-20260904 | 自建 pilot 训练 | MiniMind-Lab | `h0k5g99o` | `swanlab-url.txt` |
| S04-dense-sft-custom-pilot-8m-20260904 | 自建 pilot 评测 | MiniMind-Lab | `3jawb0uu` | `swanlab-url.txt` |
| S05-official-pilot-replay32m-lr1e5-20260907 | 官方 pilot 32M 低 LR 重放 | MiniMind-Lab | `0cj157tn` | CPFS train log |
| S06-official-base-official-pilot8m-20260907 | 官方 Base 对照 | MiniMind-Lab | `ucxyuqmu` | CPFS train log |
| S07-chat-repair-v1-20260907 | Chat 修复 v1 | MiniMind-Lab | `i4z0joa4` | CPFS train log |
| S08-chat-repair-v2-20260907 | Chat 修复 v2 | MiniMind-Lab | `4kuy7jzj` | CPFS train log |
| S09-targeted-chat-curriculum-20260907 | 定向课程训练 | MiniMind-Lab | `ipqdqgzl` | `swanlab-url.txt` |
| S09-targeted-chat-curriculum-20260907 | 行为/七项评测 | MiniMind-Lab | `pgl5rv0h` | `swanlab-url.txt` |
| S09-targeted-chat-curriculum-20260907 | P03/S02/S09 IFEval 对比 | MiniMind-Lab | `x6ps9jxa` | CPFS eval receipt |
| S10-ifeval-curriculum-v4-20260907 | IFEval 课程训练 | MiniMind-Lab | `bxdob3rh` | `swanlab-url.txt` |
| S10-ifeval-curriculum-v4-20260907 | IFEval/七项/行为评测 | MiniMind-Lab | `encs5zuk` | `swanlab-url.txt` |
## 待修正的录入缺口

- ~~E00 的机器可读证据与报告矛盾~~（2026-09-03 已修）：`metrics.csv` 拆为 `swanlab_login_first_check`（blocked, 04:22:35Z）与 `swanlab_login`（pass, 05:18:03Z），`environment.json` 新增 `swanlab_login_recheck`，`swanlab-url.txt` 改记时间线与首批云端 run（E02 三 seed）。
- ~~`registry.csv` 中 E00/E01 的 `swanlab_url` 为空~~（2026-09-03 已修）：已写入 `n/a-no-cloud-run` 与 `n/a-no-training`。
- ~~P02 的 `swanlab-url.txt` 只存在于 `stage5/p02-dense-pretrain-full`，`main` 的 `registry.csv` 尚无 P02 行~~（2026-09-03 已修）：`main` 的 registry 已补 P02 行（`completed`），实验目录也已随合并 commit `6e1c67e` 进入 main，`swanlab-url.txt` 可直接打开。
- ~~P03 目录没有 `swanlab-url.txt`，`registry.csv` 也无 P03 行~~（2026-09-03 部分修）：`main` 的 registry 已补 P03 行（`awaiting-report`，lab_commit `222e39c9…`，时间取自 `formal-b32-a1-epoch1/attempts/20260901T080746Z-542788/exit-status.json`），P03 目录也已随合并 commit `90a5c12` 进入 main；但目录仍只有 `README.md`/`command.sh`/`config.json`/`eval/`，缺 `report.md`/`run.json`/`metrics.csv`/`swanlab-url.txt`，三条 run URL 仍只能从 artifacts 日志与本地 `swanlog/` 回读。Phase 1 收口必须回填。
- ~~P01 使用 project `MiniMind-Lab-Stage3`~~（2026-09-03 已修正认知）：P01/P02 的 run 已于 2026-09-01（stage5 收口 commit `5761979`）同步到统一 project `MiniMind-Lab`，因此现存所有 run 同 project、可在 SwanLab 界面叠图；`main` 之前只是没有同步这一修正，现已回填到 P01 的 `swanlab-url.txt`、`run.json`、`report.md` 与 registry。旧 run id 作为 `source_swanlab_run_id` 保留。

## Phase 3 完整记录（2026-09-08）

- L00-code-baseline-20260907: n/a-no-training / https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/980w5n9d
- L01-code-full-ft-20260907: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/pz1x0ux9 / https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/gtetdf4e
- L02-code-lora-r16-20260907: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s0l5ishg / https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/8z9jka3t

## Phase4（2026-09-08启动）

- D01-s10-preference-baseline-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/os17qekn
- D02-chosen-only-sft-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/51vdt97x
- D03-dpo-official-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3w1au7dk
- Phase4-Evaluation-D01-D02-D03: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/csvnmhkj

## Phase5（2026-09-08）

- R01B-math-sft-control-20260908（v1 invalid）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s587mp6r
- R01C-math-grpo-20260908（v1 invalid）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/89ptosrv
- R01D-math-cispo-20260908（v1 invalid）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/ecv3zn1h
- R02B-math-sft-control-20260908：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wqhvn9wd
- R02C-math-grpo-20260908：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/6t4l6v1o
- R02D-math-cispo-20260908：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/k6ez8f9j
- Phase5-Evaluation-R02A-R02B-R02C-R02D：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3sfq26tj

## Phase6（2026-09-08）

- A00-s10-agent-baseline-20260908（validation基线）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/p3yd932a
- A01-agent-sft-pilot-20260908（轨迹SFT pilot + validation）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/jkra8o25

### Phase6 v2 attempts

- A01-v2-transfer-baseline-20260908 (transfer baseline): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1t7ndogk
- A01-agent-sft-v2-pilot-20260908 (SFT pilot): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/mabi301l
- A01-v2-counterfactual-20260908 (counterfactual evaluation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/miy5qid6
- A01-v2-rl-eligibility-probe-20260908 (sampling-only RL eligibility): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/hsn1xolt
- A00-v2-frozen-test-20260908 (S10 independent test): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/j3b8qcu0
- A01-v2-frozen-test-20260908 (A01 independent test): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/gfltoxnx
- A01-v2-evaluation-summary-20260908 (evaluation aggregation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/8yso53w9

### Phase6 v3 mixed-replay repair

- A01-v3-mixed-lr3e6-20260908 (training + graph validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/52u6fhj4
- A01-v3-mixed-lr3e6-20260908 (multi-tool validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/hauawrx1
- A01-v3-mixed-lr1e6-20260908 (training + graph validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/5h3qys7d
- A01-v3-mixed-lr1e6-20260908 (multi-tool validation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/7kagdh51
- A00-v3-s10-graph-validation-20260908 (matched validation baseline): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1a6xkwzx
- A00-v3-s10-tools-validation-20260908 (matched validation baseline): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wtets9k0
- A01-v3-validation-summary-20260908 (joint gates): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/p6sbux9p

### Phase6 Agentic RL attempts（包含审计失败）

- A01-v3-agent-rl-probe-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0a9nthb8
- A01-v3-agent-rl-fp32-probe-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/494uwnlh
- A02-agentic-grpo-v1-20260908 (aborted-audit-failed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1hza7eyl
- A03-agentic-cispo-v1-20260908 (aborted-audit-failed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/fgvlab1b
- A02-agentic-grpo-v2-20260908 (completed-not-promoted): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/1pxh36vn
- A03-agentic-cispo-v2-20260908 (completed-not-promoted): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/vdwjd3sg
- A01-agentic-rl-v2-val-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/60s7cgkf
- A01-agentic-rl-v2-test-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/37rec37k
- A02-agentic-rl-v2-val-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0l8s0wh8
- A02-agentic-rl-v2-test-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/l6kyh7yq
- A03-agentic-rl-v2-val-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/yognri61
- A03-agentic-rl-v2-test-20260908 (completed): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/op09vul1
- A02-A03-rl-v2-summary-20260908 (completed-not-promoted): n/a-local-only-approval-required

### Phase6 general regression（仅L20本地）

- A00-s10-general-regression-20260908: n/a-local-only-no-upload；完整证据见实验目录。
- A01-general-regression-20260908: n/a-local-only-no-upload；完整证据见实验目录。
- A02-general-regression-20260908: n/a-local-only-no-upload；完整证据见实验目录。
- A03-general-regression-20260908: n/a-local-only-no-upload；完整证据见实验目录。
- A00-A03-general-regression-20260908: n/a-local-only-no-upload；完整证据见实验目录。

## Phase6 SwanLab 补传完成（2026-09-08）

6个历史指标记录已获用户授权补传至MiniMind-Lab；云端均为FINISHED，282项指标回读一致。
仅上传指标/元信息，不上传权重、数据集或完整逐题文本；不改变completed-not-promoted结论。

- A02-A03-rl-v2-summary-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/563ab2ac)
- A00-A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/79f294dc)
- A00-s10-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/0cb93996)
- A01-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/da935c86)
- A02-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/c74379f7)
- A03-general-regression-20260908: [SwanLab](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/a739ed89)

## Phase7 M00架构探针

- M00-dense-probe-s42-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qtfry4f7
- M00-moe-probe-s42-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/zn6tg5eb
- M00-moe-probe-s43-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nkxulnb9
- M00-dense-probe-s43-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/lqpjih38
- M00-dense-probe-s44-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/zc825k5w
- M00-moe-probe-s44-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3dv7s2og

## Phase7 正式训练

- M01-dense-pretrain-1b28-20260908（completed）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nyy90ef5
- M02-moe-pretrain-1b28-20260908（completed）：https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/uvoc97pz

### Phase7最终评测（均FINISHED）

- M01-dense-pretrain-1b28-20260908：训练 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nyy90ef5；评测 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/opixiuyd
- M02-moe-pretrain-1b28-20260908：训练 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/uvoc97pz；评测 https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/dxfvjdxy

## Phase8 K00 teacher资格评测（running，未训练）

- K00-student-IFEval: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/15gy76hx
- K00-teacher-IFEval: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3uavnnjv

## Phase8 Qwen3-8B候选资格评测（running）

- K00-qwen3-8b-ifeval-20260909: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/fjfxv8yz
