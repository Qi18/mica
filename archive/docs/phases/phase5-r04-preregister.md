# Phase5 R04 preregistration: target extraction verifier

Date: 2026-09-08

## Why R03 was rejected before formal training

R03 used two-digit add/sub modulo 10. The S10 baseline had only 0.4% validation pass@1.
The original SFT loss supervised a one-token answer and EOS equally; higher learning rates
therefore learned immediate EOS and produced empty completions. Masking EOS fixed empty
outputs, but 400, 400 and 800 optimizer updates still produced only 10.4%, 11.2% and 11.4%
pass@1, with 50.6%-72.4% of greedy outputs concentrated on one digit. The model learned
the output type but not the arithmetic rule. No R03 formal run or test evaluation was run.

## R04 task and validation selection

R04 is a narrow marked-target extraction task. Each prompt contains six digits and one
unambiguous marker; the model must emit exactly the marked digit. Reward is one only when
the entire normalized completion parses to the correct integer, otherwise zero.

The generated dataset contains 4,000 train, 500 validation and 1,000 sealed test rows.
Answers 0-9 are exactly balanced in every split; prompt/task-key overlap is zero; four
marker templates differ by at most one row. This task measures controlled verifier
learning and exact-format compliance, not general reasoning.

Validation-only probes selected the task and fixed the formal budget:

| Candidate | Updates | pass@1 | sampled accuracy | nondegenerate groups |
|---|---:|---:|---:|---:|
| S10 baseline | 0 | 12.6% | 9.08% | 36.2% |
| SFT probe | 100 | 85.4% | 80.35% | 32.8% |
| GRPO probe | 100 | 83.4% | 66.63% | 71.6% |
| CISPO probe | 100 | 85.4% | 68.53% | 70.0% |

The probe checkpoints are tuning artifacts only. Formal runs restart independently from
the same S10 checkpoint.

## Fixed formal runs

- Common base: `D01-s10-preference-baseline-20260908/exported-fp32`
- Data: `verifiable-target-extract-v1`
- Seed: 42; exact-integer verifier; answer-stratified prompt batches
- Decode/reward: 8 rollouts, temperature 1.0, top-p 1.0, max_new_tokens 1
- R04A: no training, baseline evaluation only
- R04B: SFT, LR 2e-6, batch 10, 100 outer steps, one update/step, EOS excluded from loss
- R04C: GRPO, LR 5e-7, batch 10, 50 outer steps, two inner updates, beta 0.05,
  entropy coefficient 0.002
- R04D: CISPO, same budget and hyperparameters as R04C
- SwanLab project/group: `MiniMind-Lab` / `Phase5-Verifiable-RL`
- Formal run names: `R04B-target-sft-control-20260908`,
  `R04C-target-grpo-20260908`, `R04D-target-cispo-20260908`

The sealed test split must not be evaluated until R04B/C/D training has completed.
After unsealing, all four candidates use the same greedy and sampled protocol. Report
pass@1, sampled accuracy, pass@8, per-template/per-answer results, output-digit
distribution, paired bootstrap confidence intervals, Chat 10-case and Tool 8-case
regressions. An RL method is promoted only if its paired bootstrap improvement over R04A
has a lower 95% confidence bound above zero, no digit exceeds 35% of greedy outputs, and
Chat is at least 7/10 while Tool E2E is at least 6/8.
