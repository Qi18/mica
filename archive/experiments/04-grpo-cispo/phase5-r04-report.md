# Phase5 R04 verifier-RL report

Status: results recorded; promotion held pending SFT padding correction and revalidation.

## Diagnosis and correction

R02 failed because multiple-choice reward could be optimized by answer-position betting.
R03 removed the position shortcut but used two-digit modulo-10 arithmetic that the 64M
S10 model did not learn in these probes: after EOS masking and training for up to 800
updates, validation pass@1 remained 11.4% and one digit occupied 72.4% of greedy outputs.
R03 was rejected before any formal run or test evaluation.

R04 therefore uses a narrower marked-target extraction task. The answer is still hidden
among distractors and checked by a strict whole-completion integer verifier, but the rule
is within the model's capacity. The dataset contains 4,000 train, 500 validation and
1,000 test examples; answers are exactly balanced and cross-split task overlap is zero.
The test split remained sealed until R04B/C/D training had all completed.

## Formal results

| Candidate | Method | Updates | pass@1 | sampled accuracy | pass@8 | Max digit share | Chat | Tool E2E |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R04A | S10 baseline | 0 | 12.2% | 9.63% | 39.6% | 16.3% | 8/10 | 7/8 |
| R04B | SFT | 100 | 84.8% | 81.44% | 94.6% | 12.0% | 6/10 | 7/8 |
| R04C | GRPO | 100 | 86.1% | 67.15% | 95.0% | 12.4% | 8/10 | 7/8 |
| R04D | CISPO | 100 | 86.7% | 69.11% | 95.7% | 12.2% | 8/10 | 7/8 |

Paired bootstrap on 1,000 aligned test tasks with 10,000 replicates:

- SFT vs baseline: +72.6 percentage points, 95% CI [+69.7, +75.3].
- GRPO vs baseline: +73.9 percentage points, 95% CI [+71.0, +76.6].
- CISPO vs baseline: +74.5 percentage points, 95% CI [+71.7, +77.1].
- GRPO vs SFT: +1.3 percentage points, 95% CI [-1.4, +4.0].
- CISPO vs SFT: +1.9 percentage points, 95% CI [-0.8, +4.6].
- CISPO vs GRPO: +0.6 percentage points, 95% CI [-0.3, +1.5].

The recorded GRPO and CISPO scores meet the numerical gates versus R04A: the confidence
interval lower bound is above zero, no output digit exceeds 35%, Chat is at least 7/10,
and Tool E2E is at least 6/8. CISPO has the highest point estimate, but the experiment
does not establish that it is better than SFT or GRPO. SFT also raises task accuracy,
but its Chat probe regresses from 8/10 to 6/10. The affected SFT control prevents attributing this difference to the training algorithm.

## SwanLab

- Unified evaluation (pending revalidation): https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/fbu7s5ik

- R04B SFT: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/b3v1xgwe
- R04C GRPO: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/wg8wrx33
- R04D CISPO: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/t023y9g8

## Boundary

These measurements show increased scores on the controlled target-extraction task
without the answer-position collapse seen in R02. It is not evidence of improved general
reasoning, and one seed is insufficient for a stability claim.

## Publication audit (2026-09-08)

A read-only CPU reproduction found a label-alignment bug in the shared Phase5 SFT
trainer: input IDs are left-padded while labels are right-padded. In a two-row test,
the short row supervised token 52 at position 20, where the input token was 1; the
longest row was aligned. Variable-length SFT batches therefore supervise incorrect
positions. The recorded SFT control and prior EOS diagnosis need correction and rerun;
promotion is held. No new training was performed during this publication audit.

R03 also changed decoding from four tokens to one alongside EOS masking. The earlier
claim that EOS masking alone fixed empty answers was not established by a controlled
ablation. R04 changes the task to digit extraction, so it does not establish improved
arithmetic reasoning.

R04A/B test decoding used batch size 50, while R04C/D used 100. Sampling seeds depend on
batch start, so sampled scores used different random draws. Greedy per-task bootstrap
intervals above describe the saved predictions; they do not cover training-seed variance
or correct the affected SFT control. No official-seven or IFEval rerun is included here.
