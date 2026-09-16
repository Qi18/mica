# S04-dense-sft-custom-pilot-8m-20260904

Custom-data arm of the preregistered S03/S04 equal-budget SFT data comparison.

- Initialization: P03 64M checkpoint (SHA-256 pinned in config)
- Train budget: 8,001,601 shifted assistant targets across 55,661 rows
- Runtime: 8 x NVIDIA L20, bf16, max length 768, LR 5e-5, one epoch, seed 42
- Batch: 24 sequences/GPU; about 289 optimizer updates and 27,687 assistant targets/update
- Validation: explicit held-out validation file; no trainer-side resplit
- Tracking: SwanLab project `MiniMind-Lab`
- Comparator: `S03`

The sequence batch differs between the two arms because the official and custom datasets have different mean supervised response lengths. This keeps the effective supervised-token batch and optimizer-step count within about 2%, which is the relevant control for this comparison.
