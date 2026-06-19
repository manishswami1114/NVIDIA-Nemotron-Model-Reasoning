# Diag50 — 50-record diagnostic dataset

## What this is

A drop-in replacement of the 0.86-baseline training data with **exactly 50 cryptarithm_deduce records changed**. The other 10,495 records are byte-identical to `dont_touch_it/all_categorical_splits/`.

The 50 changed records:
- Keep the same prompt
- Have a new assistant CoT with the **same template** as the baseline cryptarithm CoT (character-for-character structure)
- Same length range (440-500 chars, matching baseline median 472)
- Same vocabulary
- Only difference: the "Symbol Mapping" section has **real digit values** (0-9 distinct) instead of baseline's poisoned "all symbols = 0", and "Operation" carries the real operator char instead of always "+"
- Final `\boxed{X}` answer is verified correct by the solver against `train.csv`

## What this experiment answers

> When we replace cryptarithm CoTs with real math at baseline length, does it regress cipher / gravity / numeral / unit_conversion?

Every prior attempt to fix cryptarithm has regressed LB from 0.86 → 0.83, and we never measured *which* categories dropped. This experiment isolates the variable.

## Training command

```python
# In your Kaggle training notebook — START from the 0.86 LoRA checkpoint
# and use a MUCH lower LR than the original to make a surgical change.

# 0.86 was trained from base with:
#   LORA_RANK = 32, LORA_ALPHA = 32, LR = 4e-4, 1 epoch, batch=2, grad_accum=2

# For diag50 continuation training, override LR only:
LORA_RANK     = 32       # same as 0.86
LORA_ALPHA    = 32       # same as 0.86
LEARNING_RATE = 4e-5     # 10x LOWER than 0.86's 4e-4 — surgical adjustment
NUM_EPOCHS    = 1
PER_DEVICE_BATCH = 2
GRAD_ACCUM    = 2

# Initialize from your 0.86 LoRA adapter (NOT from base model):
# Load the saved 0.86 adapter weights, then continue training on diag50.
# If 4e-5 still disturbs other categories, drop to 1e-5 for the next run.
```

## Eval command (after training)

1. Run inference on train.csv with the new adapter → produces `eval_diag50.csv`
2. Run the per-category eval:

```bash
cd "/Users/manishswami/developer/NVIDIA-Nemotron Model"
python3 scripts/diag50/eval_per_category.py /path/to/eval_diag50.csv
```

## Decision matrix (after eval)

| Outcome | Conclusion | Next step |
|---|---|---|
| `cryptarithm_deduce` ↑, 4-perfect categories unchanged | Replacement is **SAFE** at baseline length | Scale to 600+ records using the same generator |
| `cryptarithm_deduce` ↑, 4-perfect categories ↓ | LoRA **LEAKS** across categories | Lower LR further (5e-6), freeze more layers, or add baseline replay |
| `cryptarithm_deduce` flat | CoT format isn't being learned at 50 records | Try 200 records, or change CoT content |
| Everything ↓ | **STRUCTURAL** — can't safely modify | Stop and accept 0.86 |

## Files

- `alice_solver.py` — port of the AliceEquationSolver from the writeup (Python-only, used for sanity tests)
- `fast_solver.py` — numpy-vectorized gold-conditioned solver (the one actually used by the build)
- `gen_baseline_match_cot.py` — generates CoTs that match the baseline template byte-for-byte in structure
- `build_diag50.py` — builds `all_categorical_splits_diag50/` directory
- `eval_per_category.py` — measures per-category accuracy of a trained adapter against the 0.86 baseline
- `README.md` — this file

## Output

`all_categorical_splits_diag50/` — 9 JSONL files, drop-in replacement
`~/Downloads/diag50.zip` — submission-ready archive
