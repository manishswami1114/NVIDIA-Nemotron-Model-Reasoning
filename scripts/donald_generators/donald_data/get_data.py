import shutil, os
from pathlib import Path

DONALD_DIR = "/Users/manishswami/developer/NVIDIA-Nemotron Model/scripts/donald_generators/donald_data"
BASELINE_DIR = "/Users/manishswami/developer/NVIDIA-Nemotron Model/data/processed/all_categories_split"
OUT_DIR = Path("/Users/manishswami/developer/NVIDIA-Nemotron Model/data/processed/donald_v1_hybrid")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Donald-format files (Phase 1)
shutil.copy(f"{DONALD_DIR}/gravity_donald.jsonl",          OUT_DIR / "train_cot_gravity.jsonl")
shutil.copy(f"{DONALD_DIR}/unit_conversion_donald.jsonl",  OUT_DIR / "train_cot_unit_conversion.jsonl")
shutil.copy(f"{DONALD_DIR}/numeral_donald.jsonl",          OUT_DIR / "train_cot_numeral.jsonl")

# Original baseline files for the OTHER 6 categories (these gave 0.86)
for f in ["train_cot_bit_manipulation.jsonl",
          "train_cot_cipher.jsonl",
          "train_cot_cryptarithm_deduce.jsonl",
          "train_cot_cryptarithm_guess.jsonl",
          "train_cot_equation_numeric_deduce.jsonl",
          "train_cot_equation_numeric_guess.jsonl"]:
    shutil.copy(f"{BASELINE_DIR}/{f}", OUT_DIR / f)

# Tally
total = 0
for f in sorted(OUT_DIR.glob("*.jsonl")):
    n = sum(1 for _ in open(f))
    total += n
    print(f"  {f.name:<40} {n:>5}")
print(f"  TOTAL: {total}")