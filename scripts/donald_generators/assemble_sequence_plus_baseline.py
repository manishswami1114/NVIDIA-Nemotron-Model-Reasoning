"""
assemble_sequence_plus_baseline.py — combine sequence CoTs with 0.86 baseline.

Builds a per-category training dataset that is your proven 0.86 baseline data
PLUS our independently-solved sequence (equation_symbolic) puzzles in
Tong-style format. Outputs a directory ready to upload to Kaggle.

ZERO format mixing:
  - 9 categories from baseline (all in Tong-style — the format that gave 0.86)
  - 1 new category 'equation_symbolic' (also in Tong-style — matches baseline)

Usage:
    python assemble_sequence_plus_baseline.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=str,
                    default="../../data/processed/all_categories_split",
                    help="Your proven 0.86 baseline data directory")
    ap.add_argument("--sequence-jsonl", type=str,
                    default="sequence_donald_tong.jsonl",
                    help="Sequence CoTs from gen_sequence_cot_tong.py")
    ap.add_argument("--out-dir", type=str,
                    default="../../data/processed/baseline_plus_sequence")
    args = ap.parse_args()

    baseline = Path(args.baseline_dir)
    seq_file = Path(args.sequence_jsonl)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baseline source : {baseline}")
    print(f"Sequence source : {seq_file}")
    print(f"Output dir      : {out_dir}\n")

    if not baseline.is_dir():
        raise FileNotFoundError(f"Baseline dir not found: {baseline}")
    if not seq_file.exists():
        raise FileNotFoundError(f"Sequence jsonl not found: {seq_file}")

    # Copy all baseline files verbatim
    print("Copying baseline files:")
    for f in sorted(baseline.glob("*.jsonl")):
        dst = out_dir / f.name
        shutil.copy(f, dst)
        n = sum(1 for _ in open(dst))
        print(f"  {f.name:<46} {n:>5} records")

    # Convert sequence_donald_tong.jsonl → category file
    # Strip the internal-only _id and _answer fields for clean training format
    print("\nAdding sequence (equation_symbolic) category:")
    seq_out = out_dir / "train_cot_equation_symbolic.jsonl"
    n_seq = 0
    with open(seq_file) as f_in, open(seq_out, "w") as f_out:
        for line in f_in:
            r = json.loads(line)
            # Strip internal fields
            clean = {
                "category": r["category"],
                "messages": r["messages"],
            }
            f_out.write(json.dumps(clean, ensure_ascii=False) + "\n")
            n_seq += 1
    print(f"  {seq_out.name:<46} {n_seq:>5} records")

    # Final tally
    print(f"\n{'='*60}")
    print(f"  Final dataset: {out_dir.name}/")
    print(f"{'='*60}")
    total = 0
    for f in sorted(out_dir.glob("*.jsonl")):
        n = sum(1 for _ in open(f))
        total += n
        print(f"  {f.name:<46} {n:>5}")
    print(f"  {'-'*46} {'-'*5}")
    print(f"  {'TOTAL':<46} {total:>5}")

    print(f"\nNext steps:")
    print(f"  1. Verify {out_dir} on disk")
    print(f"  2. zip -r baseline_plus_sequence.zip {out_dir}/*.jsonl")
    print(f"  3. Upload zip to Kaggle as new dataset")
    print(f"  4. In v76 training notebook, point DATA_DIR_CANDIDATES at new dataset")
    print(f"  5. Train with proven hyperparams: alpha=64 LR=5e-5 epochs=1 grad_accum=2")
    print(f"  6. Submit. Expected: 0.86 → 0.90-0.92")


if __name__ == "__main__":
    main()
