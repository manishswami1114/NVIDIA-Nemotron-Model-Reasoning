#!/usr/bin/env python3
"""
assemble_dataset.py — Assemble the final training dataset.

Strategy:
  - For 7 non-broken categories: COPY from baseline (untouched)
  - For equation_numeric_guess: REPLACE CoT with our search-trajectory version
    (110/111 solved, 1 puzzle dropped)
  - For cryptarithm_deduce + cryptarithm_guess: REPLACE CoT with our version
    where solved; DROP unsolved puzzles

Output: a single directory with all 9 JSONL files, ready for training.

Usage:
    python assemble_dataset.py [--stats-only]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent
BASELINE = ROOT / "dont_touch_it" / "all_categorical_splits"
OUT_DIR = ROOT / "all_categorical_splits_v8_fresh"

# Our generated CoTs
COT_NUMERIC_GUESS = THIS / "cot_equation_numeric_guess.jsonl"
COT_CRYPTARITHM = THIS / "cot_cryptarithm.jsonl"

# Categories we DON'T touch
PASSTHROUGH = {
    "train_cot_bit_manipulation.jsonl",
    "train_cot_cipher.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_numeral.jsonl",
    "train_cot_unit_conversion.jsonl",
}

# Categories we REPLACE
REPLACE_MAP = {
    "train_cot_equation_numeric_guess.jsonl": COT_NUMERIC_GUESS,
    "train_cot_cryptarithm_deduce.jsonl": COT_CRYPTARITHM,
    "train_cot_cryptarithm_guess.jsonl": COT_CRYPTARITHM,
}


def load_replacement_ids(path: Path) -> dict[str, dict]:
    """Load replacement CoTs indexed by ID."""
    records = {}
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            records[rec["id"]] = rec
    return records


def main():
    stats_only = "--stats-only" in sys.argv

    if not BASELINE.is_dir():
        raise FileNotFoundError(f"Baseline not found: {BASELINE}")

    # Load replacement data
    numeric_guess_recs = load_replacement_ids(COT_NUMERIC_GUESS)
    cryptarithm_recs = load_replacement_ids(COT_CRYPTARITHM)

    print(f"Replacement CoTs available:")
    print(f"  equation_numeric_guess: {len(numeric_guess_recs)}")
    print(f"  cryptarithm (both): {len(cryptarithm_recs)}")
    print()

    if not stats_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'File':<50} {'Baseline':>5} {'Output':>5} {'Replaced':>5} {'Dropped':>5}")
    print(f"{'-'*50} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

    grand_baseline = 0
    grand_output = 0
    grand_replaced = 0
    grand_dropped = 0

    for src in sorted(BASELINE.glob("*.jsonl")):
        dst = OUT_DIR / src.name if not stats_only else None

        n_baseline = 0
        n_output = 0
        n_replaced = 0
        n_dropped = 0

        if src.name in PASSTHROUGH:
            # Copy unchanged
            with src.open() as f:
                n_baseline = sum(1 for _ in f)
            n_output = n_baseline
            if not stats_only:
                shutil.copy2(src, dst)

        elif src.name == "train_cot_equation_numeric_guess.jsonl":
            out_lines = []
            with src.open() as f:
                for line in f:
                    n_baseline += 1
                    rec = json.loads(line)
                    rid = rec["id"]
                    if rid in numeric_guess_recs:
                        out_lines.append(json.dumps(numeric_guess_recs[rid], ensure_ascii=False))
                        n_replaced += 1
                        n_output += 1
                    else:
                        n_dropped += 1
            if not stats_only:
                with dst.open("w") as f:
                    f.write("\n".join(out_lines) + "\n")

        elif src.name in ("train_cot_cryptarithm_deduce.jsonl", "train_cot_cryptarithm_guess.jsonl"):
            out_lines = []
            n_kept = 0
            with src.open() as f:
                for line in f:
                    n_baseline += 1
                    rec = json.loads(line)
                    rid = rec["id"]
                    if rid in cryptarithm_recs:
                        out_lines.append(json.dumps(cryptarithm_recs[rid], ensure_ascii=False))
                        n_replaced += 1
                        n_output += 1
                    else:
                        # Keep original (archive) record as fallback
                        out_lines.append(line.strip())
                        n_kept += 1
                        n_output += 1
            if not stats_only:
                with dst.open("w") as f:
                    if out_lines:
                        f.write("\n".join(out_lines) + "\n")
                    else:
                        f.write("")

        else:
            print(f"  WARNING: unknown file {src.name}")
            continue

        print(f"  {src.name:<50} {n_baseline:>5} {n_output:>5} {n_replaced:>5} {n_dropped:>5}")
        grand_baseline += n_baseline
        grand_output += n_output
        grand_replaced += n_replaced
        grand_dropped += n_dropped

    print(f"  {'-'*50} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    print(f"  {'TOTAL':<50} {grand_baseline:>5} {grand_output:>5} {grand_replaced:>5} {grand_dropped:>5}")

    print(f"\nSummary:")
    print(f"  Baseline records: {grand_baseline}")
    print(f"  Output records: {grand_output}")
    print(f"  Replaced with fresh CoT: {grand_replaced}")
    print(f"  Dropped (unsolvable): {grand_dropped}")
    print(f"  Passthrough (unchanged): {grand_output - grand_replaced}")
    pct_replaced = 100 * grand_replaced / grand_baseline if grand_baseline else 0
    pct_dropped = 100 * grand_dropped / grand_baseline if grand_baseline else 0
    print(f"  Replacement rate: {pct_replaced:.1f}%")
    print(f"  Drop rate: {pct_dropped:.1f}%")

    if not stats_only:
        print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
