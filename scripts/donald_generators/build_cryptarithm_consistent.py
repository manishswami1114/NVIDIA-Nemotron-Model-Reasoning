"""
build_cryptarithm_consistent.py — Option 2 (all-or-nothing real CoTs).

For cryptarithm_deduce and cryptarithm_guess:
  - If puzzle ID is in our v1 solver output → REPLACE CoT with verified math
  - If puzzle ID is NOT in our solver output → DELETE the record

This produces a smaller but CONSISTENT cryptarithm training set:
every CoT teaches the same real cipher-cracking strategy. Model can't
learn "fake template + memorized answer" because there are no fake
templates anymore.

For the other 7 categories: copy baseline unchanged.

Usage:
    python build_cryptarithm_consistent.py
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from gen_sequence_cot_tong import build_cot  # type: ignore

ROOT          = THIS_DIR.parent.parent
BASELINE_DIR  = ROOT / "dont_touch_it" / "all_categorical_splits"
SOLVED_JSONL  = THIS_DIR / "sequence_solved.jsonl"
OUT_DIR       = ROOT / "all_categorical_splits_cryptarithm_consistent"

CRYPT_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}


def load_solved(path: Path) -> dict[str, dict]:
    solved = {}
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            solved[r["id"]] = r
    return solved


def process_cryptarithm_file(src: Path, dst: Path, solved: dict) -> tuple[int, int, int]:
    """Returns (input_count, kept_count, deleted_count)."""
    n_in = n_kept = n_del = 0
    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip(): continue
            n_in += 1
            rec = json.loads(line)
            rid = rec.get("id")
            if rid in solved:
                # REPLACE with verified CoT
                solver_rec = dict(solved[rid])
                solver_rec["prompt"] = rec["messages"][0]["content"]
                new_assistant = build_cot(solver_rec)
                new_rec = {
                    "id": rid,
                    "messages": [
                        rec["messages"][0],
                        {"role": "assistant", "content": new_assistant},
                    ],
                }
                f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                n_kept += 1
            else:
                # DELETE: skip this record entirely
                n_del += 1
    return n_in, n_kept, n_del


def main():
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(f"Baseline dir not found: {BASELINE_DIR}")
    if not SOLVED_JSONL.exists():
        raise FileNotFoundError(f"Solver output not found: {SOLVED_JSONL}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    solved = load_solved(SOLVED_JSONL)
    print(f"Loaded {len(solved)} v1-solved puzzles\n")

    print(f"{'File':<45} {'In':>5} {'Keep':>5} {'Del':>5}  Action")
    print(f"{'-'*45} {'-'*5} {'-'*5} {'-'*5}  {'-'*40}")
    grand_in = grand_kept = grand_del = 0

    for src in sorted(BASELINE_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name
        if src.name in CRYPT_FILES:
            n_in, n_kept, n_del = process_cryptarithm_file(src, dst, solved)
            grand_in += n_in; grand_kept += n_kept; grand_del += n_del
            print(f"  {src.name:<45} {n_in:>5} {n_kept:>5} {n_del:>5}  REPLACE+DELETE")
        else:
            shutil.copy2(src, dst)
            n_in = sum(1 for _ in dst.open())
            grand_in += n_in; grand_kept += n_in
            print(f"  {src.name:<45} {n_in:>5} {n_in:>5} {0:>5}  COPY (untouched)")

    print(f"  {'-'*45} {'-'*5} {'-'*5} {'-'*5}")
    print(f"  {'TOTAL':<45} {grand_in:>5} {grand_kept:>5} {grand_del:>5}")
    print(f"\nOutput dir: {OUT_DIR}")
    print(f"\nKey numbers:")
    print(f"  Baseline cryptarithm records:  823")
    print(f"  Kept (verified CoTs):           {823 - grand_del}")
    print(f"  Deleted (no solver coverage):   {grand_del}")
    print(f"  Reduction:                      {100*grand_del/823:.1f}%")
    print(f"\nNext steps:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/cryptarithm_consistent.zip *.jsonl")
    print(f"  Upload to Kaggle, retrain with PROVEN hyperparams:")
    print(f"    alpha=64, LR=5e-5, 1 epoch, batch=2, grad_accum=2")


if __name__ == "__main__":
    main()
