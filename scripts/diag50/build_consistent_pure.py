"""Build consistent_pure dataset — delete poison, keep only verified.

Starts from all_categorical_splits_diag_fail_crypt/ (which has the 155 R-medium
verified records + the 668 baseline-poison records still present in the
cryptarithm files). This script:

  - Filters cryptarithm_deduce.jsonl to keep ONLY records using the R-medium
    opener ("This is a cryptarithm: each non-operator symbol...")
  - Filters cryptarithm_guess.jsonl the same way
  - Copies all 7 other category files verbatim

Output:
  all_categorical_splits_consistent_pure/
  ~/Downloads/consistent_pure.zip
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent
SRC_DIR    = ROOT / "all_categorical_splits_diag_fail_crypt"
OUT_DIR    = ROOT / "all_categorical_splits_consistent_pure"

CRYPT_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}

R_MEDIUM_MARKER = "each non-operator symbol stands for a distinct digit"


def is_rmedium(cot: str) -> bool:
    return R_MEDIUM_MARKER in cot


def main():
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(SRC_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'File':<42} {'In':>5} {'Kept':>5} {'Deleted':>8}")
    print("-"*68)
    grand_in = grand_keep = grand_del = 0

    for src in sorted(SRC_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name

        if src.name not in CRYPT_FILES:
            shutil.copy2(src, dst)
            with dst.open() as f:
                n = sum(1 for _ in f if _.strip())
            grand_in += n; grand_keep += n
            print(f"  {src.name:<42} {n:>5} {n:>5} {0:>8}  COPY")
            continue

        n_in = n_keep = n_del = 0
        with src.open() as f_in, dst.open("w") as f_out:
            for line in f_in:
                if not line.strip(): continue
                n_in += 1
                rec = json.loads(line)
                cot = rec["messages"][1]["content"]
                if is_rmedium(cot):
                    f_out.write(line)
                    n_keep += 1
                else:
                    n_del += 1
        grand_in += n_in; grand_keep += n_keep; grand_del += n_del
        print(f"  {src.name:<42} {n_in:>5} {n_keep:>5} {n_del:>8}  FILTER")

    print("-"*68)
    print(f"  {'TOTAL':<42} {grand_in:>5} {grand_keep:>5} {grand_del:>8}")

    # Verify
    print(f"\n=== Verification ===")
    for fname in sorted(CRYPT_FILES):
        with (OUT_DIR / fname).open() as f:
            lens = []
            starts = set()
            for line in f:
                r = json.loads(line)
                cot = r["messages"][1]["content"]
                lens.append(len(cot))
                starts.add(cot.split("\n", 2)[1])
        if not lens:
            print(f"  {fname}: 0 records (empty)")
            continue
        print(f"  {fname}: {len(lens)} records, {len(starts)} distinct openers, "
              f"length range {min(lens)}-{max(lens)}")
        for s in starts:
            print(f"    {s!r}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Next:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/consistent_pure.zip *.jsonl")


if __name__ == "__main__":
    main()
