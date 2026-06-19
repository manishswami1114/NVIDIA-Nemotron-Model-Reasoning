"""Build refusal_with_gold dataset — NO fake CoT anywhere in cryptarithm.

Logic per cryptarithm record (deduce + guess):
  - If the record already has an R-medium CoT in diag_fail_crypt → KEEP it
    (these are the 155 puzzles our solver verified end-to-end)
  - Otherwise → REPLACE with refusal CoT + real gold answer from train.csv

Other 7 categories: copied verbatim.

This dataset has zero fake `'+' = 0` reasoning. Every cryptarithm CoT is either:
  - Real verified math (R-medium, 134 deduce + 21 guess = 155 records)
  - Honest refusal + correct boxed answer (525 deduce + 143 guess = 668 records)
                                          + 8 deduce + 3 guess = 11 right-by-baseline
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from gen_refusal_cot import build_refusal_cot  # noqa: E402

csv.field_size_limit(sys.maxsize)

ROOT       = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV  = ROOT / "data" / "raw" / "train.csv"
SRC_DIR    = ROOT / "all_categorical_splits_diag_fail_crypt"
OUT_DIR    = ROOT / "all_categorical_splits_refusal_with_gold"

CRYPT_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}

R_MEDIUM_MARKER = "each non-operator symbol stands for a distinct digit"
REFUSAL_MARKER  = "The constraints did not resolve"


def main():
    if not SRC_DIR.is_dir():
        raise FileNotFoundError(SRC_DIR)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    # id -> gold answer
    id2gold = {}
    with TRAIN_CSV.open() as f:
        for r in csv.DictReader(f):
            id2gold[r["id"]] = r["answer"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'File':<42} {'In':>5} {'R-med':>6} {'Refusal':>8} {'Errors':>6}")
    print("-"*78)
    grand_in = grand_rmed = grand_refusal = grand_err = 0

    for src in sorted(SRC_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name

        if src.name not in CRYPT_FILES:
            shutil.copy2(src, dst)
            with dst.open() as f:
                n = sum(1 for _ in f if _.strip())
            grand_in += n
            print(f"  {src.name:<42} {n:>5} {'':>6} {'':>8} {'':>6}  COPY")
            continue

        n_in = n_rmed = n_refusal = n_err = 0
        with src.open() as f_in, dst.open("w") as f_out:
            for line in f_in:
                if not line.strip(): continue
                n_in += 1
                rec = json.loads(line)
                rid = rec.get("id")
                cot = rec["messages"][1]["content"]

                if R_MEDIUM_MARKER in cot and REFUSAL_MARKER not in cot:
                    # Already has verified R-medium CoT — keep as-is
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_rmed += 1
                else:
                    # Baseline poison record — replace with refusal + gold
                    if rid not in id2gold:
                        n_err += 1
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    gold = id2gold[rid]
                    new_cot = build_refusal_cot(gold)
                    new_rec = {
                        "id": rid,
                        "messages": [
                            rec["messages"][0],
                            {"role": "assistant", "content": new_cot},
                        ],
                    }
                    f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    n_refusal += 1

        grand_in += n_in; grand_rmed += n_rmed; grand_refusal += n_refusal; grand_err += n_err
        print(f"  {src.name:<42} {n_in:>5} {n_rmed:>6} {n_refusal:>8} {n_err:>6}  REWRITE")

    print("-"*78)
    print(f"  {'TOTAL':<42} {grand_in:>5} {grand_rmed:>6} {grand_refusal:>8} {grand_err:>6}")

    # Verify: NO fake `' = 0` tokens, both openers present
    print("\n=== Template consistency check ===")
    for fname in sorted(CRYPT_FILES):
        with (OUT_DIR / fname).open() as f:
            n_fake = 0
            n_total = 0
            lens = []
            opener_count = {}
            for line in f:
                r = json.loads(line)
                cot = r["messages"][1]["content"]
                n_total += 1
                # The fake mapping is uniquely identifiable by lines like "  'X' = 0"
                if "' = 0" in cot and "Symbol Mapping" in cot:
                    n_fake += 1
                lens.append(len(cot))
                opener = cot.split("\n", 2)[1]
                opener_count[opener] = opener_count.get(opener, 0) + 1
        print(f"  {fname}: {n_total} records, "
              f"fake-poison-CoT remaining: {n_fake}, "
              f"length range {min(lens)}-{max(lens)}")
        print(f"    Openers:")
        for o, c in opener_count.items():
            print(f"      [{c:>3}]  {o[:80]}{'...' if len(o)>80 else ''}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Next:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/refusal_with_gold.zip *.jsonl")


if __name__ == "__main__":
    main()
