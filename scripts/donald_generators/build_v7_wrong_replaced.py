"""
build_v7_wrong_replaced.py — replace ONLY wrong puzzles with long-form CoTs.

Reads:
  - dont_touch_it/all_categorical_splits/ — original baseline (preserved)
  - evaluation_results_086_with_v7_adapter.csv — which puzzles the 0.86 model
    got wrong
  - scripts/donald_generators/sequence_solved.jsonl — our v1 solver coverage

For each baseline cryptarithm record:
  - If puzzle ID is in BOTH wrong-set AND solver-coverage → REPLACE with our
    long-form CoT (5000-6000 chars, baseline template style)
  - Otherwise → KEEP baseline record unchanged

Other 7 categories: copy unchanged.

Output: all_categorical_splits_v7_wrong_replaced/

Usage:
    python build_v7_wrong_replaced.py
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from gen_long_baseline_style_cot import build_long_cot  # type: ignore

ROOT          = THIS_DIR.parent.parent
BASELINE_DIR  = ROOT / "dont_touch_it" / "all_categorical_splits"
EVAL_CSV      = ROOT / "evaluation_results_086_with_v7_adapter.csv"
SOLVED_JSONL  = THIS_DIR / "sequence_solved.jsonl"
OUT_DIR       = ROOT / "all_categorical_splits_v7_wrong_replaced"

CRYPT_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}


def main():
    # --- Load eval wrong-IDs ---
    if not EVAL_CSV.exists():
        raise FileNotFoundError(f"Missing eval CSV: {EVAL_CSV}")
    wrong = set()
    with open(EVAL_CSV) as f:
        for r in csv.DictReader(f):
            if r["is_correct"].lower() in ("false","0","no"):
                wrong.add(r["id"])
    print(f"Wrong-answer IDs in v7 eval:    {len(wrong)}")

    # --- Load solver coverage ---
    solved = {}
    with open(SOLVED_JSONL) as f:
        for line in f:
            r = json.loads(line)
            solved[r["id"]] = r
    print(f"Solver-covered IDs:             {len(solved)}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'File':<45} {'In':>5} {'Repl':>5} {'Keep':>5}")
    print(f"{'-'*45} {'-'*5} {'-'*5} {'-'*5}")
    total_in = total_repl = 0

    for src in sorted(BASELINE_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name
        if src.name not in CRYPT_FILES:
            shutil.copy2(src, dst)
            n = sum(1 for _ in dst.open())
            total_in += n
            print(f"  {src.name:<45} {n:>5} {0:>5} {n:>5}  COPY")
            continue

        n_in = n_repl = 0
        with src.open() as f_in, dst.open("w") as f_out:
            for line in f_in:
                if not line.strip(): continue
                n_in += 1
                rec = json.loads(line)
                rid = rec.get("id")
                # Only replace if puzzle was WRONG and we have solver coverage
                if rid in wrong and rid in solved:
                    solver_rec = dict(solved[rid])
                    solver_rec["prompt"] = rec["messages"][0]["content"]
                    try:
                        new_assistant = build_long_cot(solver_rec)
                    except Exception as e:
                        # Solver verified the answer but CoT generator hit a corner
                        # case — keep baseline rather than write garbage.
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue
                    new_rec = {
                        "id": rid,
                        "messages": [
                            rec["messages"][0],
                            {"role": "assistant", "content": new_assistant},
                        ],
                    }
                    f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    n_repl += 1
                else:
                    # Keep baseline record untouched
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total_in  += n_in
        total_repl += n_repl
        print(f"  {src.name:<45} {n_in:>5} {n_repl:>5} {n_in-n_repl:>5}  REPLACE WRONG-ONLY")

    print(f"  {'-'*45} {'-'*5} {'-'*5} {'-'*5}")
    print(f"  {'TOTAL':<45} {total_in:>5} {total_repl:>5}")

    # Verify length distribution and template uniformity in the new cryptarithm files
    import statistics
    for fname in CRYPT_FILES:
        lens = []
        starts_set = set()
        with (OUT_DIR / fname).open() as f:
            for line in f:
                r = json.loads(line)
                cot = r["messages"][1]["content"]
                lens.append(len(cot))
                starts_set.add(cot[:55])
        print(f"\n  {fname}:")
        print(f"    Length median = {int(statistics.median(lens))}, mean = {int(statistics.mean(lens))}, "
              f"min = {min(lens)}, max = {max(lens)}")
        print(f"    Distinct starts = {len(starts_set)}  (1 = perfectly consistent, >1 = template mixing)")

    print(f"\nOutput → {OUT_DIR}\n")
    print(f"Next: cd {OUT_DIR} && zip -r ~/Downloads/v7_wrong_replaced.zip *.jsonl")


if __name__ == "__main__":
    main()
