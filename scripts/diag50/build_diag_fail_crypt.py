"""Build diag_fail_crypt dataset — R-medium CoTs for all SOLVABLE failed cryptarithm puzzles.

Inputs:
  - data/raw/train.csv                              — id, prompt, answer
  - evaluation_results_086.csv                      — which puzzles the 0.86 LB model got wrong
  - dont_touch_it/all_categorical_splits/*.jsonl    — baseline training data

Logic:
  For both cryptarithm_deduce.jsonl and cryptarithm_guess.jsonl:
    For each record:
      If id is in failed-set AND solver can verify the gold answer:
        REPLACE with R-medium real-math CoT (~1500 chars)
      Otherwise:
        KEEP baseline record unchanged

  All 7 other category files: copied verbatim.

Output:
  all_categorical_splits_diag_fail_crypt/   — drop-in replacement
  ~/Downloads/diag_fail_crypt.zip           — submission-ready archive
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from fast_solver import solve_fast  # noqa: E402
from gen_r_medium_cot import build_r_medium_cot  # noqa: E402

csv.field_size_limit(sys.maxsize)

ROOT         = THIS_DIR.parent.parent
TRAIN_CSV    = ROOT / "data" / "raw" / "train.csv"
EVAL_CSV     = Path("/Users/manishswami/Downloads/evaluation_results_086.csv")
BASELINE_DIR = ROOT / "dont_touch_it" / "all_categorical_splits"
OUT_DIR      = ROOT / "all_categorical_splits_diag_fail_crypt"

CRYPT_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}


def load_train_csv():
    out = {}
    with TRAIN_CSV.open() as f:
        for r in csv.DictReader(f):
            out[r["id"]] = {"prompt": r["prompt"], "answer": r["answer"]}
    return out


def load_failed_ids():
    failed = set()
    with EVAL_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["is_correct"].lower() not in ("true","1","yes"):
                failed.add(r["id"])
    return failed


def main():
    for p in (TRAIN_CSV, EVAL_CSV, BASELINE_DIR):
        if not p.exists():
            raise FileNotFoundError(p)

    train = load_train_csv()
    failed = load_failed_ids()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Failed-on-086 ids: {len(failed)}")

    # Stats per cryptarithm file
    grand_in = grand_repl = 0
    grand_lens = []
    rejection_reasons = {"not_failed": 0, "no_train": 0, "no_solution": 0,
                        "answer_mismatch": 0, "cot_build_error": 0}

    print(f"\n{'File':<42} {'In':>5} {'Failed':>7} {'Repl':>5} {'Keep':>5}")
    print("-"*78)

    for src in sorted(BASELINE_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name

        if src.name not in CRYPT_FILES:
            shutil.copy2(src, dst)
            with dst.open() as f:
                n = sum(1 for _ in f if _.strip())
            grand_in += n
            print(f"  {src.name:<42} {n:>5} {0:>7} {0:>5} {n:>5}  COPY")
            continue

        t_start = time.time()
        n_in = n_failed_in_file = n_repl = 0
        with src.open() as f_in, dst.open("w") as f_out:
            for line in f_in:
                if not line.strip(): continue
                n_in += 1
                rec = json.loads(line)
                rid = rec.get("id")

                # Default: keep baseline
                should_replace = False
                new_cot = None

                if rid not in failed:
                    rejection_reasons["not_failed"] += 1
                elif rid not in train:
                    rejection_reasons["no_train"] += 1
                else:
                    n_failed_in_file += 1
                    prompt = train[rid]["prompt"]
                    gt     = train[rid]["answer"]
                    try:
                        det = solve_fast(prompt, gt)
                    except Exception:
                        det = None
                        rejection_reasons["no_solution"] += 1
                    if det is None:
                        rejection_reasons["no_solution"] += 1
                    elif det["answer"] != gt:
                        rejection_reasons["answer_mismatch"] += 1
                    else:
                        try:
                            new_cot = build_r_medium_cot(prompt, det)
                            should_replace = True
                        except Exception as e:
                            rejection_reasons["cot_build_error"] += 1

                if should_replace:
                    new_rec = {
                        "id": rid,
                        "messages": [
                            rec["messages"][0],
                            {"role": "assistant", "content": new_cot},
                        ],
                    }
                    f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    n_repl += 1
                    grand_lens.append(len(new_cot))
                else:
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if n_in % 50 == 0:
                    print(f"    ...{src.name}: examined {n_in}, replaced {n_repl}, "
                          f"elapsed {time.time()-t_start:.0f}s", flush=True)

        grand_in += n_in; grand_repl += n_repl
        keep = n_in - n_repl
        print(f"  {src.name:<42} {n_in:>5} {n_failed_in_file:>7} {n_repl:>5} {keep:>5}  REPLACE")

    print("-"*78)
    print(f"  {'TOTAL':<42} {grand_in:>5} {'':>7} {grand_repl:>5}")
    print(f"\nRejection reasons across cryptarithm files: {rejection_reasons}")
    if grand_lens:
        grand_lens.sort()
        print(f"\nReplaced-CoT length stats: "
              f"min={min(grand_lens)}, median={grand_lens[len(grand_lens)//2]}, "
              f"mean={sum(grand_lens)//len(grand_lens)}, max={max(grand_lens)}")
        print(f"  records in 1200-1800 char range: "
              f"{sum(1 for L in grand_lens if 1200 <= L <= 1800)}/{len(grand_lens)}")

    # Verify template consistency
    print("\n=== Per-cryptarithm-file template consistency ===")
    for fname in sorted(CRYPT_FILES):
        with (OUT_DIR / fname).open() as f:
            starts = set()
            lens = []
            for line in f:
                r = json.loads(line)
                cot = r["messages"][1]["content"]
                starts.add(cot.split("\n", 2)[1])
                lens.append(len(cot))
        print(f"  {fname}: {len(lens)} records, {len(starts)} distinct openers, "
              f"length range {min(lens)}–{max(lens)}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Next:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/diag_fail_crypt.zip *.jsonl")


if __name__ == "__main__":
    main()
