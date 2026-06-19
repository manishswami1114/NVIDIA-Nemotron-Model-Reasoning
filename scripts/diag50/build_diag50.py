"""Build the 50-record diagnostic dataset.

Reads:
  - data/raw/train.csv                                     — to get ID, prompt, answer
  - dont_touch_it/all_categorical_splits/*.jsonl           — baseline training data

For cryptarithm_deduce.jsonl:
  - Pick 50 puzzles that:
      * have only ONE distinct example operator (simplest case)
      * standard reading mode (no alice / little_endian)
      * unsigned answers (no operator prefix)
      * solver returns a verified answer matching train.csv ground truth
      * generated CoT length is within [440, 500] (baseline range)
  - Replace those 50 records' assistant content with the new CoT
  - Keep the other 609 records byte-identical

For the other 8 category files:
  - Copy verbatim (no changes)

Output:
  all_categorical_splits_diag50/  — drop-in replacement directory
  ~/Downloads/diag50.zip          — submission-ready archive

Run:
    python scripts/diag50/build_diag50.py
"""
from __future__ import annotations
import csv
import json
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from fast_solver import solve_fast  # noqa: E402
from gen_baseline_match_cot import build_baseline_match_cot  # noqa: E402

csv.field_size_limit(sys.maxsize)

ROOT          = THIS_DIR.parent.parent
TRAIN_CSV     = ROOT / "data" / "raw" / "train.csv"
BASELINE_DIR  = ROOT / "dont_touch_it" / "all_categorical_splits"
OUT_DIR       = ROOT / "all_categorical_splits_diag50"

CRYPT_FILE    = "train_cot_cryptarithm_deduce.jsonl"
N_TARGET      = 50
LEN_MIN, LEN_MAX = 440, 500  # baseline CoT length range


def load_train_csv():
    """id -> {'prompt': ..., 'answer': ...}"""
    out = {}
    with TRAIN_CSV.open() as f:
        for r in csv.DictReader(f):
            out[r["id"]] = {"prompt": r["prompt"], "answer": r["answer"]}
    return out


def main():
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(BASELINE_DIR)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    train = load_train_csv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: select 50 cryptarithm_deduce records ----------------------
    src = BASELINE_DIR / CRYPT_FILE
    records = []
    with src.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Baseline cryptarithm_deduce records: {len(records)}")

    selected = {}   # id -> new_cot
    examined = 0
    rejected = {"no_train": 0, "no_solution": 0, "answer_mismatch": 0,
                "length_oob": 0}

    import time
    t_start = time.time()
    for rec in records:
        if len(selected) >= N_TARGET:
            break
        examined += 1
        if examined % 25 == 0:
            print(f"  ...examined {examined}/{len(records)}  selected={len(selected)}  "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)
        rid = rec.get("id")
        if rid not in train:
            rejected["no_train"] += 1; continue

        prompt = train[rid]["prompt"]
        gt     = train[rid]["answer"]

        try:
            det = solve_fast(prompt, gt)
        except Exception:
            rejected["no_solution"] += 1; continue
        if det is None:
            rejected["no_solution"] += 1; continue
        if det["answer"] != gt:
            rejected["answer_mismatch"] += 1; continue

        cot = build_baseline_match_cot(det)
        if not (LEN_MIN <= len(cot) <= LEN_MAX):
            rejected["length_oob"] += 1; continue

        selected[rid] = cot

    print(f"\nScanned {examined} records.")
    print(f"Selected {len(selected)}/{N_TARGET}.")
    print(f"Rejected: {rejected}")
    if len(selected) < N_TARGET:
        print(f"\nWARNING: only {len(selected)} puzzles met criteria. Adjusting target.")

    # --- Pass 2: write the cryptarithm_deduce output, replacing only selected -
    dst = OUT_DIR / CRYPT_FILE
    n_in = n_repl = 0
    out_lens = []
    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip(): continue
            n_in += 1
            rec = json.loads(line)
            rid = rec.get("id")
            if rid in selected:
                new_cot = selected[rid]
                new_rec = {
                    "id": rid,
                    "messages": [
                        rec["messages"][0],
                        {"role": "assistant", "content": new_cot},
                    ],
                }
                f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                n_repl += 1
                out_lens.append(len(new_cot))
            else:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n  {CRYPT_FILE}: {n_in} in → {n_repl} replaced, {n_in - n_repl} kept")
    if out_lens:
        print(f"  Replacement CoT lengths: min={min(out_lens)}, "
              f"median={sorted(out_lens)[len(out_lens)//2]}, max={max(out_lens)}")

    # --- Pass 3: copy other 8 categories verbatim -------------------------
    print("\nCopying other 8 categories unchanged:")
    for p in sorted(BASELINE_DIR.glob("*.jsonl")):
        if p.name == CRYPT_FILE: continue
        out = OUT_DIR / p.name
        shutil.copy2(p, out)
        with out.open() as f:
            n = sum(1 for _ in f if _.strip())
        print(f"  {p.name:<42} {n:>5}  COPY")

    # --- Sanity check: verify template start consistency in cryptarithm_deduce ----
    print("\nVerifying template consistency in the rebuilt cryptarithm_deduce file:")
    with (OUT_DIR / CRYPT_FILE).open() as f:
        starts = set()
        lengths = []
        for line in f:
            r = json.loads(line)
            cot = r["messages"][1]["content"]
            starts.add(cot.split("\n", 2)[1])  # second line — the "This is..." line
            lengths.append(len(cot))
    print(f"  Distinct 2nd-line strings: {len(starts)} (should be 1 — both ours and baseline use same opener)")
    for s in starts:
        print(f"    {s!r}")
    print(f"  CoT length range across all {len(lengths)} records: "
          f"min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"\nNext:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/diag50.zip *.jsonl")


if __name__ == "__main__":
    main()
