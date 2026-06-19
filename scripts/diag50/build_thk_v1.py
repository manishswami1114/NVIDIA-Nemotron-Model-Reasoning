"""Build thk_v1 dataset — THK-style algorithmic CoT for cryptarithm.

For each cryptarithm record (deduce + guess):
  - If solver can verify → THK-medium CoT (~2000-4000 chars, derivable steps)
  - If solver can't → refusal CoT + real gold answer (preserves answer signal,
                                                      no fake reasoning)

Other 7 categories: copied verbatim.

Zero fake reasoning anywhere. Cryptarithm CoTs come in two flavors:
  1. THK-medium with full derivation (Step 1-6 algorithmic trace)
  2. Honest refusal with correct boxed answer

Both share the SAME opener so cryptarithm files stay template-consistent.
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
from gen_thk_medium_cot import build_thk_medium_cot  # noqa: E402
from gen_refusal_cot import build_refusal_cot  # noqa: E402

csv.field_size_limit(sys.maxsize)

ROOT         = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV    = ROOT / "data" / "raw" / "train.csv"
BASELINE_DIR = ROOT / "dont_touch_it" / "all_categorical_splits"
OUT_DIR      = ROOT / "all_categorical_splits_thk_v1"

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


def main():
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(BASELINE_DIR)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    train = load_train_csv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'File':<42} {'In':>5} {'THK':>5} {'Refusal':>8} {'Drop':>5}")
    print("-"*78)

    grand_thk = grand_refusal = grand_drop = 0
    thk_lens = []
    refusal_lens = []

    for src in sorted(BASELINE_DIR.glob("*.jsonl")):
        dst = OUT_DIR / src.name

        if src.name not in CRYPT_FILES:
            shutil.copy2(src, dst)
            with dst.open() as f:
                n = sum(1 for _ in f if _.strip())
            print(f"  {src.name:<42} {n:>5} {'':>5} {'':>8} {'':>5}  COPY")
            continue

        n_in = n_thk = n_refusal = n_drop = 0
        t_start = time.time()
        with src.open() as f_in, dst.open("w") as f_out:
            for line in f_in:
                if not line.strip(): continue
                n_in += 1
                rec = json.loads(line)
                rid = rec.get("id")

                if rid not in train:
                    # No gold available — drop entirely
                    n_drop += 1
                    continue

                prompt = train[rid]["prompt"]
                gold   = train[rid]["answer"]

                try:
                    det = solve_fast(prompt, gold)
                except Exception:
                    det = None

                new_cot = None
                if det is not None and det["answer"] == gold:
                    try:
                        new_cot = build_thk_medium_cot(prompt, det)
                        if new_cot:
                            thk_lens.append(len(new_cot))
                            n_thk += 1
                    except Exception:
                        new_cot = None

                if new_cot is None:
                    # Fall back to refusal + real gold
                    new_cot = build_refusal_cot(gold)
                    refusal_lens.append(len(new_cot))
                    n_refusal += 1

                new_rec = {
                    "id": rid,
                    "messages": [
                        rec["messages"][0],
                        {"role": "assistant", "content": new_cot},
                    ],
                }
                f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

                if n_in % 50 == 0:
                    print(f"    ...{src.name}: examined {n_in}, "
                          f"thk={n_thk}, refusal={n_refusal}, "
                          f"elapsed {time.time()-t_start:.0f}s", flush=True)

        grand_thk += n_thk; grand_refusal += n_refusal; grand_drop += n_drop
        print(f"  {src.name:<42} {n_in:>5} {n_thk:>5} {n_refusal:>8} {n_drop:>5}  REWRITE")

    print("-"*78)
    print(f"  {'TOTAL':<42} {'':>5} {grand_thk:>5} {grand_refusal:>8} {grand_drop:>5}")

    print(f"\n=== Length stats ===")
    if thk_lens:
        thk_lens.sort()
        print(f"  THK-medium: n={len(thk_lens)}, min={min(thk_lens)}, "
              f"median={thk_lens[len(thk_lens)//2]}, max={max(thk_lens)}")
    if refusal_lens:
        refusal_lens.sort()
        print(f"  Refusal:    n={len(refusal_lens)}, min={min(refusal_lens)}, "
              f"median={refusal_lens[len(refusal_lens)//2]}, max={max(refusal_lens)}")

    # Verify zero fake-poison records
    print(f"\n=== Verifying no fake `Symbol Mapping = 0` poison remains ===")
    for fname in sorted(CRYPT_FILES):
        with (OUT_DIR / fname).open() as f:
            n_fake = 0; n_total = 0
            opener_counts = {}
            for line in f:
                r = json.loads(line)
                cot = r["messages"][1]["content"]
                n_total += 1
                if "' = 0\n" in cot and "Symbol Mapping" in cot:
                    n_fake += 1
                op = cot.split("\n", 2)[1] if "\n" in cot else cot[:80]
                opener_counts[op] = opener_counts.get(op, 0) + 1
        print(f"  {fname}: {n_total} records, fake-poison remaining: {n_fake}")
        for o, c in opener_counts.items():
            print(f"    [{c:>3}]  {o[:80]}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"\nNext:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/thk_v1.zip *.jsonl")


if __name__ == "__main__":
    main()
