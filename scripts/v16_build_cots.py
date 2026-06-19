#!/usr/bin/env python3
"""v16 dispatcher.

Plan: v16 = baseline (proven 0.86) ⊕ 3 regenerated files in baseline-equation_numeric_deduce
exhaustive-search style.

Steps:
  1. Extract /Users/manishswami/developer/nvidia_archieve/original_data_scored_highest_LBScore.zip
     into /tmp/_v16_baseline (already done if extracted).
  2. Copy 6 baseline files verbatim into all_categorical_splits_v16/.
     (Already done; we re-copy here defensively.)
  3. Generate the 3 regenerated files using v16_writers.
"""
import json, os, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all
import v16_writers as W

BASE = Path(__file__).resolve().parent.parent
V16  = BASE / "all_categorical_splits_v16"
BASELINE_ZIP_DIR = Path("/tmp/_v16_baseline/all_categorical_splits")
SOLS_PATH = BASE / "scripts" / "crypto_family_solutions.json"

COPY_FILES = [
    "train_cot_bit_manipulation.jsonl",
    "train_cot_cipher.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_numeral.jsonl",
    "train_cot_unit_conversion.jsonl",
]


def _category_of_prompt(prompt: str) -> str:
    if 'bit manipulation rule' in prompt: return 'bit_manipulation'
    if 'encryption rules' in prompt: return 'cipher'
    if 'gravitational constant' in prompt: return 'gravity'
    if 'numeral system' in prompt: return 'numeral'
    if 'unit conversion' in prompt: return 'unit_conversion'
    if 'secret set of transformation rules' in prompt:
        for line in prompt.split('\n'):
            line = line.strip()
            if ' = ' not in line or 'determine' in line.lower(): continue
            left = line.split(' = ', 1)[0].strip()
            if len(left) == 5 and not any(c.isdigit() for c in left): return 'cryptarithm'
            if any(c.isdigit() for c in left): return 'equation_numeric'
        return 'equation_numeric'
    return 'unknown'


def main():
    V16.mkdir(parents=True, exist_ok=True)

    # Step 1: copy 6 baseline files
    print("=" * 70)
    print("STEP 1: copy 6 baseline files into v16")
    print("=" * 70)
    for fn in COPY_FILES:
        src = BASELINE_ZIP_DIR / fn
        dst = V16 / fn
        if not src.exists():
            print(f"  [!] missing baseline file: {src}")
            continue
        shutil.copyfile(src, dst)
        n = sum(1 for _ in open(dst))
        print(f"  copied {fn}  ({n} records)")

    # Step 2: load resources for regeneration
    print()
    print("=" * 70)
    print("STEP 2: regenerate 3 files (cryptarithm_deduce/guess + equation_numeric_guess)")
    print("=" * 70)
    sols = {s['id']: s for s in json.load(open(SOLS_PATH))}
    print(f"  loaded {len(sols)} cryptarithm solutions from {SOLS_PATH.name}")
    rows = load_all(BASE / 'data/raw/train.csv')
    print(f"  loaded {len(rows)} train.csv rows")

    # Group rows by category
    crypto_rows = []
    eq_guess_rows = []
    from v16_writers import _parse_eq_prompt
    for r in rows:
        cat = _category_of_prompt(r['prompt'])
        if cat == 'cryptarithm':
            crypto_rows.append(r)
        elif cat == 'equation_numeric':
            eqs, q = _parse_eq_prompt(r['prompt'])
            if not eqs or q is None: continue
            example_ops = {e[1] for e in eqs}
            if q[1] not in example_ops:
                eq_guess_rows.append(r)
    print(f"  cryptarithm rows in train.csv: {len(crypto_rows)}")
    print(f"  equation_numeric_guess rows in train.csv: {len(eq_guess_rows)}")

    # Cryptarithm — split by kind using solver dict
    crypto_deduce = []
    crypto_guess = []
    skipped_unsolved = 0
    skipped_degenerate = 0
    for r in crypto_rows:
        sol = sols.get(r['id'])
        if sol is None:
            skipped_unsolved += 1
            continue
        # Filter degenerate "all-zeros" solutions — the solver fell back to
        # mapping every symbol to 0, which doesn't actually solve the puzzle.
        # Skip these like the truly unsolved (they shouldn't pollute SFT data).
        if all(v == 0 for v in sol['map'].values()):
            skipped_degenerate += 1
            continue
        puzzle = {
            'equations': sol['equations'], 'query': sol['query'],
            'answer': r['answer'], 'family': sol['family'],
            'map': sol['map'], 'ops': sol['ops'], 'kind': sol['kind'],
        }
        content = W.write_cryptarithm(puzzle)
        rec = {
            'category': f"cryptarithm_{sol['kind']}",
            'messages': [
                {'role': 'user',      'content': r['prompt']},
                {'role': 'assistant', 'content': content},
            ],
        }
        (crypto_deduce if sol['kind'] == 'deduce' else crypto_guess).append(rec)
    print(f"  cryptarithm_deduce records: {len(crypto_deduce)}")
    print(f"  cryptarithm_guess  records: {len(crypto_guess)}")
    print(f"  skipped (unsolved):   {skipped_unsolved}")
    print(f"  skipped (degenerate): {skipped_degenerate}")

    # equation_numeric_guess
    eq_guess_recs = []
    err = 0
    for r in eq_guess_rows:
        try:
            content = W.write_equation_numeric_guess({'prompt': r['prompt'], 'answer': r['answer']})
            eq_guess_recs.append({
                'category': 'equation_numeric_guess',
                'messages': [
                    {'role': 'user',      'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ],
            })
        except Exception as e:
            err += 1
            print(f"  [err] equation_numeric_guess {r.get('id','?')}: {e}")
    print(f"  equation_numeric_guess records: {len(eq_guess_recs)}  (err={err})")

    # Write the 3 files
    def write_jsonl(path, records):
        with open(path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  wrote {path.name}: {len(records)} records")

    print()
    print("=" * 70)
    print("STEP 3: write 3 regenerated JSONL files")
    print("=" * 70)
    write_jsonl(V16 / "train_cot_cryptarithm_deduce.jsonl", crypto_deduce)
    write_jsonl(V16 / "train_cot_cryptarithm_guess.jsonl",  crypto_guess)
    write_jsonl(V16 / "train_cot_equation_numeric_guess.jsonl", eq_guess_recs)

    # Total summary
    print()
    print("=" * 70)
    print("V16 SUMMARY")
    print("=" * 70)
    total = 0
    for fp in sorted(V16.glob("train_cot_*.jsonl")):
        n = sum(1 for _ in open(fp))
        total += n
        print(f"  {fp.name:<40} {n:>5} records")
    print(f"  {'TOTAL':<40} {total:>5}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {time.time()-t0:.1f}s")
