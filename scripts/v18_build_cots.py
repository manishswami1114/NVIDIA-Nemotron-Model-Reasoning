#!/usr/bin/env python3
"""v17 dispatcher.

V17 = V16 + properly-deductive cryptarithm CoTs.

Steps:
  1. (Pre-step done by hand) `all_categorical_splits_v18/` already has 7 files
     copied from `all_categorical_splits_v16/` (the 6 baseline-verbatim files
     plus equation_numeric_guess). This script preserves them.
  2. Regenerate cryptarithm_deduce + cryptarithm_guess using v17_writers.
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all
import v18_writers as W

BASE = Path(__file__).resolve().parent.parent
V16  = BASE / "all_categorical_splits_v16"
V18  = BASE / "all_categorical_splits_v18"
# Prefer the merged strict solutions (built by v17_merge_solutions.py); fall
# back to the original json filtered at runtime.
_MERGED = BASE / "scripts" / "crypto_solutions_v17.json"
SOLS_PATH = _MERGED if _MERGED.exists() else (BASE / "scripts" / "crypto_family_solutions.json")

# Files we COPY from v16 verbatim (must already be in v17 — assert they are).
CARRY_OVER = [
    "train_cot_bit_manipulation.jsonl",
    "train_cot_cipher.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_numeral.jsonl",
    "train_cot_unit_conversion.jsonl",
]


def main():
    V18.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: verify the 7 carry-over files are present in v17")
    print("=" * 70)
    for fn in CARRY_OVER:
        p = V18 / fn
        if not p.exists():
            print(f"  [!!] missing {fn} — copy from v16 first")
            return False
        n = sum(1 for _ in open(p))
        print(f"  ok {fn}  ({n} records)")

    print()
    print("=" * 70)
    print("STEP 2: regenerate cryptarithm_deduce + cryptarithm_guess with v17_writers")
    print("=" * 70)
    sols = {s['id']: s for s in json.load(open(SOLS_PATH))}
    print(f"  loaded {len(sols)} cryptarithm solutions")
    rows = {r['id']: r for r in load_all(BASE / 'data/raw/train.csv')}
    print(f"  loaded {len(rows)} train.csv rows")

    import cryptarithm_family as CF
    def _is_neg_res(op, res):
        return len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])
    def _is_bogus(sol):
        """A solution is bogus if any equation requires zero-padding to match —
        i.e. the formula's natural-width result is shorter than the expected
        result width on a non-rev, non-concat operator."""
        mp = sol['map']; om = sol['ops']; family = sol['family']
        for n1, op, n2, res in sol['equations']:
            a = mp[n1[0]] * 10 + mp[n1[1]]
            b = mp[n2[0]] * 10 + mp[n2[1]]
            body = res[1:] if _is_neg_res(op, res) else res
            expected_width = len(body)
            applied = CF.apply_op(family, om[op], a, b)
            if applied is None:
                return True
            rs, _ = applied
            natural_width = len(rs.lstrip('-'))
            if natural_width < expected_width and 'rev(' not in om[op] and '||' not in om[op]:
                return True
        return False

    crypto_deduce = []; crypto_guess = []
    skipped_unsolved = 0; skipped_degenerate = 0; skipped_bogus = 0; err = 0
    for sid, sol in sols.items():
        r = rows.get(sid)
        if r is None:
            skipped_unsolved += 1; continue
        if all(v == 0 for v in sol['map'].values()):
            skipped_degenerate += 1; continue
        if _is_bogus(sol):
            # solver "found" a map only by zero-padding a too-short formula value;
            # these puzzles are not actually solvable under the claimed family/op —
            # they would teach the model wrong reasoning. Skip.
            skipped_bogus += 1; continue
        puzzle = {
            'equations': sol['equations'], 'query': sol['query'],
            'answer': r['answer'], 'family': sol['family'],
            'map': sol['map'], 'ops': sol['ops'], 'kind': sol['kind'],
        }
        try:
            content = W.write_cryptarithm_search(puzzle)
        except Exception as e:
            err += 1
            print(f"  [err] {sid}: {e}")
            continue
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
    print(f"  skipped (bogus zero-pad): {skipped_bogus}")
    print(f"  errors:               {err}")

    def write_jsonl(path, records):
        with open(path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  wrote {path.name}: {len(records)} records")

    print()
    print("=" * 70)
    print("STEP 3: write 2 regenerated JSONL files")
    print("=" * 70)
    write_jsonl(V18 / "train_cot_cryptarithm_deduce.jsonl", crypto_deduce)
    write_jsonl(V18 / "train_cot_cryptarithm_guess.jsonl",  crypto_guess)

    print()
    print("=" * 70)
    print("V18 SUMMARY")
    print("=" * 70)
    total = 0
    for fp in sorted(V18.glob("train_cot_*.jsonl")):
        n = sum(1 for _ in open(fp))
        total += n
        print(f"  {fp.name:<40} {n:>5} records")
    print(f"  {'TOTAL':<40} {total:>5}")
    return True


if __name__ == "__main__":
    t0 = time.time()
    ok = main()
    print(f"\nDone in {time.time()-t0:.1f}s")
    sys.exit(0 if ok else 1)
