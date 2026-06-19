#!/usr/bin/env python3
"""V19 dispatcher — unified baseline equation_numeric_deduce template.

Regenerates:
  - train_cot_cryptarithm_deduce.jsonl
  - train_cot_cryptarithm_guess.jsonl
  - train_cot_equation_numeric_guess.jsonl

All three use the same baseline equation_numeric_deduce CoT template.
The 6 carry-over files (bit_manipulation, cipher, equation_numeric_deduce,
gravity, numeral, unit_conversion) are copied from v17 verbatim.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all
import v19_writers as W

BASE = Path(__file__).resolve().parent.parent
V17  = BASE / "all_categorical_splits_v17"
V19  = BASE / "all_categorical_splits_v19"
SOLS_PATH = BASE / "scripts" / "crypto_solutions_v17.json"

CARRY_OVER = [
    "train_cot_bit_manipulation.jsonl",
    "train_cot_cipher.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_numeral.jsonl",
    "train_cot_unit_conversion.jsonl",
]


def main():
    V19.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: verify the 6 carry-over files are present in v19")
    print("=" * 70)
    for fn in CARRY_OVER:
        p = V19 / fn
        if not p.exists():
            print(f"  [!!] missing {fn} — copy from v17 first")
            return False
        n = sum(1 for _ in open(p))
        print(f"  ok {fn}  ({n} records)")

    print()
    print("=" * 70)
    print("STEP 2: regenerate cryptarithm_deduce + cryptarithm_guess +")
    print("        equation_numeric_guess with the unified baseline template")
    print("=" * 70)
    sols = {s['id']: s for s in json.load(open(SOLS_PATH))}
    print(f"  loaded {len(sols)} cryptarithm solutions")
    rows = load_all(BASE / 'data/raw/train.csv')
    rows_by_id = {r['id']: r for r in rows}
    print(f"  loaded {len(rows)} train.csv rows")

    # ---- Cryptarithm ----
    crypto_deduce = []; crypto_guess = []
    err = 0
    for sid, sol in sols.items():
        r = rows_by_id.get(sid)
        if r is None:
            continue
        puzzle = {
            'equations': sol['equations'], 'query': sol['query'],
            'answer': r['answer'], 'family': sol['family'],
            'map': sol['map'], 'ops': sol['ops'], 'kind': sol['kind'],
        }
        try:
            content = W.write_cryptarithm(puzzle)
        except Exception as e:
            err += 1
            print(f"  [err] crypto {sid}: {e}")
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
    print(f"  crypto errors: {err}")

    # ---- equation_numeric_guess ----
    from v19_writers import _crypto_to_eqnum_data  # ensure module loaded
    from v16_writers import _parse_eq_prompt
    def _is_eq(prompt):
        if 'secret set of transformation rules' not in prompt: return False
        for line in prompt.split('\n'):
            line = line.strip()
            if ' = ' in line and 'determine' not in line.lower():
                left = line.split(' = ', 1)[0].strip()
                if any(c.isdigit() for c in left): return True
                return False
        return False
    def _is_guess(prompt):
        eqs, q = _parse_eq_prompt(prompt)
        if not eqs or q is None: return False
        ex_ops = {e[1] for e in eqs}
        return q[1] not in ex_ops

    eq_guess_recs = []
    eq_err = 0
    for r in rows:
        if not _is_eq(r['prompt']): continue
        if not _is_guess(r['prompt']): continue
        try:
            content = W.write_equation_numeric_guess({'prompt': r['prompt'], 'answer': r['answer']})
        except Exception as e:
            eq_err += 1
            print(f"  [err] eq_guess {r.get('id','?')}: {e}")
            continue
        eq_guess_recs.append({
            'category': 'equation_numeric_guess',
            'messages': [
                {'role': 'user',      'content': r['prompt']},
                {'role': 'assistant', 'content': content},
            ],
        })
    print(f"  equation_numeric_guess records: {len(eq_guess_recs)}  (err={eq_err})")

    def write_jsonl(path, records):
        with open(path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  wrote {path.name}: {len(records)} records")

    print()
    print("=" * 70)
    print("STEP 3: write 3 regenerated JSONL files")
    print("=" * 70)
    write_jsonl(V19 / "train_cot_cryptarithm_deduce.jsonl", crypto_deduce)
    write_jsonl(V19 / "train_cot_cryptarithm_guess.jsonl",  crypto_guess)
    write_jsonl(V19 / "train_cot_equation_numeric_guess.jsonl", eq_guess_recs)

    print()
    print("=" * 70)
    print("V19 SUMMARY")
    print("=" * 70)
    total = 0
    for fp in sorted(V19.glob("train_cot_*.jsonl")):
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
