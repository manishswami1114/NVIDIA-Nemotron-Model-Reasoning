#!/usr/bin/env python3
"""
v15 dispatcher: loads train.csv puzzles + crypto_family_solutions.json, dispatches
each puzzle to the right writer, and writes 9 JSONL files into all_categorical_splits_v15/.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all
import cryptarithm_family as CF
import v15_writers as W

BASE = Path(__file__).resolve().parent.parent
V15 = BASE / "all_categorical_splits_v15"
SOLS_PATH = BASE / "scripts" / "crypto_family_solutions.json"


def category_of(prompt: str) -> str:
    """Return one of: bit_manipulation, cipher, equation_numeric, gravity,
    numeral, unit_conversion, cryptarithm, or 'unknown'."""
    if not isinstance(prompt, str):
        return 'unknown'
    if 'bit manipulation rule' in prompt:
        return 'bit_manipulation'
    if 'encryption rules' in prompt:
        return 'cipher'
    if 'gravitational constant' in prompt:
        return 'gravity'
    if 'numeral system' in prompt:
        return 'numeral'
    if 'unit conversion' in prompt:
        return 'unit_conversion'
    if 'secret set of transformation rules' in prompt:
        # cryptarithm = 5-symbol non-digit left side; equation_numeric = digit left side
        for line in prompt.split('\n'):
            line = line.strip()
            if ' = ' not in line or 'determine' in line.lower():
                continue
            left = line.split(' = ', 1)[0].strip()
            if len(left) == 5 and not any(c.isdigit() for c in left):
                return 'cryptarithm'
            if any(c.isdigit() for c in left):
                return 'equation_numeric'
        return 'equation_numeric'
    return 'unknown'


def equation_numeric_kind(eqs, query):
    return 'deduce' if query[1] in {e[1] for e in eqs} else 'guess'


def main():
    V15.mkdir(parents=True, exist_ok=True)
    sols = {s['id']: s for s in json.load(open(SOLS_PATH))}
    print(f"Loaded {len(sols)} cryptarithm solutions", flush=True)

    rows = load_all(BASE / 'data/raw/train.csv')
    print(f"Loaded {len(rows)} train.csv rows", flush=True)

    # buckets: subcategory → list of records
    buckets = {
        'bit_manipulation': [],
        'cipher': [],
        'cryptarithm_deduce': [],
        'cryptarithm_guess': [],
        'equation_numeric_deduce': [],
        'equation_numeric_guess': [],
        'gravity': [],
        'numeral': [],
        'unit_conversion': [],
    }
    skipped = 0
    t0 = time.time()
    for r in rows:
        cat = category_of(r['prompt'])
        try:
            if cat == 'bit_manipulation':
                content = W.write_bit_manipulation(r)
                buckets['bit_manipulation'].append({'category': 'bit_manipulation', 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            elif cat == 'cipher':
                content = W.write_cipher(r)
                buckets['cipher'].append({'category': 'cipher', 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            elif cat == 'gravity':
                content = W.write_gravity(r)
                buckets['gravity'].append({'category': 'gravity', 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            elif cat == 'numeral':
                content = W.write_numeral(r)
                buckets['numeral'].append({'category': 'numeral', 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            elif cat == 'unit_conversion':
                content = W.write_unit_conversion(r)
                buckets['unit_conversion'].append({'category': 'unit_conversion', 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            elif cat == 'equation_numeric':
                # parse equations + query
                from v15_writers import _parse_eq_prompt as parse_eq
                eqs, query = parse_eq(r['prompt'])
                if not eqs or query is None:
                    skipped += 1
                    continue
                kind = equation_numeric_kind(eqs, query)
                content = W.write_equation_numeric(r, kind=kind)
                key = f"equation_numeric_{kind}"
                buckets[key].append({'category': key, 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            elif cat == 'cryptarithm':
                # need verified solution
                sol = sols.get(r['id'])
                if sol is None:
                    skipped += 1
                    continue
                puzzle = {
                    'id': r['id'],
                    'prompt': r['prompt'],
                    'answer': r['answer'],
                    'kind': sol['kind'],
                    'equations': sol['equations'],
                    'query': sol['query'],
                    'family': sol['family'],
                    'map': sol['map'],
                    'ops': sol['ops'],
                }
                content = W.write_cryptarithm(puzzle)
                key = f"cryptarithm_{sol['kind']}"
                buckets[key].append({'category': key, 'messages': [
                    {'role': 'user', 'content': r['prompt']},
                    {'role': 'assistant', 'content': content},
                ]})
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            # keep going but log
            print(f"  err on {r.get('id','?')} ({cat}): {e}", flush=True)
    # write
    for key, recs in buckets.items():
        path = V15 / f"train_cot_{key}.jsonl"
        with open(path, 'w') as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  {key}: {len(recs)} records → {path}", flush=True)
    print(f"\nTotal records: {sum(len(v) for v in buckets.values())}; skipped: {skipped} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
