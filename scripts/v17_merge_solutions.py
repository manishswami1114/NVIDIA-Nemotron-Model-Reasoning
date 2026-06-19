#!/usr/bin/env python3
"""Merge the strict re-solved cryptarithm solutions into a combined json file.

Source files:
  - scripts/crypto_family_solutions.json (818 originals; 362 are valid, the
    rest are bogus zero-pad or all-zeros)
  - scripts/crypto_strict_solutions.json (newly re-solved with strict width
    matching, written by v17_strict_resolve.py)

Output:
  - scripts/crypto_solutions_v17.json (the union — 362 originally-valid
    + N strict re-solved, deduplicated, each verified `_is_bogus = False`).

This is what v17_build_cots.py should load.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cryptarithm_family import apply_op

BASE = HERE.parent


def _is_neg_res(op, res):
    return len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])


def _is_bogus(sol):
    mp = sol['map']; om = sol['ops']; fam = sol['family']
    if all(v == 0 for v in mp.values()): return True
    for n1, op, n2, res in sol['equations']:
        a = mp[n1[0]] * 10 + mp[n1[1]]
        b = mp[n2[0]] * 10 + mp[n2[1]]
        body = res[1:] if _is_neg_res(op, res) else res
        applied = apply_op(fam, om[op], a, b)
        if applied is None: return True
        rs, _ = applied
        if len(rs.lstrip('-')) != len(body): return True
    return False


def main():
    orig = json.load(open(HERE / "crypto_family_solutions.json"))
    strict_path = HERE / "crypto_strict_solutions.json"
    strict = json.load(open(strict_path)) if strict_path.exists() else []

    merged = {}
    valid_orig = 0
    for s in orig:
        if not _is_bogus(s):
            merged[s['id']] = s
            valid_orig += 1
    print(f"Originally-valid solutions kept: {valid_orig}")

    new_added = 0
    rejected = 0
    for s in strict:
        if _is_bogus(s):
            rejected += 1
            continue
        if s['id'] in merged:
            # Already have a valid original — prefer the original
            continue
        merged[s['id']] = s
        new_added += 1
    print(f"Strict re-solved added: {new_added}")
    print(f"Strict solutions rejected (still bogus): {rejected}")
    print(f"Total merged solutions: {len(merged)}")

    by_kind = {'deduce': 0, 'guess': 0}
    for s in merged.values():
        by_kind[s['kind']] = by_kind.get(s['kind'], 0) + 1
    print(f"Breakdown: {by_kind}")

    out_path = HERE / "crypto_solutions_v17.json"
    json.dump(list(merged.values()), open(out_path, "w"), default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
