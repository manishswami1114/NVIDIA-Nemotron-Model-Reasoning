#!/usr/bin/env python3
"""Second-pass strict re-solver for the 30 stragglers.

The first pass at 8s/family + min_distinct=4 left 30 puzzles unsolved.
This pass tries each with:
  - 90s per family (10x longer)
  - min_distinct=3 (more permissive)
  - All 4 families attempted; we keep the FIRST valid solution found.

Output: merges newly-solved puzzles into crypto_solutions_v17.json in place
(so we don't need to re-run the merger).
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cryptarithm_family import load_crypto_from_csv
from v17_strict_resolve import solve_puzzle_strict

BASE = HERE.parent
MERGED = HERE / "crypto_solutions_v17.json"


def main():
    sys.stdout.reconfigure(line_buffering=True)

    merged = json.load(open(MERGED))
    have_ids = {s['id'] for s in merged}
    print(f"Currently in crypto_solutions_v17.json: {len(merged)} puzzles", flush=True)

    all_puzzles = load_crypto_from_csv()
    missing = [p for p in all_puzzles if p['id'] not in have_ids]
    print(f"Still missing: {len(missing)} puzzles", flush=True)

    t0 = time.time()
    newly_solved = []; still_failed = []
    for i, p in enumerate(missing):
        # Two-pass: first try min_distinct=3 with long timeout, then min_distinct=2 as last resort
        r = solve_puzzle_strict(p, timeout_per_family=90, min_distinct=3)
        if r is None:
            r = solve_puzzle_strict(p, timeout_per_family=60, min_distinct=2)
        if r is None:
            still_failed.append(p)
            print(f"  [{i+1}/{len(missing)}] FAIL id={p['id']} kind={p['kind']} [{time.time()-t0:.0f}s]", flush=True)
        else:
            sol = {
                'id': p['id'],
                'kind': p['kind'],
                'equations': p['equations'],
                'query': list(p['query']),
                'answer': p['answer'],
                'family': r['family'],
                'map': r['map'],
                'ops': r['ops'],
            }
            newly_solved.append(sol)
            print(f"  [{i+1}/{len(missing)}] OK   id={p['id']} kind={p['kind']} family={r['family']} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nNewly solved: {len(newly_solved)} / {len(missing)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"Still unsolvable: {len(still_failed)}", flush=True)

    if newly_solved:
        merged.extend(newly_solved)
        json.dump(merged, open(MERGED, "w"), default=str)
        print(f"Updated {MERGED} → {len(merged)} total puzzles", flush=True)
    else:
        print("No new puzzles solved; merged file unchanged.", flush=True)


if __name__ == "__main__":
    main()
