#!/usr/bin/env python3
"""Retry only the unsolved cryptarithm puzzles with longer timeout."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryptarithm_family as F

BASE = Path(__file__).resolve().parent.parent

def main():
    all_puzzles = F.load_crypto_from_csv()
    sols_path = BASE/"scripts"/"crypto_family_solutions.json"
    existing = json.load(open(sols_path))
    solved_ids = {s['id'] for s in existing}
    unsolved = [p for p in all_puzzles if p['id'] not in solved_ids]
    print(f"All: {len(all_puzzles)}, already solved: {len(existing)}, unsolved: {len(unsolved)}", flush=True)

    new_solutions = []
    t0 = time.time()
    for i, p in enumerate(unsolved):
        r = F.solve_puzzle(p, timeout_per_family=60)  # 4 min total per puzzle worst-case
        if r:
            p.update(r)
            new_solutions.append({
                'id': p['id'], 'kind': p['kind'], 'equations': p['equations'],
                'query': list(p['query']), 'answer': p['answer'],
                'family': p['family'], 'map': p['map'], 'ops': p['ops']
            })
        elapsed = time.time() - t0
        status = "OK" if r else "FAIL"
        print(f"  [{i+1}/{len(unsolved)}] {p['id']} ({p['kind']}): {status}  [{elapsed:.0f}s total, new={len(new_solutions)}]", flush=True)

    # merge & save
    merged = existing + new_solutions
    json.dump(merged, open(sols_path, "w"), default=str)
    print(f"\nTotal solved after retry: {len(merged)} (added {len(new_solutions)} of {len(unsolved)})", flush=True)
    print(f"Time: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
