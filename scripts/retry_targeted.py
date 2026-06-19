#!/usr/bin/env python3
"""Re-solve only the IDs listed in to_resolve.json. Two passes: non-degenerate, then degenerate-fallback."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryptarithm_family as F
BASE = Path(__file__).resolve().parent.parent

def solve_relaxed(p, total_timeout=60):
    for fam in ['DIRECT_MAXMIN','DIRECT_SIMPLE','REV_AB','REV_MAXMIN']:
        r = F.solve_family(p['equations'], p['query'], p['answer'], fam, timeout=total_timeout//4, min_distinct=4)
        if r: return {'family': fam, 'map': r[0], 'ops': r[1]}
    for fam in ['DIRECT_SIMPLE','DIRECT_MAXMIN','REV_AB','REV_MAXMIN']:
        r = F.solve_family(p['equations'], p['query'], p['answer'], fam, timeout=total_timeout//4, min_distinct=1)
        if r: return {'family': fam, 'map': r[0], 'ops': r[1]}
    return None

def main():
    all_p = F.load_crypto_from_csv()
    by_id = {p['id']: p for p in all_p}
    target_ids = json.load(open(BASE/"scripts/to_resolve.json"))
    targets = [by_id[i] for i in target_ids if i in by_id]
    print(f"Re-solving {len(targets)} puzzles", flush=True)

    existing = json.load(open(BASE/"scripts/crypto_family_solutions.json"))
    new_sols = []
    t0 = time.time()
    for i, p in enumerate(targets):
        r = solve_relaxed(p, total_timeout=60)
        if r:
            p.update(r)
            new_sols.append({'id':p['id'],'kind':p['kind'],'equations':p['equations'],
                             'query':list(p['query']),'answer':p['answer'],
                             'family':p['family'],'map':p['map'],'ops':p['ops']})
        if (i+1) % 10 == 0 or i+1 == len(targets):
            print(f"  [{i+1}/{len(targets)}] new={len(new_sols)} [{time.time()-t0:.0f}s]", flush=True)

    merged = existing + new_sols
    json.dump(merged, open(BASE/"scripts/crypto_family_solutions.json","w"), default=str)
    print(f"\nFinal: total={len(merged)} (kept {len(existing)} + new {len(new_sols)}) ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
