import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryptarithm_family as F
BASE = Path(__file__).resolve().parent.parent

def solve_relaxed(p, total_timeout=60):
    for fam in ['DIRECT_MAXMIN','DIRECT_SIMPLE','REV_AB','REV_MAXMIN']:
        # try non-degenerate first
        r = F.solve_family(p['equations'], p['query'], p['answer'], fam, timeout=total_timeout//4, min_distinct=4)
        if r: return {'family': fam, 'map': r[0], 'ops': r[1]}
    for fam in ['DIRECT_SIMPLE','DIRECT_MAXMIN','REV_AB','REV_MAXMIN']:
        # fallback: allow any distinct count (including degenerate)
        r = F.solve_family(p['equations'], p['query'], p['answer'], fam, timeout=total_timeout//4, min_distinct=1)
        if r: return {'family': fam, 'map': r[0], 'ops': r[1]}
    return None

def main():
    all_puzzles = F.load_crypto_from_csv()
    existing = json.load(open(BASE/"scripts/crypto_family_solutions.json"))
    solved_ids = {s['id'] for s in existing}
    unsolved = [p for p in all_puzzles if p['id'] not in solved_ids]
    print(f"Targeting {len(unsolved)} unsolved", flush=True)
    new = []
    t0 = time.time()
    for i, p in enumerate(unsolved):
        r = solve_relaxed(p, total_timeout=60)
        if r:
            p.update(r)
            new.append({'id':p['id'],'kind':p['kind'],'equations':p['equations'],
                        'query':list(p['query']),'answer':p['answer'],
                        'family':p['family'],'map':p['map'],'ops':p['ops']})
        print(f"  [{i+1}/{len(unsolved)}] {p['id']}: {'OK' if r else 'FAIL'}  new={len(new)} [{time.time()-t0:.0f}s]", flush=True)
    merged = existing + new
    json.dump(merged, open(BASE/"scripts/crypto_family_solutions.json","w"), default=str)
    print(f"\nFinal: total={len(merged)} (added {len(new)} of {len(unsolved)}) ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
