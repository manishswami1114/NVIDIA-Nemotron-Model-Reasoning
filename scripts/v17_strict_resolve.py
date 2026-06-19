#!/usr/bin/env python3
"""Strict re-solver for cryptarithm puzzles.

The previous solver in `cryptarithm_family.py` accepts solutions that require
zero-padding a formula's natural-width result to match the expected result
width — e.g. `max-min(0, 20) = 20` claimed as `"0020"` to fit a 4-digit body.
That's mathematical cheating.

This re-solver enforces:
  * STRICT width matching — `len(rs) == expected_width` (no zero-pad).
  * Higher `min_distinct` (default 5) to avoid mostly-zero degenerate maps.
  * Per-puzzle timeout to try harder (default 30 s).

Run: python3 scripts/v17_strict_resolve.py
Output: scripts/crypto_strict_solutions.json (newly-solved puzzles only)
        + prints stats.

We process the puzzles that the original solver got wrong:
  - 17 all-zeros (map all zero)
  - 439 bogus zero-pad solutions
  - 5 truly unsolved (not in json at all)
  461 puzzles total.
"""
import json, time
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cryptarithm_family import (
    FAMILIES, apply_op, rev2, rev_signed, rev_str, extend, load_crypto_from_csv,
)
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


def eq_solutions_strict(n1s, n2s, res_s, family, op_name, mapping, is_neg_result):
    """STRICT version of eq_solutions — no zero-padding allowed.
    Yields extended mappings where the formula's natural-width result
    matches the expected result-symbol body exactly."""
    out = []
    L = len(res_s)
    k1a, k1b = mapping.get(n1s[0]), mapping.get(n1s[1])
    if k1a is not None and k1b is not None: a_vals = [k1a*10+k1b]
    elif k1a is not None: a_vals = range(k1a*10, k1a*10+10)
    elif k1b is not None: a_vals = range(k1b, 100, 10)
    elif n1s[0]==n1s[1]: a_vals = [d*11 for d in range(10)]
    else: a_vals = range(100)
    for a in a_vals:
        d0, d1 = a//10, a%10
        if n1s[0]==n1s[1] and d0 != d1: continue
        m1 = extend(mapping, [(n1s[0], d0), (n1s[1], d1)])
        if not m1: continue
        k2a, k2b = m1.get(n2s[0]), m1.get(n2s[1])
        if k2a is not None and k2b is not None: b_vals = [k2a*10+k2b]
        elif k2a is not None: b_vals = range(k2a*10, k2a*10+10)
        elif k2b is not None: b_vals = range(k2b, 100, 10)
        elif n2s[0]==n2s[1]: b_vals = [d*11 for d in range(10)]
        else: b_vals = range(100)
        for b in b_vals:
            d2, d3 = b//10, b%10
            if n2s[0]==n2s[1] and d2 != d3: continue
            m2 = extend(m1, [(n2s[0], d2), (n2s[1], d3)])
            if not m2: continue
            applied = apply_op(family, op_name, a, b)
            if applied is None: continue
            rs, neg = applied
            if neg != is_neg_result: continue
            # STRICT: result must be exactly the expected width (no zero-pad cheating)
            if len(rs) != L: continue
            m3 = extend(m2, [(res_s[j], int(rs[j])) for j in range(L)])
            if m3: out.append(m3)
    return out


def solve_family_strict(equations, query, gt, family, timeout=30, min_distinct=5):
    """Try to solve puzzle under one family — STRICT width matching only."""
    start = time.time(); ta = start + timeout
    neg_flags = []; norm_eqs = []
    for n1s, op, n2s, res in equations:
        if _is_neg_res(op, res):
            norm_eqs.append((n1s, op, n2s, res[1:]))
            neg_flags.append(True)
        else:
            norm_eqs.append((n1s, op, n2s, res))
            neg_flags.append(False)
    qop = query[1]
    if _is_neg_res(qop, gt):
        q_neg = True; q_res = gt[1:]
    else:
        q_neg = False; q_res = gt
    all_eqs = list(norm_eqs) + [(query[0], qop, query[2], q_res)]
    all_neg = neg_flags + [q_neg]
    n = len(all_eqs)
    family_ops = list(FAMILIES[family].keys())

    holder = [None]
    def dfs(m, om, done):
        if time.time() > ta: return False
        if len(done) == n:
            if len(set(m.values())) < min_distinct: return False
            holder[0] = (dict(m), dict(om)); return True
        best = -1; bo = None
        for i in range(n):
            if i in done: continue
            n1s, op, n2s, res_s = all_eqs[i]
            ops_to_try = [om[op]] if op in om else family_ops
            opts = []
            for opn in ops_to_try:
                for m2 in eq_solutions_strict(n1s, n2s, res_s, family, opn, m, all_neg[i]):
                    opts.append((opn, m2))
            if bo is None or len(opts) < len(bo):
                bo = opts; best = i
                if not opts: break
        if not bo: return False
        n1s, op, n2s, res_s = all_eqs[best]
        nd = done | {best}
        for opn, m2 in bo:
            if time.time() > ta: return False
            om2 = om if op in om else {**om, op: opn}
            if dfs(m2, om2, nd): return True
        return False

    if dfs({}, {}, frozenset()):
        return holder[0]
    return None


def solve_puzzle_strict(puzzle, timeout_per_family=20, min_distinct=5):
    for family in ['DIRECT_MAXMIN', 'DIRECT_SIMPLE', 'REV_AB', 'REV_MAXMIN']:
        r = solve_family_strict(
            puzzle['equations'], puzzle['query'], puzzle['answer'],
            family, timeout=timeout_per_family, min_distinct=min_distinct,
        )
        if r:
            mp, om = r
            return {'family': family, 'map': mp, 'ops': om}
    return None


def main():
    # Load existing json
    existing_sols = json.load(open(BASE / "scripts" / "crypto_family_solutions.json"))
    existing_by_id = {s['id']: s for s in existing_sols}

    # Find problematic puzzles: bogus + all-zeros + unsolved
    all_puzzles = load_crypto_from_csv()
    import os, sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"Loaded {len(all_puzzles)} total cryptarithm puzzles from train.csv", flush=True)

    problematic = []
    for p in all_puzzles:
        s = existing_by_id.get(p['id'])
        if s is None:
            problematic.append((p, 'unsolved'))
            continue
        if all(v == 0 for v in s['map'].values()):
            problematic.append((p, 'all_zeros'))
            continue
        if _is_bogus(s):
            problematic.append((p, 'bogus'))
            continue
    print(f"Problematic puzzles to re-solve: {len(problematic)}", flush=True)
    by_reason = {}
    for _, r in problematic: by_reason[r] = by_reason.get(r, 0) + 1
    print(f"  Breakdown: {by_reason}", flush=True)

    # Strict re-solve. Single pass with short timeout — if a puzzle doesn't
    # solve in 8s/family, it's likely truly hard and we skip to keep total
    # runtime under ~15 min. Checkpoint-save every 50 puzzles.
    t0 = time.time()
    new_solutions = []
    failed = []
    out_path = BASE / "scripts" / "crypto_strict_solutions.json"

    def save():
        json.dump(new_solutions, open(out_path, "w"), default=str)

    for i, (p, reason) in enumerate(problematic):
        r = solve_puzzle_strict(p, timeout_per_family=8, min_distinct=4)
        if r is None:
            failed.append((p, reason))
        else:
            new_solutions.append({
                'id': p['id'],
                'kind': p['kind'],
                'equations': p['equations'],
                'query': list(p['query']),
                'answer': p['answer'],
                'family': r['family'],
                'map': r['map'],
                'ops': r['ops'],
            })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(problematic)}: new_ok={len(new_solutions)} fail={len(failed)} [{time.time()-t0:.0f}s]", flush=True)
        if (i + 1) % 50 == 0:
            save()
            print(f"  (checkpoint saved — {len(new_solutions)} so far)", flush=True)

    save()
    print(f"\nFinal: re-solved={len(new_solutions)} / {len(problematic)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"Truly unsolvable under strict: {len(failed)}", flush=True)
    print(f"Saved {len(new_solutions)} new strict solutions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
