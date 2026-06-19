#!/usr/bin/env python3
"""Family-constrained cryptarithm solver.
RULES (from perfect_solver_complete.md):
  - Same operator symbol = SAME formula across all equations in a puzzle
  - All operators in one puzzle use formulas from the SAME family
  - Width-preserving rev (operates on the 2-digit string of inputs)
"""
import json, re, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE = Path(__file__).resolve().parent.parent

# ----- width-preserving primitives -----
def rev2(n):
    """Reverse a number treating it as 2 digits (preserves leading zero of input)."""
    s = f"{n:02d}"
    return int(s[::-1])

def rev_str(s):
    """Reverse a digit string preserving its width."""
    return s[::-1]

def rev_signed(v):
    """rev(-76) = '-67', rev(76) = '67'. Digits swap; sign stays put."""
    if v < 0:
        return '-' + str(-v)[::-1]
    return str(v)[::-1]

# ----- operation families -----
# Each op returns (result_value, natural_width). Width is preserved for rev ops.
def safe_pad(v, w):
    s = str(v)
    return s if len(s) >= w else s.zfill(w)

# Family 1: direct max/min (input = a,b as ints; output as int + natural digit count)
FAMILY_DIRECT_MAXMIN = {
    'max+min':       lambda a,b: max(a,b)+min(a,b),
    'max-min':       lambda a,b: max(a,b)-min(a,b),
    'max*min':       lambda a,b: max(a,b)*min(a,b),
    '(max*min)+1':   lambda a,b: max(a,b)*min(a,b)+1,
    '(max*min)-1':   lambda a,b: max(a,b)*min(a,b)-1,
    '(max+min)+1':   lambda a,b: max(a,b)+min(a,b)+1,
    '(max+min)-1':   lambda a,b: max(a,b)+min(a,b)-1,
    'max%min':       lambda a,b: max(a,b)%min(a,b) if min(a,b)>0 else None,
    '(max-min)+1':   lambda a,b: max(a,b)-min(a,b)+1,
    '(max-min)-1':   lambda a,b: max(a,b)-min(a,b)-1,
    '(min-max)+1':   lambda a,b: min(a,b)-max(a,b)+1,
    '(min-max)-1':   lambda a,b: min(a,b)-max(a,b)-1,
    'max||min':      lambda a,b: int(f"{max(a,b):02d}{min(a,b):02d}"),
    'min||max':      lambda a,b: int(f"{min(a,b):02d}{max(a,b):02d}"),
    '-(max-min)':    lambda a,b: -(max(a,b)-min(a,b)),
    'min-max':       lambda a,b: min(a,b)-max(a,b),
    'min+max':       lambda a,b: min(a,b)+max(a,b),
}

# Family 2: direct simple a/b
FAMILY_DIRECT_SIMPLE = {
    'a+b':     lambda a,b: a+b,
    'a-b':     lambda a,b: a-b,
    'b-a':     lambda a,b: b-a,
    'a*b':     lambda a,b: a*b,
    'b*a':     lambda a,b: b*a,
    '(a*b)+1': lambda a,b: a*b+1,
    '(a*b)-1': lambda a,b: a*b-1,
    '(a+b)+1': lambda a,b: a+b+1,
    '(a+b)-1': lambda a,b: a+b-1,
    '(b+a)+1': lambda a,b: b+a+1,
    '(b+a)-1': lambda a,b: b+a-1,
    '(b*a)+1': lambda a,b: b*a+1,
    '(b*a)-1': lambda a,b: b*a-1,
    'a||b':    lambda a,b: int(f"{a:02d}{b:02d}"),
    'b||a':    lambda a,b: int(f"{b:02d}{a:02d}"),
    'a%b':     lambda a,b: a%b if b>0 else None,
    'b%a':     lambda a,b: b%a if a>0 else None,
    '-(a-b)':  lambda a,b: -(a-b),
    '-(b-a)':  lambda a,b: -(b-a),
    '(a-b)+1': lambda a,b: a-b+1,
    '(a-b)-1': lambda a,b: a-b-1,
    '(b-a)+1': lambda a,b: b-a+1,
    '(b-a)-1': lambda a,b: b-a-1,
}

# Width-preserving rev on a 2-digit input means: treat as 2-char string, reverse
# After reverse, do arithmetic, then reverse the result string of width = result's natural width.

# Family 3: rev with rev(a), rev(b)  -- rev-of-2-digit input
def _ra_rb(a,b): return rev2(a), rev2(b)
FAMILY_REV_AB = {
    'rev(rev(a)+rev(b))':     lambda a,b: rev_str(str((rev2(a)+rev2(b)))),
    'rev(rev(a)-rev(b))':     lambda a,b: rev_signed(rev2(a)-rev2(b)),
    'rev(rev(b)-rev(a))':     lambda a,b: rev_signed(rev2(b)-rev2(a)),
    'rev(rev(a)*rev(b))':     lambda a,b: rev_str(str(rev2(a)*rev2(b))),
    'rev(rev(b)*rev(a))':     lambda a,b: rev_str(str(rev2(b)*rev2(a))),
    'rev(rev(a)*rev(b)+1)':   lambda a,b: rev_str(str(rev2(a)*rev2(b)+1)),
    'rev(rev(a)*rev(b)-1)':   lambda a,b: rev_signed(rev2(a)*rev2(b)-1),
    'rev(rev(a)+rev(b)+1)':   lambda a,b: rev_str(str(rev2(a)+rev2(b)+1)),
    'rev(rev(a)+rev(b)-1)':   lambda a,b: rev_signed(rev2(a)+rev2(b)-1),
    'rev(rev(b)*rev(a)+1)':   lambda a,b: rev_str(str(rev2(b)*rev2(a)+1)),
    'rev(rev(b)+rev(a)+1)':   lambda a,b: rev_str(str(rev2(b)+rev2(a)+1)),
    'rev(rev(b)+rev(a)-1)':   lambda a,b: rev_signed(rev2(b)+rev2(a)-1),
    'rev(rev(a)||rev(b))':    lambda a,b: rev_str(f"{rev2(a):02d}{rev2(b):02d}"),
    'rev(rev(b)||rev(a))':    lambda a,b: rev_str(f"{rev2(b):02d}{rev2(a):02d}"),
    'rev(rev(a)%rev(b))':     lambda a,b: rev_str(str(rev2(a)%rev2(b))) if rev2(b)>0 else None,
    'rev(rev(b)%rev(a))':     lambda a,b: rev_str(str(rev2(b)%rev2(a))) if rev2(a)>0 else None,
    '-rev(rev(a)-rev(b))':    lambda a,b: rev_str(str(abs(rev2(a)-rev2(b)))),
    '-rev(rev(b)-rev(a))':    lambda a,b: rev_str(str(abs(rev2(b)-rev2(a)))),
    'rev(rev(a)-rev(b)+1)':   lambda a,b: rev_signed(rev2(a)-rev2(b)+1),
    'rev(rev(a)-rev(b)-1)':   lambda a,b: rev_signed(rev2(a)-rev2(b)-1),
    'rev(rev(b)-rev(a)+1)':   lambda a,b: rev_signed(rev2(b)-rev2(a)+1),
    'rev(rev(b)-rev(a)-1)':   lambda a,b: rev_signed(rev2(b)-rev2(a)-1),
    'rev(rev(b)*rev(a)-1)':   lambda a,b: rev_signed(rev2(b)*rev2(a)-1),
}

# Family 4: rev with max/min of rev(a),rev(b)
def _rmx_rmn(a,b):
    ra,rb=rev2(a),rev2(b);return max(ra,rb),min(ra,rb)
FAMILY_REV_MAXMIN = {
    'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))':   lambda a,b: rev_str(str(_rmx_rmn(a,b)[0]+_rmx_rmn(a,b)[1])),
    'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':   lambda a,b: rev_signed(_rmx_rmn(a,b)[0]-_rmx_rmn(a,b)[1]),
    'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))':   lambda a,b: rev_str(str(_rmx_rmn(a,b)[0]*_rmx_rmn(a,b)[1])),
    'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)': lambda a,b: rev_str(str(_rmx_rmn(a,b)[0]+_rmx_rmn(a,b)[1]+1)),
    'rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)': lambda a,b: rev_signed(_rmx_rmn(a,b)[0]+_rmx_rmn(a,b)[1]-1),
    'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)': lambda a,b: rev_str(str(_rmx_rmn(a,b)[0]*_rmx_rmn(a,b)[1]+1)),
    'rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)': lambda a,b: rev_signed(_rmx_rmn(a,b)[0]*_rmx_rmn(a,b)[1]-1),
    'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))':   lambda a,b: rev_str(str(_rmx_rmn(a,b)[0]%_rmx_rmn(a,b)[1])) if _rmx_rmn(a,b)[1]>0 else None,
    'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))':  lambda a,b: rev_str(f"{_rmx_rmn(a,b)[0]:02d}{_rmx_rmn(a,b)[1]:02d}"),
    'rev(min(rev(a),rev(b))||max(rev(a),rev(b)))':  lambda a,b: rev_str(f"{_rmx_rmn(a,b)[1]:02d}{_rmx_rmn(a,b)[0]:02d}"),
    'rev(min(rev(a),rev(b))-max(rev(a),rev(b)))':   lambda a,b: rev_signed(_rmx_rmn(a,b)[1]-_rmx_rmn(a,b)[0]),
    '-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':  lambda a,b: rev_str(str(_rmx_rmn(a,b)[0]-_rmx_rmn(a,b)[1])),
    'rev(max(rev(a),rev(b))-min(rev(a),rev(b))+1)': lambda a,b: rev_signed(_rmx_rmn(a,b)[0]-_rmx_rmn(a,b)[1]+1),
    'rev(max(rev(a),rev(b))-min(rev(a),rev(b))-1)': lambda a,b: rev_signed(_rmx_rmn(a,b)[0]-_rmx_rmn(a,b)[1]-1),
    'rev(min(rev(a),rev(b))-max(rev(a),rev(b))+1)': lambda a,b: rev_signed(_rmx_rmn(a,b)[1]-_rmx_rmn(a,b)[0]+1),
}

FAMILIES = {
    'DIRECT_MAXMIN':   FAMILY_DIRECT_MAXMIN,
    'DIRECT_SIMPLE':   FAMILY_DIRECT_SIMPLE,
    'REV_AB':          FAMILY_REV_AB,
    'REV_MAXMIN':      FAMILY_REV_MAXMIN,
}

def apply_op(family, op_name, a, b):
    """Return (result_string, is_negative)."""
    fn = FAMILIES[family].get(op_name)
    if fn is None: return None
    res = fn(a, b)
    if res is None: return None
    explicit_neg = op_name.startswith('-')
    if isinstance(res, str):
        if res.startswith('-'):
            return res[1:], True
        return res, explicit_neg
    if res < 0:
        return str(-res), True
    return str(res), explicit_neg

# ----- puzzle loading -----
def load_crypto_from_csv():
    from csv_loader import load_all
    rows = load_all(BASE/"data/raw/train.csv")
    def is_crypto(p):
        if not isinstance(p, str): return False
        if 'secret set of transformation' not in p: return False
        for line in p.split('\n'):
            line = line.strip()
            if ' = ' in line and 'determine' not in line.lower():
                left = line.split(' = ', 1)[0].strip()
                if len(left)==5 and not any(c.isdigit() for c in left):
                    return True
        return False
    out = []
    for r in rows:
        if not is_crypto(r['prompt']): continue
        eqs = []; query = None
        for line in r['prompt'].split('\n'):
            line = line.strip()
            if 'determine the result for:' in line.lower():
                m = re.search(r':\s*(.+?)$', line)
                if m and len(m.group(1).strip()) >= 5:
                    q = m.group(1).strip()
                    query = (q[0:2], q[2], q[3:5])
            elif ' = ' in line:
                left, right = line.split(' = ', 1)
                left = left.strip(); right = right.strip()
                if len(left) == 5:
                    eqs.append((left[0:2], left[2], left[3:5], right))
        if eqs and query:
            ex_ops = set(e[1] for e in eqs)
            kind = 'deduce' if query[1] in ex_ops else 'guess'
            out.append({'id': r['id'], 'kind': kind, 'equations': eqs, 'query': query, 'answer': r['answer'], 'prompt': r['prompt']})
    return out

# ----- solver: family-constrained DFS -----
def extend(m, pairs):
    m = dict(m)
    for s,d in pairs:
        if s in m:
            if m[s] != d: return None
        else: m[s] = d
    return m

def eq_solutions(n1s, n2s, res_s, family, op_name, mapping, is_neg_result):
    """All extended maps where eq holds with this family+op (width-preserving)."""
    out = []
    L = len(res_s)
    # num1 candidates
    k1a, k1b = mapping.get(n1s[0]), mapping.get(n1s[1])
    if k1a is not None and k1b is not None: a_vals = [k1a*10+k1b]
    elif k1a is not None: a_vals = range(k1a*10, k1a*10+10)
    elif k1b is not None: a_vals = range(k1b, 100, 10)
    elif n1s[0]==n1s[1]: a_vals = [d*11 for d in range(10)]
    else: a_vals = range(100)
    for a in a_vals:
        d0, d1 = a//10, a%10
        if n1s[0]==n1s[1] and d0!=d1: continue
        m1 = extend(mapping, [(n1s[0],d0),(n1s[1],d1)])
        if not m1: continue
        k2a, k2b = m1.get(n2s[0]), m1.get(n2s[1])
        if k2a is not None and k2b is not None: b_vals = [k2a*10+k2b]
        elif k2a is not None: b_vals = range(k2a*10, k2a*10+10)
        elif k2b is not None: b_vals = range(k2b, 100, 10)
        elif n2s[0]==n2s[1]: b_vals = [d*11 for d in range(10)]
        else: b_vals = range(100)
        for b in b_vals:
            d2, d3 = b//10, b%10
            if n2s[0]==n2s[1] and d2!=d3: continue
            m2 = extend(m1, [(n2s[0],d2),(n2s[1],d3)])
            if not m2: continue
            applied = apply_op(family, op_name, a, b)
            if applied is None: continue
            rs, neg = applied
            if neg != is_neg_result: continue
            # Width match: rev/concat ops keep their natural width; simple direct ops may zero-pad
            if len(rs) > L: continue
            if len(rs) < L:
                # Only allow zero-pad for non-rev, non-concat ops
                if 'rev(' in op_name or '||' in op_name: continue
                rs = rs.zfill(L)
            m3 = extend(m2, [(res_s[j], int(rs[j])) for j in range(L)])
            if m3: out.append(m3)
    return out

def solve_family(equations, query, gt, family, timeout=8, min_distinct=4):
    """Try to solve puzzle with ALL operators using operations from this family."""
    start = time.time(); ta = start + timeout
    # detect negative example equations (result starts with operator char)
    neg_flags = []
    norm_eqs = []
    for n1s, op, n2s, res in equations:
        if len(res) >= 2 and res[0] == op and all(c != op for c in res[1:]):
            norm_eqs.append((n1s, op, n2s, res[1:]))
            neg_flags.append(True)
        else:
            norm_eqs.append((n1s, op, n2s, res))
            neg_flags.append(False)
    # build query eq
    qop = query[1]
    if len(gt) >= 2 and gt[0] == qop and all(c != qop for c in gt[1:]):
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
        # MRV: pick eq with fewest candidates given current state
        best = -1; bo = None
        for i in range(n):
            if i in done: continue
            n1s, op, n2s, res_s = all_eqs[i]
            ops_to_try = [om[op]] if op in om else family_ops
            opts = []
            for opn in ops_to_try:
                for m2 in eq_solutions(n1s, n2s, res_s, family, opn, m, all_neg[i]):
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

def solve_puzzle(puzzle, timeout_per_family=15):
    """Try each family; first one that yields a non-degenerate global solution wins."""
    for family in ['DIRECT_MAXMIN', 'DIRECT_SIMPLE', 'REV_AB', 'REV_MAXMIN']:
        r = solve_family(puzzle['equations'], puzzle['query'], puzzle['answer'], family, timeout=timeout_per_family)
        if r:
            mp, om = r
            return {'family': family, 'map': mp, 'ops': om}
    return None

if __name__ == "__main__":
    puzzles = load_crypto_from_csv()
    print(f"Loaded {len(puzzles)} cryptarithm puzzles from train.csv", flush=True)
    t0 = time.time()
    solved = []; failed = []
    for i, p in enumerate(puzzles):
        r = solve_puzzle(p)
        if r:
            p.update(r); solved.append(p)
        else:
            failed.append(p)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(puzzles)}: ok={len(solved)} fail={len(failed)} [{time.time()-t0:.0f}s]", flush=True)
    print(f"\nFinal: solved={len(solved)} failed={len(failed)} ({time.time()-t0:.0f}s)", flush=True)
    # save
    out = [{'id':p['id'],'kind':p['kind'],'equations':p['equations'],'query':list(p['query']),'answer':p['answer'],'family':p['family'],'map':p['map'],'ops':p['ops']} for p in solved]
    json.dump(out, open(BASE/"scripts"/"crypto_family_solutions.json","w"), default=str)
    print(f"Saved {len(out)} family-verified solutions.", flush=True)
