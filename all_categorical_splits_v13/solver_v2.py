#!/usr/bin/env python3
"""
SOLVER V2: The key insight is that within each block, ALL example visible ops
must be CONSISTENTLY explained by ONE domain. We need to find which
(domain, {visop→(inner_op,adj)}) assignment is consistent, then apply
the REMAINING operation to the query.

The algorithm:
1. For each candidate domain D:
   a. For each visible op in examples, find which (inner_op, adj) works in D
   b. Check that different visible ops map to DIFFERENT inner_ops
   c. The query gets an inner_op NOT used by any example visop
2. If only ONE (domain, query_inner_op, query_adj) produces a valid answer → done
3. If multiple → we need further disambiguation

The critical constraint is: DIFFERENT visible ops must use DIFFERENT base operations.
"""
import json, re
from collections import defaultdict, Counter

def rev_int(s): return int(s[::-1])

def compute(a_str, b_str, domain, inner_op, adj):
    a, b = int(a_str), int(b_str)
    ra, rb = rev_int(a_str), rev_int(b_str)
    ops = {
        'revab': (ra, rb, 'rev'), 'revba': (rb, ra, 'rev'),
        'maxmin': (max(a,b), min(a,b), 'plain'), 'minmax': (min(a,b), max(a,b), 'plain'),
        'maxmin_r': (max(ra,rb), min(ra,rb), 'rev'), 'minmax_r': (min(ra,rb), max(ra,rb), 'rev'),
        'ab': (a, b, 'plain'), 'ba': (b, a, 'plain'),
    }
    if domain not in ops: return None
    L, R, outer = ops[domain]
    if inner_op == '+': val = L + R
    elif inner_op == '-': val = L - R
    elif inner_op == '*': val = L * R
    elif inner_op == '%':
        if R == 0: return None
        val = L % R
    elif inner_op == '||':
        s = str(L) + str(R)
        return s[::-1] if outer == 'rev' else s
    else: return None
    val += adj
    if outer == 'rev':
        neg = val < 0; r = str(abs(val))[::-1]
        return ('-' + r) if neg else r
    return str(val)

def matches(computed, shown, op_char=''):
    if computed is None: return False
    if computed == shown: return True
    try:
        if int(computed) == int(shown): return True
    except: pass
    clean = shown; sign = 1
    if clean.startswith('-'): sign = -1; clean = clean[1:]
    elif op_char and len(op_char)==1 and clean.startswith(op_char): sign = -1; clean = clean[len(op_char):]
    if op_char and len(op_char)==1 and clean.endswith(op_char): clean = clean[:-len(op_char)]
    if not clean: return False
    try:
        if int(computed) == sign * int(clean): return True
    except: pass
    return False

DOMAINS = ['revab', 'revba', 'maxmin', 'maxmin_r', 'ab', 'ba', 'minmax', 'minmax_r']
INNER_OPS = ['+', '-', '*', '%', '||']
ADJS = [0, 1, -1]

def get_valid_ops(eqs, domain):
    """For a set of equations with the same visible op, find all (inner_op, adj) valid in domain."""
    valid = []
    for op in INNER_OPS:
        for adj in ADJS:
            if op in ('%', '||') and adj != 0: continue
            if all(matches(compute(e['a'],e['b'],domain,op,adj), e['result'], e['op']) for e in eqs):
                valid.append((op, adj))
    return valid

def solve(examples, q_a, q_b, q_op):
    """
    Return list of (answer, domain, inner_op, adj, confidence_score).
    """
    by_visop = defaultdict(list)
    for ex in examples: by_visop[ex['op']].append(ex)
    visops = list(by_visop.keys())

    results = []

    for domain in DOMAINS:
        # Step 1: For each visible op, find valid (inner_op, adj) in this domain
        visop_valid = {}
        ok = True
        for vop in visops:
            v = get_valid_ops(by_visop[vop], domain)
            if not v: ok = False; break
            visop_valid[vop] = v
        if not ok: continue

        # Step 2: Find all CONSISTENT assignments where different visops get different BASE ops
        # (Simple approach: enumerate assignments for 1-3 visops)
        def enumerate_assignments(visop_list, idx, used_base, assignment):
            if idx == len(visop_list):
                yield dict(assignment)
                return
            vop = visop_list[idx]
            for (op, adj) in visop_valid[vop]:
                if op in used_base: continue
                assignment[vop] = (op, adj)
                yield from enumerate_assignments(visop_list, idx+1, used_base | {op}, assignment)
                del assignment[vop]

        for assignment in enumerate_assignments(visops, 0, set(), {}):
            used_bases = set(op for op, adj in assignment.values())

            # Step 3: Query gets an UNUSED base op
            unused = [op for op in INNER_OPS if op not in used_bases]
            for q_inner in unused:
                for q_adj in ADJS:
                    if q_inner in ('%', '||') and q_adj != 0: continue
                    ans = compute(q_a, q_b, domain, q_inner, q_adj)
                    if ans is not None:
                        # Score: prefer adj=0, prefer fewer assignments
                        score = (0 if q_adj == 0 else 1)
                        results.append((ans, domain, q_inner, q_adj, score))

    return results

# Load puzzles
puzzles = []
with open('train_cot_equation_numeric_guess.jsonl') as f:
    for line in f:
        d = json.loads(line)
        user = d['messages'][0]['content']
        asst = d['messages'][1]['content']
        lines = user.split('\n'); examples = []; query_str = None
        for l in lines:
            l = l.strip()
            det = re.match(r'Now, determine the result for:\s*(.+)', l)
            if det: query_str = det.group(1).strip(); continue
            eq = re.match(r'^(\d+)([^\d]+?)(\d+)\s*=\s*(\S+)', l)
            if eq: examples.append({'a':eq.group(1),'op':eq.group(2),'b':eq.group(3),'result':eq.group(4)})
        qm = re.match(r'^(\d+)([^\d]+?)(\d+)', query_str) if query_str else None
        boxed = re.findall(r'boxed\{([^}]+)\}', asst)
        answer = boxed[-1] if boxed else None
        if qm and answer:
            puzzles.append({'examples':examples,'q_a':qm.group(1),'q_op':qm.group(2),'q_b':qm.group(3),'answer':answer})

print(f"Loaded {len(puzzles)} puzzles\n")

# Test
correct = 0; wrong = 0; n_unique = 0
answer_counts = []

for i, p in enumerate(puzzles):
    results = solve(p['examples'], p['q_a'], p['q_b'], p['q_op'])
    actual = p['answer']

    # Deduplicate by answer
    unique_answers = {}
    for ans, domain, op, adj, score in results:
        if ans not in unique_answers or score < unique_answers[ans][1]:
            unique_answers[ans] = ((domain, op, adj), score)

    # Check if correct answer is in candidates
    matching = [(ans, info) for ans, info in unique_answers.items()
                if matches(ans, actual, p['q_op'])]

    n_cands = len(unique_answers)
    answer_counts.append(n_cands)

    if matching:
        correct += 1
        if n_cands == 1:
            n_unique += 1
        if n_cands > 1 and i < 20:
            print(f"  P{i+1:3d}: ✓ {n_cands:2d} candidates | correct={actual:>8s}")
            for ans, (info, score) in sorted(unique_answers.items(), key=lambda x: x[1][1]):
                mark = " ◄" if matches(ans, actual, p['q_op']) else ""
                print(f"         {ans:>10s} domain={info[0]:>10s} op={info[1]}{'+1' if info[2]==1 else '-1' if info[2]==-1 else '':>3s} score={score}{mark}")
    else:
        wrong += 1
        print(f"  P{i+1:3d}: ✗ MISSED | actual={actual}")

print(f"\n{'='*80}")
print(f"CORRECT:    {correct}/{len(puzzles)} ({correct*100//len(puzzles)}%)")
print(f"UNIQUE:     {n_unique} (only 1 candidate answer)")
print(f"WRONG:      {wrong}")
print(f"Avg candidates: {sum(answer_counts)/len(answer_counts):.1f}")
print(f"{'='*80}")

# Distribution of candidate counts
cc = Counter(answer_counts)
print(f"\nCandidate count distribution:")
for n, cnt in sorted(cc.items()):
    print(f"  {n:3d} candidates: {cnt:3d} puzzles")
