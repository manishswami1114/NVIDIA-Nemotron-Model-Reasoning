#!/usr/bin/env python3
"""
Analyze the GAP in the current CoT:
For each puzzle, how many candidate operations could the query use?
If there's only 1 possible answer → CoT doesn't need to show elimination
If there are multiple → CoT MUST show why the others were rejected

This tells us exactly what the CoT is missing.
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
            puzzles.append({'examples':examples,'q_a':qm.group(1),'q_op':qm.group(2),
                          'q_b':qm.group(3),'answer':answer,'query_str':query_str})

print(f"Analyzing {len(puzzles)} puzzles\n")

# For each puzzle, find ALL possible candidate answers
print("="*100)
print("PUZZLE-BY-PUZZLE: How many candidate answers exist?")
print("What must the CoT SHOW to disambiguate?")
print("="*100)

ambiguity_stats = []

for i, p in enumerate(puzzles):
    by_visop = defaultdict(list)
    for ex in p['examples']: by_visop[ex['op']].append(ex)

    # Find which (domain, inner_op, adj) works for each example visop
    visop_valid = {}
    for vop, eqs in by_visop.items():
        valid = set()
        for d in DOMAINS:
            for op in INNER_OPS:
                for adj in ADJS:
                    if op in ('%','||') and adj != 0: continue
                    if all(matches(compute(e['a'],e['b'],d,op,adj), e['result'], e['op']) for e in eqs):
                        valid.add((d, op, adj))
        visop_valid[vop] = valid

    # Find shared domains
    shared = None
    for vop, tmpls in visop_valid.items():
        doms = set(d for d,o,a in tmpls)
        shared = doms if shared is None else (shared & doms)
    if not shared: shared = set()

    # For query: try ALL (domain, inner_op, adj) and collect unique answers
    all_answers = {}  # answer → list of (domain, inner_op, adj)
    for d in shared:
        # Find used base ops in this domain
        used = set()
        for vop, tmpls in visop_valid.items():
            for dd, op, adj in tmpls:
                if dd == d: used.add(op)

        for op in INNER_OPS:
            for adj in ADJS:
                if op in ('%','||') and adj != 0: continue
                if op in used: continue  # only unused ops
                ans = compute(p['q_a'], p['q_b'], d, op, adj)
                if ans is not None:
                    all_answers.setdefault(ans, []).append((d, op, adj))

    # Check which answer matches actual
    correct_ans = None
    for ans in all_answers:
        if matches(ans, p['answer'], p['q_op']):
            correct_ans = ans
            break

    n_candidates = len(all_answers)
    ambiguity_stats.append(n_candidates)

    # Show details for first 10 puzzles
    if i < 10:
        print(f"\nPuzzle {i+1}: query={p['q_a']}{p['q_op']}{p['q_b']} | actual={p['answer']} | {n_candidates} candidate answers")
        print(f"  Shared domains: {shared}")
        print(f"  Example ops decoded:")
        for vop, tmpls in visop_valid.items():
            # Show in first shared domain
            for d in sorted(shared):
                ops_in_d = [(op,adj) for dd,op,adj in tmpls if dd == d]
                if ops_in_d:
                    print(f"    '{vop}' in {d}: {ops_in_d}")
                    break
        print(f"  ALL candidate answers for query:")
        for ans, templates in sorted(all_answers.items(), key=lambda x: -len(x[1])):
            mark = " ◄◄◄ CORRECT" if ans == correct_ans else ""
            print(f"    {ans:>10s} ← {templates[0]}{mark}")

# Summary
print("\n" + "="*100)
print("AMBIGUITY SUMMARY")
print("="*100)
cc = Counter(ambiguity_stats)
for n, cnt in sorted(cc.items()):
    pct = cnt * 100 // len(puzzles)
    bar = "█" * (cnt // 2)
    print(f"  {n:3d} candidate answers: {cnt:3d} puzzles ({pct:2d}%) {bar}")

n_unique = sum(1 for x in ambiguity_stats if x == 1)
n_ambig = sum(1 for x in ambiguity_stats if x > 1)
print(f"\n  UNIQUE (1 answer):   {n_unique} puzzles → CoT is fine")
print(f"  AMBIGUOUS (>1):      {n_ambig} puzzles → CoT MUST show elimination reasoning")
print(f"  Average candidates:  {sum(ambiguity_stats)/len(ambiguity_stats):.1f}")

# The key question: WHAT distinguishes the correct answer?
print("\n" + "="*100)
print("WHAT DISTINGUISHES THE CORRECT ANSWER?")
print("(Among multiple candidates, what makes the right one right?)")
print("="*100)

adj_pattern = Counter()
for i, p in enumerate(puzzles):
    by_visop = defaultdict(list)
    for ex in p['examples']: by_visop[ex['op']].append(ex)
    visop_valid = {}
    for vop, eqs in by_visop.items():
        valid = set()
        for d in DOMAINS:
            for op in INNER_OPS:
                for adj in ADJS:
                    if op in ('%','||') and adj != 0: continue
                    if all(matches(compute(e['a'],e['b'],d,op,adj), e['result'], e['op']) for e in eqs):
                        valid.add((d, op, adj))
        visop_valid[vop] = valid
    shared = None
    for vop, tmpls in visop_valid.items():
        doms = set(d for d,o,a in tmpls)
        shared = doms if shared is None else (shared & doms)
    if not shared: continue

    all_answers = {}
    for d in shared:
        used = set()
        for vop, tmpls in visop_valid.items():
            for dd, op, adj in tmpls:
                if dd == d: used.add(op)
        for op in INNER_OPS:
            for adj in ADJS:
                if op in ('%','||') and adj != 0: continue
                if op in used: continue
                ans = compute(p['q_a'], p['q_b'], d, op, adj)
                if ans is not None:
                    all_answers.setdefault(ans, []).append((d, op, adj))

    for ans, templates in all_answers.items():
        if matches(ans, p['answer'], p['q_op']):
            d, op, adj = templates[0]
            adj_pattern[adj] += 1
            break

print(f"\nCorrect answer's adjustment: {adj_pattern.most_common()}")
print(f"  adj=0:  {adj_pattern[0]} ({adj_pattern[0]*100//sum(adj_pattern.values())}%) → no adjustment")
print(f"  adj=+1: {adj_pattern[1]} ({adj_pattern[1]*100//sum(adj_pattern.values())}%)")
print(f"  adj=-1: {adj_pattern[-1]} ({adj_pattern[-1]*100//sum(adj_pattern.values())}%)")
