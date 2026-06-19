#!/usr/bin/env python3
"""
Analyze the disambiguation problem:
When multiple candidate answers exist, what rule picks the correct one?

The failures fall into categories:
1. Concatenation (||) not found because it was already "used" by examples
2. Negative sign handling with operator-as-sign
3. Domain ambiguity (revab vs maxmin_r etc.)

Let's fix these and find the disambiguation pattern.
"""
import json, re
from collections import defaultdict, Counter

def rev_int(s): return int(s[::-1])

def compute(a_str, b_str, domain, inner_op, adj):
    a, b = int(a_str), int(b_str)
    ra, rb = rev_int(a_str), rev_int(b_str)
    ops = {
        'revab':    (ra, rb, 'rev'),
        'revba':    (rb, ra, 'rev'),
        'maxmin':   (max(a,b), min(a,b), 'plain'),
        'minmax':   (min(a,b), max(a,b), 'plain'),
        'maxmin_r': (max(ra,rb), min(ra,rb), 'rev'),
        'minmax_r': (min(ra,rb), max(ra,rb), 'rev'),
        'ab':       (a, b, 'plain'),
        'ba':       (b, a, 'plain'),
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
        neg = val < 0
        r = str(abs(val))[::-1]
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

def all_templates():
    for d in DOMAINS:
        for op in INNER_OPS:
            for adj in ADJS:
                if op in ('%', '||') and adj != 0: continue
                yield (d, op, adj)

# Load puzzles
puzzles = []
with open('train_cot_equation_numeric_guess.jsonl') as f:
    for line in f:
        d = json.loads(line)
        user = d['messages'][0]['content']
        asst = d['messages'][1]['content']
        lines = user.split('\n')
        examples = []; query_str = None
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

def solve_full(examples, q_a, q_b, q_op):
    """Return ALL candidate (answer, domain, inner_op, adj, is_unused)."""
    by_visop = defaultdict(list)
    for ex in examples: by_visop[ex['op']].append(ex)

    visop_templates = {}
    for vop, eqs in by_visop.items():
        valid = set()
        for tmpl in all_templates():
            d, op, adj = tmpl
            if all(matches(compute(e['a'],e['b'],d,op,adj), e['result'], e['op']) for e in eqs):
                valid.add(tmpl)
        visop_templates[vop] = valid

    shared_domains = None
    for vop, tmpls in visop_templates.items():
        domains = set(d for d,op,adj in tmpls)
        shared_domains = domains if shared_domains is None else (shared_domains & domains)
    if not shared_domains: return []

    candidates = []
    for domain in shared_domains:
        used = set()
        for vop, tmpls in visop_templates.items():
            for d,op,adj in tmpls:
                if d == domain: used.add((op,adj))
        used_base = set(op for op,adj in used)

        for op in INNER_OPS:
            for adj in ADJS:
                if op in ('%','||') and adj != 0: continue
                is_unused = op not in used_base
                result = compute(q_a, q_b, domain, op, adj)
                if result is not None:
                    candidates.append({
                        'answer': result, 'domain': domain,
                        'inner_op': op, 'adj': adj,
                        'is_unused': is_unused
                    })
    return candidates

# ============================================================================
# ANALYZE: For correct predictions, which candidate properties distinguish
# the correct answer?
# ============================================================================

print("="*120)
print("DISAMBIGUATION ANALYSIS: What makes the correct candidate unique?")
print("="*120)

# For each puzzle, find which candidate(s) match, and analyze their properties
domain_rank = Counter()
op_rank = Counter()
adj_rank = Counter()
unused_rank = Counter()

for i, p in enumerate(puzzles):
    cands = solve_full(p['examples'], p['q_a'], p['q_b'], p['q_op'])
    correct = [c for c in cands if matches(c['answer'], p['answer'], p['q_op'])]
    if not correct: continue

    c = correct[0]
    domain_rank[c['domain']] += 1
    op_rank[c['inner_op']] += 1
    adj_rank[c['adj']] += 1
    unused_rank[c['is_unused']] += 1

    # How many candidates total?
    n_total = len(cands)
    n_unused = len([c for c in cands if c['is_unused']])

    # Among unused candidates only, how many match?
    unused_correct = [c for c in cands if c['is_unused'] and matches(c['answer'], p['answer'], p['q_op'])]

print(f"\nCorrect answer properties:")
print(f"  Domain: {domain_rank.most_common()}")
print(f"  Inner op: {op_rank.most_common()}")
print(f"  Adjustment: {adj_rank.most_common()}")
print(f"  Is unused (not in examples): {unused_rank.most_common()}")

# ============================================================================
# KEY: Can we reduce candidates by DOMAIN PRIORITY?
# ============================================================================

print("\n" + "="*120)
print("DOMAIN PRIORITY: When multiple domains are valid, which is correct?")
print("="*120)

domain_winner = Counter()
domain_coexist = Counter()

for i, p in enumerate(puzzles):
    cands = solve_full(p['examples'], p['q_a'], p['q_b'], p['q_op'])
    correct = [c for c in cands if matches(c['answer'], p['answer'], p['q_op'])]
    if not correct: continue

    all_domains = set(c['domain'] for c in cands)
    correct_domain = correct[0]['domain']

    for d in all_domains:
        if d == correct_domain:
            domain_winner[d] += 1
        else:
            domain_coexist[(d, correct_domain)] += 1

print(f"\nWinning domains: {domain_winner.most_common()}")
print(f"\nDomain rivalries (loser, winner):")
for (loser, winner), cnt in domain_coexist.most_common(20):
    print(f"  {loser:>10s} lost to {winner:>10s} : {cnt}x")

# ============================================================================
# EQUIVALENCE CLASSES
# ============================================================================

print("\n" + "="*120)
print("DOMAIN EQUIVALENCE: Which domains are mathematically equivalent?")
print("="*120)

# For domains that always co-occur, check if they give the same answers
equiv_cases = Counter()
for i, p in enumerate(puzzles[:50]):
    cands = solve_full(p['examples'], p['q_a'], p['q_b'], p['q_op'])
    by_answer = defaultdict(set)
    for c in cands:
        by_answer[c['answer']].add((c['domain'], c['inner_op'], c['adj']))

    for ans, templates in by_answer.items():
        if len(templates) > 1:
            domains = set(d for d,op,adj in templates)
            if len(domains) > 1:
                equiv_cases[tuple(sorted(domains))] += 1

print("Domain pairs that produce identical answers:")
for pair, cnt in equiv_cases.most_common(15):
    print(f"  {pair}: {cnt}x")

# ============================================================================
# FINAL: Build the complete disambiguation rule
# ============================================================================

print("\n" + "="*120)
print("FINAL TEST: Apply domain priority + unused-only filter")
print("="*120)

# Domain priority based on frequency analysis
DOMAIN_PRIORITY = {
    'revab': 0, 'maxmin_r': 1, 'revba': 2, 'minmax_r': 3,
    'maxmin': 4, 'ab': 5, 'minmax': 6, 'ba': 7
}

correct_count = 0
unique_count = 0
total = len(puzzles)

for i, p in enumerate(puzzles):
    cands = solve_full(p['examples'], p['q_a'], p['q_b'], p['q_op'])

    # Filter 1: prefer unused ops
    unused = [c for c in cands if c['is_unused']]
    pool = unused if unused else cands

    # Filter 2: domain priority (pick highest priority domain)
    if pool:
        best_domain_score = min(DOMAIN_PRIORITY.get(c['domain'], 99) for c in pool)
        pool = [c for c in pool if DOMAIN_PRIORITY.get(c['domain'], 99) == best_domain_score]

    # Filter 3: prefer adj=0, then +1, then -1
    adj_prio = {0: 0, 1: 1, -1: 2}
    if pool:
        best_adj = min(adj_prio.get(c['adj'], 99) for c in pool)
        pool = [c for c in pool if adj_prio.get(c['adj'], 99) == best_adj]

    predicted = pool[0]['answer'] if pool else None
    actual = p['answer']

    is_correct = matches(predicted, actual, p['q_op']) if predicted else False

    if is_correct:
        correct_count += 1
        if len(pool) == 1:
            unique_count += 1
    else:
        print(f"  P{i+1:3d}: ✗ predicted={predicted:>8s} actual={actual:>8s} "
              f"domain={pool[0]['domain'] if pool else 'N/A'} "
              f"op={pool[0]['inner_op'] if pool else 'N/A'}")

print(f"\n{'='*60}")
print(f"CORRECT: {correct_count}/{total} ({correct_count*100//total}%)")
print(f"UNIQUE:  {unique_count}")
print(f"WRONG:   {total - correct_count}")
print(f"{'='*60}")
