#!/usr/bin/env python3
"""
THE ACTUAL SOLVER: Given ONLY example lines, predict the query answer.
Tests against all 136 puzzles from the JSONL file.

Algorithm:
1. Try all domains on examples → find which domains explain ALL examples
2. Within each valid domain, find which inner_ops are "used" by examples
3. For the query: try all UNUSED (domain, inner_op, adj) combos
4. Collect all candidate answers
5. If exactly 1 candidate → that's the answer
   If multiple → we need a disambiguation rule
"""
import json
import re
from collections import defaultdict

# ============================================================================
# COMPUTATION ENGINE
# ============================================================================

def rev_int(s):
    return int(s[::-1])

def compute(a_str, b_str, domain, inner_op, adj):
    a, b = int(a_str), int(b_str)
    ra, rb = rev_int(a_str), rev_int(b_str)

    # Operands
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

    # Inner operation
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
    """Check if computed matches the shown answer, handling display quirks."""
    if computed is None: return False
    if computed == shown: return True
    # Numeric equality (handles zero-padding)
    try:
        if int(computed) == int(shown): return True
    except: pass
    # Op-as-sign: shown starts with op_char (negative) or ends with it (artifact)
    for s in [shown]:
        clean = s
        sign = 1
        if clean.startswith('-'):
            sign = -1; clean = clean[1:]
        elif op_char and len(op_char)==1 and clean.startswith(op_char):
            sign = -1; clean = clean[len(op_char):]
        if op_char and len(op_char)==1 and clean.endswith(op_char):
            clean = clean[:-len(op_char)]
        if not clean: continue
        try:
            if int(computed) == sign * int(clean): return True
        except: pass
    return False


# ============================================================================
# CANDIDATE SPACE
# ============================================================================

DOMAINS = ['revab', 'revba', 'maxmin', 'maxmin_r', 'ab', 'ba', 'minmax', 'minmax_r']
INNER_OPS = ['+', '-', '*', '%', '||']
ADJS = [0, 1, -1]

def all_templates():
    for d in DOMAINS:
        for op in INNER_OPS:
            for adj in ADJS:
                if op in ('%', '||') and adj != 0: continue
                yield (d, op, adj)


# ============================================================================
# PARSE JSONL
# ============================================================================

puzzles = []
with open('train_cot_equation_numeric_guess.jsonl') as f:
    for line in f:
        d = json.loads(line)
        user = d['messages'][0]['content']
        asst = d['messages'][1]['content']

        # Extract examples and query from user message
        lines = user.split('\n')
        examples = []
        query_str = None
        for l in lines:
            l = l.strip()
            det = re.match(r'Now, determine the result for:\s*(.+)', l)
            if det:
                query_str = det.group(1).strip()
                continue
            eq = re.match(r'^(\d+)([^\d]+?)(\d+)\s*=\s*(\S+)', l)
            if eq:
                examples.append({
                    'a': eq.group(1), 'op': eq.group(2),
                    'b': eq.group(3), 'result': eq.group(4)
                })

        # Extract query a, op, b
        qm = re.match(r'^(\d+)([^\d]+?)(\d+)', query_str) if query_str else None

        # Extract boxed answer
        boxed = re.findall(r'boxed\{([^}]+)\}', asst)
        answer = boxed[-1] if boxed else None

        if qm and answer:
            puzzles.append({
                'examples': examples,
                'q_a': qm.group(1), 'q_op': qm.group(2), 'q_b': qm.group(3),
                'answer': answer,
                'query_str': query_str
            })

print(f"Loaded {len(puzzles)} puzzles")

# ============================================================================
# SOLVER
# ============================================================================

def solve(examples, q_a, q_b, q_op):
    """
    Given example equations and query operands, return list of candidate answers.
    """
    # Group examples by visible op
    by_visop = defaultdict(list)
    for ex in examples:
        by_visop[ex['op']].append(ex)

    # Step 1: For each visible op, find all (domain, inner_op, adj) that explain ALL lines
    visop_templates = {}
    for vop, eqs in by_visop.items():
        valid = set()
        for tmpl in all_templates():
            d, op, adj = tmpl
            if all(matches(compute(e['a'], e['b'], d, op, adj), e['result'], e['op']) for e in eqs):
                valid.add(tmpl)
        visop_templates[vop] = valid

    # Step 2: Find domains that work for ALL visible ops
    shared_domains = None
    for vop, tmpls in visop_templates.items():
        domains = set(d for d, op, adj in tmpls)
        shared_domains = domains if shared_domains is None else (shared_domains & domains)
    if not shared_domains:
        return []

    # Step 3: For each shared domain, find which inner_ops are used by examples
    candidates = []
    for domain in shared_domains:
        used_ops = set()
        for vop, tmpls in visop_templates.items():
            for d, op, adj in tmpls:
                if d == domain:
                    used_ops.add(op)

        # Step 4: Try all UNUSED inner_ops for the query
        for inner_op in INNER_OPS:
            if inner_op in used_ops:
                continue
            for adj in ADJS:
                if inner_op in ('%', '||') and adj != 0: continue
                result = compute(q_a, q_b, domain, inner_op, adj)
                if result is not None:
                    candidates.append({
                        'answer': result,
                        'domain': domain,
                        'inner_op': inner_op,
                        'adj': adj
                    })

    # Also try used ops (for edge cases — ~2%)
    for domain in shared_domains:
        for inner_op in INNER_OPS:
            for adj in ADJS:
                if inner_op in ('%', '||') and adj != 0: continue
                result = compute(q_a, q_b, domain, inner_op, adj)
                if result is not None:
                    already = any(c['answer'] == result for c in candidates)
                    if not already:
                        candidates.append({
                            'answer': result,
                            'domain': domain,
                            'inner_op': inner_op,
                            'adj': adj,
                            'fallback': True
                        })
    return candidates


# ============================================================================
# TEST
# ============================================================================

print("\n" + "="*120)
print("TESTING SOLVER ON ALL 136 PUZZLES")
print("="*120)

correct = 0
multi_correct = 0
wrong = 0

for i, p in enumerate(puzzles):
    candidates = solve(p['examples'], p['q_a'], p['q_b'], p['q_op'])
    actual = p['answer']

    # Check if actual answer is among candidates
    matching = [c for c in candidates if matches(c['answer'], actual, p['q_op'])]
    non_fallback = [c for c in matching if not c.get('fallback')]

    if non_fallback:
        correct += 1
        total_non_fb = [c for c in candidates if not c.get('fallback')]
        if len(total_non_fb) == 1:
            status = "✓ UNIQUE"
        else:
            status = f"✓ FOUND (1 of {len(total_non_fb)} candidates)"
            multi_correct += 1
        if i < 30 or len(total_non_fb) > 1:
            c = non_fallback[0]
            print(f"  P{i+1:3d}: {status:30s} ans={actual:>8s} domain={c['domain']:>10s} op={c['inner_op']}{'+1' if c['adj']==1 else '-1' if c['adj']==-1 else '':>3s} | candidates={len(total_non_fb)}")
    elif matching:
        correct += 1
        status = "✓ FALLBACK"
        c = matching[0]
        print(f"  P{i+1:3d}: {status:30s} ans={actual:>8s} domain={c['domain']:>10s} op={c['inner_op']}")
    else:
        wrong += 1
        print(f"  P{i+1:3d}: ✗ MISSED                       ans={actual:>8s} query={p['query_str']}")
        # Debug: show what candidates we got
        if candidates:
            for c in candidates[:5]:
                print(f"         candidate: {c['answer']:>8s} domain={c['domain']:>10s} op={c['inner_op']}{'+1' if c['adj']==1 else '-1' if c['adj']==-1 else ''}")

print(f"\n{'='*60}")
print(f"CORRECT: {correct}/{len(puzzles)} ({correct*100//len(puzzles)}%)")
print(f"  Unique prediction: {correct - multi_correct}")
print(f"  Multiple candidates (need disambiguation): {multi_correct}")
print(f"WRONG: {wrong}")
print(f"{'='*60}")
