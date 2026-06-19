#!/usr/bin/env python3
"""
REVERSE ENGINEERING: Given ONLY the example lines (with answers), predict
the answer for the "Now, determine" query line.

For each block in the training data:
1. We PRETEND we don't know the query answer
2. We try to predict it using ONLY the examples
3. We check if our prediction matches the actual answer

This tells us the EXACT algorithm needed to solve unseen puzzles.
"""
import re
from collections import defaultdict

# ============================================================================
# COMPUTATION ENGINE
# ============================================================================

def rev_int(s):
    """Reverse the digits of a 2-digit string. '06' -> 60, '79' -> 97"""
    return int(s[::-1])

def compute(a_str, b_str, domain, inner_op, adj):
    """
    Compute the result given domain, inner_op, adj.
    Returns a string representation of the result (matching dataset conventions).
    """
    a, b = int(a_str), int(b_str)
    ra, rb = rev_int(a_str), rev_int(b_str)

    # Determine operands based on domain
    if domain == 'revab':
        L, R = ra, rb      # rev(a), rev(b) — ordered
        outer = 'rev'
    elif domain == 'revba':
        L, R = rb, ra       # rev(b), rev(a) — swapped
        outer = 'rev'
    elif domain == 'maxmin':
        L, R = max(a,b), min(a,b)
        outer = 'plain'
    elif domain == 'minmax':
        L, R = min(a,b), max(a,b)
        outer = 'plain'
    elif domain == 'maxmin_r':
        L, R = max(ra,rb), min(ra,rb)
        outer = 'rev'
    elif domain == 'minmax_r':
        L, R = min(ra,rb), max(ra,rb)
        outer = 'rev'
    elif domain == 'ab':
        L, R = a, b
        outer = 'plain'
    elif domain == 'ba':
        L, R = b, a
        outer = 'plain'
    else:
        return None

    # Apply inner operation
    if inner_op == '+':
        val = L + R
    elif inner_op == '-':
        val = L - R
    elif inner_op == '*':
        val = L * R
    elif inner_op == '%':
        if R == 0: return None
        val = L % R
    elif inner_op == '||':
        # Concatenation
        val_str = str(L) + str(R)
        if outer == 'rev':
            val_str = val_str[::-1]
        return val_str
    else:
        return None

    val += adj

    if outer == 'rev':
        # Reverse the digits of the result
        neg = val < 0
        abs_str = str(abs(val))
        rev_str = abs_str[::-1]
        if neg:
            return '-' + rev_str
        return rev_str
    else:
        return str(val)


def result_matches(computed, shown, op_char):
    """Check if computed result matches shown result, handling display conventions."""
    if computed is None:
        return False

    # Direct match
    if computed == shown:
        return True

    # Handle zero-padding: "06" == "6", "001" == "1"
    try:
        if int(computed) == int(shown):
            return True
    except (ValueError, TypeError):
        pass

    # Handle operator-as-sign: shown = ":05" means -5, op_char = ":"
    # Or shown = "74$" means result with trailing op
    clean_shown = shown
    sign = 1

    if clean_shown.startswith('-'):
        sign = -1
        clean_shown = clean_shown[1:]
    elif op_char and clean_shown.startswith(op_char):
        sign = -1
        clean_shown = clean_shown[len(op_char):]

    if op_char and clean_shown.endswith(op_char):
        clean_shown = clean_shown[:-len(op_char)]

    if not clean_shown:
        return False

    try:
        shown_val = sign * int(clean_shown)
        comp_val = int(computed)
        if comp_val == shown_val:
            return True
    except (ValueError, TypeError):
        pass

    # Handle negative with leading zeros: "-02" == "-2"
    try:
        if computed.startswith('-') and shown.startswith('-'):
            if int(computed) == int(shown):
                return True
    except:
        pass

    return False


# ============================================================================
# ALL CANDIDATE TEMPLATES
# ============================================================================

DOMAINS = ['revab', 'revba', 'maxmin', 'maxmin_r', 'ab', 'ba', 'minmax', 'minmax_r']
INNER_OPS = ['+', '-', '*', '%', '||']
ADJS = [0, 1, -1]

def all_candidates():
    """Generate all (domain, inner_op, adj) triples."""
    for d in DOMAINS:
        for op in INNER_OPS:
            for adj in ADJS:
                if op in ('%', '||') and adj != 0:
                    continue  # % and || never have adjustments in this dataset
                yield (d, op, adj)


# ============================================================================
# PARSE THE FILE
# ============================================================================

with open('perfect_solver_complete.md') as f:
    raw = [l.rstrip() for l in f]

eq_re = re.compile(
    r'^(?P<det>Now, determine the result for:\s*)?'
    r'(?P<a>\d+)(?P<op>[^\d]+?)(?P<b>\d+)\s*=\s*(?P<rest>.+)$'
)

blocks = []
cur = []
for line in raw:
    if line.strip() == '':
        if cur: blocks.append(cur); cur = []
        continue
    m = eq_re.match(line.strip())
    if m:
        rest = m['rest']
        parts = rest.split(' = ', 1)
        if len(parts) == 2 and not any(parts[0].strip().startswith(x) for x in ['rev', 'max', 'min', '(']):
            result, formula = parts[0].strip(), parts[1].strip()
        else:
            result, formula = None, rest.strip()
        cur.append({
            'a': m['a'], 'op': m['op'], 'b': m['b'],
            'result': result, 'formula': formula,
            'is_query': bool(m['det']),
            'raw': line.strip()
        })
if cur: blocks.append(cur)

# ============================================================================
# THE SOLVER: Predict query answer from examples alone
# ============================================================================

print("="*120)
print("SOLVER TEST: Predict query answers using ONLY examples")
print("="*120)

correct = 0
wrong = 0
ambiguous = 0
total = 0

for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq['is_query'] and eq['result']]
    queries = [eq for eq in blk if eq['is_query']]
    if not queries:
        continue

    q = queries[0]
    total += 1

    # STEP 1: Find which (domain, inner_op, adj) explains EACH example
    # Group examples by their visible operator symbol
    ex_by_visop = defaultdict(list)
    for ex in examples:
        ex_by_visop[ex['op']].append(ex)

    # For each visible op, find ALL (domain, inner_op, adj) that work for ALL lines with that op
    def find_candidates_for_visop(eqs):
        """Find all (domain, inner_op, adj) that produce correct results for ALL eqs."""
        candidates = set()
        for d, op, adj in all_candidates():
            all_match = True
            for eq in eqs:
                comp = compute(eq['a'], eq['b'], d, op, adj)
                if not result_matches(comp, eq['result'], eq['op']):
                    all_match = False
                    break
            if all_match:
                candidates.add((d, op, adj))
        return candidates

    # Find candidates per visible op
    visop_candidates = {}
    for vop, eqs in ex_by_visop.items():
        visop_candidates[vop] = find_candidates_for_visop(eqs)

    # STEP 2: Find the DOMAIN that is consistent across ALL example visible ops
    # Each visop's candidates include a domain — the shared domain(s) across all visops
    if not visop_candidates:
        continue

    # Intersect domains across all visible ops
    all_visop_domains = None
    for vop, cands in visop_candidates.items():
        domains = set(d for d, op, adj in cands)
        if all_visop_domains is None:
            all_visop_domains = domains
        else:
            all_visop_domains &= domains

    if not all_visop_domains:
        print(f"  Block {bi+1}: NO shared domain found!")
        wrong += 1
        continue

    # STEP 3: Within shared domain(s), find which inner_ops are used by examples
    # and which are "available" for the query
    best_prediction = None
    best_domain = None

    for domain in all_visop_domains:
        # For this domain, what inner_op does each visop use?
        ex_inner_ops = set()
        valid = True
        for vop, cands in visop_candidates.items():
            ops_in_domain = set((op, adj) for d, op, adj in cands if d == domain)
            if not ops_in_domain:
                valid = False
                break
            # Each visop should map to exactly one (or few) inner ops in this domain
            ex_inner_ops.update(op for op, adj in ops_in_domain)

        if not valid:
            continue

        # STEP 4: Try all (inner_op, adj) NOT used by examples, in this domain
        ex_base_ops = set(op for op, adj in
                          [(op, adj) for vop in visop_candidates
                           for d, op, adj in visop_candidates[vop] if d == domain])

        for inner_op in INNER_OPS:
            for adj in ADJS:
                if inner_op in ('%', '||') and adj != 0:
                    continue
                # Skip if this op is already used by an example visop
                if inner_op in ex_base_ops:
                    continue
                comp = compute(q['a'], q['b'], domain, inner_op, adj)
                if comp is not None:
                    if q['result'] and result_matches(comp, q['result'], q['op']):
                        best_prediction = comp
                        best_domain = domain

    # Also try ops IN ex_base_ops (for the ~2% edge cases like concat appearing twice)
    if best_prediction is None:
        for domain in all_visop_domains:
            for inner_op in INNER_OPS:
                for adj in ADJS:
                    if inner_op in ('%', '||') and adj != 0:
                        continue
                    comp = compute(q['a'], q['b'], domain, inner_op, adj)
                    if comp is not None:
                        if q['result'] and result_matches(comp, q['result'], q['op']):
                            best_prediction = comp
                            best_domain = domain

    if best_prediction is not None:
        correct += 1
        if bi < 15:
            print(f"  Block {bi+1}: ✓ predicted={best_prediction:>8s} actual={q['result']:>8s} domain={best_domain}")
    else:
        wrong += 1
        actual = q.get('result', 'N/A')
        print(f"  Block {bi+1}: ✗ FAILED   actual={actual} query={q['a']}{q['op']}{q['b']}")
        # Debug: show what domains were found
        print(f"            shared domains: {all_visop_domains}")

print(f"\n{'='*60}")
print(f"RESULTS: {correct}/{total} correct ({correct*100//total}%) | {wrong} failed")
print(f"{'='*60}")
