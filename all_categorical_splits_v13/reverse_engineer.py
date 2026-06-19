#!/usr/bin/env python3
"""
Reverse-engineer: given the example lines in a block, how is the query line's
formula determined? What is the RULE that maps examples → query prediction?

For each block we extract:
  - examples: their (visible_op, inner_op, domain)
  - query:    its (visible_op, inner_op, domain)

Then we look for the pattern connecting them.
"""
import re
from collections import Counter, defaultdict

# ============================================================================
# PARSING
# ============================================================================

with open('perfect_solver_complete.md') as f:
    raw = [l.rstrip() for l in f]

eq_re = re.compile(
    r'^(?P<det>Now, determine the result for:\s*)?'
    r'(?P<a>\d+)(?P<op>[^\d]+?)(?P<b>\d+)\s*=\s*(?P<rest>.+)$'
)

def parse_formula(formula):
    """Classify formula → (domain, inner_op, adj, outer, sign)"""
    f = re.sub(r'\s*\[op suffix\]', '', formula).strip()
    nf = re.sub(r'\s+', '', f)

    sign = '+'
    inner = nf

    # Outer sign
    if inner.startswith('-rev('):
        sign = '-'; outer = 'rev'; inner = inner[5:-1] if inner.endswith(')') else inner[5:]
    elif inner.startswith('-('):
        sign = '-'; outer = 'plain'; inner = inner[2:-1] if inner.endswith(')') else inner[2:]
    elif inner.startswith('rev(') and inner.endswith(')'):
        outer = 'rev'; inner = inner[4:-1]
    else:
        outer = 'plain'

    # Adjustment
    adj = 0
    if inner.endswith('+1'): adj = 1; inner = inner[:-2]
    elif inner.endswith('-1'): adj = -1; inner = inner[:-2]

    # Strip outer parens
    while inner.startswith('(') and inner.endswith(')'):
        # Check balanced
        depth = 0; balanced = True
        for i, c in enumerate(inner):
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if depth == 0 and i < len(inner)-1: balanced = False; break
        if balanced: inner = inner[1:-1]
        else: break

    # Identify domain and inner_op
    patterns = [
        (r'max\(rev\(a\),rev\(b\)\)([\+\-\*%]|[\|]{2})min\(rev\(a\),rev\(b\)\)', 'maxmin_r'),
        (r'min\(rev\(a\),rev\(b\)\)([\+\-\*%]|[\|]{2})max\(rev\(a\),rev\(b\)\)', 'minmax_r'),
        (r'max\(a,b\)([\+\-\*%]|[\|]{2})min\(a,b\)', 'maxmin'),
        (r'min\(a,b\)([\+\-\*%]|[\|]{2})max\(a,b\)', 'minmax'),
        (r'rev\(a\)([\+\-\*%]|[\|]{2})rev\(b\)', 'revab'),
        (r'rev\(b\)([\+\-\*%]|[\|]{2})rev\(a\)', 'revba'),
        (r'a([\+\-\*%]|[\|]{2})b$', 'ab'),
        (r'b([\+\-\*%]|[\|]{2})a$', 'ba'),
    ]

    for pat, domain in patterns:
        m = re.match(pat, inner)
        if m:
            return (domain, m.group(1), adj, outer, sign)

    return ('UNPARSED', inner, adj, outer, sign)


def parse_line(line):
    m = eq_re.match(line.strip())
    if not m: return None
    rest = m['rest']
    is_query = bool(m['det'])
    parts = rest.split(' = ', 1)
    if len(parts) == 2 and not parts[0].startswith('rev') and not parts[0].startswith('max') and not parts[0].startswith('(') and not parts[0].startswith('min'):
        result, formula = parts[0].strip(), parts[1].strip()
    else:
        result, formula = None, rest.strip()

    cls = parse_formula(formula)
    return {
        'a': m['a'], 'op': m['op'], 'b': m['b'],
        'result': result, 'formula': formula,
        'is_query': is_query,
        'domain': cls[0], 'inner_op': cls[1], 'adj': cls[2],
        'outer': cls[3], 'sign': cls[4],
        # Compact representation: what the visible op "does"
        'action': f"{cls[1]}{'+1' if cls[2]==1 else '-1' if cls[2]==-1 else ''}"
    }

# Parse into blocks
blocks = []; cur = []
for line in raw:
    if line.strip() == '':
        if cur: blocks.append(cur); cur = []
        continue
    p = parse_line(line)
    if p: cur.append(p)
if cur: blocks.append(cur)

# ============================================================================
# ANALYSIS 1: For each block, what inner ops do examples have, what does query get?
# ============================================================================

print("="*120)
print("BLOCK-BY-BLOCK: Example inner ops → Query inner op")
print("="*120)

transition_counter = Counter()  # (example_ops_tuple, query_op) → count
domain_counter = Counter()

for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq['is_query']]
    queries  = [eq for eq in blk if eq['is_query']]
    if not queries: continue
    q = queries[0]

    # Get unique inner ops from examples (by visible op groups)
    ex_by_visop = defaultdict(list)
    for ex in examples:
        ex_by_visop[ex['op']].append(ex)

    ex_actions = set()
    for vop, eqs in ex_by_visop.items():
        ex_actions.add(eqs[0]['action'])

    q_action = q['action']
    q_domain = q['domain']

    # Check: is query action same as one of the example actions?
    q_in_examples = q_action in ex_actions

    # Block's primary domain
    domain_votes = Counter(eq['domain'] for eq in examples if eq['domain'] != 'UNPARSED')
    primary_domain = domain_votes.most_common(1)[0][0] if domain_votes else 'UNKNOWN'

    ex_sorted = tuple(sorted(ex_actions))
    transition_counter[(ex_sorted, q_action)] += 1
    domain_counter[(primary_domain, q_domain)] += 1

    if bi < 30:  # Show first 30
        print(f"\nBlock {bi+1} | domain: {primary_domain} → query domain: {q_domain}")
        for vop, eqs in ex_by_visop.items():
            print(f"  example '{vop}' → {eqs[0]['action']:6s} (domain={eqs[0]['domain']})")
        print(f"  QUERY   '{q['op']}' → {q_action:6s} (domain={q_domain})")
        print(f"  Query action {'IN' if q_in_examples else 'NOT IN'} examples: {ex_sorted}")

# ============================================================================
# ANALYSIS 2: Does query domain always match example domain?
# ============================================================================

print("\n" + "="*120)
print("DOMAIN CONSISTENCY: Does query always use the same domain as examples?")
print("="*120)

for (ex_dom, q_dom), cnt in domain_counter.most_common():
    match = "✓ SAME" if ex_dom == q_dom else "✗ DIFFERENT"
    print(f"  Example domain={ex_dom:12s} → Query domain={q_dom:12s} | {cnt:3d} blocks | {match}")

# ============================================================================
# ANALYSIS 3: Which inner op does the query get?
# ============================================================================

print("\n" + "="*120)
print("QUERY ACTION ANALYSIS: What inner op does the query use?")
print("="*120)

# For each block, collect the set of example actions and the query action
# Question: is the query action always the "missing" one from a known set?

FULL_OP_SET = {'+', '-', '*', '%', '||'}

missing_pattern = Counter()
for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq['is_query']]
    queries  = [eq for eq in blk if eq['is_query']]
    if not queries: continue
    q = queries[0]

    ex_by_visop = defaultdict(list)
    for ex in examples:
        ex_by_visop[ex['op']].append(ex)

    ex_inner_ops = set()
    for vop, eqs in ex_by_visop.items():
        ex_inner_ops.add(eqs[0]['inner_op'])

    q_inner = q['inner_op']
    q_adj = q['adj']

    # What's "missing" from {+,-,*,%,||}?
    missing = FULL_OP_SET - ex_inner_ops
    q_is_missing = q_inner in missing

    missing_pattern[(tuple(sorted(ex_inner_ops)), q_inner, q_adj, q_is_missing)] += 1

print("\nExample inner ops → Query inner op (adj) | Is it a 'missing' op?")
for (ex_ops, q_op, q_adj, is_missing), cnt in sorted(missing_pattern.items(), key=lambda x: -x[1]):
    adj_s = f"({'+' if q_adj>0 else ''}{q_adj})" if q_adj else ""
    miss = "✓ missing" if is_missing else "✗ NOT missing"
    print(f"  [{cnt:2d}x] examples={set(ex_ops)} → query={q_op}{adj_s:4s} | {miss}")

# ============================================================================
# ANALYSIS 4: The visible operator symbol relationship
# ============================================================================

print("\n" + "="*120)
print("VISIBLE OP SYMBOL: Relationship between example ops and query op")
print("="*120)

# In each block, what visible op symbols are used by examples, and which by query?
# Is the query symbol always new?
query_new_counter = Counter()
for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq['is_query']]
    queries  = [eq for eq in blk if eq['is_query']]
    if not queries: continue

    ex_ops = set(eq['op'] for eq in examples)
    q_op = queries[0]['op']
    is_new = q_op not in ex_ops
    query_new_counter[is_new] += 1

print(f"  Query uses a NEW symbol (not in examples): {query_new_counter[True]}")
print(f"  Query uses an EXISTING symbol:             {query_new_counter[False]}")

# ============================================================================
# ANALYSIS 5: Per-block deep dive — the "operation menu" pattern
# ============================================================================

print("\n" + "="*120)
print("THE OPERATION MENU: Each block offers a 'menu' of operations")
print("="*120)

# For each block, list ALL operations (example + query) as a complete set
menu_counter = Counter()
for bi, blk in enumerate(blocks):
    all_ops_in_block = set()
    for eq in blk:
        if eq['domain'] != 'UNPARSED':
            all_ops_in_block.add(eq['action'])
    menu = tuple(sorted(all_ops_in_block))
    menu_counter[menu] += 1

print("\nComplete operation menus (examples + query combined):")
for menu, cnt in menu_counter.most_common():
    print(f"  [{cnt:2d}x] {set(menu)}")

# ============================================================================
# ANALYSIS 6: Does the # of example ops predict the query op?
# ============================================================================

print("\n" + "="*120)
print("NUMBER OF DISTINCT EXAMPLE OPS vs QUERY OP")
print("="*120)

n_ex_ops_vs_query = defaultdict(Counter)
for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq['is_query']]
    queries  = [eq for eq in blk if eq['is_query']]
    if not queries: continue

    ex_by_visop = defaultdict(list)
    for ex in examples:
        ex_by_visop[ex['op']].append(ex)

    n_distinct = len(set(eqs[0]['action'] for eqs in ex_by_visop.values()))
    q_action = queries[0]['action']
    n_ex_ops_vs_query[n_distinct][q_action] += 1

for n, actions in sorted(n_ex_ops_vs_query.items()):
    total = sum(actions.values())
    print(f"\n  {n} distinct example actions ({total} blocks):")
    for act, cnt in actions.most_common():
        print(f"    query → {act:6s} ({cnt}x)")
