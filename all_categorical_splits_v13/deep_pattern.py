#!/usr/bin/env python3
"""
Deep pattern analysis: discover the UNIVERSAL STRUCTURE governing
how operator symbols map to arithmetic formulas within each block.

KEY HYPOTHESIS: Each block uses ONE of ~6 "template families", and within that
family, each visible operator symbol (-,+,*,||,etc) maps to a specific
arithmetic operation (+,-,*,%,||) CONSISTENTLY WITHIN that family.
"""
import re
from collections import Counter, defaultdict

with open('perfect_solver_complete.md') as f:
    raw = [l.rstrip() for l in f]

# Parse with single-char operator (handle "" as single op)
eq_re = re.compile(
    r'^(?P<det>Now, determine the result for:\s*)?'
    r'(?P<a>\d+)(?P<op>[^\d]{1,2})(?P<b>\d+)\s*=\s*(?P<rest>.+)$'
)

def parse_line(line):
    m = eq_re.match(line.strip())
    if not m: return None
    rest = m['rest']
    is_query = bool(m['det'])
    # Split rest into result and formula
    # For query lines without result: "06@77 = rev(rev(a)+rev(b))"
    # For normal lines: "79-12 = 67 = rev(rev(a)-rev(b))"
    parts = rest.split(' = ', 1)
    if len(parts) == 2 and not parts[0].startswith('rev') and not parts[0].startswith('max') and not parts[0].startswith('('):
        result = parts[0].strip()
        formula = parts[1].strip()
    else:
        result = None
        formula = rest.strip()
    return {
        'a': m['a'], 'op': m['op'], 'b': m['b'],
        'result': result, 'formula': formula,
        'is_query': is_query
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
# NORMALIZE formulas to canonical form
# ============================================================================
def normalize(f):
    f = re.sub(r'\s*\[op suffix\]\s*', '', f)
    f = re.sub(r'\s+', '', f)  # remove all whitespace for comparison
    return f

# ============================================================================
# CLASSIFY each formula into a canonical template
# ============================================================================

def classify_formula(f):
    """Return (operand_domain, inner_op, adjustment, outer_sign)

    operand_domain:
      'ab'       = plain a, b with order (a OP b)
      'ba'       = plain b, a with order (b OP a)
      'maxmin'   = max(a,b), min(a,b) — order-independent plain
      'revab'    = rev(a), rev(b) — reversed, with order
      'revba'    = rev(b), rev(a) — reversed, swapped order
      'maxmin_r' = max(rev(a),rev(b)), min(rev(a),rev(b))
      'minmax_r' = min(rev(a),rev(b)), max(rev(a),rev(b))

    inner_op: '+', '-', '*', '%', '||'
    adjustment: -1, 0, +1
    outer: 'plain', 'rev', '-rev', '-plain'
    """
    nf = normalize(f)

    # Handle annotations
    nf = nf.replace('[opsuffix]', '')

    # Check for outer rev()
    outer = 'plain'
    inner = nf

    if inner.startswith('-rev(') and inner.endswith(')'):
        outer = '-rev'
        inner = inner[5:-1]
    elif inner.startswith('rev(') and inner.endswith(')'):
        outer = 'rev'
        inner = inner[4:-1]
    elif inner.startswith('-(') and inner.endswith(')'):
        outer = '-plain'
        inner = inner[2:-1]

    # Now parse inner expression
    # Patterns to match:
    adj = 0
    # Check for trailing +1 or -1
    if inner.endswith('+1'):
        adj = 1
        inner = inner[:-2]
    elif inner.endswith('-1'):
        adj = -1
        inner = inner[:-2]

    # Strip outer parens if present
    if inner.startswith('(') and inner.endswith(')'):
        inner = inner[1:-1]

    # Now identify domain and op
    # max/min patterns
    m_maxmin_rev = re.match(r'max\(rev\(a\),rev\(b\)\)([+\-*%]|[\|]{2})min\(rev\(a\),rev\(b\)\)', inner)
    m_minmax_rev = re.match(r'min\(rev\(a\),rev\(b\)\)([+\-*%]|[\|]{2})max\(rev\(a\),rev\(b\)\)', inner)
    m_maxmin = re.match(r'max\(a,b\)([+\-*%]|[\|]{2})min\(a,b\)', inner)
    m_minmax = re.match(r'min\(a,b\)([+\-*%]|[\|]{2})max\(a,b\)', inner)

    # rev(a) op rev(b) patterns
    m_revab = re.match(r'rev\(a\)([+\-*%]|[\|]{2})rev\(b\)', inner)
    m_revba = re.match(r'rev\(b\)([+\-*%]|[\|]{2})rev\(a\)', inner)

    # plain a op b
    m_ab = re.match(r'a([+\-*%]|[\|]{2})b$', inner)
    m_ba = re.match(r'b([+\-*%]|[\|]{2})a$', inner)

    if m_maxmin_rev:
        return ('maxmin_r', m_maxmin_rev.group(1), adj, outer)
    if m_minmax_rev:
        return ('minmax_r', m_minmax_rev.group(1), adj, outer)
    if m_maxmin:
        return ('maxmin', m_maxmin.group(1), adj, outer)
    if m_minmax:
        return ('minmax', m_minmax.group(1), adj, outer)
    if m_revab:
        return ('revab', m_revab.group(1), adj, outer)
    if m_revba:
        return ('revba', m_revba.group(1), adj, outer)
    if m_ab:
        return ('ab', m_ab.group(1), adj, outer)
    if m_ba:
        return ('ba', m_ba.group(1), adj, outer)

    # Special: rev(rev(b))*(rev(a)) style (typo or special)
    # These are malformed — flag them
    return ('UNPARSED', inner, adj, outer)

# ============================================================================
# CORE ANALYSIS: What is the block-level template domain?
# ============================================================================

print("="*110)
print("BLOCK-LEVEL ANALYSIS: Template Domain & Operator Mapping")
print("="*110)

# For each block, determine:
# 1. The DOMAIN used (the operand_domain + outer from classify)
# 2. The operator symbol → inner_op mapping

block_domain_counter = Counter()
op_to_innerop = defaultdict(Counter)  # visible_op_symbol -> inner arithmetic op

for bi, blk in enumerate(blocks):
    domains = set()
    ops_map = {}  # op_symbol -> (inner_op, adj)

    for eq in blk:
        cls = classify_formula(eq['formula'])
        domain, inner_op, adj, outer = cls

        # Block domain = (domain, outer)
        block_dom = (domain, outer)
        domains.add(block_dom)

        # Track op symbol mapping
        op_sym = eq['op']  # visible operator like +, -, *, etc.
        ops_map.setdefault(op_sym, []).append((inner_op, adj))

    block_domain_counter[tuple(sorted(domains))] += 1

print("\n### How many blocks use each domain combination? ###")
for dom_combo, cnt in block_domain_counter.most_common():
    items = [f"({d},{o})" for d,o in dom_combo]
    print(f"  [{cnt:3d} blocks] {', '.join(items)}")

# ============================================================================
# THE BIG QUESTION: Within a given domain, how does visible op → inner op?
# ============================================================================

print("\n" + "="*110)
print("VISIBLE OPERATOR SYMBOL → ARITHMETIC OPERATION (within each domain)")
print("="*110)

# Collect all mappings: (block_domain, visible_op) → inner_op
domain_op_map = defaultdict(lambda: defaultdict(Counter))

for bi, blk in enumerate(blocks):
    # Determine this block's primary domain
    domain_votes = Counter()
    for eq in blk:
        cls = classify_formula(eq['formula'])
        domain_votes[(cls[0], cls[3])] += 1
    primary_domain = domain_votes.most_common(1)[0][0]

    for eq in blk:
        cls = classify_formula(eq['formula'])
        domain, inner_op, adj, outer = cls
        vis_op = eq['op']
        adj_str = f"{'+' if adj > 0 else ''}{adj}" if adj != 0 else ''
        domain_op_map[primary_domain][vis_op][f"{inner_op}{adj_str}"] += 1

for dom in sorted(domain_op_map.keys()):
    print(f"\n  Domain: {dom}")
    for vis_op in sorted(domain_op_map[dom].keys()):
        mappings = domain_op_map[dom][vis_op]
        items = [f"{k}({v}x)" for k,v in mappings.most_common()]
        print(f"    '{vis_op}' → {', '.join(items)}")

# ============================================================================
# MOST IMPORTANT: The relationship between visible_op and inner_op
# ============================================================================

print("\n" + "="*110)
print("### KEY DISCOVERY: VISIBLE OP → INNER OP CONSISTENCY ###")
print("(Is each visible operator symbol always the same inner arithmetic op?)")
print("="*110)

# Global: visible_op → inner_op (ignoring domain)
global_vis_to_inner = defaultdict(Counter)
for bi, blk in enumerate(blocks):
    for eq in blk:
        cls = classify_formula(eq['formula'])
        if cls[0] == 'UNPARSED': continue
        vis_op = eq['op']
        inner_op = cls[1]
        global_vis_to_inner[vis_op][inner_op] += 1

print("\nVisible operator → Inner arithmetic operation (GLOBAL):")
for vis in sorted(global_vis_to_inner.keys()):
    inner = global_vis_to_inner[vis]
    total = sum(inner.values())
    items = [f"{k}({v})" for k,v in inner.most_common()]
    consistent = "✓ ALWAYS SAME" if len(inner) == 1 else "✗ VARIES"
    print(f"  '{vis}' → {', '.join(items):40s} {consistent}")

# ============================================================================
# ADJUSTMENT PATTERN: When +1/-1 is used
# ============================================================================

print("\n" + "="*110)
print("### ADJUSTMENT PATTERN (+1/-1) BY VISIBLE OP ###")
print("="*110)

adj_counter = defaultdict(Counter)
for bi, blk in enumerate(blocks):
    for eq in blk:
        cls = classify_formula(eq['formula'])
        if cls[0] == 'UNPARSED': continue
        vis_op = eq['op']
        adj = cls[2]
        adj_counter[vis_op][adj] += 1

for vis in sorted(adj_counter.keys()):
    adjs = adj_counter[vis]
    items = [f"adj={k}({v})" for k,v in adjs.most_common()]
    print(f"  '{vis}' → {', '.join(items)}")

# ============================================================================
# DOMAIN PATTERN: does visible op dictate the domain?
# ============================================================================

print("\n" + "="*110)
print("### DOMAIN BY VISIBLE OP ###")
print("="*110)

domain_by_op = defaultdict(Counter)
for bi, blk in enumerate(blocks):
    for eq in blk:
        cls = classify_formula(eq['formula'])
        if cls[0] == 'UNPARSED': continue
        domain_by_op[eq['op']][(cls[0], cls[3])] += 1

for vis in sorted(domain_by_op.keys()):
    doms = domain_by_op[vis]
    items = [f"{k}({v})" for k,v in doms.most_common()]
    consistent = "✓" if len(doms) == 1 else "✗"
    print(f"  '{vis}' → {', '.join(items):60s} {consistent}")
