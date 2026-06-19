#!/usr/bin/env python3
"""
Analyze the template pattern system in perfect_solver_complete.md.
Goal: discover the universal template taxonomy — how many distinct template
families exist, how the operator symbol maps to an arithmetic operation within
each block, and what the structural rules are.
"""
import re
from collections import Counter, defaultdict

with open('perfect_solver_complete.md') as f:
    raw = [l.rstrip('\n') for l in f]

# Parse into blocks (separated by blank lines)
eq_re = re.compile(
    r'^(?P<determine>Now, determine the result for:\s*)?'
    r'(?P<a>\d+)(?P<op>.{1,2})(?P<b>\d+)\s*=\s*(?P<res>\S+)\s*=\s*(?P<formula>.+?)\s*$'
)
# Also handle "Now, determine" lines without an answer (just formula)
det_re = re.compile(
    r'^Now, determine the result for:\s*(?P<a>\d+)(?P<op>.{1,2})(?P<b>\d+)\s*=\s*(?P<formula>.+?)\s*$'
)

blocks = []
cur_block = []
for i, line in enumerate(raw):
    if line.strip() == '':
        if cur_block:
            blocks.append(cur_block)
            cur_block = []
        continue
    # Try full match (with answer)
    m = eq_re.match(line.strip())
    if m:
        cur_block.append({
            'line': i+1, 'raw': line.strip(),
            'a': m['a'], 'op': m['op'], 'b': m['b'],
            'res': m['res'] if m['res'] else None,
            'formula': m['formula'].strip(),
            'is_query': bool(m['determine'])
        })
        continue
    m2 = det_re.match(line.strip())
    if m2:
        cur_block.append({
            'line': i+1, 'raw': line.strip(),
            'a': m2['a'], 'op': m2['op'], 'b': m2['b'],
            'res': None,
            'formula': m2['formula'].strip(),
            'is_query': True
        })
        continue
    # non-matching line
    if line.strip():
        cur_block.append({'line': i+1, 'raw': line.strip(), 'unparsed': True})
if cur_block:
    blocks.append(cur_block)

# ============================================================================
# 1. Normalize formulas to identify template families
# ============================================================================

def normalize_formula(f):
    """Strip annotations like [op suffix], whitespace variations."""
    f = re.sub(r'\s*\[op suffix\]\s*', '', f)
    f = f.strip()
    # normalize spacing
    f = re.sub(r'\s+', ' ', f)
    return f

# Collect all unique normalized formulas
all_formulas = Counter()
for blk in blocks:
    for eq in blk:
        if 'unparsed' in eq:
            continue
        nf = normalize_formula(eq['formula'])
        all_formulas[nf] += 1

print("="*100)
print(f"TOTAL BLOCKS: {len(blocks)}")
print(f"TOTAL EQUATION LINES: {sum(len(b) for b in blocks)}")
print(f"UNIQUE FORMULA TEMPLATES: {len(all_formulas)}")
print("="*100)

# ============================================================================
# 2. Classify formulas into families
# ============================================================================

FAMILIES = {
    'PLAIN_OP': [],          # (a OP b), max(a,b) OP min(a,b)
    'REV_REV': [],           # rev(rev(a) OP rev(b))
    'REV_MAXMIN_REV': [],    # rev(max(rev(a),rev(b)) OP min(rev(a),rev(b)))
    'MAXMIN_PLAIN': [],      # max(a,b) OP min(a,b)
    'CONCAT': [],            # a||b, b||a, rev variants
    'OTHER': [],
}

def classify(f):
    nf = normalize_formula(f)
    if '||' in nf:
        return 'CONCAT'
    if nf.startswith('rev(max(rev(a),rev(b))') or nf.startswith('-rev(max(rev(a),rev(b))'):
        return 'REV_MAXMIN_REV'
    if nf.startswith('rev(rev(') or nf.startswith('-rev(rev(') or nf.startswith('rev(min(rev('):
        return 'REV_REV'
    if 'max(a,b)' in nf or 'min(a,b)' in nf:
        return 'MAXMIN_PLAIN'
    if nf in ('(a+b)', '(a-b)', '(a*b)', '(a%b)', 'a+b', 'a-b', 'a*b',
              '(b+a)', '(b-a)', '(b*a)', '(b%a)', 'b+a', 'b-a', 'b*a',
              '(a+b)+1', '(a+b)-1', '(a*b)+1', '(a*b)-1',
              '(b+a)+1', '(b+a)-1', '(b*a)+1', '(b*a)-1',
              '-(b-a)', '-(a-b)'):
        return 'PLAIN_OP'
    return 'OTHER'

family_counts = Counter()
for f, cnt in all_formulas.items():
    fam = classify(f)
    family_counts[fam] += cnt

print("\n### FORMULA FAMILY DISTRIBUTION ###")
for fam, cnt in family_counts.most_common():
    print(f"  {fam:20s}: {cnt:4d} lines")

# ============================================================================
# 3. Analyze the TEMPLATE STRUCTURE within each block
# ============================================================================

print("\n" + "="*100)
print("### BLOCK-LEVEL TEMPLATE ANALYSIS ###")
print("="*100)

# For each block, identify the template "signature":
#   - What operand domain is used? (plain a,b | rev(a),rev(b) | max/min of rev)
#   - What arithmetic ops appear? (+, -, *, %, ||)
#   - Is the result reversed (outer rev)?

def extract_template_sig(formula):
    """Extract a structural signature from a formula."""
    nf = normalize_formula(formula)

    # Determine operand domain
    if 'max(rev(a),rev(b))' in nf:
        domain = 'max/min(rev)'
    elif 'rev(a)' in nf or 'rev(b)' in nf:
        if 'max(' in nf or 'min(' in nf:
            domain = 'max/min(rev)'
        else:
            domain = 'rev(a,b)'
    elif 'max(a,b)' in nf or 'min(a,b)' in nf:
        domain = 'max/min(a,b)'
    else:
        domain = 'plain(a,b)'

    # Determine outer rev
    outer_rev = nf.startswith('rev(') or nf.startswith('-rev(')

    # Determine core operation
    if '||' in nf:
        core_op = '||'
    elif '*' in nf and '+' not in nf and '-' not in nf:
        core_op = '*'
    elif '%' in nf:
        core_op = '%'
    elif '+' in nf and '-' not in nf:
        core_op = '+'
    elif '-' in nf:
        core_op = '-'
    else:
        core_op = '?'

    # Determine adjustment
    adj = 0
    if '+1' in nf and '||' not in nf:
        adj = 1
    elif '-1' in nf and '||' not in nf:
        adj = -1

    # Determine negation
    negated = nf.startswith('-') or nf.startswith('-(')

    return (domain, outer_rev, core_op, adj, negated)

# ============================================================================
# 4. Key insight: within each block, how do operator symbols map to templates?
# ============================================================================

print("\n" + "="*100)
print("### OPERATOR SYMBOL → TEMPLATE MAPPING (per block) ###")
print("="*100)

# Group by block, then by operator symbol within block
block_patterns = []
for bi, blk in enumerate(blocks):
    ops_in_block = defaultdict(list)
    for eq in blk:
        if 'unparsed' in eq:
            continue
        ops_in_block[eq['op']].append(normalize_formula(eq['formula']))

    # Check: is each operator consistently mapped to one template?
    consistent = True
    op_map = {}
    for op, formulas in ops_in_block.items():
        unique = set(formulas)
        if len(unique) > 1:
            consistent = False
        op_map[op] = formulas[0] if len(unique) == 1 else list(unique)

    block_patterns.append({
        'block_idx': bi,
        'ops': dict(ops_in_block),
        'op_map': op_map,
        'consistent': consistent,
        'num_ops': len(ops_in_block),
        'lines': [eq.get('line') for eq in blk if 'unparsed' not in eq]
    })

# Count how many blocks are fully consistent
n_consistent = sum(1 for bp in block_patterns if bp['consistent'])
print(f"\nBlocks with CONSISTENT op→template mapping: {n_consistent}/{len(block_patterns)}")

# ============================================================================
# 5. Identify the UNIVERSAL template vocabulary
# ============================================================================

print("\n" + "="*100)
print("### UNIVERSAL TEMPLATE VOCABULARY ###")
print("  (all distinct formula templates, sorted by frequency)")
print("="*100)

for f, cnt in all_formulas.most_common():
    fam = classify(f)
    print(f"  [{cnt:3d}x] {fam:20s} | {f}")

# ============================================================================
# 6. Detect the CORE PATTERN: how blocks compose templates
# ============================================================================

print("\n" + "="*100)
print("### BLOCK COMPOSITION PATTERNS ###")
print("  (what combination of template families appear in each block)")
print("="*100)

composition_counter = Counter()
for bi, blk in enumerate(blocks):
    families_in_block = set()
    for eq in blk:
        if 'unparsed' in eq:
            continue
        families_in_block.add(classify(eq['formula']))
    comp = tuple(sorted(families_in_block))
    composition_counter[comp] += 1

for comp, cnt in composition_counter.most_common():
    print(f"  [{cnt:3d} blocks] {' + '.join(comp)}")

# ============================================================================
# 7. The KEY DISCOVERY: How many distinct "operand domains" exist?
# ============================================================================

print("\n" + "="*100)
print("### THE CORE TEMPLATE SYSTEM ###")
print("="*100)

# Each formula maps to one of these template slots:
# DOMAIN × OPERATION × ADJUSTMENT × SIGN
# Domain: {plain, rev, max/min(plain), max/min(rev)}
# Operation: {+, -, *, %, ||}
# Adjustment: {-1, 0, +1}
# Sign: {positive, negative}
# Outer-rev: {yes, no}

template_slots = Counter()
for f, cnt in all_formulas.items():
    sig = extract_template_sig(f)
    template_slots[sig] += cnt

print("\nTemplate slots (domain, outer_rev, core_op, adj, negated) → count:")
for sig, cnt in sorted(template_slots.items(), key=lambda x: -x[1]):
    domain, outer_rev, core_op, adj, negated = sig
    adj_s = f"+{adj}" if adj > 0 else str(adj)
    print(f"  {domain:15s} | rev={str(outer_rev):5s} | op={core_op:2s} | adj={adj_s:3s} | neg={str(negated):5s} | count={cnt}")

# ============================================================================
# 8. Per-block: show the operator→formula mapping
# ============================================================================

print("\n" + "="*100)
print("### DETAILED BLOCK MAPPINGS (first 20 blocks) ###")
print("="*100)

for bi, bp in enumerate(block_patterns[:30]):
    lines = bp['lines']
    print(f"\nBlock {bi+1} (lines {lines[0]}-{lines[-1]}):")
    for op, formulas in bp['ops'].items():
        unique = set(formulas)
        if len(unique) == 1:
            print(f"  '{op}' → {formulas[0]}  [{len(formulas)} lines]")
        else:
            for uf in unique:
                cnt = formulas.count(uf)
                print(f"  '{op}' → {uf}  [{cnt}x]")

# ============================================================================
# 9. WITHIN a block, how does the QUERY line's template relate to examples?
# ============================================================================

print("\n" + "="*100)
print("### QUERY LINE TEMPLATE vs EXAMPLE TEMPLATES ###")
print("="*100)

query_same_as_examples = 0
query_different = 0
query_ops = Counter()

for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq.get('is_query') and 'unparsed' not in eq]
    queries = [eq for eq in blk if eq.get('is_query') and 'unparsed' not in eq]

    if not queries:
        continue

    q = queries[0]
    q_formula = normalize_formula(q['formula'])
    q_op = q['op']

    # Find example lines with same operator
    same_op_examples = [e for e in examples if e['op'] == q_op]

    if same_op_examples:
        # Query op was already seen in examples
        ex_formula = normalize_formula(same_op_examples[0]['formula'])
        if ex_formula == q_formula:
            query_same_as_examples += 1
        else:
            query_different += 1
            print(f"  Block {bi+1}: query '{q_op}' uses {q_formula} but examples use {ex_formula}")
    else:
        # Query op is NEW (not seen in examples) → this is the puzzle!
        query_ops[q_op] += 1

print(f"\nQuery op SAME as example op in block: {query_same_as_examples}")
print(f"Query op DIFFERENT from example op:    {query_different}")
print(f"Query op is NEW (not in examples):     {sum(query_ops.values())}")
print(f"  Distribution of new query ops: {dict(query_ops.most_common())}")
