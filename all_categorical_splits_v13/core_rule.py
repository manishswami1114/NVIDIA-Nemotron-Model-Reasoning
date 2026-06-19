#!/usr/bin/env python3
"""
THE CORE RULE: How the query operation is determined.

Key findings so far:
1. Query ALWAYS uses a NEW visible symbol (136/136)
2. Query ALWAYS stays in the SAME domain as examples (~90%, rest are typos)
3. Query action is ALWAYS different from example actions (it's "missing")
4. Each block has 1-2 example operator types, query adds one more

Now: can we PREDICT the query action from the examples alone?
"""
import re
from collections import Counter, defaultdict

with open('perfect_solver_complete.md') as f:
    raw = [l.rstrip() for l in f]

eq_re = re.compile(
    r'^(?P<det>Now, determine the result for:\s*)?'
    r'(?P<a>\d+)(?P<op>[^\d]+?)(?P<b>\d+)\s*=\s*(?P<rest>.+)$'
)

def parse_formula(formula):
    f = re.sub(r'\s*\[op suffix\]', '', formula).strip()
    nf = re.sub(r'\s+', '', f)
    sign = '+'
    inner = nf
    if inner.startswith('-rev('):
        sign = '-'; outer = 'rev'; inner = inner[5:-1] if inner.endswith(')') else inner[5:]
    elif inner.startswith('-('):
        sign = '-'; outer = 'plain'; inner = inner[2:-1] if inner.endswith(')') else inner[2:]
    elif inner.startswith('rev(') and inner.endswith(')'):
        outer = 'rev'; inner = inner[4:-1]
    else:
        outer = 'plain'
    adj = 0
    if inner.endswith('+1'): adj = 1; inner = inner[:-2]
    elif inner.endswith('-1'): adj = -1; inner = inner[:-2]
    while inner.startswith('(') and inner.endswith(')'):
        depth = 0; balanced = True
        for i, c in enumerate(inner):
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            if depth == 0 and i < len(inner)-1: balanced = False; break
        if balanced: inner = inner[1:-1]
        else: break
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
    adj_s = f"+{cls[2]}" if cls[2] > 0 else str(cls[2]) if cls[2] < 0 else ""
    return {
        'a': m['a'], 'op': m['op'], 'b': m['b'],
        'result': result, 'formula': formula,
        'is_query': is_query,
        'domain': cls[0], 'inner_op': cls[1], 'adj': cls[2],
        'outer': cls[3], 'sign': cls[4],
        'action': f"{cls[1]}{adj_s}"
    }

blocks = []; cur = []
for line in raw:
    if line.strip() == '':
        if cur: blocks.append(cur); cur = []
        continue
    p = parse_line(line)
    if p: cur.append(p)
if cur: blocks.append(cur)

# ============================================================================
# THE PREDICTION RULE
# ============================================================================

# Canonical operation set within each domain family:
# For revab/revba: {+, -, *, ||} with optional adj {+1, -1}
# For maxmin/maxmin_r: {+, -, *, %, ||} with optional adj
# For ab/ba: {+, -, *, %, ||} with optional adj

# Let's see: given examples, what EXACTLY predicts the query?

print("="*120)
print("COMPLETE BLOCK TABLE: examples → query (clean blocks only, no UNPARSED)")
print("="*120)
print(f"{'Blk':>4} | {'Domain':>10} | {'Example actions':>30} | {'Query action':>12} | {'Query is new?':>13} | Num ex ops")

clean_blocks = []
for bi, blk in enumerate(blocks):
    examples = [eq for eq in blk if not eq['is_query'] and eq['domain'] != 'UNPARSED']
    queries  = [eq for eq in blk if eq['is_query']]
    if not queries: continue
    q = queries[0]
    if q['domain'] == 'UNPARSED': continue

    ex_by_visop = defaultdict(list)
    for ex in examples:
        ex_by_visop[ex['op']].append(ex)

    ex_actions = sorted(set(eqs[0]['action'] for eqs in ex_by_visop.values()))

    domain_votes = Counter(eq['domain'] for eq in examples)
    primary_domain = domain_votes.most_common(1)[0][0] if domain_votes else q['domain']

    q_action = q['action']
    is_new = q_action not in ex_actions

    clean_blocks.append({
        'bi': bi, 'domain': primary_domain,
        'ex_actions': ex_actions, 'q_action': q_action,
        'is_new': is_new, 'n_ex': len(ex_actions)
    })

    print(f"{bi+1:4d} | {primary_domain:>10} | {str(ex_actions):>30} | {q_action:>12} | {'✓ NEW' if is_new else '✗ SAME':>13} | {len(ex_actions)}")

# ============================================================================
# THE MAPPING TABLE: What examples → what query
# ============================================================================

print("\n" + "="*120)
print("MAPPING TABLE: (sorted example actions) → query action")
print("="*120)

mapping = Counter()
for cb in clean_blocks:
    key = (tuple(cb['ex_actions']), cb['q_action'])
    mapping[key] += 1

for (ex, qa), cnt in sorted(mapping.items(), key=lambda x: (-x[1], x[0])):
    print(f"  [{cnt:2d}x] {str(list(ex)):>30} → {qa}")

# ============================================================================
# IS THERE A "MISSING OP" RULE?
# ============================================================================

print("\n" + "="*120)
print("'MISSING OP' HYPOTHESIS TEST")
print("For each block, is the query action the one NOT present in examples?")
print("="*120)

# Define the canonical base ops for each domain
# Most blocks use 2 example ops, and query adds a 3rd from {+,-,*,||,%, +1/-1 variants}

# Let's check: if we define the "available pool" as {+,-,*,||,%} (ignoring adj),
# is the query's base op always missing from examples?

base_op_pool = {'+', '-', '*', '||', '%'}
missing_correct = 0
missing_wrong = 0
missing_ambiguous = 0

for cb in clean_blocks:
    ex_base = set()
    for a in cb['ex_actions']:
        # strip adjustment
        base = a.rstrip('+-1').rstrip('+').rstrip('-') if len(a) > 1 else a
        if base in ('', '+'): base = a[0] if a else a
        base = a[0] if a[0] in '+-*%' else ('||' if '||' in a else a[0])
        ex_base.add(base)

    q_base = cb['q_action'][0] if cb['q_action'][0] in '+-*%' else ('||' if '||' in cb['q_action'] else cb['q_action'][0])

    missing = base_op_pool - ex_base
    if q_base in missing:
        missing_correct += 1
    elif len(missing) == 0:
        missing_ambiguous += 1
    else:
        missing_wrong += 1

print(f"  Query base op IS in the missing set: {missing_correct} ({missing_correct*100//len(clean_blocks)}%)")
print(f"  Query base op NOT in missing set:     {missing_wrong}")
print(f"  No missing ops (ambiguous):           {missing_ambiguous}")

# ============================================================================
# COMPREHENSIVE: Show every block's full operation spectrum
# ============================================================================

print("\n" + "="*120)
print("FULL OPERATION SPECTRUM PER BLOCK")
print("(All operations present: examples + query)")
print("="*120)

spectrum_counter = Counter()
for cb in clean_blocks:
    all_actions = sorted(set(cb['ex_actions'] + [cb['q_action']]))
    spectrum_counter[tuple(all_actions)] += 1

print(f"\n{'Count':>5} | Operations in block (examples + query)")
for spec, cnt in spectrum_counter.most_common():
    print(f"  {cnt:3d}x | {list(spec)}")

# ============================================================================
# THE ACTUAL RULE: Pattern of triplets
# ============================================================================

print("\n" + "="*120)
print("TRIPLET PATTERNS (most blocks have 3 operations total)")
print("="*120)

# Group by how many total operations
by_total = defaultdict(list)
for cb in clean_blocks:
    total = len(set(cb['ex_actions'] + [cb['q_action']]))
    by_total[total].append(cb)

for total, cbs in sorted(by_total.items()):
    print(f"\n  {total} total operations: {len(cbs)} blocks")
    if total <= 3:
        trip_counter = Counter()
        for cb in cbs:
            all_ops = tuple(sorted(set(cb['ex_actions'] + [cb['q_action']])))
            trip_counter[all_ops] += 1
        for trip, cnt in trip_counter.most_common(10):
            print(f"    [{cnt:2d}x] {list(trip)}")
