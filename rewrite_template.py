#!/usr/bin/env python3
"""Rewrite every line into the universal max/min(+rev) template, SET-AWARE:
all lines that share an operator within the same block must use the same rule
signature (operand-set, operation, adjustment, outer-rev). Only the sign
(negate) varies per line, as dictated by its own shown answer.
Answers are never changed."""
import re

def rev_int(s): return int(s[::-1])

def clean(op, res):
    s = res; sign = 1
    if s.startswith('-'):         sign = -1; s = s[1:]
    elif op and s.startswith(op): sign = -1; s = s[len(op):]
    if op and s.endswith(op):     s = s[:-len(op)]
    return sign, s

# ---- rule signatures -------------------------------------------------------
# A signature = (operand, op, adj, outer_rev)
#   operand : 'ab' | 'rev'
#   op      : '+' | '-' | '*' | '%'
#   adj     : -1 | 0 | 1
#   outer_rev: bool   (whether result digits are reversed, i.e. "= rev(c)")
# Concatenation is handled separately as pseudo-signatures.
SIGS = []
for operand in ('ab', 'rev'):
    for op in ('+', '-', '*', '%'):
        for adj in (0, 1, -1):
            if op in ('-', '%') and adj != 0:   # adjustments only seen on + and *
                continue
            for outer in (False, True):
                SIGS.append((operand, op, adj, outer))
CONCATS = ['a || b', 'b || a', 'rev(a) || rev(b)']

def base_value(sig, a, b, ra, rb):
    operand, op, adj, outer = sig
    x, y = (a, b) if operand == 'ab' else (ra, rb)
    M, m = max(x, y), min(x, y)
    if op == '+': v = M + m
    elif op == '-': v = M - m            # always >= 0
    elif op == '*': v = M * m
    elif op == '%':
        if m == 0: return None
        v = M % m
    v += adj
    return v

def concat_value(c, a_str, b_str):
    if c == 'a || b':           return a_str + b_str
    if c == 'b || a':           return b_str + a_str
    if c == 'rev(a) || rev(b)': return a_str[::-1] + b_str[::-1]

def line_matches(sig, a_str, b_str, op, res):
    """Return the per-line negate (True/False) under which sig reproduces res,
    or None if it doesn't match."""
    a, b = int(a_str), int(b_str)
    ra, rb = rev_int(a_str), rev_int(b_str)
    base = base_value(sig, a, b, ra, rb)
    if base is None: return None
    operand, op2, adj, outer = sig
    sign, mag = clean(op, res)
    if mag == '' or not mag.isdigit(): return None
    mi = int(mag)
    for negate in (False, True):
        val = -base if negate else base
        if not outer:
            if val == sign * mi:
                return negate
        else:
            av = abs(val)
            magok = (int(mag[::-1]) == av) or (int(str(av)[::-1]) == mi)
            signok = (sign == (-1 if val < 0 else 1)) or val == 0
            if magok and signok:
                return negate
    return None

def concat_matches(c, a_str, b_str, op, res):
    sign, mag = clean(op, res)
    if not mag.isdigit(): return False
    cv = concat_value(c, a_str, b_str)
    return cv == mag or int(cv) == int(mag)

# priority: simpler first  (ab<rev, no-outer<outer, +,-,*,% , adj 0 first)
def sig_priority(sig):
    operand, op, adj, outer = sig
    return (0 if operand == 'ab' else 1,
            0 if not outer else 1,
            {'+':0,'-':1,'*':2,'%':3}[op],
            0 if adj == 0 else 1)

def render(sig, negate):
    operand, op, adj, outer = sig
    X, Y = ('a', 'b') if operand == 'ab' else ('rev(a)', 'rev(b)')
    inner = f"max({X}, {Y}) {op} min({X}, {Y})"
    if adj != 0:
        sgn = '+' if adj > 0 else '-'
        if op == '*':
            inner = f"({inner}) {sgn} 1"
        else:
            inner = f"{inner} {sgn} 1"
    expr = f"rev({inner})" if outer else inner
    if negate:
        expr = f"-{expr}" if outer else f"-({inner})"
    return expr

# ---- parse file into blocks of equation lines ------------------------------
with open('puzzles_guess_logic.md') as f:
    raw = [l.rstrip('\n') for l in f]

eq_re = re.compile(r'^(?P<pre>(?:Now, determine the result for:\s*)?)'
                   r'(?P<a>\d+)(?P<op>.)(?P<b>\d+)\s*=\s*(?P<res>\S+)\s*'
                   r'\((?P<sig>a.b)\)\s*=\s*(?P<formula>.+?)\s*$')

# identify blocks (separated by blank lines)
blocks = []; cur = []
parsed = [None] * len(raw)
for i, line in enumerate(raw):
    m = eq_re.match(line.strip())
    if line.strip() == '':
        if cur: blocks.append(cur); cur = []
    if m:
        parsed[i] = m
        cur.append(i)
if cur: blocks.append(cur)

chosen_render = {}   # line index -> render string
unmatched = []

for blk in blocks:
    # group line indices by operator symbol
    groups = {}
    for i in blk:
        op = parsed[i]['op']
        groups.setdefault(op, []).append(i)
    for op, idxs in groups.items():
        # intersect matching signatures across the group
        inter = None
        per_line = {}      # i -> dict(sig->negate)
        concat_ok = {}     # i -> set of concats matching
        for i in idxs:
            a, b, res = parsed[i]['a'], parsed[i]['b'], parsed[i]['res']
            ms = {}
            for sig in SIGS:
                neg = line_matches(sig, a, b, op, res)
                if neg is not None:
                    ms[sig] = neg
            per_line[i] = ms
            cset = {c for c in CONCATS if concat_matches(c, a, b, op, res)}
            concat_ok[i] = cset
            keys = set(ms.keys())
            inter = keys if inter is None else (inter & keys)
        cinter = None
        for i in idxs:
            cinter = concat_ok[i] if cinter is None else (cinter & concat_ok[i])

        if inter:
            best = sorted(inter, key=sig_priority)[0]
            for i in idxs:
                chosen_render[i] = render(best, per_line[i][best])
        elif cinter:
            c = sorted(cinter)[0]
            for i in idxs:
                chosen_render[i] = c
        else:
            # no group-consistent rule; fall back to per-line best
            for i in idxs:
                a, b, res = parsed[i]['a'], parsed[i]['b'], parsed[i]['res']
                ms = per_line[i]
                if ms:
                    best = sorted(ms.keys(), key=sig_priority)[0]
                    chosen_render[i] = render(best, ms[best])
                elif concat_ok[i]:
                    chosen_render[i] = sorted(concat_ok[i])[0]
                else:
                    chosen_render[i] = parsed[i]['formula']
                    unmatched.append((i + 1, raw[i]))

# ---- write output ----------------------------------------------------------
out = []
for i, line in enumerate(raw):
    if parsed[i] is None:
        out.append(line); continue
    m = parsed[i]
    out.append(f"{m['pre']}{m['a']}{m['op']}{m['b']} = {m['res']} ({m['sig']}) = {chosen_render[i]}")

with open('puzzles_guess_logic.md', 'w') as f:
    f.write('\n'.join(out) + '\n')

print(f"Rewrote {sum(1 for p in parsed if p)} equation lines (set-aware).")
print(f"Unmatched: {len(unmatched)}")
for ln, l in unmatched:
    print(f"  !! L{ln}: {l}")
