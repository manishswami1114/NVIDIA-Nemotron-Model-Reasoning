#!/usr/bin/env python3
"""Independently evaluate the rewritten template formulas and confirm each
reproduces its shown answer (honoring sign-symbol + padding + rev conventions)."""
import re

def rev_digits(n):
    s = str(abs(n))
    return int(s[::-1])

def evaluate(expr, a_str, b_str):
    """Evaluate a template expression -> ('NUM', signed_value, is_rev) or ('STR', s)."""
    a, b = int(a_str), int(b_str)
    ra, rb = int(a_str[::-1]), int(b_str[::-1])
    e = expr.strip()

    # concatenation
    if e == 'a || b':           return ('STR', a_str + b_str)
    if e == 'b || a':           return ('STR', b_str + a_str)
    if e == 'rev(a) || rev(b)': return ('STR', a_str[::-1] + b_str[::-1])

    neg = False
    if e.startswith('-'):
        neg = True
        e = e[1:]

    is_rev = False
    # outer rev(...) wrap?  (only if it wraps the WHOLE expression)
    mrev = re.match(r'^rev\((.*)\)$', e)
    if mrev and e.startswith('rev(') and not e.startswith('rev(max(rev') is not None:
        pass
    if e.startswith('rev(') and e.endswith(')'):
        # ensure these are matched outer parens
        depth = 0; outer = True
        inner = e[4:-1]
        for i, ch in enumerate(e[4:], start=4):
            if ch == '(': depth += 1
            elif ch == ')':
                depth -= 1
                if depth < 0 and i != len(e) - 1:
                    outer = False; break
        if outer:
            is_rev = True
            e = inner

    # now e is an arithmetic expression over max/min of (a,b) or (rev(a),rev(b))
    # substitute
    def sub(s):
        s = s.replace('max(rev(a), rev(b))', str(max(ra, rb)))
        s = s.replace('min(rev(a), rev(b))', str(min(ra, rb)))
        s = s.replace('max(a, b)', str(max(a, b)))
        s = s.replace('min(a, b)', str(min(a, b)))
        return s
    py = sub(e)
    # only digits, operators, spaces, parens, % * + -
    if not re.match(r'^[\d\s\+\-\*\%\(\)]+$', py):
        return ('ERR', py)
    val = eval(py)
    if neg: val = -val
    return ('NUM', val, is_rev)

def clean(op, res):
    s = res; sign = 1
    if s.startswith('-'):         sign = -1; s = s[1:]
    elif op and s.startswith(op): sign = -1; s = s[len(op):]
    if op and s.endswith(op):     s = s[:-len(op)]
    return sign, s

def ok(op, ev, res):
    sign, mag = clean(op, res)
    if mag == '' or not mag.isdigit(): return False, "bad-token"
    if ev[0] == 'STR':
        return (ev[1] == mag or int(ev[1]) == int(mag)), f"str={ev[1]}"
    if ev[0] == 'ERR':
        return False, f"ERR:{ev[1]}"
    val, is_rev = ev[1], ev[2]
    mi = int(mag)
    if not is_rev:
        return (val == sign * mi), f"val={val}"
    av = abs(val)
    magok = (int(mag[::-1]) == av) or (rev_digits(av) == mi)
    signok = (sign == (-1 if val < 0 else 1)) or val == 0
    return (magok and signok), f"val={val}->rev"

eq_re = re.compile(r'^(?:Now, determine the result for:\s*)?'
                   r'(\d+)(.)(\d+)\s*=\s*(\S+)\s*\(a.b\)\s*=\s*(.+?)\s*$')
fail = []; n = 0
with open('puzzles_guess_logic.md') as f:
    for ln, raw in enumerate(f, 1):
        m = eq_re.match(raw.strip())
        if not m: continue
        n += 1
        a, op, b, res, formula = m.groups()
        ev = evaluate(formula, a, b)
        good, info = ok(op, ev, res)
        if not good:
            fail.append((ln, raw.strip(), info))

print(f"Verified {n} template lines. PASS: {n-len(fail)}  FAIL: {len(fail)}")
for ln, line, info in fail:
    print(f"  FAIL L{ln}: {line}   [{info}]")
