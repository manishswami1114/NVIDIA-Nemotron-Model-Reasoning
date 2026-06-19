#!/usr/bin/env python3
"""
Check each equation: plug a and b into the solver formula on the right side
and verify it produces the stated result. Operator symbols are NOT changed.

Dataset display conventions handled:
  * results may be zero-padded ("06" == 6, "-02" == -2)
  * the operator symbol can stand in for a sign or be a trailing artifact
    e.g.  82:87 = :05  means -5 ;  46$71 = 74$  has a trailing $ artifact
  * "= rev(c)" means the displayed result's digits are the reverse of the
    computed value's digits (sign stays in front): value -58 -> "-85"
"""
import re

def rev_int(n_str):
    return int(n_str[::-1])

def parse_line(line):
    m = re.match(r'^\s*(\d+)(.)(\d+)\s*=\s*(\S+)\s*\(a.b\)\s*=\s*(.+?)\s*$', line)
    if not m:
        return None
    a_str, op, b_str, result_str, formula = m.groups()
    return {'a_str': a_str, 'op': op, 'b_str': b_str,
            'result_str': result_str, 'formula': formula.strip()}

def evaluate(a_str, b_str, formula):
    """Return ('UNKNOWN',), ('STR', s), or ('NUM', value, target)."""
    a, b = int(a_str), int(b_str)
    ra, rb = rev_int(a_str), rev_int(b_str)

    parts = formula.rsplit('=', 1)
    if len(parts) == 2:
        expr, target = parts[0].strip(), parts[1].strip()
    else:
        expr, target = formula.strip(), 'c'

    if expr == 'unknown logic':
        return ('UNKNOWN',)

    def smod(x, y): return x % y if y else None

    if   expr == 'a-b':            val = a - b
    elif expr == 'a+b':            val = a + b
    elif expr == 'a*b':            val = a * b
    elif expr == 'b-a':            val = b - a
    elif expr == 'abs(a-b)':       val = abs(a - b)
    elif expr == '-abs(a-b)':      val = -abs(a - b)
    elif expr == 'a+b-1':          val = a + b - 1
    elif expr == 'a+b+1':          val = a + b + 1
    elif expr == '(a*b)+1':        val = a*b + 1
    elif expr == '(a*b)-1':        val = a*b - 1
    elif expr == 'a||b':           return ('STR', a_str + b_str)
    elif expr == 'b||a':           return ('STR', b_str + a_str)
    elif expr == 'rev(a)||rev(b)': return ('STR', a_str[::-1] + b_str[::-1])
    elif expr == 'rev(a)+rev(b)':       val = ra + rb
    elif expr == 'rev(a)-rev(b)':       val = ra - rb
    elif expr == 'rev(b)-rev(a)':       val = rb - ra
    elif expr == 'rev(a)*rev(b)':       val = ra * rb
    elif expr == 'rev(a)+rev(b)+1':     val = ra + rb + 1
    elif expr == 'rev(a)+rev(b)-1':     val = ra + rb - 1
    elif expr == '(rev(a)*rev(b))+1':   val = ra*rb + 1
    elif expr == '(rev(a)*rev(b))-1':   val = ra*rb - 1
    elif expr == 'abs(rev(a)-rev(b))':  val = abs(ra - rb)
    elif expr == '-abs(rev(a)-rev(b))': val = -abs(ra - rb)
    elif expr == 'rev(b)-rev(a)':       val = rb - ra
    elif expr == 'max(a,b)%min(a,b)':                       val = smod(max(a,b), min(a,b))
    elif expr == 'max(rev(a),rev(b))%min(rev(a),rev(b))':   val = smod(max(ra,rb), min(ra,rb))
    elif expr == 'rev(b)-rev(a)':       val = rb - ra
    else:
        return ('UNRECOGNIZED', expr)

    if val is None:
        return ('MODZERO',)
    return ('NUM', val, target)

def clean_result(op, result_str):
    """
    Return (sign, magnitude_string) after stripping the operator symbol used
    as a sign-placeholder/artifact. A leading '-' or leading op-symbol => negative.
    """
    s = result_str
    sign = 1
    # leading minus
    if s.startswith('-'):
        sign = -1
        s = s[1:]
    # leading operator symbol acting as negative sign
    elif op and s.startswith(op):
        sign = -1
        s = s[len(op):]
    # trailing operator symbol artifact
    if op and s.endswith(op):
        s = s[:-len(op)]
    return sign, s

def compare(op, computed, result_str):
    kind = computed[0]
    if kind == 'UNKNOWN':
        return ('unknown', '')
    if kind in ('UNRECOGNIZED', 'MODZERO'):
        return ('skip', kind)

    sign, mag = clean_result(op, result_str)
    if mag == '' or not mag.lstrip('-').isdigit():
        return ('REVIEW', f"result token not numeric: {result_str!r}")

    if kind == 'STR':
        # concatenation: compare digit strings directly (ignore any sign)
        got = computed[1]
        # allow padding differences via int compare too
        ok = (got == mag) or (int(got) == int(mag))
        return ('match' if ok else 'MISMATCH', f"solver={got} shown={result_str}")

    # NUM
    val, target = computed[1], computed[2]
    mag_int = int(mag)
    if target == 'c':
        ok = (val == sign * mag_int)
        return ('match' if ok else 'MISMATCH',
                f"solver={val} shown={result_str}({sign*mag_int})")
    else:  # rev(c)
        absval = abs(val)
        # match if reversing displayed digits gives |value|,
        # OR reversing |value| gives displayed (handles dropped leading zero)
        rev_shown = int(mag[::-1])
        rev_val   = int(str(absval)[::-1])
        magval_ok = (rev_shown == absval) or (rev_val == mag_int)
        sign_ok   = (sign == (-1 if val < 0 else 1)) or (val == 0)
        ok = magval_ok and sign_ok
        return ('match' if ok else 'MISMATCH',
                f"solver={val} -> rev shown={result_str}")

with open('puzzles_guess_logic.md') as f:
    lines = f.readlines()

buckets = {'match': [], 'MISMATCH': [], 'unknown': [], 'REVIEW': [], 'skip': []}
for i, raw in enumerate(lines, 1):
    line = raw.rstrip('\n')
    if not line.strip():
        continue
    p = parse_line(line)
    if not p:
        continue
    computed = evaluate(p['a_str'], p['b_str'], p['formula'])
    status, info = compare(p['op'], computed, p['result_str'])
    buckets[status].append((i, line.strip(), info))

print("="*92)
print(f"  {len(buckets['match'])} MATCH | {len(buckets['MISMATCH'])} MISMATCH | "
      f"{len(buckets['unknown'])} unknown-logic | {len(buckets['REVIEW'])} needs-review | "
      f"{len(buckets['skip'])} skipped")
print("="*92)

print("\n##### GENUINE MISMATCHES (solver does NOT yield the shown result) #####")
if not buckets['MISMATCH']:
    print("  (none)")
for i, line, info in buckets['MISMATCH']:
    print(f"Line {i:3d}: {line}\n          -> {info}")

print("\n##### NEEDS REVIEW (result token malformed / non-numeric) #####")
for i, line, info in buckets['REVIEW']:
    print(f"Line {i:3d}: {line}\n          -> {info}")

print("\n##### UNKNOWN LOGIC (no solver written) #####")
for i, line, info in buckets['unknown']:
    print(f"Line {i:3d}: {line}")
