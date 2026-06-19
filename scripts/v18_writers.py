#!/usr/bin/env python3
"""
V18 cryptarithm writer — sequential equation-by-equation formula search.

Walks the equations in order. For each operator's FIRST appearance, iterate
candidate formulas (width/sign-filtered, simplest first), substitute the
verified operand digits, compute the formula's value, compare to the expected
result-as-integer. Mark `wrong` or `match — lock '<op>' = <formula>`. Stop at
first match. For subsequent appearances of the same operator, APPLY the
locked formula and verify.

Style: short one-line per attempt, no algebra, like baseline equation_numeric_deduce.

No multiplication decomposition. No `Family elimination` handwave. The search
trace IS the elimination evidence.
"""
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cryptarithm_family as CF
from v16_writers import _verification_epilogue


# ============================================================
# Equation introspection (same helpers as V17)
# ============================================================

def _is_neg_res(op, res):
    return len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])


def _result_body(op, res):
    return res[1:] if _is_neg_res(op, res) else res


def _result_width(op, res):
    return len(_result_body(op, res))


def _digits_of(symbols, mp):
    return mp[symbols[0]] * 10 + mp[symbols[1]]


def _decode_int(body, mp, neg=False):
    s = ''.join(str(mp[c]) for c in body)
    n = int(s)
    return -n if neg else n


# ============================================================
# ONE-LINE arithmetic per formula (no decomposition)
# ============================================================

def _short_arith(family, op_name, a, b):
    """Return one-line substituted arithmetic for op_name applied to (a, b),
    ending in '= <value>'. No multiplication decomposition. ≤ 60 chars typical.
    """
    if family == 'DIRECT_SIMPLE':
        if op_name == 'a+b':       return f"{a} + {b} = {a+b}"
        if op_name == 'a-b':       return f"{a} - {b} = {a-b}"
        if op_name == 'b-a':       return f"{b} - {a} = {b-a}"
        if op_name == 'a*b':       return f"{a} * {b} = {a*b}"
        if op_name == 'b*a':       return f"{b} * {a} = {b*a}"
        if op_name == '(a*b)+1':   return f"{a} * {b} + 1 = {a*b+1}"
        if op_name == '(a*b)-1':   return f"{a} * {b} - 1 = {a*b-1}"
        if op_name == '(b*a)+1':   return f"{b} * {a} + 1 = {b*a+1}"
        if op_name == '(b*a)-1':   return f"{b} * {a} - 1 = {b*a-1}"
        if op_name == '(a+b)+1':   return f"{a} + {b} + 1 = {a+b+1}"
        if op_name == '(a+b)-1':   return f"{a} + {b} - 1 = {a+b-1}"
        if op_name == '(b+a)+1':   return f"{b} + {a} + 1 = {b+a+1}"
        if op_name == '(b+a)-1':   return f"{b} + {a} - 1 = {b+a-1}"
        if op_name == '(a-b)+1':   return f"{a} - {b} + 1 = {a-b+1}"
        if op_name == '(a-b)-1':   return f"{a} - {b} - 1 = {a-b-1}"
        if op_name == '(b-a)+1':   return f"{b} - {a} + 1 = {b-a+1}"
        if op_name == '(b-a)-1':   return f"{b} - {a} - 1 = {b-a-1}"
        if op_name == 'a||b':      return f"{a} || {b} = {a:02d}{b:02d}"
        if op_name == 'b||a':      return f"{b} || {a} = {b:02d}{a:02d}"
        if op_name == 'a%b':       return f"{a} mod {b} = {(a%b) if b else 0}"
        if op_name == 'b%a':       return f"{b} mod {a} = {(b%a) if a else 0}"
        if op_name == '-(a-b)':    return f"-({a} - {b}) = {-(a-b)}"
        if op_name == '-(b-a)':    return f"-({b} - {a}) = {-(b-a)}"
    if family == 'DIRECT_MAXMIN':
        mx, mn = max(a, b), min(a, b)
        if op_name == 'max+min':       return f"max({a},{b}) + min({a},{b}) = {mx} + {mn} = {mx+mn}"
        if op_name == 'max-min':       return f"max({a},{b}) - min({a},{b}) = {mx} - {mn} = {mx-mn}"
        if op_name == 'max*min':       return f"max({a},{b}) * min({a},{b}) = {mx} * {mn} = {mx*mn}"
        if op_name == '(max+min)+1':   return f"max({a},{b}) + min({a},{b}) + 1 = {mx+mn+1}"
        if op_name == '(max+min)-1':   return f"max({a},{b}) + min({a},{b}) - 1 = {mx+mn-1}"
        if op_name == '(max-min)+1':   return f"max({a},{b}) - min({a},{b}) + 1 = {mx-mn+1}"
        if op_name == '(max-min)-1':   return f"max({a},{b}) - min({a},{b}) - 1 = {mx-mn-1}"
        if op_name == '(min-max)+1':   return f"min({a},{b}) - max({a},{b}) + 1 = {mn-mx+1}"
        if op_name == '(min-max)-1':   return f"min({a},{b}) - max({a},{b}) - 1 = {mn-mx-1}"
        if op_name == '(max*min)+1':   return f"max({a},{b}) * min({a},{b}) + 1 = {mx*mn+1}"
        if op_name == '(max*min)-1':   return f"max({a},{b}) * min({a},{b}) - 1 = {mx*mn-1}"
        if op_name == 'max%min':       return f"max({a},{b}) mod min({a},{b}) = {(mx%mn) if mn else 0}"
        if op_name == 'max||min':      return f"max({a},{b}) || min({a},{b}) = {mx:02d}{mn:02d}"
        if op_name == 'min||max':      return f"min({a},{b}) || max({a},{b}) = {mn:02d}{mx:02d}"
        if op_name == '-(max-min)':    return f"-(max({a},{b}) - min({a},{b})) = {-(mx-mn)}"
        if op_name == 'min-max':       return f"min({a},{b}) - max({a},{b}) = {mn-mx}"
        if op_name == 'min+max':       return f"min({a},{b}) + max({a},{b}) = {mn+mx}"
    if family == 'REV_AB':
        ra, rb = CF.rev2(a), CF.rev2(b)
        if op_name == 'rev(rev(a)+rev(b))':
            return f"rev({ra} + {rb}) = rev({ra+rb})"
        if op_name == 'rev(rev(a)-rev(b))':
            return f"rev({ra} - {rb}) = rev({ra-rb})"
        if op_name == 'rev(rev(b)-rev(a))':
            return f"rev({rb} - {ra}) = rev({rb-ra})"
        if op_name == 'rev(rev(a)*rev(b))':
            return f"rev({ra} * {rb}) = rev({ra*rb})"
        if op_name == 'rev(rev(b)*rev(a))':
            return f"rev({rb} * {ra}) = rev({rb*ra})"
        if op_name == 'rev(rev(a)*rev(b)+1)':
            return f"rev({ra} * {rb} + 1) = rev({ra*rb+1})"
        if op_name == 'rev(rev(a)*rev(b)-1)':
            return f"rev({ra} * {rb} - 1) = rev({ra*rb-1})"
        if op_name == 'rev(rev(a)+rev(b)+1)':
            return f"rev({ra} + {rb} + 1) = rev({ra+rb+1})"
        if op_name == 'rev(rev(a)+rev(b)-1)':
            return f"rev({ra} + {rb} - 1) = rev({ra+rb-1})"
        if op_name == 'rev(rev(a)-rev(b)+1)':
            return f"rev({ra} - {rb} + 1) = rev({ra-rb+1})"
        if op_name == 'rev(rev(a)-rev(b)-1)':
            return f"rev({ra} - {rb} - 1) = rev({ra-rb-1})"
        if op_name == 'rev(rev(b)-rev(a)+1)':
            return f"rev({rb} - {ra} + 1) = rev({rb-ra+1})"
        if op_name == 'rev(rev(b)-rev(a)-1)':
            return f"rev({rb} - {ra} - 1) = rev({rb-ra-1})"
        if op_name == 'rev(rev(b)+rev(a)+1)':
            return f"rev({rb} + {ra} + 1) = rev({rb+ra+1})"
        if op_name == 'rev(rev(b)+rev(a)-1)':
            return f"rev({rb} + {ra} - 1) = rev({rb+ra-1})"
        if op_name == 'rev(rev(b)*rev(a)+1)':
            return f"rev({rb} * {ra} + 1) = rev({rb*ra+1})"
        if op_name == 'rev(rev(b)*rev(a)-1)':
            return f"rev({rb} * {ra} - 1) = rev({rb*ra-1})"
        if op_name == 'rev(rev(a)||rev(b))':
            return f"rev({ra:02d} || {rb:02d}) = rev({ra:02d}{rb:02d})"
        if op_name == 'rev(rev(b)||rev(a))':
            return f"rev({rb:02d} || {ra:02d}) = rev({rb:02d}{ra:02d})"
        if op_name == 'rev(rev(a)%rev(b))':
            return f"rev({ra} mod {rb}) = rev({(ra%rb) if rb else 0})"
        if op_name == 'rev(rev(b)%rev(a))':
            return f"rev({rb} mod {ra}) = rev({(rb%ra) if ra else 0})"
        if op_name == '-rev(rev(a)-rev(b))':
            return f"-rev({ra} - {rb}) = -rev({ra-rb})"
        if op_name == '-rev(rev(b)-rev(a))':
            return f"-rev({rb} - {ra}) = -rev({rb-ra})"
    if family == 'REV_MAXMIN':
        ra, rb = CF.rev2(a), CF.rev2(b)
        rmx, rmn = max(ra, rb), min(ra, rb)
        if op_name == 'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) + min({ra},{rb})) = rev({rmx+rmn})"
        if op_name == 'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) - min({ra},{rb})) = rev({rmx-rmn})"
        if op_name == 'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) * min({ra},{rb})) = rev({rmx*rmn})"
        if op_name == 'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)':
            return f"rev(max({ra},{rb}) + min({ra},{rb}) + 1) = rev({rmx+rmn+1})"
        if op_name == 'rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)':
            return f"rev(max({ra},{rb}) + min({ra},{rb}) - 1) = rev({rmx+rmn-1})"
        if op_name == 'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)':
            return f"rev(max({ra},{rb}) * min({ra},{rb}) + 1) = rev({rmx*rmn+1})"
        if op_name == 'rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)':
            return f"rev(max({ra},{rb}) * min({ra},{rb}) - 1) = rev({rmx*rmn-1})"
        if op_name == 'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) mod min({ra},{rb})) = rev({(rmx%rmn) if rmn else 0})"
        if op_name == 'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))':
            return f"rev({rmx:02d} || {rmn:02d}) = rev({rmx:02d}{rmn:02d})"
        if op_name == 'rev(min(rev(a),rev(b))||max(rev(a),rev(b)))':
            return f"rev({rmn:02d} || {rmx:02d}) = rev({rmn:02d}{rmx:02d})"
        if op_name == 'rev(min(rev(a),rev(b))-max(rev(a),rev(b)))':
            return f"rev(min({ra},{rb}) - max({ra},{rb})) = rev({rmn-rmx})"
        if op_name == '-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':
            return f"-rev(max({ra},{rb}) - min({ra},{rb})) = -rev({rmx-rmn})"
        if op_name == 'rev(max(rev(a),rev(b))-min(rev(a),rev(b))+1)':
            return f"rev(max({ra},{rb}) - min({ra},{rb}) + 1) = rev({rmx-rmn+1})"
        if op_name == 'rev(max(rev(a),rev(b))-min(rev(a),rev(b))-1)':
            return f"rev(max({ra},{rb}) - min({ra},{rb}) - 1) = rev({rmx-rmn-1})"
        if op_name == 'rev(min(rev(a),rev(b))-max(rev(a),rev(b))+1)':
            return f"rev(min({ra},{rb}) - max({ra},{rb}) + 1) = rev({rmn-rmx+1})"
    # Fallback
    applied = CF.apply_op(family, op_name, a, b)
    if applied:
        v, neg = applied
        return f"{op_name}({a},{b}) = {('-' + v) if neg else v}"
    return f"{op_name}({a},{b}) = UNDEF"


# ============================================================
# Compute formula value as a SIGNED INTEGER (no zero-padding).
# Returns (int_value, is_negative_marker_bool) — or None if undefined.
# ============================================================

def _formula_int_value(family, op_name, a, b):
    """Return (signed_int, is_neg_marker). Returns None if undefined.
    is_neg_marker means the operator-as-sign prefix applies (negative)."""
    applied = CF.apply_op(family, op_name, a, b)
    if applied is None:
        return None
    res_str, is_neg = applied
    try:
        val = int(res_str)
        return (-val if is_neg else val), is_neg
    except ValueError:
        return None


# ============================================================
# Candidate formula ordering — simplest first, baseline-style.
# Width/sign filter applied at evaluation time.
# ============================================================

# Order matters: simplest formulas first (a+b before a*b before reversed-result).
_FORMULA_ORDER = (
    # Direct simple — simplest
    [('DIRECT_SIMPLE', n) for n in [
        'a+b', 'a-b', 'b-a', 'a*b', 'a||b', 'b||a',
        '(a*b)+1', '(a*b)-1', '(a+b)+1', '(a+b)-1',
        '(a-b)+1', '(a-b)-1', '(b-a)+1', '(b-a)-1',
        'a%b', 'b%a', '-(a-b)', '-(b-a)',
    ]]
    # Direct max/min
    + [('DIRECT_MAXMIN', n) for n in [
        'max+min', 'max-min', 'max*min',
        'max||min', 'min||max',
        '(max*min)+1', '(max*min)-1',
        '(max+min)+1', '(max+min)-1',
        '(max-min)+1', '(max-min)-1',
        '(min-max)+1', '(min-max)-1',
        '-(max-min)', 'min-max', 'min+max',
        'max%min',
    ]]
    # Reversed-AB family
    + [('REV_AB', n) for n in [
        'rev(rev(a)+rev(b))', 'rev(rev(a)-rev(b))', 'rev(rev(b)-rev(a))',
        'rev(rev(a)*rev(b))', 'rev(rev(b)*rev(a))',
        'rev(rev(a)*rev(b)+1)', 'rev(rev(a)*rev(b)-1)',
        'rev(rev(a)+rev(b)+1)', 'rev(rev(a)+rev(b)-1)',
        'rev(rev(a)-rev(b)+1)', 'rev(rev(a)-rev(b)-1)',
        'rev(rev(b)-rev(a)+1)', 'rev(rev(b)-rev(a)-1)',
        'rev(rev(b)+rev(a)+1)', 'rev(rev(b)+rev(a)-1)',
        'rev(rev(b)*rev(a)+1)', 'rev(rev(b)*rev(a)-1)',
        'rev(rev(a)||rev(b))', 'rev(rev(b)||rev(a))',
        'rev(rev(a)%rev(b))', 'rev(rev(b)%rev(a))',
        '-rev(rev(a)-rev(b))', '-rev(rev(b)-rev(a))',
    ]]
    # Reversed max/min
    + [('REV_MAXMIN', n) for n in [
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))',
        'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))',
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))',
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)',
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)',
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)',
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)',
        'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))',
        'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))',
        'rev(min(rev(a),rev(b))||max(rev(a),rev(b)))',
        'rev(min(rev(a),rev(b))-max(rev(a),rev(b)))',
        '-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))',
        'rev(max(rev(a),rev(b))-min(rev(a),rev(b))+1)',
        'rev(max(rev(a),rev(b))-min(rev(a),rev(b))-1)',
        'rev(min(rev(a),rev(b))-max(rev(a),rev(b))+1)',
    ]]
)


# ============================================================
# Main writer — sequential equation-by-equation search
# ============================================================

def write_cryptarithm_search(puzzle):
    """V18 cryptarithm CoT — sequential per-equation formula search.

    puzzle keys: equations, query, answer, family, map, ops, kind.
    """
    equations = puzzle['equations']
    query     = puzzle['query']
    answer    = puzzle['answer']
    family    = puzzle['family']
    mp        = puzzle['map']
    om        = puzzle['ops']
    kind      = puzzle.get('kind', 'deduce')
    q1, qop, q2 = query

    L = []
    L.append("<think>")
    L.append("We need to find the operator family and digit map satisfying all examples.")
    L.append("Search: walk equations in order. For each operator try formulas one by one;")
    L.append("substitute the operand digits and check the value against the expected result.")
    L.append("Lock the first matching formula and move on. Backtrack on contradiction.")
    L.append("I will put my final answer inside \\boxed{}.\n")
    L.append("Examples:")
    for n1, op, n2, res in equations:
        L.append(f"  {n1}{op}{n2} = {res}")
    L.append(f"  Query: {q1}{qop}{q2}")
    L.append("")

    # Symbol inventory
    syms = []
    for n1, op, n2, res in equations:
        for c in n1 + n2 + _result_body(op, res):
            if c not in syms: syms.append(c)
    for c in q1 + q2:
        if c not in syms: syms.append(c)
    L.append(f"Symbols (non-operator): {' '.join(syms)}")

    L.append("Operators in examples:")
    ops_seen = []
    for _, op, _, _ in equations:
        if op not in ops_seen: ops_seen.append(op)
    for op in ops_seen:
        L.append(op)
    L.append("")

    L.append("Looking at the question")
    L.append(f"{q1}{qop}{q2} -> {qop}")
    if qop in ops_seen:
        L.append("The question operator is found in the examples.")
    else:
        L.append("The question operator is NOT in the examples — we will infer it from the family of example operators.")
    L.append("")

    # Brief per-equation summary
    L.append("Per-equation result widths and signs:")
    for i, (n1, op, n2, res) in enumerate(equations, 1):
        body = _result_body(op, res)
        sign = "negative" if _is_neg_res(op, res) else "positive"
        L.append(f"  eq{i} {n1}{op}{n2} = {res}: {len(body)}-digit {sign} result.")
    L.append("")

    # ---- Sequential search ----
    L.append("Sequential search:")
    locked = {}  # operator -> formula_name (locked after first match)

    for i, (n1, op, n2, res) in enumerate(equations, 1):
        a = _digits_of(n1, mp)
        b = _digits_of(n2, mp)
        is_neg = _is_neg_res(op, res)
        body = _result_body(op, res)
        expected_int = _decode_int(body, mp, neg=is_neg)
        sign_label = "negative " if is_neg else ""
        L.append(f"")
        L.append(f"eq{i}: {n1}{op}{n2} = {res}  (a={a}, b={b}, expected={'-' if is_neg else ''}{_decode_int(body, mp)})")

        if op in locked:
            # Already locked — apply and check
            f_name = locked[op]
            r = _formula_int_value(family, f_name, a, b)
            if r is None:
                L.append(f"  apply locked '{op}' = {f_name}: UNDEFINED. (contradiction — would backtrack)")
                continue
            val, val_neg = r
            arith = _short_arith(family, f_name, a, b)
            ok = (val == expected_int)
            if ok:
                L.append(f"  apply locked '{op}' = {f_name}: {arith}  match")
            else:
                L.append(f"  apply locked '{op}' = {f_name}: {arith}  ≠ {expected_int}  (would backtrack — but the verified solver confirms no real conflict)")
            continue

        # First occurrence — iterate formulas. Show only width-compatible
        # candidates (those whose computed value has the right digit count
        # AND right sign) so we focus on the meaningful "wrong" cases.
        target_family = family
        correct_formula = om[op]
        expected_abs = abs(expected_int)
        expected_width = len(body)
        expected_sign_neg = is_neg
        shown = 0
        for (fam_cand, op_name) in _FORMULA_ORDER:
            r = _formula_int_value(fam_cand, op_name, a, b)
            if r is None:
                continue
            val, val_neg = r
            # Width filter: skip if computed value has wrong digit count
            v_abs = abs(val)
            v_width = len(str(v_abs))
            # Allow value to be shown if width matches OR it's the correct formula
            width_ok = (v_width == expected_width)
            sign_ok = (val_neg == expected_sign_neg)
            is_correct = (op_name == correct_formula and fam_cand == target_family)
            if not (width_ok and sign_ok) and not is_correct:
                continue
            arith = _short_arith(fam_cand, op_name, a, b)
            if val == expected_int and is_correct:
                L.append(f"  try {op_name}: {arith}  match — lock '{op}' = {op_name} (family {fam_cand})")
                locked[op] = op_name
                break
            elif val == expected_int:
                # value matches but wrong family/op — would conflict later
                L.append(f"  try {op_name}: {arith}  value matches but family-inconsistent")
            else:
                L.append(f"  try {op_name}: {arith}  wrong")
            shown += 1
        else:
            # Fall back if nothing matched (shouldn't happen for verified puzzles)
            L.append(f"  (no formula found — defaulting to verified solver's choice)")
            locked[op] = correct_formula
    L.append("")

    # Digit map summary
    L.append("Locked operator formulas:")
    for op in ops_seen:
        L.append(f"  '{op}' = {locked.get(op, om[op])}  (family {family})")
    L.append("")
    L.append(f"Digit map (consistent with every locked formula):")
    map_disp = ', '.join(f"'{s}'={mp[s]}" for s in sorted(mp))
    L.append(f"  {map_disp}")
    L.append("")

    # Verification — recompute each equation under final formulas
    L.append("Verification:")
    for i, (n1, op, n2, res) in enumerate(equations, 1):
        a = _digits_of(n1, mp)
        b = _digits_of(n2, mp)
        is_neg = _is_neg_res(op, res)
        body = _result_body(op, res)
        expected_int = _decode_int(body, mp, neg=is_neg)
        f_name = locked.get(op, om[op])
        r = _formula_int_value(family, f_name, a, b)
        if r is None:
            L.append(f"  eq{i} {n1}{op}{n2} = {res}: UNDEFINED (would not happen for verified puzzle)")
            continue
        val, _ = r
        arith = _short_arith(family, f_name, a, b)
        ok = (val == expected_int)
        L.append(f"  eq{i} {n1}{op}{n2} = {res}: {arith}  {'match' if ok else 'mismatch'}")
    L.append("")

    # Apply to query
    qa = mp[q1[0]] * 10 + mp[q1[1]]
    qb = mp[q2[0]] * 10 + mp[q2[1]]
    L.append(f"Applying to {q1}{qop}{q2}:")
    L.append(f"  a={qa}, b={qb}")
    if qop in ops_seen:
        qformula = locked.get(qop, om[qop])
        L.append(f"  '{qop}' = {qformula}  (already locked from examples)")
        arith = _short_arith(family, qformula, qa, qb)
        L.append(f"  {arith}")
    else:
        # GUESS — enumerate family formulas, find one matching answer
        L.append(f"  '{qop}' is NEW. All example operators are in family {family}, so '{qop}' is too.")
        ans_is_neg = _is_neg_res(qop, answer)
        ans_body = answer[1:] if ans_is_neg else answer
        try:
            expected_q = _decode_int(ans_body, mp, neg=ans_is_neg)
        except KeyError:
            expected_q = None
        qformula = None
        for op_name in list(CF.FAMILIES[family].keys()):
            r = _formula_int_value(family, op_name, qa, qb)
            if r is None: continue
            val, _ = r
            if expected_q is not None and val == expected_q:
                arith = _short_arith(family, op_name, qa, qb)
                L.append(f"  try {op_name}: {arith}  match — adopt '{qop}' = {op_name}")
                qformula = op_name
                break
        if qformula is None:
            qformula = om.get(qop, list(CF.FAMILIES[family].keys())[0])
            arith = _short_arith(family, qformula, qa, qb)
            L.append(f"  fallback '{qop}' = {qformula}: {arith}")

    L.append(f"  read back through digit→symbol map: {answer}")
    L.append(f"  Result: 【{answer}】")
    L.extend(_verification_epilogue(answer))
    return '\n'.join(L)
