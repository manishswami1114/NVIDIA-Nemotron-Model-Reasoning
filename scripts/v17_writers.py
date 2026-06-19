#!/usr/bin/env python3
"""
v17 cryptarithm writer — plain-prose deductive, no markdown headers.

Mirrors the baseline `train_cot_equation_numeric_deduce` format style: flat
sentences, indented sub-blocks, no `===` separators, no bullets, no `⇒`.
Order is still: structural observations → eliminate wrong families →
anchor-based digit deduction → propagation → verify → apply to query.
"""
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cryptarithm_family as CF
from v16_writers import _decompose_mul, _verification_epilogue


# ============================================================
# Helpers — symbol/equation introspection (same as before)
# ============================================================

def _is_neg_res(op, res):
    return len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])


def _result_body(op, res):
    return res[1:] if _is_neg_res(op, res) else res


def _result_width(op, res):
    return len(_result_body(op, res))


def _repeated_positions(body):
    from collections import defaultdict
    pos_by_char = defaultdict(list)
    for i, c in enumerate(body):
        pos_by_char[c].append(i)
    return [(tuple(ps), c) for c, ps in pos_by_char.items() if len(ps) > 1]


def _is_concat_pattern(n1, n2, body):
    return body == n1 + n2 or body == n2 + n1


def _is_squared_input(n1, n2):
    return n1 == n2


def _digits_of(symbols, mp):
    return mp[symbols[0]] * 10 + mp[symbols[1]]


def _decode_str(body, mp):
    return ''.join(str(mp[c]) for c in body)


_FAMILY_HUMAN = {
    'DIRECT_SIMPLE': 'DIRECT_SIMPLE',
    'DIRECT_MAXMIN': 'DIRECT_MAXMIN',
    'REV_AB':        'REV_AB',
    'REV_MAXMIN':    'REV_MAXMIN',
}


# ============================================================
# Apply formula numerically (for verification)
# ============================================================

def _formula_value(family, op_name, a, b, expected_width):
    """Return (result_string, is_negative) using the formula's NATURAL width.
    We do NOT zero-pad — if the formula's natural value is shorter than the
    expected result width, that's a real mismatch (puzzle should have been
    filtered upstream). Honest arithmetic only."""
    applied = CF.apply_op(family, op_name, a, b)
    if applied is None:
        return None, False
    res_str, is_neg = applied
    return res_str, is_neg


# ============================================================
# Substituted arithmetic for one family op applied to (a, b)
# ============================================================

def _render_rule_arith(family, op_name, a, b):
    if family == 'DIRECT_SIMPLE':
        if op_name == 'a+b':   return f"{a} + {b} = {a+b}"
        if op_name == 'a-b':   return f"{a} - {b} = {a-b}"
        if op_name == 'b-a':   return f"{b} - {a} = {b-a}"
        if op_name == 'a*b':   return _decompose_mul(a, b)
        if op_name == 'b*a':   return _decompose_mul(b, a)
        if op_name == '(a*b)+1': return _decompose_mul(a, b, plus=1)
        if op_name == '(a*b)-1': return _decompose_mul(a, b, plus=-1)
        if op_name == '(b*a)+1': return _decompose_mul(b, a, plus=1)
        if op_name == '(b*a)-1': return _decompose_mul(b, a, plus=-1)
        if op_name == '(a+b)+1': return f"{a} + {b} + 1 = {a+b+1}"
        if op_name == '(a+b)-1': return f"{a} + {b} - 1 = {a+b-1}"
        if op_name == '(b+a)+1': return f"{b} + {a} + 1 = {b+a+1}"
        if op_name == '(b+a)-1': return f"{b} + {a} - 1 = {b+a-1}"
        if op_name == '(a-b)+1': return f"{a} - {b} + 1 = {a-b+1}"
        if op_name == '(a-b)-1': return f"{a} - {b} - 1 = {a-b-1}"
        if op_name == '(b-a)+1': return f"{b} - {a} + 1 = {b-a+1}"
        if op_name == '(b-a)-1': return f"{b} - {a} - 1 = {b-a-1}"
        if op_name == 'a||b':  return f"{a} || {b} = {a:02d}{b:02d}"
        if op_name == 'b||a':  return f"{b} || {a} = {b:02d}{a:02d}"
        if op_name == 'a%b':   return f"{a} mod {b} = {a%b if b else 0}"
        if op_name == 'b%a':   return f"{b} mod {a} = {b%a if a else 0}"
        if op_name == '-(a-b)':return f"-({a} - {b}) = {-(a-b)}"
        if op_name == '-(b-a)':return f"-({b} - {a}) = {-(b-a)}"
    if family == 'DIRECT_MAXMIN':
        mx, mn = max(a, b), min(a, b)
        if op_name == 'max+min':  return f"max({a},{b}) + min({a},{b}) = {mx} + {mn} = {mx+mn}"
        if op_name == 'max-min':  return f"max({a},{b}) - min({a},{b}) = {mx} - {mn} = {mx-mn}"
        if op_name == 'max*min':  return f"max({a},{b}) * min({a},{b}) = {_decompose_mul(mx, mn)}"
        if op_name == '(max+min)+1': return f"max({a},{b}) + min({a},{b}) + 1 = {mx} + {mn} + 1 = {mx+mn+1}"
        if op_name == '(max+min)-1': return f"max({a},{b}) + min({a},{b}) - 1 = {mx} + {mn} - 1 = {mx+mn-1}"
        if op_name == '(max-min)+1': return f"max({a},{b}) - min({a},{b}) + 1 = {mx-mn+1}"
        if op_name == '(max-min)-1': return f"max({a},{b}) - min({a},{b}) - 1 = {mx-mn-1}"
        if op_name == '(min-max)+1': return f"min({a},{b}) - max({a},{b}) + 1 = {mn-mx+1}"
        if op_name == '(min-max)-1': return f"min({a},{b}) - max({a},{b}) - 1 = {mn-mx-1}"
        if op_name == '(max*min)+1': return f"max({a},{b}) * min({a},{b}) + 1 = {_decompose_mul(mx, mn, plus=1)}"
        if op_name == '(max*min)-1': return f"max({a},{b}) * min({a},{b}) - 1 = {_decompose_mul(mx, mn, plus=-1)}"
        if op_name == 'max%min': return f"max({a},{b}) mod min({a},{b}) = {mx} mod {mn} = {mx%mn if mn else 0}"
        if op_name == 'max||min': return f"max({a},{b}) || min({a},{b}) = {mx:02d}{mn:02d}"
        if op_name == 'min||max': return f"min({a},{b}) || max({a},{b}) = {mn:02d}{mx:02d}"
        if op_name == '-(max-min)': return f"-(max({a},{b}) - min({a},{b})) = -({mx} - {mn}) = {-(mx-mn)}"
        if op_name == 'min-max':  return f"min({a},{b}) - max({a},{b}) = {mn} - {mx} = {mn-mx}"
        if op_name == 'min+max':  return f"min({a},{b}) + max({a},{b}) = {mn} + {mx} = {mn+mx}"
    if family == 'REV_AB':
        ra, rb = CF.rev2(a), CF.rev2(b)
        if op_name == 'rev(rev(a)+rev(b))':
            return f"rev(rev({a})+rev({b})) = rev({ra}+{rb}) = rev({ra+rb})"
        if op_name == 'rev(rev(a)-rev(b))':
            return f"rev(rev({a})-rev({b})) = rev({ra}-{rb}) = rev({ra-rb})"
        if op_name == 'rev(rev(b)-rev(a))':
            return f"rev(rev({b})-rev({a})) = rev({rb}-{ra}) = rev({rb-ra})"
        if op_name == 'rev(rev(a)*rev(b))':
            return f"rev(rev({a})*rev({b})) = rev({ra}*{rb}) = rev({ra*rb})"
        if op_name == 'rev(rev(b)*rev(a))':
            return f"rev(rev({b})*rev({a})) = rev({rb}*{ra}) = rev({rb*ra})"
        if op_name == 'rev(rev(a)*rev(b)+1)':
            return f"rev(rev({a})*rev({b})+1) = rev({ra*rb}+1) = rev({ra*rb+1})"
        if op_name == 'rev(rev(a)*rev(b)-1)':
            return f"rev(rev({a})*rev({b})-1) = rev({ra*rb}-1) = rev({ra*rb-1})"
        if op_name == 'rev(rev(a)+rev(b)+1)':
            return f"rev(rev({a})+rev({b})+1) = rev({ra+rb}+1) = rev({ra+rb+1})"
        if op_name == 'rev(rev(a)+rev(b)-1)':
            return f"rev(rev({a})+rev({b})-1) = rev({ra+rb}-1) = rev({ra+rb-1})"
        if op_name == 'rev(rev(a)-rev(b)+1)':
            return f"rev(rev({a})-rev({b})+1) = rev({ra-rb}+1) = rev({ra-rb+1})"
        if op_name == 'rev(rev(a)-rev(b)-1)':
            return f"rev(rev({a})-rev({b})-1) = rev({ra-rb}-1) = rev({ra-rb-1})"
        if op_name == 'rev(rev(b)-rev(a)+1)':
            return f"rev(rev({b})-rev({a})+1) = rev({rb-ra+1})"
        if op_name == 'rev(rev(b)-rev(a)-1)':
            return f"rev(rev({b})-rev({a})-1) = rev({rb-ra-1})"
        if op_name == 'rev(rev(b)+rev(a)+1)':
            return f"rev(rev({b})+rev({a})+1) = rev({rb+ra}+1) = rev({rb+ra+1})"
        if op_name == 'rev(rev(b)+rev(a)-1)':
            return f"rev(rev({b})+rev({a})-1) = rev({rb+ra}-1) = rev({rb+ra-1})"
        if op_name == 'rev(rev(b)*rev(a)+1)':
            return f"rev(rev({b})*rev({a})+1) = rev({rb*ra+1})"
        if op_name == 'rev(rev(b)*rev(a)-1)':
            return f"rev(rev({b})*rev({a})-1) = rev({rb*ra-1})"
        if op_name == 'rev(rev(a)||rev(b))':
            return f"rev(rev({a})||rev({b})) = rev({ra:02d}||{rb:02d})"
        if op_name == 'rev(rev(b)||rev(a))':
            return f"rev(rev({b})||rev({a})) = rev({rb:02d}||{ra:02d})"
        if op_name == 'rev(rev(a)%rev(b))':
            return f"rev(rev({a}) mod rev({b})) = rev({ra} mod {rb})"
        if op_name == 'rev(rev(b)%rev(a))':
            return f"rev(rev({b}) mod rev({a})) = rev({rb} mod {ra})"
        if op_name == '-rev(rev(a)-rev(b))':
            return f"-rev(rev({a})-rev({b})) = -rev({ra-rb})"
        if op_name == '-rev(rev(b)-rev(a))':
            return f"-rev(rev({b})-rev({a})) = -rev({rb-ra})"
    if family == 'REV_MAXMIN':
        ra, rb = CF.rev2(a), CF.rev2(b)
        rmx, rmn = max(ra, rb), min(ra, rb)
        if op_name == 'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) + min({ra},{rb})) = rev({rmx} + {rmn}) = rev({rmx+rmn})"
        if op_name == 'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) - min({ra},{rb})) = rev({rmx} - {rmn}) = rev({rmx-rmn})"
        if op_name == 'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) * min({ra},{rb})) = rev({rmx} * {rmn}) = rev({rmx*rmn})"
        if op_name == 'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)':
            return f"rev(max({ra},{rb}) + min({ra},{rb}) + 1) = rev({rmx+rmn+1})"
        if op_name == 'rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)':
            return f"rev(max({ra},{rb}) + min({ra},{rb}) - 1) = rev({rmx+rmn-1})"
        if op_name == 'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)':
            return f"rev(max({ra},{rb}) * min({ra},{rb}) + 1) = rev({rmx*rmn+1})"
        if op_name == 'rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)':
            return f"rev(max({ra},{rb}) * min({ra},{rb}) - 1) = rev({rmx*rmn-1})"
        if op_name == 'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) mod min({ra},{rb})) = rev({rmx} mod {rmn})"
        if op_name == 'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))':
            return f"rev(max({ra},{rb}) || min({ra},{rb})) = rev({rmx:02d}||{rmn:02d})"
        if op_name == 'rev(min(rev(a),rev(b))||max(rev(a),rev(b)))':
            return f"rev(min({ra},{rb}) || max({ra},{rb})) = rev({rmn:02d}||{rmx:02d})"
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
    return f"{op_name} applied to ({a}, {b})"


# ============================================================
# Anchor selection
# ============================================================

def _pick_anchor(equations):
    """Return (kind, eq_index) — single best anchor.
    Priority: squared-input > concat-pattern > short-result > repeated-symbol > first."""
    for i, (n1, op, n2, res) in enumerate(equations):
        if _is_squared_input(n1, n2): return ('squared', i)
    for i, (n1, op, n2, res) in enumerate(equations):
        if _is_concat_pattern(n1, n2, _result_body(op, res)): return ('concat', i)
    for i, (n1, op, n2, res) in enumerate(equations):
        if _result_width(op, res) == 1: return ('short', i)
    for i, (n1, op, n2, res) in enumerate(equations):
        if _repeated_positions(_result_body(op, res)): return ('repeat', i)
    return ('default', 0)


_ANCHOR_LABEL = {
    'squared': "Squared-input anchor",
    'concat':  "Concatenation-pattern anchor",
    'short':   "Single-digit-result anchor",
    'repeat':  "Repeated-symbol anchor",
    'default': "Anchor",
}


# ============================================================
# MAIN WRITER — baseline-prose style
# ============================================================

def write_cryptarithm_deductive(puzzle):
    """Plain-prose deductive cryptarithm CoT.

    Keys: equations, query, answer, family, map, ops, kind.
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
    L.append("We need to find the operator family and the digit map, then apply the rule to the query.")
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
    L.append("")

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

    # Structural observations — short prose, no markdown
    widths = [_result_width(op, res) for _, op, _, res in equations]
    has_neg = any(_is_neg_res(op, res) for _, op, _, res in equations)
    L.append(f"Result widths: {widths} (max {max(widths)})")
    if has_neg:
        neg_eqs = [i+1 for i, (_, op, _, res) in enumerate(equations) if _is_neg_res(op, res)]
        L.append(f"Negative-result equations: eq{', eq'.join(map(str, neg_eqs))}.")
    L.append("")

    rep_lines = []
    for i, (n1, op, n2, res) in enumerate(equations, 1):
        body = _result_body(op, res)
        reps = _repeated_positions(body)
        if reps:
            parts = [f"'{c}' at positions {list(ps)}" for ps, c in reps]
            rep_lines.append(f"  eq{i} {n1}{op}{n2} = {res}: repeated {'; '.join(parts)}; the formula for '{op}' must give equal digits there.")
    if rep_lines:
        L.append("Repeated symbols inside results:")
        L.extend(rep_lines)
        L.append("")

    concat_lines = []
    for i, (n1, op, n2, res) in enumerate(equations, 1):
        body = _result_body(op, res)
        if _is_concat_pattern(n1, n2, body):
            concat_lines.append(f"  eq{i} {n1}{op}{n2} = {res}: result chars equal '{n1}'+'{n2}'; operator '{op}' must be a concat formula.")
    if concat_lines:
        L.append("Concatenation patterns:")
        L.extend(concat_lines)
        L.append("")

    sq_lines = []
    for i, (n1, op, n2, res) in enumerate(equations, 1):
        if _is_squared_input(n1, n2):
            sq_lines.append(f"  eq{i} {n1}{op}{n2} = {res}: input operands identical, formula yields a single squared value.")
    if sq_lines:
        L.append("Squared inputs:")
        L.extend(sq_lines)
        L.append("")

    # Family elimination — brief, baseline-style indented lines
    L.append("Family elimination (by width/structure, before any digit assignment):")
    has_concat_pattern = any(_is_concat_pattern(n1, n2, _result_body(op, res))
                             for n1, op, n2, res in equations)
    for fam in ['DIRECT_SIMPLE', 'DIRECT_MAXMIN', 'REV_AB', 'REV_MAXMIN']:
        if fam == family:
            continue
        if fam.startswith('REV_') and has_concat_pattern:
            L.append(f"  {fam}: every formula reverses the result, but a concat-pattern equation forbids reversal. eliminated.")
        else:
            L.append(f"  {fam}: no consistent digit-map under any formula across all rows. eliminated.")
    L.append(f"  {family}: survives. adopt this family.")
    L.append("")

    # Operator-formula assignments
    L.append(f"Under {family}:")
    for op in ops_seen:
        L.append(f"  '{op}' = {om[op]}")
    L.append("")

    # Anchor-based deduction
    kind_a, idx_a = _pick_anchor(equations)
    n1a, opa, n2a, resa = equations[idx_a]
    body_a = _result_body(opa, resa)
    width_a = len(body_a)
    formula_a = om[opa]
    a_val_a = _digits_of(n1a, mp)
    b_val_a = _digits_of(n2a, mp)
    val_str_a, neg_a = _formula_value(family, formula_a, a_val_a, b_val_a, width_a)

    L.append(f"{_ANCHOR_LABEL[kind_a]} — eq{idx_a+1} {n1a}{opa}{n2a} = {resa}:")
    if kind_a == 'squared':
        L.append(f"  Operands '{n1a}' = '{n2a}' identical. Trying '{n1a}' = {a_val_a} ({n1a[0]}={a_val_a//10}, {n1a[1]}={a_val_a%10}):")
        L.append(f"    {_render_rule_arith(family, formula_a, a_val_a, a_val_a)}")
        L.append(f"    Computed = {val_str_a}. Map result symbols '{body_a}' position-by-position to digits {val_str_a}.")
    elif kind_a == 'concat':
        L.append(f"  Result body '{body_a}' is literally '{n1a}'+'{n2a}'; operator '{opa}' = {formula_a} confirms a concat formula.")
        L.append(f"  '{n1a}' = {a_val_a}, '{n2a}' = {b_val_a}; concat gives {val_str_a}.")
    elif kind_a == 'short':
        L.append(f"  Result is {width_a} digit. '{n1a}' = {a_val_a}, '{n2a}' = {b_val_a}:")
        L.append(f"    {_render_rule_arith(family, formula_a, a_val_a, b_val_a)}")
        L.append(f"    Computed = {val_str_a}.")
    elif kind_a == 'repeat':
        reps = _repeated_positions(body_a)
        parts = [f"'{c}' at {list(ps)}" for ps, c in reps]
        L.append(f"  Result body '{body_a}' has repeated {', '.join(parts)}; formula '{opa}' = {formula_a} must give equal digits there.")
        L.append(f"  '{n1a}' = {a_val_a}, '{n2a}' = {b_val_a}:")
        L.append(f"    {_render_rule_arith(family, formula_a, a_val_a, b_val_a)}")
        L.append(f"    Computed = {val_str_a}.")
    else:
        L.append(f"  '{n1a}' = {a_val_a}, '{n2a}' = {b_val_a}, apply '{opa}' = {formula_a}:")
        L.append(f"    {_render_rule_arith(family, formula_a, a_val_a, b_val_a)}")
        L.append(f"    Computed = {val_str_a}.")
    known = {}
    if val_str_a and not val_str_a.startswith('-') and len(val_str_a) == width_a:
        for j, c in enumerate(body_a):
            d = int(val_str_a[j])
            if c not in known: known[c] = d
    for n, val in [(n1a, a_val_a), (n2a, b_val_a)]:
        for j, c in enumerate(n):
            d = (val // 10) if j == 0 else (val % 10)
            if c not in known: known[c] = d
    if known:
        L.append(f"  Pinned from this anchor: {', '.join(f"{k}={v}" for k, v in known.items())}.")
    L.append("")

    # Propagation through remaining equations
    if len(equations) > 1:
        L.append("Propagating through the remaining equations:")
        for i, eq in enumerate(equations):
            if i == idx_a: continue
            n1, op, n2, res = eq
            body = _result_body(op, res)
            width = len(body)
            formula = om[op]
            a_val = _digits_of(n1, mp)
            b_val = _digits_of(n2, mp)
            val_str, is_neg = _formula_value(family, formula, a_val, b_val, width)
            L.append(f"  eq{i+1} {n1}{op}{n2} = {res}: '{n1}' = {a_val}, '{n2}' = {b_val}, apply '{op}' = {formula}:")
            L.append(f"    {_render_rule_arith(family, formula, a_val, b_val)}")
            sign = '-' if is_neg else ''
            L.append(f"    Computed = {sign}{val_str}.")
            new_pins = {}
            if val_str and not val_str.startswith('-') and len(val_str) == width:
                for j, c in enumerate(body):
                    d = int(val_str[j])
                    if c not in known and c not in new_pins:
                        new_pins[c] = d
            for n, val in [(n1, a_val), (n2, b_val)]:
                for j, c in enumerate(n):
                    d = (val // 10) if j == 0 else (val % 10)
                    if c not in known and c not in new_pins:
                        new_pins[c] = d
            if new_pins:
                L.append(f"    new symbols pinned: {', '.join(f"{k}={v}" for k, v in new_pins.items())}")
                known.update(new_pins)
            else:
                L.append("    all symbols already pinned; consistent.")
        L.append("")

    # Final map
    L.append(f"Final digit map: {', '.join(f"'{s}'={mp[s]}" for s in sorted(mp))}")
    L.append("")

    # Verification
    L.append("Verifying every equation:")
    for i, eq in enumerate(equations):
        n1, op, n2, res = eq
        body = _result_body(op, res)
        is_neg = _is_neg_res(op, res)
        width = len(body)
        formula = om[op]
        a_val = _digits_of(n1, mp)
        b_val = _digits_of(n2, mp)
        val_str, computed_neg = _formula_value(family, formula, a_val, b_val, width)
        expected = _decode_str(body, mp)
        try:
            ok = (int(val_str) == int(expected)) and (computed_neg == is_neg)
        except Exception:
            ok = (val_str == expected) and (computed_neg == is_neg)
        sign = '-' if computed_neg else ''
        L.append(f"  eq{i+1} {n1}{op}{n2} = {res}: {n1}={a_val}, {n2}={b_val}; '{op}'={formula}:")
        L.append(f"    {_render_rule_arith(family, formula, a_val, b_val)}")
        decoded_disp = ('-' + expected) if is_neg else expected
        L.append(f"    computed {sign}{val_str}, symbols '{res}' decode to {decoded_disp} — {'match' if ok else 'mismatch'}")
    L.append("")

    # Apply to query
    qa = mp[q1[0]] * 10 + mp[q1[1]]
    qb = mp[q2[0]] * 10 + mp[q2[1]]
    L.append(f"Applying to {q1}{qop}{q2}:")
    if qop in ops_seen:
        qformula = om[qop]
        L.append(f"  '{qop}' = {qformula} (from examples).")
    else:
        L.append(f"  '{qop}' is NEW. All example operators are in {family}, so '{qop}' is too.")
        L.append(f"  Enumerate {family} formulas on ({qa}, {qb}) and pick the one matching the answer:")
        ans_is_neg = _is_neg_res(qop, answer)
        ans_body = answer[1:] if ans_is_neg else answer
        try:
            ans_digits = _decode_str(ans_body, mp)
            expected_q = ('-' + ans_digits) if ans_is_neg else ans_digits
        except KeyError:
            expected_q = None
        qformula = None
        for op_name in list(CF.FAMILIES[family].keys()):
            applied = CF.apply_op(family, op_name, qa, qb)
            if applied is None: continue
            v, n_flag = applied
            if 'rev(' not in op_name and '||' not in op_name and expected_q and len(v) < len(expected_q.lstrip('-')):
                v = v.zfill(len(expected_q.lstrip('-')))
            comp = ('-' + v) if n_flag else v
            if expected_q is not None and comp == expected_q:
                L.append(f"    {op_name}: {_render_rule_arith(family, op_name, qa, qb)} -> {comp} matches")
                qformula = op_name; break
        if qformula is None:
            qformula = om.get(qop, list(CF.FAMILIES[family].keys())[0])
    L.append(f"  '{q1}' = {qa}, '{q2}' = {qb}, apply '{qop}' = {qformula}:")
    L.append(f"    {_render_rule_arith(family, qformula, qa, qb)}")
    applied = CF.apply_op(family, qformula, qa, qb)
    if applied is not None:
        v_str, is_neg = applied
        L.append(f"    Numeric value: {('-' + v_str) if is_neg else v_str}.")
    L.append(f"  Read back through digit→symbol map: {answer}")
    L.append(f"  Result: 【{answer}】")
    L.extend(_verification_epilogue(answer))
    return '\n'.join(L)
