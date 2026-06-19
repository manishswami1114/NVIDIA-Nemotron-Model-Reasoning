#!/usr/bin/env python3
"""V19 unified writer.

All four "find the rule" categories use the SAME baseline
equation_numeric_deduce template (8-block search per operator + apply block
+ verification + boxed answer).

For cryptarithm, symbols are converted to integers via the verified solver
map BEFORE writing the CoT, so the search arithmetic is identical to
equation_numeric_deduce. For equation_numeric_guess, after the example
operators are searched we ALSO do a full 8-block search for the query
operator anchored on the single `(qa, qb, train_csv_answer)` row.
"""
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the baseline machinery from v16_writers.
import v16_writers as _v16
from v16_writers import (
    _header_section,
    _verification_epilogue,
    _render_operator_search,
    _render_block,
    _render_apply,
    _normalize_result_for_compare,
    _detect_sign_prefix,
    _resolve_answer_for_value,
    _apply_with_transform,
    _build_actions_str,
    rev_int,
    rev_str_pad,
    _TRANSFORMS,
)


# ============================================================
# V19-extended formula library — add max||min and min||max so the
# baseline search can recognise cryptarithm puzzles whose verified
# formula comes from DIRECT_MAXMIN's concat side. We monkey-patch v16's
# RARE_OPS, _FORMULA_RENDERERS, and _compute_value_str so the existing
# _render_block / _render_operator_search machinery picks them up.
# ============================================================

def _ren_max_concat_min(a, b):
    mx, mn = max(a, b), min(a, b)
    return f"max({a},{b}) || min({a},{b}) = {mx:02d}{mn:02d}"


def _ren_min_concat_max(a, b):
    mx, mn = max(a, b), min(a, b)
    return f"min({a},{b}) || max({a},{b}) = {mn:02d}{mx:02d}"


def _ren_max_sub_min_p1(a, b):
    mx, mn = max(a, b), min(a, b)
    return f"max({a},{b}) - min({a},{b}) + 1 = {mx} - {mn} + 1 = {mx - mn + 1}"


def _ren_max_sub_min_m1(a, b):
    mx, mn = max(a, b), min(a, b)
    return f"max({a},{b}) - min({a},{b}) - 1 = {mx} - {mn} - 1 = {mx - mn - 1}"


def _ren_min_sub_max_p1(a, b):
    mx, mn = max(a, b), min(a, b)
    return f"min({a},{b}) - max({a},{b}) + 1 = {mn} - {mx} + 1 = {mn - mx + 1}"


def _ren_min_sub_max_m1(a, b):
    mx, mn = max(a, b), min(a, b)
    return f"min({a},{b}) - max({a},{b}) - 1 = {mn} - {mx} - 1 = {mn - mx - 1}"


_V19_EXTRA = {
    'max||min':       (_ren_max_concat_min, lambda a, b: f"{max(a,b):02d}{min(a,b):02d}"),
    'min||max':       (_ren_min_concat_max, lambda a, b: f"{min(a,b):02d}{max(a,b):02d}"),
    '(max-min)+1':    (_ren_max_sub_min_p1, lambda a, b: str(max(a,b) - min(a,b) + 1)),
    '(max-min)-1':    (_ren_max_sub_min_m1, lambda a, b: str(max(a,b) - min(a,b) - 1)),
    '(min-max)+1':    (_ren_min_sub_max_p1, lambda a, b: str(min(a,b) - max(a,b) + 1)),
    '(min-max)-1':    (_ren_min_sub_max_m1, lambda a, b: str(min(a,b) - max(a,b) - 1)),
}

# Patch v16's _FORMULA_RENDERERS
for _name, (_render_fn, _) in _V19_EXTRA.items():
    _v16._FORMULA_RENDERERS[_name] = _render_fn

# Wrap _compute_value_str
_orig_compute = _v16._compute_value_str
def _compute_value_str_v19(name, a, b):
    if name in _V19_EXTRA:
        return _V19_EXTRA[name][1](a, b)
    return _orig_compute(name, a, b)
_v16._compute_value_str = _compute_value_str_v19

# Extend RARE_OPS — append the two new formulas at the end
COMMON_OPS = list(_v16.COMMON_OPS)
RARE_OPS = list(_v16.RARE_OPS)
for _name in _V19_EXTRA:
    if _name not in RARE_OPS:
        RARE_OPS.append(_name)
        _v16.RARE_OPS.append(_name)
_FORMULA_RENDERERS = _v16._FORMULA_RENDERERS
_compute_value_str = _compute_value_str_v19


# ============================================================
# Cryptarithm puzzle → equation_numeric-shaped data
# ============================================================

def _is_neg_res_crypto(op, res):
    return len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])


def _crypto_to_eqnum_data(puzzle):
    """Convert a cryptarithm puzzle into the same data shape used by
    write_equation_numeric_guess: list of (a:int, op:str, b:int, result_str:str)
    and a query (qa:int, qop:str, qb:int) + numeric answer string.

    The `result_str` preserves the operator-as-sign prefix exactly as in the
    original cryptarithm result (e.g. "^44" stays "^44" so the sign-prefix
    detection in _header_section works).

    Returns (eqs, query, answer_numeric) where answer_numeric is the symbol
    train.csv answer translated through the digit map (with operator-as-sign
    prefix preserved if present).
    """
    mp = puzzle['map']
    eqs = []
    for n1s, op, n2s, res in puzzle['equations']:
        a = mp[n1s[0]] * 10 + mp[n1s[1]]
        b = mp[n2s[0]] * 10 + mp[n2s[1]]
        if _is_neg_res_crypto(op, res):
            body_digits = ''.join(str(mp[c]) for c in res[1:])
            result_str = op + body_digits
        else:
            result_str = ''.join(str(mp[c]) for c in res)
        eqs.append((a, op, b, result_str))
    q1, qop, q2 = puzzle['query']
    qa = mp[q1[0]] * 10 + mp[q1[1]]
    qb = mp[q2[0]] * 10 + mp[q2[1]]
    # Decode the train.csv answer (symbol form) to numeric form.
    answer = puzzle['answer']
    if _is_neg_res_crypto(qop, answer):
        # answer like "^44" — operator-as-sign prefix
        ans_body = answer[1:]
        try:
            ans_num = qop + ''.join(str(mp[c]) for c in ans_body)
        except KeyError:
            ans_num = answer
    else:
        try:
            ans_num = ''.join(str(mp[c]) for c in answer)
        except KeyError:
            ans_num = answer
    return eqs, (qa, qop, qb), ans_num


# ============================================================
# Shared writer body — does the common header + per-operator searches.
# Returns (lines_list, formulas_used_dict).
# ============================================================

def _search_example_operators(L, examples, query, ops_seen, op_to_eqs):
    """For each example operator, render the full 8-block search.
    Returns formulas_used = {op: (formula_name, rev_ops, rev_result)}."""
    formulas_used = {}
    for op in ops_seen:
        eqs = op_to_eqs[op]
        expected_strs = []
        for a, oo, b, r in eqs:
            n, _ = _normalize_result_for_compare(op, r)
            expected_strs.append(n)
        anchor_disp = ', '.join(f"{a:02d}{op}{b:02d} = {r}" for a, _, b, r in eqs)
        L.append(f"Looking at operator 【{op}】 [{anchor_disp}]:")
        block_lines, matched, m_rev_ops, m_rev_res = _render_operator_search(
            op, eqs, expected_strs)
        L.extend(block_lines)
        if matched is None:
            L.append("  (no formula in the library matched all examples for this operator)")
        else:
            formulas_used[op] = (matched, m_rev_ops, m_rev_res)
        L.append("")
    return formulas_used


def _search_query_operator(L, qa, qop, qb, answer):
    """Full 8-block search for the query operator anchored on (qa, qop, qb, answer).
    Returns (formula_name, rev_ops, rev_result) or (None, None, None) if no match."""
    norm_answer, _ = _normalize_result_for_compare(qop, answer)
    anchor = [(qa, qop, qb, answer)]
    L.append(f"Looking at operator 【{qop}】 [{qa:02d}{qop}{qb:02d} = {answer}]:")
    transform_kinds = [('common', COMMON_OPS), ('rare', RARE_OPS)]
    for transform_kind, ops_list in transform_kinds:
        for ro, rr in _TRANSFORMS:
            block_lines, matched = _render_block(
                transform_kind, ops_list, anchor, ro, rr, [norm_answer])
            L.extend(block_lines)
            if matched is not None:
                L.append("")
                return matched, ro, rr
    L.append("")
    # Brute fallback (shouldn't happen for verified puzzles)
    for op_name in COMMON_OPS + RARE_OPS:
        for ro, rr in _TRANSFORMS:
            aa, bb = (rev_int(qa), rev_int(qb)) if ro else (qa, qb)
            val = _compute_value_str(op_name, aa, bb)
            if val is None: continue
            comp = rev_str_pad(val) if rr else val
            if _resolve_answer_for_value(comp, qop, answer):
                return op_name, ro, rr
    return None, None, None


# ============================================================
# WRITER: equation_numeric_deduce-style
# Works for both equation_numeric and cryptarithm (after symbol→int convert).
# ============================================================

def _write_unified_format(examples, query, answer_for_search, answer_for_box=None,
                          symbol_decode_note=False):
    """Render the full baseline CoT.

    answer_for_search: numeric-form answer used inside the formula search and
        the apply block (e.g. "66" or "-66" for cryptarithm; "1162" or "*53"
        for equation_numeric where the operator-as-sign is preserved).
    answer_for_box: original train.csv answer for the \\boxed{...} line. For
        cryptarithm this is the symbol form (e.g. "??"); for equation_numeric
        it equals answer_for_search.
    symbol_decode_note: if True, append a "decoded back to symbols" note after
        the apply block before the boxed answer.
    """
    if answer_for_box is None:
        answer_for_box = answer_for_search
    qa, qop, qb = query
    L = _header_section(examples, query)

    # Extended: examples + query (with numeric answer)
    extended = list(examples)
    extended.append((qa, qop, qb, answer_for_search))

    op_order = []
    op_to_eqs = {}
    for a, op, b, r in extended:
        if op not in op_order:
            op_order.append(op); op_to_eqs[op] = []
        op_to_eqs[op].append((a, op, b, r))

    formulas_used = _search_example_operators(L, extended, query, op_order, op_to_eqs)

    if qop in formulas_used:
        name, ro, rr = formulas_used[qop]
        L.extend(_render_apply(qop, qa, qb, name, ro, rr, answer_for_search))
    else:
        L.append(f"Applying to {qa:02d}{qop}{qb:02d}:")
        L.append(f"  Result: 【{answer_for_search}】")

    if symbol_decode_note and answer_for_box != answer_for_search:
        L.append(f"  Read back through the digit→symbol map: {answer_for_box}")

    L.extend(_verification_epilogue(answer_for_box))
    return '\n'.join(L)


# ============================================================
# Public API
# ============================================================

def write_equation_numeric_guess(puzzle):
    """V19 equation_numeric_guess — query operator search uses the same
    8-block template as a deduce operator."""
    from v16_writers import _parse_eq_prompt
    examples, query = _parse_eq_prompt(puzzle['prompt'])
    return _write_unified_format(examples, query, puzzle['answer'])


def write_cryptarithm(puzzle):
    """V19 cryptarithm (deduce + guess) — same unified template."""
    examples, query, answer_numeric = _crypto_to_eqnum_data(puzzle)
    return _write_unified_format(
        examples, query,
        answer_for_search=answer_numeric,
        answer_for_box=puzzle['answer'],
        symbol_decode_note=True,
    )
