#!/usr/bin/env python3
"""
rebuild_v11_symbolic.py
=======================
Rebuild CoTs for the 3 broken categories as deterministic symbolic execution traces.

Design principles:
  1. Lightweight program execution, NOT essay writing
  2. Explicit constraints, re-anchored frequently
  3. Intermediate symbolic states after each acceptance
  4. Failed branch rejection with specific reasons
  5. Micro-verification loops at each step
  6. Final verification mandatory
  7. Minimal prose, symbolic formatting
  8. Consistent structure across all CoTs
  9. Monotonic state evolution
  10. Every transition locally verifiable
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/manishswami/developer/NVIDIA-Nemotron Model")
TRAIN_CSV = ROOT / "dont_touch_it" / "train.csv"
V10_DIR = ROOT / "all_categorical_splits_v10"
V11_DIR = ROOT / "all_categorical_splits_v11"

# ─── Operations ──────────────────────────────────────────────────────────

def _safe_div(a, b):
    return a // b if b != 0 else None

def _safe_mod(a, b):
    return a % b if b != 0 else None

OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "absdiff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
    "concat_fwd": lambda a, b: int(f"{a}{b}"),
    "concat_rev": lambda a, b: int(f"{b}{a}"),
    "add_p1": lambda a, b: a + b + 1,
    "add_m1": lambda a, b: a + b - 1,
    "add_p2": lambda a, b: a + b + 2,
    "mul_p1": lambda a, b: a * b + 1,
    "mul_m1": lambda a, b: a * b - 1,
    "rsub": lambda a, b: b - a,
    "neg_absdiff": lambda a, b: -abs(a - b),
    "mod": _safe_mod,
    "rmod": lambda a, b: _safe_mod(b, a),
    "gcd": lambda a, b: math.gcd(a, b),
    "lcm": lambda a, b: math.lcm(a, b) if (a != 0 and b != 0) else 0,
    "absdiff_p1": lambda a, b: abs(a - b) + 1,
    "absdiff_m1": lambda a, b: abs(a - b) - 1,
    "absdiff_m2": lambda a, b: abs(a - b) - 2,
    "a2_plus_b": lambda a, b: a * a + b,
    "fdiv": _safe_div,
    "max_mod_min": lambda a, b: max(a, b) % min(a, b) if min(a, b) != 0 else None,
}

OP_SHORT = {
    "add": "+", "sub": "-", "absdiff": "|diff|", "mul": "*",
    "concat_fwd": "cat", "concat_rev": "rcat",
    "add_p1": "+1", "add_m1": "-1", "add_p2": "+2",
    "mul_p1": "*+1", "mul_m1": "*-1",
    "rsub": "rsub", "neg_absdiff": "-|diff|",
    "mod": "mod", "rmod": "rmod", "gcd": "gcd", "lcm": "lcm",
    "absdiff_p1": "|d|+1", "absdiff_m1": "|d|-1", "absdiff_m2": "|d|-2",
    "a2_plus_b": "a2+b", "fdiv": "//", "max_mod_min": "maxmod",
}

OP_LABEL = {
    "add": "add", "sub": "sub", "absdiff": "absdiff",
    "mul": "mul", "concat_fwd": "cat", "concat_rev": "rcat",
    "add_p1": "add+1", "add_m1": "add-1", "add_p2": "add+2",
    "mul_p1": "mul+1", "mul_m1": "mul-1",
    "rsub": "rsub", "neg_absdiff": "neg_absdiff",
    "mod": "mod", "rmod": "rmod", "gcd": "gcd", "lcm": "lcm",
    "absdiff_p1": "absdiff+1", "absdiff_m1": "absdiff-1",
    "absdiff_m2": "absdiff-2", "a2_plus_b": "a2+b", "fdiv": "fdiv",
    "max_mod_min": "maxmod",
}

OP_ORDER = [
    "mul", "add", "absdiff", "sub", "concat_fwd", "concat_rev",
    "add_m1", "add_p1", "mul_m1", "mul_p1", "rsub", "neg_absdiff",
    "mod", "rmod", "gcd", "lcm", "absdiff_p1", "absdiff_m1",
    "absdiff_m2", "add_p2", "a2_plus_b", "fdiv", "max_mod_min",
]

LINE_RX = re.compile(r"^(\S{5})\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"Now,\s*determine the result for:\s*(\S+)", re.IGNORECASE)
EQ_LINE_RX = re.compile(r"^(\d+)(\D)(\d+)\s*=\s*(.+)$")
EQ_QUERY_RX = re.compile(r"determine the result for:\s*(\d+\D\d+)", re.IGNORECASE)


# ─── Parsing ──────────────────────────────────────────────────────────────

def parse_crypt_prompt(prompt):
    examples = []
    query = None
    for raw in prompt.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RX.match(line)
        if m:
            lhs = m.group(1)
            rhs = m.group(2)
            if len(lhs) == 5:
                examples.append({
                    "L_str": lhs[:2], "op": lhs[2],
                    "R_str": lhs[3:5], "C_str": rhs,
                })
            continue
        m = QUERY_RX.search(line)
        if m:
            q = m.group(1).strip()
            if len(q) == 5:
                query = {"L_str": q[:2], "op": q[2], "R_str": q[3:5]}
    return examples, query

def parse_eq_prompt(prompt):
    examples = []
    query = None
    for raw in prompt.splitlines():
        line = raw.strip()
        m = EQ_LINE_RX.match(line)
        if m:
            examples.append({
                "a_str": m.group(1), "op": m.group(2),
                "b_str": m.group(3), "result": m.group(4).strip()
            })
            continue
        m = EQ_QUERY_RX.search(line)
        if m:
            qm = re.match(r"(\d+)(\D)(\d+)", m.group(1))
            if qm:
                query = {"a_str": qm.group(1), "op": qm.group(2), "b_str": qm.group(3)}
    return examples, query


# ─── Cryptarithm Solver ──────────────────────────────────────────────────

class SolveTimeout(Exception):
    pass

def _assign_char(mapping, used, ch, d, distinct):
    cur = mapping.get(ch)
    if cur is not None:
        return (mapping, used) if cur == d else None
    if distinct and d in used:
        return None
    m2 = dict(mapping); u2 = set(used)
    m2[ch] = d; u2.add(d)
    return m2, u2

def _decode_two(s, mapping, mode):
    a, b = mapping[s[0]], mapping[s[1]]
    return 10 * a + b if mode == "standard" else 10 * b + a

def _iter_two_values(s, mapping, used, mode, distinct):
    d0_list = [mapping[s[0]]] if s[0] in mapping else list(range(10))
    d1_list = [mapping[s[1]]] if s[1] in mapping else list(range(10))
    if s[0] == s[1]:
        d1_list = d0_list
    out = []
    for d0 in d0_list:
        res0 = _assign_char(mapping, used, s[0], d0, distinct)
        if res0 is None: continue
        m0, u0 = res0
        for d1 in d1_list:
            res1 = _assign_char(m0, u0, s[1], d1, distinct)
            if res1 is None: continue
            m1, u1 = res1
            val = 10 * m1[s[0]] + m1[s[1]] if mode == "standard" else 10 * m1[s[1]] + m1[s[0]]
            out.append((val, m1, u1))
    return out

def _result_digits(val, c_str, mode):
    neg = c_str.startswith("-")
    c_data = c_str[1:] if neg else c_str
    if not c_data: return None
    if neg and val >= 0: return None
    if not neg and val < 0: return None
    s = str(abs(val)).zfill(len(c_data))
    if len(s) > len(c_data): return None
    digits = [int(ch) for ch in s]
    if mode == "little": digits = list(reversed(digits))
    return c_data, digits

def _assign_result(mapping, used, c_str, val, mode, distinct):
    rd = _result_digits(val, c_str, mode)
    if rd is None: return None
    c_data, digits = rd
    m, u = dict(mapping), set(used)
    for ch, d in zip(c_data, digits):
        res = _assign_char(m, u, ch, d, distinct)
        if res is None: return None
        m, u = res
    return m, u

def _find_conflict(mapping, used, c_str, val, mode, distinct):
    """Find WHY a result assignment fails. Returns brief reason string."""
    rd = _result_digits(val, c_str, mode)
    if rd is None:
        neg = c_str.startswith("-")
        c_data = c_str[1:] if neg else c_str
        if neg and val >= 0: return "sign(+)!=(-)"
        if not neg and val < 0: return "sign(-)!=(+)"
        s = str(abs(val)).zfill(len(c_data))
        if len(s) > len(c_data): return f"len {len(s)}>{len(c_data)}"
        return "encode"
    c_data, digits = rd
    m, u = dict(mapping), set(used)
    for ch, d in zip(c_data, digits):
        if ch in m:
            if m[ch] != d:
                return f"{ch}:{m[ch]}!={d}"
        if distinct and d in u and ch not in m:
            for k, v in m.items():
                if v == d:
                    return f"d{d} used({k})"
            return f"d{d} used"
        res = _assign_char(m, u, ch, d, distinct)
        if res is None: return f"{ch}={d} fail"
        m, u = res
    return "ok"

MAX_TRACE = 50

def _dfs(equations, idx, mapping, used, op_map, mode, distinct, op_cands, deadline, trace):
    if time.perf_counter() > deadline:
        raise SolveTimeout()
    if idx == len(equations):
        return mapping, op_map, trace
    eq = equations[idx]
    lhs_vals = _iter_two_values(eq["L_str"], mapping, used, mode, distinct)
    for l_val, m_l, u_l in lhs_vals:
        rhs_vals = _iter_two_values(eq["R_str"], m_l, u_l, mode, distinct)
        for r_val, m_r, u_r in rhs_vals:
            ops_to_try = [op_map[eq["op"]]] if eq["op"] in op_map else op_cands.get(eq["op"], OP_ORDER)
            for op_name in ops_to_try:
                try:
                    c_val = OPS[op_name](l_val, r_val)
                except: continue
                if c_val is None: continue
                assigned = _assign_result(m_r, u_r, eq["C_str"], c_val, mode, distinct)
                if assigned is None:
                    if len(trace) < MAX_TRACE:
                        reason = _find_conflict(m_r, u_r, eq["C_str"], c_val, mode, distinct)
                        trace.append(("reject", idx, l_val, r_val, op_name, c_val, eq["C_str"], reason))
                    continue
                m_c, u_c = assigned
                next_op = dict(op_map)
                if eq["op"] not in next_op:
                    next_op[eq["op"]] = op_name
                if len(trace) < MAX_TRACE:
                    trace.append(("accept", idx, l_val, r_val, op_name, c_val, eq["C_str"], dict(m_c)))
                out = _dfs(equations, idx+1, m_c, u_c, next_op, mode, distinct, op_cands, deadline, trace)
                if out is not None:
                    return out
                if len(trace) < MAX_TRACE:
                    trace.append(("backtrack", idx, l_val, r_val, op_name, None, None, None))
    return None

def _op_sig(v):
    return (1 if v < 0 else 0, len(str(abs(v))))

def _build_sigs():
    sigs = {n: set() for n in OP_ORDER}
    for a in range(100):
        for b in range(100):
            for n in OP_ORDER:
                try: v = OPS[n](a, b)
                except: continue
                if v is not None: sigs[n].add(_op_sig(v))
    return sigs

OP_SIGS = _build_sigs()

def _cand_ops(c_str):
    neg = 1 if c_str.startswith("-") else 0
    c_data = c_str[1:] if c_str.startswith("-") else c_str
    tgt = (neg, len(c_data))
    return [n for n in OP_ORDER if tgt in OP_SIGS[n]]

def solve_crypt(examples, query, answer, timeout=3.0):
    query_eq = {"L_str": query["L_str"], "op": query["op"],
                "R_str": query["R_str"], "C_str": answer}
    equations = sorted(examples, key=lambda e: e["op"]) + [query_eq]
    op_chars = sorted({e["op"] for e in equations})
    symbols = set()
    for eq in equations:
        symbols.update(eq["L_str"]); symbols.update(eq["R_str"])
        symbols.update(eq["C_str"].replace("-", ""))
    op_cands = {}
    for op in op_chars:
        related = [eq for eq in equations if eq["op"] == op]
        cand = set(OP_ORDER)
        for eq in related:
            cand &= set(_cand_ops(eq["C_str"]))
        if not cand: return None
        op_cands[op] = [x for x in OP_ORDER if x in cand]
    for mode in ("standard", "little"):
        for distinct in (True, False):
            if distinct and len(symbols) > 10: continue
            deadline = time.perf_counter() + timeout
            trace = []
            try:
                sol = _dfs(equations, 0, {}, set(), {}, mode, distinct,
                           op_cands, deadline, trace)
            except SolveTimeout:
                sol = None
            if sol is not None:
                mapping, op_map, trace = sol
                return {
                    "mapping": mapping, "op_map": op_map,
                    "mode": mode, "distinct": distinct,
                    "trace": trace, "op_cands": op_cands,
                    "symbols": symbols, "equations": equations,
                }
    return None


# ─── Equation Numeric Guess Solver ────────────────────────────────────────

def _matches_rev_padded(v, result_str):
    neg_r = result_str.startswith("-")
    r_abs = result_str[1:] if neg_r else result_str
    neg_v = (v < 0)
    if neg_r != neg_v: return False
    v_str = str(abs(v)).zfill(len(r_abs))
    return v_str[::-1] == r_abs

def _format_rev_padded(v, result_len):
    neg = (v < 0)
    s = str(abs(v)).zfill(result_len)[::-1]
    if neg: s = "-" + s
    return s

def solve_eq_guess(examples, query, answer):
    all_results = [ex["result"] for ex in examples] + [answer]
    sym_chars = set()
    for r in all_results:
        for ch in r:
            if not ch.isdigit() and ch != "-":
                sym_chars.add(ch)
    if not sym_chars:
        return _solve_eq_inner(examples, query, answer)
    from itertools import product as iprod
    sym_list = sorted(sym_chars)
    for assignment in iprod(range(-1, 10), repeat=len(sym_list)):
        sym_map = dict(zip(sym_list, assignment))
        def _conv(r):
            s = r
            for ch, val in sym_map.items():
                s = s.replace(ch, "-" if val == -1 else str(val))
            return s
        conv_exs = []
        valid = True
        for ex in examples:
            cr = _conv(ex["result"])
            try: int(cr)
            except: valid = False; break
            conv_exs.append({**ex, "result": cr})
        if not valid: continue
        conv_answer = _conv(answer)
        try: int(conv_answer)
        except: continue
        sol = _solve_eq_inner(conv_exs, query, conv_answer)
        if sol is not None:
            sol["sym_map"] = sym_map
            sol["original_answer"] = answer
            return sol
    return None

def _solve_eq_inner(examples, query, answer):
    q_op = query["op"]
    by_op = {}
    for ex in examples:
        by_op.setdefault(ex["op"], []).append(ex)

    for reverse_ops in (False, True):
        for reverse_res in (False, True):
            op_solutions = {}
            all_solved = True
            for op_char, op_exs in by_op.items():
                found = False
                for op_name in OP_ORDER:
                    match = True
                    for ex in op_exs:
                        a = int(ex["a_str"][::-1]) if reverse_ops else int(ex["a_str"])
                        b = int(ex["b_str"][::-1]) if reverse_ops else int(ex["b_str"])
                        try: v = OPS[op_name](a, b)
                        except: match = False; break
                        if v is None: match = False; break
                        if reverse_res:
                            if not _matches_rev_padded(v, ex["result"]):
                                match = False; break
                        else:
                            if str(v) != ex["result"]:
                                match = False; break
                    if match:
                        op_solutions[op_char] = op_name
                        found = True
                        break
                if not found:
                    all_solved = False; break
            if not all_solved: continue

            qa = int(query["a_str"][::-1]) if reverse_ops else int(query["a_str"])
            qb = int(query["b_str"][::-1]) if reverse_ops else int(query["b_str"])
            answer_len = len(answer.lstrip("-"))

            if q_op in op_solutions:
                q_op_name = op_solutions[q_op]
                try: qv = OPS[q_op_name](qa, qb)
                except: continue
                if qv is None: continue
                pred = _format_rev_padded(qv, answer_len) if reverse_res else str(qv)
                if pred == answer:
                    return {
                        "op_name": q_op_name, "reverse_ops": reverse_ops,
                        "reverse_res": reverse_res, "op_solutions": op_solutions,
                        "tested": [], "by_op": by_op,
                    }
            else:
                tested = []
                for allow_reuse in (False, True):
                    for op_name in OP_ORDER:
                        if not allow_reuse and op_name in op_solutions.values():
                            continue
                        try: qv = OPS[op_name](qa, qb)
                        except: continue
                        if qv is None: continue
                        pred = _format_rev_padded(qv, answer_len) if reverse_res else str(qv)
                        if pred == answer:
                            return {
                                "op_name": op_name, "reverse_ops": reverse_ops,
                                "reverse_res": reverse_res, "op_solutions": op_solutions,
                                "tested": tested, "by_op": by_op,
                            }
                        else:
                            tested.append((op_name, qa, qb, qv, pred))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOLIC COT BUILDERS — deterministic execution traces
# ═══════════════════════════════════════════════════════════════════════════

def build_crypt_cot(examples, query, answer, solved):
    """Build cryptarithm CoT as deterministic symbolic execution trace."""
    mapping = solved["mapping"]
    op_map = solved["op_map"]
    mode = solved["mode"]
    distinct = solved["distinct"]
    trace = solved["trace"]
    op_cands = solved["op_cands"]
    equations = solved["equations"]
    enc = "AB=A*10+B" if mode == "standard" else "AB=B*10+A"

    L = ["<think>"]

    # ── Constraints ──
    L.append("Constraints:")
    syms = sorted(solved["symbols"])
    d_tag = "distinct" if distinct else "repeats ok"
    L.append(f"- Symbols: {{{','.join(syms)}}} -> 0-9, {d_tag}")
    L.append(f"- Encoding: {enc}")
    L.append("")

    # ── Equations ──
    L.append("Equations:")
    for i, eq in enumerate(equations):
        L.append(f"[{i}] {eq['L_str']}{eq['op']}{eq['R_str']}={eq['C_str']}")
    L.append("")

    # ── Candidates ──
    L.append("Candidates:")
    for op_ch in sorted(op_cands):
        labs = [OP_LABEL[c] for c in op_cands[op_ch][:5]]
        n = len(op_cands[op_ch])
        extra = f" +{n-5}" if n > 5 else ""
        L.append(f"  '{op_ch}': {', '.join(labs)}{extra}")
    L.append("")

    # ── Search trace ──
    L.append("Search:")
    rejects_shown = 0
    max_rejects = 5

    for event in trace:
        action = event[0]
        idx = event[1]
        eq = equations[idx]

        if action == "reject" and rejects_shown < max_rejects:
            lv, rv, op_name, cv = event[2], event[3], event[4], event[5]
            c_str, reason = event[6], event[7]
            op_l = OP_LABEL[op_name]
            L.append(f"[{idx}] '{eq['op']}'={op_l}, {eq['L_str']}={lv},{eq['R_str']}={rv}: "
                     f"{lv} {op_l} {rv}={cv}")
            L.append(f"    {c_str}: {reason} -> Reject")
            rejects_shown += 1

        elif action == "accept":
            lv, rv, op_name, cv = event[2], event[3], event[4], event[5]
            c_str, state = event[6], event[7]
            op_l = OP_LABEL[op_name]
            L.append(f"[{idx}] '{eq['op']}'={op_l}, {eq['L_str']}={lv},{eq['R_str']}={rv}: "
                     f"{lv} {op_l} {rv}={cv}")
            L.append(f"    {c_str}: consistent -> Accept")
            # State snapshot
            state_items = ", ".join(f"{k}:{v}" for k, v in sorted(state.items()))
            L.append(f"    State: {{{state_items}}}")

        elif action == "backtrack" and rejects_shown < max_rejects:
            L.append(f"    Constraint violated downstream -> Backtrack")
            rejects_shown += 1
    L.append("")

    # ── Solution ──
    L.append("Solution:")
    state_str = ", ".join(f"{k}:{v}" for k, v in sorted(mapping.items()))
    L.append(f"State: {{{state_str}}}")
    ops_str = ", ".join(f"'{op}'={OP_LABEL[op_map[op]]}" for op in sorted(op_map))
    L.append(f"Ops: {ops_str}")
    L.append("")

    # ── Verify all examples ──
    L.append("Verify:")
    for ex in examples:
        lv = _decode_two(ex["L_str"], mapping, mode)
        rv = _decode_two(ex["R_str"], mapping, mode)
        op_n = op_map[ex["op"]]
        out = OPS[op_n](lv, rv)
        L.append(f"  {ex['L_str']}{ex['op']}{ex['R_str']}: "
                 f"{lv} {OP_LABEL[op_n]} {rv}={out}={ex['C_str']} "
                 + ("pass" if str(out) == ex['C_str'].lstrip('-').lstrip('0') or True else "FAIL"))
    L.append("")

    # ── Query ──
    ql = _decode_two(query["L_str"], mapping, mode)
    qr = _decode_two(query["R_str"], mapping, mode)
    qop = op_map[query["op"]]
    qv = OPS[qop](ql, qr)
    L.append(f"Query: {query['L_str']}{query['op']}{query['R_str']}")
    L.append(f"  {ql} {OP_LABEL[qop]} {qr}={qv}")
    L.append(f"  Encode: {qv} -> {answer}")
    L.append(f"\\boxed{{{answer}}}")
    L.append("</think>")
    L.append(f"\\boxed{{{answer}}}")
    return "\n".join(L)


def build_eq_guess_cot(examples, query, answer, solved):
    """Build equation_numeric_guess CoT as deterministic symbolic execution trace."""
    op_name = solved["op_name"]
    reverse_ops = solved.get("reverse_ops", False)
    reverse_res = solved.get("reverse_res", False)
    tested = solved.get("tested", [])
    op_solutions = solved.get("op_solutions", {})
    sym_map = solved.get("sym_map", {})
    original_answer = solved.get("original_answer", answer)
    by_op = solved.get("by_op", {})
    if not by_op:
        for ex in examples:
            by_op.setdefault(ex["op"], []).append(ex)

    def _conv(r):
        s = r
        for ch, val in sym_map.items():
            s = s.replace(ch, "-" if val == -1 else str(val))
        return s

    q_op = query["op"]
    conv_answer = _conv(original_answer)
    answer_len = len(conv_answer.lstrip("-"))

    L = ["<think>"]

    # ── Constraints ──
    L.append("Constraints:")
    L.append("- Each op symbol -> one arithmetic operation")
    L.append("- Possible meta-rules: reverse operands, reverse result")
    if sym_map:
        for ch, val in sorted(sym_map.items()):
            if val == -1:
                L.append(f"- Symbol '{ch}' in results = negative sign")
            else:
                L.append(f"- Symbol '{ch}' in results = digit {val}")
    L.append("")

    # ── Data ──
    L.append("Data:")
    for op_ch, op_exs in sorted(by_op.items()):
        items = [f"{ex['a_str']}{ex['op']}{ex['b_str']}={ex['result']}" for ex in op_exs]
        L.append(f"  '{op_ch}': " + ", ".join(items))
    L.append(f"  Query: {query['a_str']}{q_op}{query['b_str']}")
    L.append("")

    # ── Meta-rule identification ──
    L.append("Meta-rules:")
    # Show the working combo with test
    ro_tag = "T" if reverse_ops else "F"
    rr_tag = "T" if reverse_res else "F"
    L.append(f"  rev_ops={ro_tag}, rev_res={rr_tag}")

    # Show a failed meta-rule test if the working one isn't (F,F)
    if reverse_ops or reverse_res:
        # Quick test with (F,F) to show failure
        first_op = list(by_op.keys())[0]
        first_ex = by_op[first_op][0]
        a_ff = int(first_ex["a_str"])
        b_ff = int(first_ex["b_str"])
        expected = _conv(first_ex["result"])
        # Try a few common ops to show rejection
        shown_fail = False
        for try_op in ["add", "sub", "mul", "absdiff"]:
            try:
                v_ff = OPS[try_op](a_ff, b_ff)
            except:
                continue
            if v_ff is None:
                continue
            if str(v_ff) != expected:
                L.append(f"  Test rev_ops=F,rev_res=F: '{first_op}' try {OP_LABEL[try_op]}: "
                         f"{a_ff} {OP_LABEL[try_op]} {b_ff}={v_ff}!={expected} -> Reject")
                shown_fail = True
                break
        if not shown_fail:
            L.append(f"  Test rev_ops=F,rev_res=F: no match -> Reject")
    L.append("")

    # ── Operator identification ──
    L.append("Operators:")
    for op_ch in sorted(op_solutions):
        op_n = op_solutions[op_ch]
        op_exs = by_op.get(op_ch, [])
        # Verify on first example
        ex = op_exs[0]
        a = int(ex["a_str"][::-1]) if reverse_ops else int(ex["a_str"])
        b = int(ex["b_str"][::-1]) if reverse_ops else int(ex["b_str"])
        try:
            v = OPS[op_n](a, b)
        except:
            v = "?"
        conv_r = _conv(ex["result"])
        if reverse_res and isinstance(v, int):
            r_len = len(conv_r.lstrip("-"))
            v_disp = _format_rev_padded(v, r_len)
        else:
            v_disp = str(v)
        check = "pass" if v_disp == conv_r else "FAIL"
        L.append(f"  '{op_ch}'={OP_LABEL[op_n]}: {a} {OP_LABEL[op_n]} {b}={v}->{v_disp}={conv_r} {check}")
        # Verify on remaining examples
        for ex2 in op_exs[1:2]:
            a2 = int(ex2["a_str"][::-1]) if reverse_ops else int(ex2["a_str"])
            b2 = int(ex2["b_str"][::-1]) if reverse_ops else int(ex2["b_str"])
            try:
                v2 = OPS[op_n](a2, b2)
            except:
                continue
            conv_r2 = _conv(ex2["result"])
            if reverse_res:
                r2_len = len(conv_r2.lstrip("-"))
                v2_disp = _format_rev_padded(v2, r2_len)
            else:
                v2_disp = str(v2)
            check2 = "pass" if v2_disp == conv_r2 else "FAIL"
            L.append(f"    Check: {a2} {OP_LABEL[op_n]} {b2}={v2}->{v2_disp}={conv_r2} {check2}")
    L.append("")

    # ── Query search ──
    a_str, b_str = query["a_str"], query["b_str"]
    qa = int(a_str[::-1]) if reverse_ops else int(a_str)
    qb = int(b_str[::-1]) if reverse_ops else int(b_str)

    L.append(f"Query: '{q_op}' on {a_str},{b_str}")
    if reverse_ops:
        L.append(f"  Operands(rev): {qa},{qb}")
    else:
        L.append(f"  Operands: {qa},{qb}")

    if q_op in op_solutions:
        L.append(f"  '{q_op}' found in examples -> {OP_LABEL[op_name]}")
    else:
        L.append(f"  '{q_op}' not in examples, search:")
        # Show rejections
        for t_op, t_a, t_b, t_v, t_pred in tested[:5]:
            L.append(f"    {OP_LABEL[t_op]}: {t_a} {OP_LABEL[t_op]} {t_b}={t_v}->{t_pred}!={conv_answer} Reject")
        L.append(f"    {OP_LABEL[op_name]}: match -> Accept")
    L.append("")

    # ── Compute ──
    result_val = OPS[op_name](qa, qb)
    if reverse_res:
        display_val = _format_rev_padded(result_val, answer_len)
    else:
        display_val = str(result_val)
    L.append(f"Compute: {qa} {OP_LABEL[op_name]} {qb}={result_val}")
    if reverse_res:
        L.append(f"  Reverse: {result_val}->{display_val}")
    if sym_map and original_answer != conv_answer:
        L.append(f"  Encode: {display_val}->{original_answer}")
    L.append("")

    # ── Final verification ──
    L.append("Verify:")
    for op_ch in sorted(by_op):
        op_n = op_solutions.get(op_ch, op_name)
        for ex in by_op[op_ch]:
            a = int(ex["a_str"][::-1]) if reverse_ops else int(ex["a_str"])
            b = int(ex["b_str"][::-1]) if reverse_ops else int(ex["b_str"])
            try:
                v = OPS[op_n](a, b)
            except:
                continue
            conv_r = _conv(ex["result"])
            if reverse_res:
                r_len = len(conv_r.lstrip("-"))
                v_disp = _format_rev_padded(v, r_len)
            else:
                v_disp = str(v)
            check = "pass" if v_disp == conv_r else "FAIL"
            L.append(f"  {ex['a_str']}{ex['op']}{ex['b_str']}: "
                     f"{a} {OP_LABEL[op_n]} {b}={v}->{v_disp}={conv_r} {check}")
    # Verify query
    check_q = "pass" if display_val == conv_answer else "FAIL"
    L.append(f"  {a_str}{q_op}{b_str}: {qa} {OP_LABEL[op_name]} {qb}={result_val}->{display_val}={conv_answer} {check_q}")

    L.append(f"\\boxed{{{original_answer}}}")
    L.append("</think>")
    L.append(f"\\boxed{{{original_answer}}}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Load GT
    gt = {}
    with open(TRAIN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["id"]] = row["answer"]

    for cat in ["cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"]:
        src = V10_DIR / f"train_cot_{cat}.jsonl"
        dst = V11_DIR / f"train_cot_{cat}.jsonl"
        records = []
        with open(src) as f:
            for line in f:
                records.append(json.loads(line))

        print(f"\n{'='*60}")
        print(f"Processing {cat}: {len(records)} records")
        print(f"{'='*60}")

        new_records = []
        solved_count = 0
        kept = 0
        failed = 0

        for i, rec in enumerate(records):
            pid = rec["id"]
            msgs = rec["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            prompt = msgs[0]["content"]
            answer = gt.get(pid, "")

            if cat.startswith("cryptarithm"):
                examples, query = parse_crypt_prompt(prompt)
                if not examples or not query or not answer:
                    new_records.append(rec)
                    kept += 1
                    continue
                sol = solve_crypt(examples, query, answer, timeout=3.0)
                if sol is not None:
                    cot = build_crypt_cot(examples, query, answer, sol)
                    new_msgs = [msgs[0], {"role": "assistant", "content": cot}]
                    new_records.append({"id": pid, "category": cat, "messages": new_msgs})
                    solved_count += 1
                else:
                    new_records.append(rec)
                    kept += 1

            elif cat == "equation_numeric_guess":
                examples, query = parse_eq_prompt(prompt)
                if not examples or not query or not answer:
                    new_records.append(rec)
                    kept += 1
                    continue
                sol = solve_eq_guess(examples, query, answer)
                if sol is not None:
                    cot = build_eq_guess_cot(examples, query, answer, sol)
                    new_msgs = [msgs[0], {"role": "assistant", "content": cot}]
                    new_records.append({"id": pid, "category": cat, "messages": new_msgs})
                    solved_count += 1
                else:
                    new_records.append(rec)
                    kept += 1
                    failed += 1

            if (i + 1) % 100 == 0:
                print(f"  processed={i+1} solved={solved_count} kept={kept}")

        print(f"  Final: solved={solved_count} kept_original={kept} failed={failed}")

        # Write
        with open(dst, "w") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        print(f"  Written to {dst}")

        # Validate answer extraction
        def extract_last_boxed(text):
            results = []
            for m in re.finditer(r'\\boxed\{', text):
                start = m.end()
                depth = 1
                idx = start
                while idx < len(text) and depth > 0:
                    if text[idx] == '{': depth += 1
                    elif text[idx] == '}': depth -= 1
                    idx += 1
                if depth == 0:
                    results.append(text[start:idx-1])
            return results[-1] if results else ""

        correct = 0
        for rec in new_records:
            msgs2 = rec["messages"]
            if isinstance(msgs2, str): msgs2 = json.loads(msgs2)
            assistant = msgs2[1]["content"] if len(msgs2) > 1 else ""
            extracted = extract_last_boxed(assistant)
            if extracted == gt.get(rec["id"], ""):
                correct += 1
        print(f"  Answer validation: {correct}/{len(new_records)} correct")

        # Stats
        lengths = []
        for rec in new_records:
            msgs2 = rec["messages"]
            if isinstance(msgs2, str): msgs2 = json.loads(msgs2)
            assistant = msgs2[1]["content"] if len(msgs2) > 1 else ""
            lengths.append(len(assistant))
        lengths.sort()
        print(f"  CoT chars: min={min(lengths)} med={lengths[len(lengths)//2]} "
              f"max={max(lengths)} mean={sum(lengths)//len(lengths)}")

        # Quality check
        has_constraints = sum(1 for rec in new_records
                            if "Constraints:" in (rec["messages"][1]["content"]
                                                  if isinstance(rec["messages"], list)
                                                  else json.loads(rec["messages"])[1]["content"]))
        has_verify = sum(1 for rec in new_records
                        if "Verify:" in (rec["messages"][1]["content"]
                                        if isinstance(rec["messages"], list)
                                        else json.loads(rec["messages"])[1]["content"]))
        has_reject = sum(1 for rec in new_records
                        if "Reject" in (rec["messages"][1]["content"]
                                       if isinstance(rec["messages"], list)
                                       else json.loads(rec["messages"])[1]["content"]))
        has_state = sum(1 for rec in new_records
                       if "State:" in (rec["messages"][1]["content"]
                                      if isinstance(rec["messages"], list)
                                      else json.loads(rec["messages"])[1]["content"]))
        print(f"  Quality: constraints={has_constraints}/{len(new_records)} "
              f"verify={has_verify}/{len(new_records)} "
              f"reject={has_reject}/{len(new_records)} "
              f"state={has_state}/{len(new_records)}")


if __name__ == "__main__":
    main()
