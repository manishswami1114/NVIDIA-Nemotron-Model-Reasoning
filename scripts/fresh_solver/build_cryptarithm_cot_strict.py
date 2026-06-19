#!/usr/bin/env python3
"""
build_cryptarithm_cot_strict.py
================================

Builds strict, internally consistent CoT data for all cryptarithm rows in train.csv.

Design goals:
1. Solve every cryptarithm row by finding a full witness:
   - symbol -> digit mapping
   - operator -> operation mapping
   - endian mode (standard / little-endian)
   that satisfies all examples and the query->answer relation.
2. Never patch symbols after solving.
3. Emit CoT that matches the solved witness exactly.
4. Write category-split JSONL files compatible with training.
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
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
PROBLEMS = ROOT / "tong_reasoners" / "problems.jsonl"
OUT_DIR = ROOT / "all_categorical_splits_v9_crypt_strict"

PROMPT_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`."
MAX_SECONDS_PER_ROW = 2.0
RETRY_SECONDS_PER_ROW = 8.0


def _safe_div(a: int, b: int) -> int | None:
    return a // b if b != 0 else None


def _safe_mod(a: int, b: int) -> int | None:
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
}

OP_ORDER = [
    "mul",
    "add",
    "absdiff",
    "sub",
    "concat_fwd",
    "concat_rev",
    "add_m1",
    "add_p1",
    "mul_m1",
    "mul_p1",
    "rsub",
    "neg_absdiff",
    "mod",
    "rmod",
    "gcd",
    "lcm",
    "absdiff_p1",
    "absdiff_m1",
    "absdiff_m2",
    "add_p2",
    "a2_plus_b",
    "fdiv",
]

OP_LABEL = {
    "add": "addition",
    "sub": "subtraction",
    "absdiff": "absolute difference",
    "mul": "multiplication",
    "concat_fwd": "concatenation",
    "concat_rev": "reverse concatenation",
    "add_p1": "add+1",
    "add_m1": "add-1",
    "add_p2": "add+2",
    "mul_p1": "multiply+1",
    "mul_m1": "multiply-1",
    "rsub": "reverse subtraction",
    "neg_absdiff": "negated absolute difference",
    "mod": "modulo",
    "rmod": "reverse modulo",
    "gcd": "gcd",
    "lcm": "lcm",
    "absdiff_p1": "absdiff+1",
    "absdiff_m1": "absdiff-1",
    "absdiff_m2": "absdiff-2",
    "a2_plus_b": "a squared plus b",
    "fdiv": "floor division",
}


LINE_RX = re.compile(r"^(\S{5})\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"Now,\s*determine the result for:\s*(\S+)", re.IGNORECASE)


def parse_prompt(prompt: str):
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
                examples.append(
                    {
                        "L_str": lhs[:2],
                        "op": lhs[2],
                        "R_str": lhs[3:5],
                        "C_str": rhs,
                    }
                )
            continue
        m = QUERY_RX.search(line)
        if m:
            q = m.group(1).strip()
            if len(q) == 5:
                query = {"L_str": q[:2], "op": q[2], "R_str": q[3:5]}
    return examples, query


class SolveTimeout(Exception):
    pass


def _value_signature(v: int):
    return (1 if v < 0 else 0, len(str(abs(v))))


def _build_op_signatures():
    signatures = {name: set() for name in OP_ORDER}
    for a in range(100):
        for b in range(100):
            for name in OP_ORDER:
                try:
                    v = OPS[name](a, b)
                except Exception:
                    continue
                if v is None:
                    continue
                signatures[name].add(_value_signature(v))
    return signatures


OP_SIGNATURES = _build_op_signatures()


def _assign_char(mapping: dict[str, int], used: set[int], ch: str, d: int, distinct: bool):
    cur = mapping.get(ch)
    if cur is not None:
        if cur != d:
            return None
        return mapping, used
    if distinct and d in used:
        return None
    m2 = dict(mapping)
    u2 = set(used)
    m2[ch] = d
    u2.add(d)
    return m2, u2


def _decode_two(s: str, mapping: dict[str, int], mode: str) -> int:
    a = mapping[s[0]]
    b = mapping[s[1]]
    if mode == "standard":
        return 10 * a + b
    return 10 * b + a


def _iter_two_values(s: str, mapping: dict[str, int], used: set[int], mode: str, distinct: bool):
    if s[0] == s[1]:
        if s[0] in mapping:
            d0_list = [mapping[s[0]]]
        else:
            d0_list = list(range(10))
        d1_list = d0_list
    else:
        d0_list = [mapping[s[0]]] if s[0] in mapping else list(range(10))
        d1_list = [mapping[s[1]]] if s[1] in mapping else list(range(10))
    out = []
    for d0 in d0_list:
        res0 = _assign_char(mapping, used, s[0], d0, distinct)
        if res0 is None:
            continue
        m0, u0 = res0
        for d1 in d1_list:
            res1 = _assign_char(m0, u0, s[1], d1, distinct)
            if res1 is None:
                continue
            m1, u1 = res1
            if mode == "standard":
                val = 10 * m1[s[0]] + m1[s[1]]
            else:
                val = 10 * m1[s[1]] + m1[s[0]]
            out.append((val, m1, u1))
    return out


def _result_digits(val: int, c_str: str, mode: str):
    neg = c_str.startswith("-")
    c_data = c_str[1:] if neg else c_str
    if not c_data:
        return None
    if neg and val >= 0:
        return None
    if (not neg) and val < 0:
        return None

    abs_v = abs(val)
    s = str(abs_v)
    if len(s) > len(c_data):
        return None
    s = s.zfill(len(c_data))
    digits = [int(ch) for ch in s]
    if mode == "little":
        digits = list(reversed(digits))
    return c_data, digits


def _assign_result_symbols(mapping: dict[str, int], used: set[int], c_str: str, val: int, mode: str, distinct: bool):
    rd = _result_digits(val, c_str, mode)
    if rd is None:
        return None
    c_data, digits = rd
    m = dict(mapping)
    u = set(used)
    for ch, d in zip(c_data, digits):
        res = _assign_char(m, u, ch, d, distinct)
        if res is None:
            return None
        m, u = res
    return m, u


def _equation_order(examples: list[dict], query_eq: dict):
    by_op = Counter(e["op"] for e in examples)
    ordered = sorted(examples, key=lambda e: (-by_op[e["op"]], e["op"]))
    return ordered + [query_eq]


def _candidate_ops_for_cstr(c_str: str):
    neg = 1 if c_str.startswith("-") else 0
    c_data = c_str[1:] if c_str.startswith("-") else c_str
    target = (neg, len(c_data))
    out = []
    for op_name in OP_ORDER:
        if target in OP_SIGNATURES[op_name]:
            out.append(op_name)
    return out


def solve_cryptarithm(examples: list[dict], query: dict, answer: str):
    query_eq = {
        "L_str": query["L_str"],
        "op": query["op"],
        "R_str": query["R_str"],
        "C_str": answer,
    }

    equations = _equation_order(examples, query_eq)
    op_chars = sorted({e["op"] for e in equations})

    symbols = set()
    for eq in equations:
        symbols.update(eq["L_str"])
        symbols.update(eq["R_str"])
        symbols.update(eq["C_str"].replace("-", ""))

    op_candidates = {}
    for op in op_chars:
        related = [eq for eq in equations if eq["op"] == op]
        cand = set(OP_ORDER)
        for eq in related:
            cand &= set(_candidate_ops_for_cstr(eq["C_str"]))
        if not cand:
            return None
        op_candidates[op] = [x for x in OP_ORDER if x in cand]

    for mode in ("standard", "little"):
        for distinct in (True, False):
            if distinct and len(symbols) > 10:
                continue
            deadline = time.perf_counter() + MAX_SECONDS_PER_ROW
            failed_states = set()
            try:
                sol = _dfs_equations(
                    equations=equations,
                    idx=0,
                    mapping={},
                    used=set(),
                    op_map={},
                    mode=mode,
                    distinct=distinct,
                    op_candidates=op_candidates,
                    deadline=deadline,
                    failed_states=failed_states,
                )
            except SolveTimeout:
                sol = None
            if sol is not None:
                mapping, op_map = sol
                qn = _decode_two(query["L_str"], mapping, mode), _decode_two(query["R_str"], mapping, mode)
                qv = OPS[op_map[query["op"]]](qn[0], qn[1])
                return {
                    "mapping": mapping,
                    "op_map": op_map,
                    "mode": mode,
                    "distinct": distinct,
                    "query_numeric": qv,
                    "op_chars": op_chars,
                }
    return None


def _dfs_equations(equations, idx, mapping, used, op_map, mode, distinct, op_candidates, deadline, failed_states):
    if time.perf_counter() > deadline:
        raise SolveTimeout()

    if idx == len(equations):
        return mapping, op_map

    state_key = (
        idx,
        mode,
        distinct,
        tuple(sorted(mapping.items())),
        tuple(sorted(op_map.items())),
    )
    if state_key in failed_states:
        return None

    eq = equations[idx]
    lhs_vals = _iter_two_values(eq["L_str"], mapping, used, mode, distinct)
    for l_val, m_l, u_l in lhs_vals:
        rhs_vals = _iter_two_values(eq["R_str"], m_l, u_l, mode, distinct)
        for r_val, m_r, u_r in rhs_vals:
            if eq["op"] in op_map:
                op_name = op_map[eq["op"]]
                try:
                    c_val = OPS[op_name](l_val, r_val)
                except Exception:
                    continue
                if c_val is None:
                    continue
                assigned = _assign_result_symbols(m_r, u_r, eq["C_str"], c_val, mode, distinct)
                if assigned is None:
                    continue
                m_c, u_c = assigned
                out = _dfs_equations(
                    equations,
                    idx + 1,
                    m_c,
                    u_c,
                    op_map,
                    mode,
                    distinct,
                    op_candidates,
                    deadline,
                    failed_states,
                )
                if out is not None:
                    return out
                continue

            for op_name in op_candidates[eq["op"]]:
                try:
                    c_val = OPS[op_name](l_val, r_val)
                except Exception:
                    continue
                if c_val is None:
                    continue
                assigned = _assign_result_symbols(m_r, u_r, eq["C_str"], c_val, mode, distinct)
                if assigned is None:
                    continue
                m_c, u_c = assigned
                next_op_map = dict(op_map)
                next_op_map[eq["op"]] = op_name
                out = _dfs_equations(
                    equations,
                    idx + 1,
                    m_c,
                    u_c,
                    next_op_map,
                    mode,
                    distinct,
                    op_candidates,
                    deadline,
                    failed_states,
                )
                if out is not None:
                    return out

    failed_states.add(state_key)
    return None


def build_cot(record_id: str, category: str, examples: list[dict], query: dict, answer: str, solved: dict):
    mapping = solved["mapping"]
    op_map = solved["op_map"]
    mode = solved["mode"]
    mode_label = "standard (AB = A*10 + B)" if mode == "standard" else "little-endian (AB = B*10 + A)"

    lines = ["<think>"]
    lines.append("I will solve this cryptarithm by enforcing one consistent symbol-to-digit mapping across all equations.")
    lines.append("Each operator symbol corresponds to one operation, and the same operator must behave consistently.")
    lines.append("")
    lines.append("Solved settings:")
    lines.append(f"- Number mode: {mode_label}")
    lines.append(f"- Distinct digits required: {'YES' if solved['distinct'] else 'NO'}")
    lines.append("- Symbol mapping:")
    for ch, d in sorted(mapping.items(), key=lambda kv: (kv[1], kv[0])):
        lines.append(f"  '{ch}' = {d}")
    lines.append("- Operator mapping:")
    for op_ch in sorted(op_map):
        lines.append(f"  '{op_ch}' -> {OP_LABEL[op_map[op_ch]]}")
    lines.append("")
    lines.append("Verification on examples:")

    for ex in examples:
        l = _decode_two(ex["L_str"], mapping, mode)
        r = _decode_two(ex["R_str"], mapping, mode)
        op_name = op_map[ex["op"]]
        out = OPS[op_name](l, r)
        lines.append(f"  {ex['L_str']} {ex['op']} {ex['R_str']} = {ex['C_str']}")
        lines.append(f"    decoded: {l} {OP_LABEL[op_name]} {r} = {out}")
        lines.append("    [verified]")

    ql = _decode_two(query["L_str"], mapping, mode)
    qr = _decode_two(query["R_str"], mapping, mode)
    qop = op_map[query["op"]]
    qv = OPS[qop](ql, qr)
    lines.append("")
    lines.append(f"Apply to query: {query['L_str']} {query['op']} {query['R_str']}")
    lines.append(f"  decoded: {ql} {OP_LABEL[qop]} {qr} = {qv}")
    lines.append(f"  encoded result: {answer}")
    lines.append("")
    lines.append("All equations are consistent with one mapping and operator assignment.")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def load_categories():
    out = {}
    with PROBLEMS.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["category"]
    return out


def main():
    global MAX_SECONDS_PER_ROW

    categories = load_categories()
    records = {"cryptarithm_deduce": [], "cryptarithm_guess": []}

    solved_ct = Counter()
    failed = []
    unresolved_rows = []

    total_crypt = 0
    with TRAIN_CSV.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for i, row in enumerate(reader, start=1):
            rid = row["id"].strip()
            cat = categories.get(rid)
            if cat not in records:
                continue
            total_crypt += 1

            prompt = row["prompt"]
            answer = row["answer"]
            examples, query = parse_prompt(prompt)
            if not examples or query is None:
                failed.append((rid, cat, "parse_failed"))
                continue

            solved = solve_cryptarithm(examples, query, answer)
            if solved is None:
                failed.append((rid, cat, "unsolved"))
                unresolved_rows.append((rid, cat, prompt, answer))
            else:
                cot = build_cot(rid, cat, examples, query, answer, solved)
                user_prompt = prompt + PROMPT_SUFFIX

                rec = {
                    "id": rid,
                    "category": cat,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": cot},
                    ],
                }
                records[cat].append(rec)
                solved_ct[cat] += 1

            if total_crypt % 50 == 0:
                print(
                    f"processed={total_crypt} solved={sum(solved_ct.values())} "
                    f"failed={len(failed)}"
                )

    # Retry only unresolved rows with a higher per-row budget.
    if unresolved_rows:
        print(f"retrying unresolved rows with budget={RETRY_SECONDS_PER_ROW}s")
        unresolved_set = {rid for rid, _, _, _ in unresolved_rows}
        old_budget = MAX_SECONDS_PER_ROW
        MAX_SECONDS_PER_ROW = RETRY_SECONDS_PER_ROW

        resolved_in_retry = 0
        still_failed = []
        for rid, cat, prompt, answer in unresolved_rows:
            examples, query = parse_prompt(prompt)
            solved = solve_cryptarithm(examples, query, answer)
            if solved is None:
                still_failed.append((rid, cat, "unsolved_after_retry"))
                continue
            cot = build_cot(rid, cat, examples, query, answer, solved)
            user_prompt = prompt + PROMPT_SUFFIX
            rec = {
                "id": rid,
                "category": cat,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": cot},
                ],
            }
            # Replace if already present (defensive), else append.
            replaced = False
            for i, old in enumerate(records[cat]):
                if old["id"] == rid:
                    records[cat][i] = rec
                    replaced = True
                    break
            if not replaced:
                records[cat].append(rec)
                solved_ct[cat] += 1
            resolved_in_retry += 1

        MAX_SECONDS_PER_ROW = old_budget

        if unresolved_set:
            failed = [f for f in failed if f[0] not in unresolved_set]
        failed.extend(still_failed)
        print(f"retry solved={resolved_in_retry} still_failed={len(still_failed)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cat in ("cryptarithm_deduce", "cryptarithm_guess"):
        out_path = OUT_DIR / f"train_cot_{cat}.jsonl"
        with out_path.open("w") as f:
            for rec in sorted(records[cat], key=lambda x: x["id"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Solved counts:", dict(solved_ct))
    print("Total solved:", sum(solved_ct.values()))
    print("Total crypt rows:", total_crypt)
    print("Failed:", len(failed))
    if failed:
        print("Sample failed:", failed[:20])


if __name__ == "__main__":
    main()
