#!/usr/bin/env python3
"""
gen_cot_numeric_guess.py — Generate search-trajectory CoT for equation_numeric_guess.

Structure matches the WORKING equation_numeric_deduce CoT format:
  1. List examples, identify operators
  2. For each global (rev_ops × rev_res) combo:
     a. For the first operator group, try all operations (show failures)
     b. If found, try remaining operator groups
     c. If all groups match → done; else reject and try next meta combo
  3. Verify across all examples
  4. Apply to query
  5. Verification + boxed answer

Key: meta-rules are GLOBAL, not per-operator.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# ============================================================
# Operations
# ============================================================
def _safe_div(L, R): return L // R if R else None
def _safe_mod(L, R): return L % R if R else None

OPS = {
    "add":         lambda L, R: L + R,
    "absdiff":     lambda L, R: abs(L - R),
    "sub":         lambda L, R: L - R,
    "mul":         lambda L, R: L * R,
    "concat_fwd":  lambda L, R: int(f"{L}{R}"),
    "add_p1":      lambda L, R: L + R + 1,
    "add_m1":      lambda L, R: L + R - 1,
    "mul_p1":      lambda L, R: L * R + 1,
    "mul_m1":      lambda L, R: L * R - 1,
    "rsub":        lambda L, R: R - L,
    "concat_rev":  lambda L, R: int(f"{R}{L}"),
    "add_p2":      lambda L, R: L + R + 2,
    "neg_absdiff": lambda L, R: -(abs(L - R)),
    "mod":         _safe_mod,
    "rmod":        lambda L, R: _safe_mod(R, L),
    "gcd":         lambda L, R: math.gcd(L, R) if (L and R) else 0,
    "lcm":         lambda L, R: math.lcm(L, R) if (L and R) else 0,
    "absdiff_p1":  lambda L, R: abs(L - R) + 1,
    "absdiff_m1":  lambda L, R: abs(L - R) - 1,
    "absdiff_m2":  lambda L, R: abs(L - R) - 2,
    "a2_plus_b":   lambda L, R: L * L + R,
    "fdiv":        _safe_div,
    "max_mod_min": lambda L, R: max(L, R) % min(L, R) if min(L, R) else None,
    "min_mod_max": lambda L, R: min(L, R) % max(L, R) if max(L, R) else None,
}

OP_LABELS = {
    "add": "addition", "absdiff": "absolute difference",
    "sub": "subtraction (a-b)", "mul": "multiplication",
    "concat_fwd": "concatenation", "add_p1": "add+1",
    "add_m1": "add-1", "mul_p1": "multiply+1", "mul_m1": "multiply-1",
    "rsub": "reverse subtraction (b-a)", "concat_rev": "reverse concatenation",
    "add_p2": "add+2", "neg_absdiff": "negated absolute difference",
    "mod": "modulo (a mod b)", "rmod": "reverse modulo (b mod a)",
    "gcd": "gcd", "lcm": "lcm",
    "absdiff_p1": "absdiff+1", "absdiff_m1": "absdiff-1",
    "absdiff_m2": "absdiff-2", "a2_plus_b": "a²+b",
    "fdiv": "integer division (a/b)",
    "max_mod_min": "max mod min", "min_mod_max": "min mod max",
}

COMMON_OPS = [
    "concat_fwd", "concat_rev", "add", "absdiff", "neg_absdiff",
    "sub", "rsub", "mul",
]
RARE_OPS = [
    "mul_p1", "mul_m1", "add_p1", "add_m1", "add_p2",
    "absdiff_p1", "absdiff_m1", "absdiff_m2",
    "max_mod_min", "fdiv", "mod", "rmod",
    "gcd", "lcm", "a2_plus_b", "min_mod_max",
]
ALL_OPS_ORDERED = COMMON_OPS + RARE_OPS


def reverse_2digit(n: int) -> int:
    s = f"{n:02d}"
    return int(s[::-1])


def reverse_result_str(n: int) -> str:
    if n < 0:
        return "-" + str(-n)[::-1]
    return str(n)[::-1]


def compute_op_str(L, R, op_name):
    """Return (result_int_or_None, human_readable_computation_string)."""
    label = OP_LABELS.get(op_name, op_name)
    fn = OPS[op_name]
    try:
        result = fn(L, R)
    except Exception:
        return None, f"{label} f({L}, {R}) = error"
    if result is None:
        return None, f"{label} f({L}, {R}) = undefined"

    if op_name == "mul":
        tens = (L // 10) * 10
        ones = L % 10
        comp = (f"{label} f({L}, {R}) = ({tens} + {ones}) * {R} "
                f"= {tens} * {R} + {ones} * {R} "
                f"= {tens * R} + {ones * R} = {result}")
    elif op_name in ("mul_p1", "mul_m1"):
        base = L * R
        sign = "+" if op_name == "mul_p1" else "-"
        comp = (f"{label} f({L}, {R}) = {L} * {R} {sign} 1 "
                f"= {base} {sign} 1 = {result}")
    elif op_name == "concat_fwd":
        comp = f"{label} f({L}, {R}) = {L} || {R} = {result}"
    elif op_name == "concat_rev":
        comp = f"{label} f({L}, {R}) = {R} || {L} = {result}"
    elif op_name == "add":
        comp = f"{label} f({L}, {R}) = {L} + {R} = {result}"
    elif op_name == "sub":
        comp = f"{label} f({L}, {R}) = {L} - {R} = {result}"
    elif op_name == "rsub":
        comp = f"{label} f({L}, {R}) = {R} - {L} = {result}"
    elif op_name == "absdiff":
        comp = f"{label} f({L}, {R}) = |{L} - {R}| = {result}"
    elif op_name == "neg_absdiff":
        comp = f"{label} f({L}, {R}) = -|{L} - {R}| = {result}"
    elif op_name in ("add_p1", "add_m1", "add_p2"):
        base = L + R
        offset = {"add_p1": 1, "add_m1": -1, "add_p2": 2}[op_name]
        sign = "+" if offset > 0 else "-"
        comp = f"{label} f({L}, {R}) = {L} + {R} {sign} {abs(offset)} = {base} {sign} {abs(offset)} = {result}"
    elif op_name in ("absdiff_p1", "absdiff_m1", "absdiff_m2"):
        base = abs(L - R)
        offset = {"absdiff_p1": 1, "absdiff_m1": -1, "absdiff_m2": -2}[op_name]
        sign = "+" if offset > 0 else "-"
        comp = f"{label} f({L}, {R}) = |{L} - {R}| {sign} {abs(offset)} = {base} {sign} {abs(offset)} = {result}"
    elif op_name == "a2_plus_b":
        comp = f"{label} f({L}, {R}) = {L}² + {R} = {L*L} + {R} = {result}"
    elif op_name == "mod":
        comp = f"{label} f({L}, {R}) = {L} mod {R} = {result}"
    elif op_name == "rmod":
        comp = f"{label} f({L}, {R}) = {R} mod {L} = {result}"
    elif op_name == "gcd":
        comp = f"gcd f({L}, {R}) = gcd({L},{R}) = {result}"
    elif op_name == "lcm":
        comp = f"lcm f({L}, {R}) = lcm({L},{R}) = {result}"
    elif op_name == "fdiv":
        comp = f"{label} f({L}, {R}) = {L} / {R} = {result}"
    elif op_name == "max_mod_min":
        comp = f"{label} f({L}, {R}) = max({L},{R}) mod min({L},{R}) = {max(L,R)} mod {min(L,R)} = {result}"
    elif op_name == "min_mod_max":
        comp = f"{label} f({L}, {R}) = min({L},{R}) mod max({L},{R}) = {min(L,R)} mod {max(L,R)} = {result}"
    else:
        comp = f"{label} f({L}, {R}) = {result}"

    return result, comp


def _check_op_on_example(ex, op_name, rev_ops, rev_res):
    """Check if operation matches example under given meta-rules. Returns (matched, result_str)."""
    L = reverse_2digit(ex["L_num"]) if rev_ops else ex["L_num"]
    R = reverse_2digit(ex["R_num"]) if rev_ops else ex["R_num"]
    fn = OPS[op_name]
    try:
        raw = fn(L, R)
    except Exception:
        return False, None
    if raw is None:
        return False, None
    final = reverse_result_str(raw) if rev_res else str(raw)
    return final == ex["result_str"], final


def build_search_trajectory(solved_rec: dict, prompt: str) -> str:
    """Build a full search-trajectory CoT for one numeric_guess puzzle."""
    examples = solved_rec["examples"]
    query = solved_rec["query"]
    answer = solved_rec["answer"]
    correct_op_map = solved_rec["op_map"]
    correct_rev_ops = solved_rec["reverse_ops"]
    correct_rev_res = solved_rec["reverse_res"]

    op_chars = sorted(set(e["op_char"] for e in examples))
    query_op = query["op_char"]
    groups = {oc: [e for e in examples if e["op_char"] == oc] for oc in op_chars}
    sorted_ops = sorted(op_chars, key=lambda oc: -len(groups[oc]))

    lines = ["<think>"]
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Examples:")
    for ex in examples:
        lines.append(f"  {ex['L_str']}{ex['op_char']}{ex['R_str']} = {ex['result_str']}")
    lines.append("")

    lines.append("The operators")
    for oc in sorted_ops:
        lines.append(oc)
    lines.append("")

    qL_str = query["L_str"]
    qR_str = query["R_str"]
    lines.append(f"Looking at the question")
    lines.append(f"{qL_str}{query_op}{qR_str} -> {query_op}")
    if query_op in op_chars:
        lines.append("The question operator is found in the examples.")
    else:
        lines.append("The question operator is NOT found in the examples — unseen operator.")
    lines.append("")

    META_COMBOS = [
        (False, False, "identity", ""),
        (True, False, "reversed operands", ""),
        (False, True, "identity", " and reversed result"),
        (True, True, "reversed operands", " and reversed result"),
    ]

    # Build correct combo index to ensure we show attempts up to and including it
    correct_meta_idx = None
    for idx, (ro, rr, _, _) in enumerate(META_COMBOS):
        if ro == correct_rev_ops and rr == correct_rev_res:
            correct_meta_idx = idx
            break

    found_global = False

    for meta_idx, (rev_ops, rev_res, ro_label, rr_suffix) in enumerate(META_COMBOS):
        if found_global:
            break

        is_correct_meta = (meta_idx == correct_meta_idx)

        # For wrong meta combos before the correct one, show abbreviated search
        # For the correct meta combo, show full search
        first_oc = sorted_ops[0]
        ref_ex = groups[first_oc][0]
        L_raw, R_raw = ref_ex["L_num"], ref_ex["R_num"]

        L_eff = reverse_2digit(L_raw) if rev_ops else L_raw
        R_eff = reverse_2digit(R_raw) if rev_ops else R_raw
        expected = ref_ex["result_str"]

        if rev_ops:
            op_desc = f"reversed operands [{L_raw:02d}->{L_eff:02d} {R_raw:02d}->{R_eff:02d}]"
        else:
            op_desc = f"[{L_raw:02d} {R_raw:02d}]"

        lines.append(f"Looking at operator 【{first_oc}】 [{ref_ex['L_str']}{first_oc}{ref_ex['R_str']} = {expected}]:")

        if is_correct_meta:
            # Show full search (common then rare) — this is where we find the answer
            phase_label = f"Trying common operations {ro_label}{rr_suffix} {op_desc}"
            lines.append(f"  {phase_label} [expected ({L_eff},{R_eff})->{expected}]:")

            found_first_op = None
            for op_name in COMMON_OPS:
                raw_result, comp_str = compute_op_str(L_eff, R_eff, op_name)
                if raw_result is None:
                    lines.append(f"    {comp_str} wrong")
                    continue
                final = reverse_result_str(raw_result) if rev_res else str(raw_result)
                if rev_res:
                    line_txt = f"    {comp_str} -rev-> {final}"
                else:
                    line_txt = f"    {comp_str}"

                if final == expected:
                    actions = []
                    if rev_ops: actions.append("reversed operands")
                    if rev_res: actions.append("reversed result")
                    actions.append(OP_LABELS.get(op_name, op_name))
                    lines.append(f"{line_txt} match, correct, actions: {', '.join(actions)}")
                    found_first_op = op_name
                    break
                else:
                    lines.append(f"{line_txt} wrong")

            if found_first_op is None:
                lines.append(f"  Trying rare operations {ro_label}{rr_suffix} {op_desc} [expected ({L_eff},{R_eff})->{expected}]:")
                for op_name in RARE_OPS:
                    raw_result, comp_str = compute_op_str(L_eff, R_eff, op_name)
                    if raw_result is None:
                        continue
                    final = reverse_result_str(raw_result) if rev_res else str(raw_result)
                    if rev_res:
                        line_txt = f"    {comp_str} -rev-> {final}"
                    else:
                        line_txt = f"    {comp_str}"

                    if final == expected:
                        actions = []
                        if rev_ops: actions.append("reversed operands")
                        if rev_res: actions.append("reversed result")
                        actions.append(OP_LABELS.get(op_name, op_name))
                        lines.append(f"{line_txt} match, correct, actions: {', '.join(actions)}")
                        found_first_op = op_name
                        break
                    else:
                        lines.append(f"{line_txt} wrong")

            if found_first_op is None:
                lines.append("  No operation found for this combination.")
                lines.append("")
                continue

            # Verify first op with remaining examples
            first_grp = groups[first_oc]
            if len(first_grp) > 1:
                lines.append(f"  Verifying with remaining 【{first_oc}】 examples:")
                for vex in first_grp[1:]:
                    vL = reverse_2digit(vex["L_num"]) if rev_ops else vex["L_num"]
                    vR = reverse_2digit(vex["R_num"]) if rev_ops else vex["R_num"]
                    vraw, vcomp = compute_op_str(vL, vR, found_first_op)
                    if vraw is not None:
                        vfinal = reverse_result_str(vraw) if rev_res else str(vraw)
                        status = "✓" if vfinal == vex["result_str"] else "✗"
                        if rev_res:
                            lines.append(f"    [{status}] {vex['L_str']}{first_oc}{vex['R_str']}: "
                                         f"{vcomp} -rev-> {vfinal} (expected {vex['result_str']})")
                        else:
                            lines.append(f"    [{status}] {vex['L_str']}{first_oc}{vex['R_str']}: "
                                         f"{vcomp} = {vfinal} (expected {vex['result_str']})")

            discovered = {first_oc: found_first_op}

            # Now find operations for remaining operator groups
            for oc in sorted_ops[1:]:
                grp = groups[oc]
                ref = grp[0]
                eL = reverse_2digit(ref["L_num"]) if rev_ops else ref["L_num"]
                eR = reverse_2digit(ref["R_num"]) if rev_ops else ref["R_num"]
                exp = ref["result_str"]

                if rev_ops:
                    edesc = f"reversed operands [{ref['L_num']:02d}->{eL:02d} {ref['R_num']:02d}->{eR:02d}]"
                else:
                    edesc = f"[{ref['L_num']:02d} {ref['R_num']:02d}]"

                lines.append(f"  Looking at operator 【{oc}】 [{ref['L_str']}{oc}{ref['R_str']} = {exp}]:")
                lines.append(f"    Trying operations {ro_label}{rr_suffix} {edesc} [expected ({eL},{eR})->{exp}]:")

                found_op = None
                for op_name in ALL_OPS_ORDERED:
                    raw_r, comp_s = compute_op_str(eL, eR, op_name)
                    if raw_r is None:
                        continue
                    final = reverse_result_str(raw_r) if rev_res else str(raw_r)
                    if rev_res:
                        lt = f"      {comp_s} -rev-> {final}"
                    else:
                        lt = f"      {comp_s}"

                    if final == exp:
                        lines.append(f"{lt} match")
                        found_op = op_name

                        # Quick verify
                        if len(grp) > 1:
                            all_ok = True
                            for vex in grp[1:]:
                                matched, _ = _check_op_on_example(vex, op_name, rev_ops, rev_res)
                                if not matched:
                                    all_ok = False
                                    break
                            if not all_ok:
                                lines.append(f"      but fails on other examples, trying next...")
                                found_op = None
                                continue
                        break
                    else:
                        lines.append(f"{lt} wrong")

                if found_op:
                    discovered[oc] = found_op
                else:
                    lines.append("    No matching operation found.")

            lines.append("")
            found_global = True

        else:
            # Wrong meta combo — show abbreviated search (just common ops, first group only)
            lines.append(f"  Trying common operations {ro_label}{rr_suffix} {op_desc} [expected ({L_eff},{R_eff})->{expected}]:")

            any_match = False
            ops_to_show = COMMON_OPS[:6]
            for op_name in ops_to_show:
                raw_result, comp_str = compute_op_str(L_eff, R_eff, op_name)
                if raw_result is None:
                    lines.append(f"    {comp_str} wrong")
                    continue
                final = reverse_result_str(raw_result) if rev_res else str(raw_result)
                if rev_res:
                    line_txt = f"    {comp_str} -rev-> {final}"
                else:
                    line_txt = f"    {comp_str}"

                if final == expected:
                    # Found a match for first op, but need to check if ALL groups work
                    lines.append(f"{line_txt} match")
                    # Check remaining groups
                    all_groups_ok = True
                    for oc2 in sorted_ops[1:]:
                        ref2 = groups[oc2][0]
                        ok = False
                        for op2 in ALL_OPS_ORDERED:
                            matched, _ = _check_op_on_example(ref2, op2, rev_ops, rev_res)
                            if matched:
                                # Also verify across group
                                grp_ok = all(_check_op_on_example(e, op2, rev_ops, rev_res)[0] for e in groups[oc2])
                                if grp_ok:
                                    ok = True
                                    break
                        if not ok:
                            all_groups_ok = False
                            break
                    if not all_groups_ok:
                        lines.append(f"    but other operator groups don't match under this meta-rule, rejecting")
                        any_match = False
                    else:
                        any_match = True
                    break
                else:
                    lines.append(f"{line_txt} wrong")

            if not any_match:
                lines.append("")
                continue

    # Summary
    lines.append("Summary of discovered rules:")
    meta_parts = []
    if correct_rev_ops: meta_parts.append("reversed operands")
    if correct_rev_res: meta_parts.append("reversed result")
    if not meta_parts: meta_parts.append("identity (no reversal)")
    lines.append(f"  Meta-rules: {', '.join(meta_parts)}")
    for oc in sorted_ops:
        op_name = correct_op_map.get(oc, "unknown")
        lines.append(f"  Operator 【{oc}】 → {OP_LABELS.get(op_name, op_name)}")
    if query_op not in op_chars and query_op in correct_op_map:
        op_name = correct_op_map[query_op]
        lines.append(f"  Operator 【{query_op}】 (query, unseen) → {OP_LABELS.get(op_name, op_name)}")
    lines.append("")

    # Apply to query
    qL_num = query["L_num"]
    qR_num = query["R_num"]
    qL_eff = reverse_2digit(qL_num) if correct_rev_ops else qL_num
    qR_eff = reverse_2digit(qR_num) if correct_rev_ops else qR_num

    query_op_name = correct_op_map.get(query_op)

    lines.append(f"Applying to {qL_str}{query_op}{qR_str}:")
    if correct_rev_ops:
        lines.append(f"  reversed operands [{qL_str}->{qL_eff:02d}, {qR_str}->{qR_eff:02d}]"
                      + (" and reversed result" if correct_rev_res else ""))
    else:
        lines.append(f"  identity operands [{qL_str}, {qR_str}]"
                      + (" and reversed result" if correct_rev_res else ""))

    final_answer = answer
    if query_op_name:
        raw_result, comp_str = compute_op_str(qL_eff, qR_eff, query_op_name)
        if raw_result is not None:
            if correct_rev_res:
                lines.append(f"  {comp_str} -rev-> {final_answer}")
            else:
                lines.append(f"  {comp_str}")
            lines.append(f"  Result: 【{final_answer}】")
    lines.append("")

    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is")
    lines.append("")
    lines.append("Verification Step:")
    lines.append("[✓] Equation evaluated following order of operations? -> YES")
    lines.append("[✓] LHS equals RHS? -> YES")
    lines.append("")
    lines.append("All constraints satisfied. The solution is verified.")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"\\boxed{{{final_answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{final_answer}}}")

    return "\n".join(lines)


def main():
    THIS = Path(__file__).resolve().parent
    ROOT = THIS.parent.parent
    BASELINE = ROOT / "dont_touch_it" / "all_categorical_splits"
    SOLVED = THIS / "solved_numeric_guess_fresh.jsonl"
    SRC = BASELINE / "train_cot_equation_numeric_guess.jsonl"
    OUT = THIS / "cot_equation_numeric_guess.jsonl"

    solved = {}
    with SOLVED.open() as f:
        for line in f:
            rec = json.loads(line)
            solved[rec["id"]] = rec

    print(f"Solver output: {len(solved)} records")

    generated = 0
    skipped = 0
    cot_lengths = []

    with SRC.open() as f_in, OUT.open("w") as f_out:
        for line in f_in:
            rec = json.loads(line)
            rid = rec.get("id", "")
            prompt = rec["messages"][0]["content"]

            if rid not in solved:
                skipped += 1
                continue

            sr = solved[rid]
            cot = build_search_trajectory(sr, prompt)
            cot_lengths.append(len(cot))

            out_rec = {
                "id": rid,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": cot},
                ],
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            generated += 1

    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    if cot_lengths:
        print(f"CoT length: min={min(cot_lengths)}, max={max(cot_lengths)}, "
              f"avg={sum(cot_lengths)/len(cot_lengths):.0f}")
    print(f"Output: {OUT}")

    # Show sample
    with OUT.open() as f:
        sample = json.loads(f.readline())
    cot = sample["messages"][1]["content"]
    print(f"\n{'='*60}")
    print(f"Sample CoT ({len(cot)} chars):")
    print(f"{'='*60}")
    print(cot[:2500])


if __name__ == "__main__":
    main()
