"""
gen_long_baseline_style_cot.py — long-form CoT matching baseline template structure.

Output target: 5000-6000 chars per CoT, matching the verbose style of
baseline's cipher (5411 median) and bit_manipulation (8959 median).

Template structure (preserved from baseline cryptarithm):
  <think>
  This is a cryptarithm solved using base-10 arithmetic and logical constraints.
  Deduced properties:
    - Base
    - Operation
    - Reverse Operands / Result
    - Distinct values
  [VERBOSE EXAMPLE BREAKDOWN]
  [STEP-BY-STEP VERIFICATION OF EACH EXAMPLE]
  Symbol Mapping:
    'x' = N
  [APPLY TO QUERY VERBOSELY]
  The exact value calculation yields the correct answer sequence.
  The answer is \\boxed{X}
  </think>
  \\boxed{X}

Difference from baseline: the math is REAL (verified by solver),
not fake all-zeros. The verbose verification block is what gives us
the 5000-6000 char target length.
"""
from __future__ import annotations
import json
import math


# ============================================================
# Op semantics (must match solve_sequence.py)
# ============================================================
def _safe_div(L, R): return L // R if R else None
def _safe_mod(L, R): return L % R if R else None

OPS = {
    "mul":         (lambda L, R: L * R,                       "multiplication (L × R)"),
    "add":         (lambda L, R: L + R,                       "addition (L + R)"),
    "absdiff":     (lambda L, R: abs(L - R),                  "absolute difference (|L − R|)"),
    "sub_signed":  (lambda L, R: L - R,                       "signed subtraction (L − R)"),
    "concat_fwd":  (lambda L, R: int(f"{L}{R}"),              "forward concatenation (L||R)"),
    "add_m1":      (lambda L, R: L + R - 1,                   "addition minus one (L + R − 1)"),
    "mul_m1":      (lambda L, R: L * R - 1,                   "multiplication minus one (L × R − 1)"),
    "mul_p1":      (lambda L, R: L * R + 1,                   "multiplication plus one (L × R + 1)"),
    "add_p1":      (lambda L, R: L + R + 1,                   "addition plus one (L + R + 1)"),
    "rsub_signed": (lambda L, R: R - L,                       "reverse subtraction (R − L)"),
    "neg_absdiff": (lambda L, R: -abs(L - R),                 "negated absolute difference (−|L − R|)"),
    "concat_rev":  (lambda L, R: int(f"{R}{L}"),              "reverse concatenation (R||L)"),
    "mod":         (_safe_mod,                                "modulo (L mod R)"),
    "gcd":         (lambda L, R: math.gcd(L, R),              "greatest common divisor (gcd(L, R))"),
    "rmod":        (lambda L, R: _safe_mod(R, L),             "reverse modulo (R mod L)"),
    "lcm":         (lambda L, R: math.lcm(L, R) if (L or R) else 0, "least common multiple (lcm(L, R))"),
    "absdiff_p1":  (lambda L, R: abs(L - R) + 1,              "absolute difference plus one (|L − R| + 1)"),
    "absdiff_m1":  (lambda L, R: abs(L - R) - 1,              "absolute difference minus one (|L − R| − 1)"),
    "absdiff_m2":  (lambda L, R: abs(L - R) - 2,              "absolute difference minus two (|L − R| − 2)"),
    "add_p2":      (lambda L, R: L + R + 2,                   "addition plus two (L + R + 2)"),
    "a2_plus_b":   (lambda L, R: L * L + R,                   "square plus right (L² + R)"),
    "fdiv":        (_safe_div,                                "floor division (L ÷ R)"),
}

OP_SHORT = {  # short op-name to display in "Operation:" line (matches baseline style)
    "mul": "*", "add": "+", "absdiff": "|−|", "sub_signed": "−",
    "concat_fwd": "||", "concat_rev": "rev||", "mul_p1": "*+1", "mul_m1": "*−1",
    "add_p1": "++1", "add_m1": "+−1", "rsub_signed": "rev−",
    "neg_absdiff": "−|−|", "mod": "mod", "rmod": "rev_mod", "gcd": "gcd",
    "lcm": "lcm", "absdiff_p1": "|−|+1", "absdiff_m1": "|−|−1",
    "absdiff_m2": "|−|−2", "add_p2": "++2", "a2_plus_b": "L²+R", "fdiv": "÷",
}


def decode_two(s, mapping, mode):
    a, b = mapping[s[0]], mapping[s[1]]
    return 10 * a + b if mode == "standard" else 10 * b + a


def encode_n(n, length, inv_map, mode):
    s = str(n).zfill(length)
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return "".join(inv_map[d] for d in digits)


def build_long_cot(rec):
    """Generate a 5000-6000 char CoT in baseline template style with real math."""
    examples = rec["examples"]
    query    = rec["query"]
    mapping  = rec["mapping"]
    op_map   = rec["op_map"]
    mode     = rec["mode"]
    answer   = rec["answer"]
    inv_map  = {v: k for k, v in mapping.items()}

    rev_ops = (mode == "little_endian")
    rev_res = (mode == "little_endian")
    op_chars = sorted({e["op"] for e in examples} | {query["op"]})

    lines = []
    lines.append("<think>")
    lines.append("This is a cryptarithm solved using base-10 arithmetic and logical constraints.")
    lines.append("Deduced properties:")
    lines.append("- Base: 10")
    # Operation summary — show every operator + its full operation name
    op_summary = ", ".join(f"'{o}' = {op_map[o]}" for o in op_chars)
    lines.append(f"- Operations: {op_summary}")
    lines.append(f"- Reverse Operands: {rev_ops}")
    lines.append(f"- Reverse Result: {rev_res}")
    lines.append("- All symbols map to distinct values: True")
    lines.append("")

    # === Verbose example enumeration with per-character decoding ===
    lines.append("We analyze each example by decoding its left and right operands using the")
    lines.append("symbol→digit mapping, applying the discovered operation, then encoding the")
    lines.append("result and comparing it to the target.")
    lines.append("")

    lines.append("Examples (encrypted form):")
    for i, e in enumerate(examples, 1):
        lines.append(f"  EX{i}: 【{e['L_str']}{e['op']}{e['R_str']}】 = 【{e['C_str']}】"
                     f"   (L={e['L_str']!r}, op={e['op']!r}, R={e['R_str']!r}, "
                     f"length of C = {len(e['C_str'])} chars)")
    lines.append(f"  QUERY: 【{query['L_str']}{query['op']}{query['R_str']}】 = ?")
    lines.append("")

    # === Symbol mapping block (closely mirrors baseline format) ===
    lines.append("Symbol Mapping:")
    by_digit = sorted(mapping.items(), key=lambda kv: kv[1])
    for c, d in by_digit:
        lines.append(f"  '{c}' = {d}")
    lines.append("")
    lines.append("Operator Mapping:")
    for o in op_chars:
        op_name = op_map[o]
        _, desc = OPS[op_name]
        lines.append(f"  '{o}' = {desc}")
    lines.append("")

    # === Per-example verification — verbose to bulk up character count ===
    lines.append("Per-example verification:")
    for i, e in enumerate(examples, 1):
        op_name = op_map[e["op"]]
        op_fn, op_desc = OPS[op_name]
        L = decode_two(e["L_str"], mapping, mode)
        R = decode_two(e["R_str"], mapping, mode)
        try:
            C_val = op_fn(L, R)
        except Exception:
            C_val = None
        C_pred = encode_n(C_val, len(e["C_str"]), inv_map, mode) if C_val is not None and C_val >= 0 else None
        ok = (C_pred == e["C_str"])

        lines.append(f"")
        lines.append(f"EX{i}: 【{e['L_str']}{e['op']}{e['R_str']}】 = 【{e['C_str']}】")
        # Decode L per character
        L_chars = e["L_str"]
        L_digits_per_char = [(c, mapping[c]) for c in L_chars]
        ld_str = ", ".join(f"'{c}'={d}" for c, d in L_digits_per_char)
        lines.append(f"  L characters: {ld_str}")
        if mode == "standard":
            lines.append(f"    Reading mode = standard, so L = 10·{L_digits_per_char[0][1]} + "
                         f"{L_digits_per_char[1][1]} = {L}.")
        else:
            lines.append(f"    Reading mode = little_endian, so L = 10·{L_digits_per_char[1][1]} + "
                         f"{L_digits_per_char[0][1]} = {L}.")

        R_chars = e["R_str"]
        R_digits_per_char = [(c, mapping[c]) for c in R_chars]
        rd_str = ", ".join(f"'{c}'={d}" for c, d in R_digits_per_char)
        lines.append(f"  R characters: {rd_str}")
        if mode == "standard":
            lines.append(f"    Reading mode = standard, so R = 10·{R_digits_per_char[0][1]} + "
                         f"{R_digits_per_char[1][1]} = {R}.")
        else:
            lines.append(f"    Reading mode = little_endian, so R = 10·{R_digits_per_char[1][1]} + "
                         f"{R_digits_per_char[0][1]} = {R}.")

        lines.append(f"  Apply operator '{e['op']}' which decodes to {op_name} = {op_desc}:")
        lines.append(f"    {op_desc.split(' (')[0]} of L={L} and R={R} gives {C_val}.")
        # Show the digits of C_val
        if C_val is not None and C_val >= 0:
            c_digits = list(str(C_val).zfill(len(e["C_str"])))
            if mode == "little_endian":
                c_digits = list(reversed(c_digits))
            ce_str = " ".join(f"{d}→'{inv_map[int(d)]}'" for d in c_digits)
            lines.append(f"  Encode {C_val} as a {len(e['C_str'])}-character string using")
            lines.append(f"  the inverse mapping (mode={mode}): {ce_str}")
            lines.append(f"  Encoded result = 【{C_pred}】.")
            lines.append(f"  Target was 【{e['C_str']}】 — match: {'YES ✓' if ok else 'NO ✗'}.")
        else:
            lines.append(f"  Result is out of range; cannot encode.")
    lines.append("")

    # === Final query application (also verbose) ===
    qL = decode_two(query["L_str"], mapping, mode)
    qR = decode_two(query["R_str"], mapping, mode)
    q_op_name = op_map[query["op"]]
    q_op_fn, q_op_desc = OPS[q_op_name]
    qC_val = q_op_fn(qL, qR)
    qC_str = encode_n(qC_val, len(answer), inv_map, mode)

    lines.append("Apply the discovered transformation to the query:")
    lines.append(f"  Query: 【{query['L_str']}{query['op']}{query['R_str']}】")
    qL_chars = [(c, mapping[c]) for c in query["L_str"]]
    qR_chars = [(c, mapping[c]) for c in query["R_str"]]
    lines.append(f"  Query L characters: " + ", ".join(f"'{c}'={d}" for c, d in qL_chars))
    lines.append(f"    L = {qL} ({mode} reading)")
    lines.append(f"  Query R characters: " + ", ".join(f"'{c}'={d}" for c, d in qR_chars))
    lines.append(f"    R = {qR} ({mode} reading)")
    lines.append(f"  Operator '{query['op']}' is {q_op_name} = {q_op_desc}")
    lines.append(f"  Numeric result of {q_op_name}({qL}, {qR}) = {qC_val}")
    qd = list(str(qC_val).zfill(len(answer)))
    if mode == "little_endian":
        qd = list(reversed(qd))
    qe_str = " ".join(f"{d}→'{inv_map[int(d)]}'" for d in qd)
    lines.append(f"  Encode {qC_val} as {len(answer)}-char output using inverse mapping ({mode}):")
    lines.append(f"    {qe_str}")
    lines.append(f"  Encoded query result = 【{qC_str}】.")
    lines.append("")

    lines.append(f"The answer is \\boxed{{{qC_str}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{qC_str}}}")

    cot = "\n".join(lines)
    assert qC_str == answer, f"Wrong answer: got {qC_str!r} vs gt {answer!r}"
    return cot


if __name__ == "__main__":
    # Smoke test
    import json
    SOLVED = "/Users/manishswami/developer/NVIDIA-Nemotron Model/scripts/donald_generators/sequence_solved.jsonl"
    with open(SOLVED) as f:
        rec = json.loads(f.readline())
    cot = build_long_cot(rec)
    print(f"Generated CoT for {rec['id']}: {len(cot)} chars")
    print()
    print(cot[:3500])
    print("\n... [truncated, showing first 3500 chars] ...")
