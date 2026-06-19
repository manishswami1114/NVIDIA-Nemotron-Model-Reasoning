"""
gen_sequence_cot_tong.py — Tong-style CoT generator for sequence puzzles.

Reads `sequence_solved.jsonl` (from solve_sequence.py — our INDEPENDENT
solver, no parquet dependency) and generates Tong-style CoTs matching
the 0.86 baseline format.

Format conventions (matching baseline cipher/bit-manip CoTs):
  - Open with <think>\\n
  - First sentence: "We need to <task>."
  - Followed by "I will put my final answer inside \\boxed{}."
  - Brackets-and-bullets exposition style
  - End with \\boxed{answer} both inside and after </think>

Usage:
    python gen_sequence_cot_tong.py \\
        --solved sequence_solved.jsonl \\
        --out   sequence_donald_tong.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# Same op semantics as solve_sequence.py — must match exactly.
def _safe_div(L, R):
    return L // R if R else None
def _safe_mod(L, R):
    return L % R if R else None

OPS = {
    "mul":         (lambda L, R: L * R,                       "L × R"),
    "add":         (lambda L, R: L + R,                       "L + R"),
    "absdiff":     (lambda L, R: abs(L - R),                  "|L − R|"),
    "sub_signed":  (lambda L, R: L - R,                       "L − R"),
    "concat_fwd":  (lambda L, R: int(f"{L}{R}"),              "concat(L, R)"),
    "add_m1":      (lambda L, R: L + R - 1,                   "L + R − 1"),
    "mul_m1":      (lambda L, R: L * R - 1,                   "L × R − 1"),
    "mul_p1":      (lambda L, R: L * R + 1,                   "L × R + 1"),
    "add_p1":      (lambda L, R: L + R + 1,                   "L + R + 1"),
    "rsub_signed": (lambda L, R: R - L,                       "R − L"),
    "neg_absdiff": (lambda L, R: -abs(L - R),                 "−|L − R|"),
    "concat_rev":  (lambda L, R: int(f"{R}{L}"),              "concat(R, L)"),
    "mod":         (_safe_mod,                                "L mod R"),
    "gcd":         (lambda L, R: math.gcd(L, R),              "gcd(L, R)"),
    "rmod":        (lambda L, R: _safe_mod(R, L),             "R mod L"),
    "lcm":         (lambda L, R: math.lcm(L, R) if (L or R) else 0, "lcm(L, R)"),
    "absdiff_p1":  (lambda L, R: abs(L - R) + 1,              "|L − R| + 1"),
    "absdiff_m1":  (lambda L, R: abs(L - R) - 1,              "|L − R| − 1"),
    "absdiff_m2":  (lambda L, R: abs(L - R) - 2,              "|L − R| − 2"),
    "add_p2":      (lambda L, R: L + R + 2,                   "L + R + 2"),
    "a2_plus_b":   (lambda L, R: L * L + R,                   "L² + R"),
    "fdiv":        (_safe_div,                                "L ÷ R"),
}


def decode_two_digit(s, mapping, mode):
    d0, d1 = mapping[s[0]], mapping[s[1]]
    return 10 * d0 + d1 if mode == "standard" else 10 * d1 + d0


def encode_result(n, length, inv_map, mode):
    s = str(n).zfill(length)
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return "".join(inv_map[d] for d in digits)


def build_cot(rec):
    """Produce Tong-style CoT for one solved sequence puzzle."""
    examples  = rec["examples"]
    query     = rec["query"]
    mapping   = rec["mapping"]
    op_map    = rec["op_map"]
    mode      = rec["mode"]
    answer    = rec["answer"]
    inv_map   = {v: k for k, v in mapping.items()}

    # ---- Opening + task identification ----
    parts = []
    parts.append("<think>")
    parts.append("We need to crack a cipher-digit puzzle. Each character in "
                 "the equations is an encrypted digit, and the operator "
                 "symbols are also encrypted.")
    parts.append("I will put my final answer inside \\boxed{}.\n")

    # ---- List examples ----
    parts.append("Examples (encrypted):")
    for i, e in enumerate(examples, 1):
        parts.append(f"  EX{i}: 【{e['L_str']}{e['op']}{e['R_str']}】 = 【{e['C_str']}】")
    parts.append(f"Query: 【{query['L_str']}{query['op']}{query['R_str']}】 = ?\n")

    # ---- Identify operator positions ----
    op_chars = sorted(set(op_map.keys()))
    parts.append("Each example has form 【L op R】 = 【C】 with the operator "
                 "at position 2 of the 5-char LHS.")
    parts.append(f"Operator symbols present: {', '.join(repr(o) for o in op_chars)}.")
    parts.append(f"Reading mode: {mode}.\n")

    # ---- Discovered cipher mapping ----
    parts.append("After testing assignments, the symbol→digit mapping that "
                 "satisfies all examples is:")
    # Sort by digit for readable display
    by_digit = sorted(mapping.items(), key=lambda kv: kv[1])
    mapping_lines = "  " + "    ".join(f"{c}={d}" for c, d in by_digit)
    parts.append(mapping_lines)
    parts.append("")

    parts.append("Operator-symbol → operation:")
    for op_char in op_chars:
        op_name = op_map[op_char]
        _, op_desc = OPS[op_name]
        parts.append(f"  {op_char!r} → {op_name} ({op_desc})")
    parts.append("")

    # ---- Verify each example ----
    parts.append("Verification:")
    for i, e in enumerate(examples, 1):
        L = decode_two_digit(e["L_str"], mapping, mode)
        R = decode_two_digit(e["R_str"], mapping, mode)
        op_name = op_map[e["op"]]
        op_fn, op_desc = OPS[op_name]
        C_val = op_fn(L, R)
        C_pred = encode_result(C_val, len(e["C_str"]), inv_map, mode)
        ok = "✓" if C_pred == e["C_str"] else "✗"
        parts.append(f"  EX{i}: L={L} (from {e['L_str']}), R={R} (from {e['R_str']}), "
                     f"op={op_name}")
        parts.append(f"        {op_desc} = {C_val} → encode → {C_pred}  "
                     f"(target {e['C_str']}) {ok}")
    parts.append("")

    # ---- Apply to query ----
    qL = decode_two_digit(query["L_str"], mapping, mode)
    qR = decode_two_digit(query["R_str"], mapping, mode)
    q_op_name = op_map[query["op"]]
    q_op_fn, q_op_desc = OPS[q_op_name]
    qC_val = q_op_fn(qL, qR)
    qC_str = encode_result(qC_val, len(answer), inv_map, mode)

    parts.append("Apply to query:")
    parts.append(f"  Query 【{query['L_str']}{query['op']}{query['R_str']}】")
    parts.append(f"  L = {qL} (from {query['L_str']}, {mode})")
    parts.append(f"  R = {qR} (from {query['R_str']}, {mode})")
    parts.append(f"  Operation = {q_op_name} ({q_op_desc})")
    parts.append(f"  Numeric result = {qC_val}")
    parts.append(f"  Encode {qC_val} → {qC_str}")
    parts.append("")
    parts.append(f"I will now return the answer in \\boxed{{}}")
    parts.append(f"\\boxed{{{qC_str}}}")
    parts.append("</think>")
    parts.append(f"\\boxed{{{qC_str}}}")

    cot = "\n".join(parts)

    # Sanity: prediction must match GT answer
    assert qC_str == answer, f"CoT generated wrong answer: {qC_str} vs {answer}"

    return cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", type=str, default="sequence_solved.jsonl")
    ap.add_argument("--out",    type=str, default="sequence_donald_tong.jsonl")
    args = ap.parse_args()

    n_in, n_out, n_bad = 0, 0, 0
    with open(args.solved) as f_in, open(args.out, "w") as f_out:
        for line in f_in:
            rec = json.loads(line)
            n_in += 1
            try:
                cot = build_cot(rec)
            except AssertionError as e:
                n_bad += 1
                continue

            # Training record in the same format as your other JSONLs
            record = {
                "category": "equation_symbolic",
                "messages": [
                    {"role": "user",      "content": rec["prompt"]},
                    {"role": "assistant", "content": cot},
                ],
                "_id": rec["id"],
                "_answer": rec["answer"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Wrote {n_out} CoTs ({n_bad} dropped as malformed) from {n_in} solved")
    if n_out:
        import statistics
        with open(args.out) as f:
            lens = [len(json.loads(l)["messages"][-1]["content"]) for l in f]
        print(f"CoT chars: median={statistics.median(lens):.0f} "
              f"mean={statistics.mean(lens):.0f} max={max(lens)}")
        print(f"Estimated tokens: median≈{statistics.median(lens)/3.5:.0f}")


if __name__ == "__main__":
    main()
