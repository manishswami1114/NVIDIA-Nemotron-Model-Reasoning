"""
gen_multipos_cot.py — generate Tong-style CoTs for multipos-solved puzzles.

Mirrors gen_sequence_cot_tong.py but supports variable operand widths
(1, 2, or 3 digit operands depending on operator position).

Usage:
    python gen_multipos_cot.py \\
        --solved sequence_solved_multipos.jsonl \\
        --out   sequence_multipos_tong.jsonl
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def _safe_div(L, R): return L // R if R else None
def _safe_mod(L, R): return L % R if R else None

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
    "add_p2":      (lambda L, R: L + R + 2,                   "L + R + 2"),
    "a2_plus_b":   (lambda L, R: L * L + R,                   "L² + R"),
    "fdiv":        (_safe_div,                                "L ÷ R"),
}


def decode_digits(s, mapping, mode):
    digits = [mapping[c] for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    n = 0
    for d in digits:
        n = n * 10 + d
    return n


def encode_result(n, length, inv_map, mode):
    s = str(n).zfill(length)
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return "".join(inv_map[d] for d in digits)


def build_cot(rec):
    examples  = rec["examples"]
    query     = rec["query"]
    mapping   = rec["mapping"]
    op_map    = rec["op_map"]
    mode      = rec["mode"]
    answer    = rec["answer"]
    op_pos    = rec.get("op_position", 2)
    inv_map   = {v: k for k, v in mapping.items()}

    parts = ["<think>"]
    parts.append("We need to crack a cipher-equation puzzle. Each character "
                 "in the equations is an encrypted digit, and the operator "
                 "symbols are also encrypted.")
    parts.append("I will put my final answer inside \\boxed{}.\n")

    parts.append("Examples (encrypted):")
    for i, e in enumerate(examples, 1):
        # Reconstruct the original 5-char LHS by placing op at op_pos
        parts.append(f"  EX{i}: L=【{e['L_str']}】 op=【{e['op']}】 R=【{e['R_str']}】  →  C=【{e['C_str']}】")
    parts.append(f"Query: L=【{query['L_str']}】 op=【{query['op']}】 R=【{query['R_str']}】  →  ?\n")

    op_chars = sorted({e["op"] for e in examples} | {query["op"]})
    L_width = len(examples[0]["L_str"])
    R_width = len(examples[0]["R_str"])
    parts.append(f"In each example the LHS has the operator at position {op_pos}.")
    parts.append(f"  Left operand: {L_width} digit(s);  Right operand: {R_width} digit(s).")
    parts.append(f"  Operator symbols: {', '.join(repr(o) for o in op_chars)}.")
    parts.append(f"  Reading mode: {mode}.\n")

    parts.append("Symbol → digit (bijection):")
    by_digit = sorted(mapping.items(), key=lambda kv: kv[1])
    parts.append("  " + "    ".join(f"{c}={d}" for c, d in by_digit))
    parts.append("")

    parts.append("Operator-symbol → operation:")
    for o in op_chars:
        op_name = op_map[o]
        _, desc = OPS[op_name]
        parts.append(f"  '{o}' → {op_name} ({desc})")
    parts.append("")

    parts.append("Verification:")
    for i, e in enumerate(examples, 1):
        L = decode_digits(e["L_str"], mapping, mode)
        R = decode_digits(e["R_str"], mapping, mode)
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

    qL = decode_digits(query["L_str"], mapping, mode)
    qR = decode_digits(query["R_str"], mapping, mode)
    q_op_name = op_map[query["op"]]
    q_op_fn, q_op_desc = OPS[q_op_name]
    qC_val = q_op_fn(qL, qR)
    qC_str = encode_result(qC_val, len(answer), inv_map, mode)

    parts.append("Apply to query:")
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
    assert qC_str == answer, f"CoT generated wrong answer: {qC_str} vs {answer}"
    return cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", default="sequence_solved_multipos.jsonl")
    ap.add_argument("--out",    default="sequence_multipos_tong.jsonl")
    args = ap.parse_args()

    n_ok = 0; n_bad = 0
    with open(args.solved) as f_in, open(args.out, "w") as f_out:
        for line in f_in:
            rec = json.loads(line)
            try:
                cot = build_cot(rec)
            except AssertionError as e:
                n_bad += 1; continue
            record = {
                "category": "cryptarithm_deduce",  # match competition labels
                "messages": [
                    {"role": "user",      "content": rec["prompt"]},
                    {"role": "assistant", "content": cot},
                ],
                "_id":     rec["id"],
                "_answer": rec["answer"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} multipos CoTs ({n_bad} dropped)")


if __name__ == "__main__":
    main()
