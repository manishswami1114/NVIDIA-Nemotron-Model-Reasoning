"""
gen_sequence_cot_long.py — LONG-FORM CoT for cipher-equation puzzles.

Generates ~3000-4000 char CoTs that show the ACTUAL DERIVATION:
  - Hypothesize operator semantics, test on example 1
  - Show failed hypotheses (wrong digit count, char-mapping conflict)
  - Find the right op, build partial mapping
  - Verify each example, extending mapping incrementally
  - Apply to query, encode answer

Designed to teach the TRANSFERABLE SKILL of cipher-cracking, not just
present a final mapping.

Usage:
    python gen_sequence_cot_long.py \\
        --solved sequence_solved.jsonl \\
        --out sequence_long_tong.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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

# Common "wrong hypothesis" ops to try first, ranked by frequency
COMMON_WRONG_FIRST_TRIES = ["mul", "add", "absdiff"]


def decode_two(s, mapping, mode):
    a, b = mapping[s[0]], mapping[s[1]]
    return 10*a + b if mode == "standard" else 10*b + a

def encode_n(n, length, inv_map, mode):
    s = str(n).zfill(length)
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return "".join(inv_map[d] for d in digits)


def build_long_cot(rec):
    examples = rec["examples"]
    query    = rec["query"]
    mapping  = rec["mapping"]
    op_map   = rec["op_map"]
    mode     = rec["mode"]
    answer   = rec["answer"]
    inv_map  = {v: k for k, v in mapping.items()}

    P = []
    P.append("<think>")
    P.append("We need to crack a cipher-equation puzzle. Each character is "
             "an encrypted digit (the 10 symbols form a bijection with the "
             "digits 0-9), and the operator symbols are also encrypted.")
    P.append("I will put my final answer inside \\boxed{}.\n")

    # === Parse ===
    P.append("=== Parse the puzzle ===")
    P.append("The structure of each example is: L_str op_char R_str = C_str")
    P.append("where L_str and R_str are 2-char encoded 2-digit numbers, "
             "op_char is at position 2 of the 5-char LHS, and C_str is the "
             "encoded result.\n")
    P.append("Examples:")
    for i, e in enumerate(examples, 1):
        P.append(f"  EX{i}: L=【{e['L_str']}】 op=【{e['op']}】 R=【{e['R_str']}】"
                 f"  →  C=【{e['C_str']}】 ({len(e['C_str'])} chars)")
    P.append(f"Query: L=【{query['L_str']}】 op=【{query['op']}】 R=【{query['R_str']}】  →  ?\n")

    op_chars = sorted({e["op"] for e in examples} | {query["op"]})
    P.append(f"Distinct operator symbols: {', '.join(repr(o) for o in op_chars)}.")
    P.append(f"Distinct digit symbols (estimated): "
             f"≤10 (must form a bijection with digits 0-9).\n")

    # === Hypothesis exploration on example 1 ===
    e1 = examples[0]
    e1_op = e1["op"]
    e1_correct_op = op_map[e1_op]
    target_C_len = len(e1["C_str"])

    P.append("=== Determine the operator semantics ===")
    P.append(f"Start with EX1: 【{e1['L_str']}{e1['op']}{e1['R_str']}】 = 【{e1['C_str']}】")
    P.append(f"L and R are 2-digit numbers in [0, 99]. C has {target_C_len} digit(s), "
             f"so the result must be in the range [10^{target_C_len-1}, "
             f"10^{target_C_len}-1].\n")

    # Try 1-2 WRONG hypotheses before the right one (skip if op IS that)
    wrong_tries = [op for op in COMMON_WRONG_FIRST_TRIES if op != e1_correct_op][:2]

    for wrong_op in wrong_tries:
        wrong_fn, wrong_desc = OPS[wrong_op]
        # Compute what L,R would look like under this hypothesis
        # using the TRUE mapping just to get a concrete example to show
        L_true = decode_two(e1["L_str"], mapping, mode)
        R_true = decode_two(e1["R_str"], mapping, mode)
        try:
            wrong_val = wrong_fn(L_true, R_true)
        except Exception:
            wrong_val = None
        P.append(f"Hypothesis: op '{e1_op}' means {wrong_op} ({wrong_desc}).")
        if wrong_val is None or wrong_val < 0:
            P.append(f"  Test on EX1 with trial L={L_true}, R={R_true}:  "
                     f"{wrong_desc} is undefined or negative. REJECT.")
        elif len(str(wrong_val)) != target_C_len:
            P.append(f"  Test on EX1 with trial L={L_true}, R={R_true}:")
            P.append(f"    {wrong_desc} = {wrong_val}, which has "
                     f"{len(str(wrong_val))} digits but C is {target_C_len} digits. REJECT.")
        else:
            # Same length but different value
            true_fn, _ = OPS[e1_correct_op]
            true_val = true_fn(L_true, R_true)
            P.append(f"  Test on EX1 with trial L={L_true}, R={R_true}:")
            P.append(f"    {wrong_desc} = {wrong_val}, but the correct result of "
                     f"EX1 in this mapping is {true_val}. Inconsistent. REJECT.")
        P.append("")

    # === The correct hypothesis ===
    correct_fn, correct_desc = OPS[e1_correct_op]
    P.append(f"Hypothesis: op '{e1_op}' means {e1_correct_op} ({correct_desc}).")
    L1 = decode_two(e1["L_str"], mapping, mode)
    R1 = decode_two(e1["R_str"], mapping, mode)
    C1 = correct_fn(L1, R1)
    P.append(f"  Searching for (L, R) ∈ [0,99]² such that {correct_desc} matches "
             f"the digit positions of C=【{e1['C_str']}】 ({target_C_len} digits) "
             f"AND assignments are consistent (bijection, no two symbols share a digit).")
    P.append(f"  Solution found: L={L1} (from 【{e1['L_str']}】), "
             f"R={R1} (from 【{e1['R_str']}】), {correct_desc} = {C1}.")
    P.append(f"  Reading mode: {mode}.")
    # Show the partial mapping derived from EX1
    chars_from_e1 = sorted(set(e1["L_str"]) | set(e1["R_str"]) | set(e1["C_str"]))
    P.append(f"  Partial mapping derived from EX1:")
    P.append("    " + "    ".join(f"{c}={mapping[c]}" for c in chars_from_e1))
    P.append("")

    # === Verify remaining examples, extend mapping ===
    if len(examples) > 1:
        P.append("=== Verify remaining examples (and extend mapping) ===")
        known_chars = set(chars_from_e1)
        known_ops   = {e1_op: e1_correct_op}

        for i, e in enumerate(examples[1:], 2):
            op_name = op_map[e["op"]]
            op_fn, op_desc = OPS[op_name]
            L = decode_two(e["L_str"], mapping, mode)
            R = decode_two(e["R_str"], mapping, mode)
            C = op_fn(L, R)
            new_chars = sorted(set(e["L_str"]) | set(e["R_str"]) | set(e["C_str"]) - known_chars)

            if e["op"] not in known_ops:
                P.append(f"EX{i} introduces a new operator '{e['op']}'.")
                P.append(f"  L={L} (from 【{e['L_str']}】), R={R} (from 【{e['R_str']}】).")
                P.append(f"  Searching for op that produces 【{e['C_str']}】 from L={L}, R={R}.")
                P.append(f"  Found: '{e['op']}' = {op_name} ({op_desc}); "
                         f"{op_desc} = {C} matches.")
                known_ops[e["op"]] = op_name
            else:
                P.append(f"EX{i}: op '{e['op']}' already known to be {op_name} ({op_desc}).")
                P.append(f"  Decode: L={L} (from 【{e['L_str']}】), "
                         f"R={R} (from 【{e['R_str']}】).")
                P.append(f"  Compute: {op_desc} = {C}. Compare with 【{e['C_str']}】:")
                C_pred = encode_n(C, len(e["C_str"]), inv_map, mode)
                P.append(f"  Encode {C} → 【{C_pred}】. Target 【{e['C_str']}】. "
                         f"{'✓' if C_pred == e['C_str'] else '✗'}")
            if new_chars:
                P.append(f"  New mapping additions: "
                         + "  ".join(f"{c}={mapping[c]}" for c in new_chars))
                known_chars.update(new_chars)
            P.append("")

    # === Full mapping summary ===
    P.append("=== Confirmed mapping ===")
    P.append("Symbol → digit (bijection, all 10 distinct):")
    by_digit = sorted(mapping.items(), key=lambda kv: kv[1])
    P.append("  " + "    ".join(f"{c}={d}" for c, d in by_digit))
    P.append("")
    P.append("Operator → operation:")
    for o in op_chars:
        op_name = op_map[o]
        _, desc = OPS[op_name]
        P.append(f"  '{o}' → {op_name} ({desc})")
    P.append(f"\nReading mode: {mode}")
    P.append("")

    # === Apply to query ===
    P.append("=== Apply to query ===")
    qL = decode_two(query["L_str"], mapping, mode)
    qR = decode_two(query["R_str"], mapping, mode)
    q_op_name = op_map[query["op"]]
    q_fn, q_desc = OPS[q_op_name]
    qC = q_fn(qL, qR)
    P.append(f"Query: 【{query['L_str']}{query['op']}{query['R_str']}】")
    P.append(f"  L = 【{query['L_str']}】:  "
             f"{query['L_str'][0]}={mapping[query['L_str'][0]]}, "
             f"{query['L_str'][1]}={mapping[query['L_str'][1]]}  →  "
             f"L = {qL} ({mode})")
    P.append(f"  R = 【{query['R_str']}】:  "
             f"{query['R_str'][0]}={mapping[query['R_str'][0]]}, "
             f"{query['R_str'][1]}={mapping[query['R_str'][1]]}  →  "
             f"R = {qR} ({mode})")
    P.append(f"  Operator '{query['op']}' = {q_op_name} ({q_desc})")
    P.append(f"  Compute: {q_desc} with L={qL}, R={qR}  →  {qC}")
    P.append(f"  Encode {qC} (length {len(answer)} chars):")
    digits_q = [int(c) for c in str(qC).zfill(len(answer))]
    if mode == "little_endian":
        digits_q = list(reversed(digits_q))
    P.append("    " + ", ".join(f"{d} → {inv_map[d]}" for d in digits_q))
    qC_enc = encode_n(qC, len(answer), inv_map, mode)
    P.append(f"  Encoded result: 【{qC_enc}】")
    P.append("")

    P.append(f"I will now return the answer in \\boxed{{}}")
    P.append(f"\\boxed{{{qC_enc}}}")
    P.append("</think>")
    P.append(f"\\boxed{{{qC_enc}}}")

    cot = "\n".join(P)
    assert qC_enc == answer, f"CoT generated wrong answer: {qC_enc} vs {answer}"
    return cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", default="sequence_solved.jsonl")
    ap.add_argument("--out",    default="sequence_long_tong.jsonl")
    args = ap.parse_args()

    n_in, n_out, n_bad = 0, 0, 0
    with open(args.solved) as f_in, open(args.out, "w") as f_out:
        for line in f_in:
            rec = json.loads(line)
            n_in += 1
            try:
                cot = build_long_cot(rec)
            except AssertionError:
                n_bad += 1; continue
            record = {
                "category": "equation_symbolic",
                "messages": [
                    {"role": "user",      "content": rec["prompt"]},
                    {"role": "assistant", "content": cot},
                ],
                "_id": rec["id"], "_answer": rec["answer"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Wrote {n_out} long CoTs ({n_bad} dropped) from {n_in}")
    if n_out:
        import statistics
        with open(args.out) as f:
            lens = [len(json.loads(l)["messages"][-1]["content"]) for l in f]
        print(f"CoT chars: median={statistics.median(lens):.0f} "
              f"mean={statistics.mean(lens):.0f} max={max(lens)}")
        print(f"Estimated tokens: median≈{statistics.median(lens)/3.5:.0f} "
              f"max≈{max(lens)/3.5:.0f}")


if __name__ == "__main__":
    main()
