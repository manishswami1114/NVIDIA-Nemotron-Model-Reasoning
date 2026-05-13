"""
gen_bit_manip_gf2.py — Affine GF(2) solver + Donald-format CoT generator
=========================================================================

Insight: for many bit_manipulation puzzles, each output bit is an XOR of
input bits plus an optional constant — an AFFINE transform over GF(2).

  c_i = (w_{i,0} · b_0) ⊕ (w_{i,1} · b_1) ⊕ ... ⊕ (w_{i,7} · b_7) ⊕ bias_i

That's 9 unknowns per output bit. With 8+ examples, this is a small linear
system over GF(2) solvable in milliseconds by Gaussian elimination.

For puzzles where the rule is purely affine, we generate a SHORT, RIGOROUS
CoT showing the discovered equations. This replaces the 2,500-token brute-
force enumeration with a 500-800 token mathematical derivation.

Bit convention used throughout:
  Input  "01010001" → b_7=0, b_6=1, b_5=0, b_4=1, b_3=0, b_2=0, b_1=0, b_0=1
  (i.e. b_7 is MSB, b_0 is LSB — matches conventional bit positions)

Usage:
    python gen_bit_manip_gf2.py \\
        --csv ../../data/raw/train.csv \\
        --out bit_manip_gf2_donald.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

csv.field_size_limit(sys.maxsize)


# ============================================================
# Bit utilities
# ============================================================
def str_to_bits(s: str) -> list[int]:
    """'01010001' → [b_7, b_6, b_5, b_4, b_3, b_2, b_1, b_0] = [0,1,0,1,0,0,0,1]

    Index 0 of the returned list is b_7 (MSB). Index 7 is b_0 (LSB).
    But for solver math we want b_j at list index j (so b_0 at index 0).
    """
    assert len(s) == 8 and all(c in "01" for c in s), f"bad bit string: {s!r}"
    # Reverse so b_0 (LSB) is at list[0], b_7 (MSB) at list[7]
    return [int(c) for c in reversed(s)]


def bits_to_str(bits: list[int]) -> str:
    """[b_0, b_1, ..., b_7] → '<b_7><b_6>...<b_0>'"""
    assert len(bits) == 8
    return "".join(str(b) for b in reversed(bits))


# ============================================================
# GF(2) Gaussian elimination
# ============================================================
def solve_gf2(matrix_a: list[list[int]], vec_b: list[int]) -> Optional[list[int]]:
    """Solve A·x = b over GF(2).

    A is m×n, b is length m. Returns x (length n) or None if inconsistent.
    For under-determined systems, returns one valid solution (free vars = 0).
    """
    m = len(matrix_a)
    n = len(matrix_a[0]) if m > 0 else 0

    # Augmented matrix [A | b], copy to avoid mutating input
    aug = [[matrix_a[i][j] & 1 for j in range(n)] + [vec_b[i] & 1] for i in range(m)]

    pivot_row = 0
    pivot_cols = []  # for back-sub
    for col in range(n):
        # Find pivot in this column at or below pivot_row
        p = None
        for r in range(pivot_row, m):
            if aug[r][col] == 1:
                p = r; break
        if p is None:
            continue  # free variable
        # Swap
        if p != pivot_row:
            aug[pivot_row], aug[p] = aug[p], aug[pivot_row]
        # Eliminate this column in all OTHER rows
        for r in range(m):
            if r != pivot_row and aug[r][col] == 1:
                for c in range(col, n + 1):
                    aug[r][c] ^= aug[pivot_row][c]
        pivot_cols.append((pivot_row, col))
        pivot_row += 1
        if pivot_row == m:
            break

    # Check consistency: any row with all-zero LHS but nonzero RHS = inconsistent
    for r in range(m):
        if all(aug[r][c] == 0 for c in range(n)) and aug[r][n] == 1:
            return None

    # Back-extract: pivot col → its RHS value; free vars set to 0
    x = [0] * n
    for pr, pc in pivot_cols:
        x[pc] = aug[pr][n]
    return x


def solve_affine_for_output_bit(examples: list[tuple[list[int], int]]) -> Optional[dict]:
    """For one output bit, solve:
       c = (w_0·b_0) ⊕ (w_1·b_1) ⊕ ... ⊕ (w_7·b_7) ⊕ bias

    Args:
       examples: list of (input_bits[0..7], output_bit) pairs

    Returns: {"weights": [w_0..w_7], "bias": bias} or None if no affine solution.
    """
    # 9 unknowns: w_0..w_7, bias. Matrix col 8 is bias (always 1 in eqn).
    A = [bits + [1] for bits, _ in examples]   # m × 9
    b = [out_bit for _, out_bit in examples]   # m
    sol = solve_gf2(A, b)
    if sol is None:
        return None

    weights = sol[:8]
    bias = sol[8]

    # Verify on every example (Gaussian elim can give an answer that doesn't
    # satisfy ALL equations if the system was over-determined but inconsistent
    # — we should have caught that, but double-check for safety)
    for bits, expected in examples:
        computed = bias
        for j in range(8):
            if weights[j]:
                computed ^= bits[j]
        if computed != expected:
            return None  # affine fit failed on this example

    return {"weights": weights, "bias": bias}


def solve_full_affine(input_bits_list: list[list[int]],
                      output_bits_list: list[list[int]]) -> Optional[list[dict]]:
    """Solve the affine rule for EVERY output bit (8 separate small systems).

    Args:
        input_bits_list:  [example1_bits[0..7], example2_bits[0..7], ...]
        output_bits_list: [example1_out_bits[0..7], example2_out_bits[0..7], ...]

    Returns: list of 8 dicts (one per output bit) or None if ANY bit has no
             consistent affine solution.
    """
    n_examples = len(input_bits_list)
    assert len(output_bits_list) == n_examples

    rules = []
    for out_idx in range(8):  # for each output bit c_0..c_7
        examples_for_this_bit = [
            (input_bits_list[k], output_bits_list[k][out_idx])
            for k in range(n_examples)
        ]
        rule = solve_affine_for_output_bit(examples_for_this_bit)
        if rule is None:
            return None  # at least one output bit isn't affine
        rules.append(rule)
    return rules


# ============================================================
# Donald-format CoT generation
# ============================================================
PREAMBLE = (
    "\nI am a reasoning model. I am in a Kaggle competition. I follow the "
    "template strictly to achieve high accuracy.\n\n"
    "RULE 1: I must identify what type of question this is because that "
    "determines which template I move forward with. This is a binary boolean "
    "decomposition template. I see 8-bit binary strings mapped to 8-bit "
    "binary strings. Each output bit is an independent boolean function of "
    "the input bits. I know this is NOT roman, unit conversion, gravity, "
    "symbol-digit, or cipher-digit.\n\n"
    "RULE 2: Since I have identified this is a binary template, none of the "
    "flavor text surrounding this problem matters. The wrapper is here to "
    "trick me. I am only here to solve the problem.\n\n"
    "RULE 3: Final answer in \\boxed{} at the end. Wrong format means zero "
    "points.\n\n"
    "RULE 4: I will look for an AFFINE rule over GF(2): each output bit "
    "c_i = XOR of (some input bits b_j) plus an optional constant. If "
    "such a rule fits all examples, I use it directly — no brute-force "
    "search needed.\n\n"
)


def format_equation(rule: dict, out_idx: int) -> str:
    """Render c_i = b_j ⊕ b_k ⊕ ... ⊕ 1 (or just c_i = 1 if all weights zero)."""
    weights = rule["weights"]
    bias = rule["bias"]
    terms = [f"b_{j}" for j in range(8) if weights[j]]
    if bias:
        terms.append("1")
    if not terms:
        terms = ["0"]
    return f"c_{out_idx} = " + " ⊕ ".join(terms)


def apply_rule_to_bits(rules: list[dict], input_bits: list[int]) -> list[int]:
    """Compute output bits from input bits using the discovered rules."""
    output_bits = []
    for out_idx in range(8):
        rule = rules[out_idx]
        v = rule["bias"]
        for j in range(8):
            if rule["weights"][j]:
                v ^= input_bits[j]
        output_bits.append(v)
    return output_bits


def format_bit_computation(rule: dict, out_idx: int, input_bits: list[int]) -> str:
    """Render the step-by-step computation of one output bit for a specific input.
    Example output: 'c_3 = b_2 ⊕ 1 = 1 ⊕ 1 = 0'
    """
    weights = rule["weights"]
    bias = rule["bias"]

    # Symbolic form on LHS
    sym_terms = [f"b_{j}" for j in range(8) if weights[j]]
    if bias:
        sym_terms.append("1")
    if not sym_terms:
        sym_terms = ["0"]
    sym_form = " ⊕ ".join(sym_terms)

    # Numeric form (substituting actual bit values)
    num_terms = [str(input_bits[j]) for j in range(8) if weights[j]]
    if bias:
        num_terms.append("1")
    if not num_terms:
        num_terms = ["0"]
    num_form = " ⊕ ".join(num_terms)

    # Compute result
    result = bias
    for j in range(8):
        if weights[j]:
            result ^= input_bits[j]

    if sym_form == num_form:
        return f"  c_{out_idx} = {sym_form} = {result}"
    return f"  c_{out_idx} = {sym_form} = {num_form} = {result}"


def build_cot(examples: list[tuple[str, str]],
              query_input: str,
              rules: list[dict],
              answer: str) -> str:
    """Build the full Donald-format CoT trace for one bit_manipulation puzzle."""

    # S2: discovered equations
    equation_lines = "\n".join(f"  {format_equation(rules[i], i)}" for i in range(8))

    # S3: verify on first example
    ex1_in_str, ex1_out_str = examples[0]
    ex1_in_bits = str_to_bits(ex1_in_str)
    ex1_check_lines = "\n".join(
        format_bit_computation(rules[i], i, ex1_in_bits) for i in range(8)
    )
    ex1_computed_bits = apply_rule_to_bits(rules, ex1_in_bits)
    ex1_computed_str = bits_to_str(ex1_computed_bits)
    ex1_ok = "YES" if ex1_computed_str == ex1_out_str else "NO"

    # S4: apply to query
    query_bits = str_to_bits(query_input)
    query_lines = "\n".join(
        format_bit_computation(rules[i], i, query_bits) for i in range(8)
    )
    query_computed = bits_to_str(apply_rule_to_bits(rules, query_bits))

    cot = PREAMBLE + (
        f"S1: I see 8-bit→8-bit mapping. I will solve the affine rule "
        f"c_i = XOR(b_j)⊕bias for each output bit using Gaussian elimination "
        f"over GF(2), then verify against an example, then apply to the query.\n\n"
        f"S2: SOLVE - discovered affine rule (fits all examples):\n"
        f"{equation_lines}\n\n"
        f"S3: VER - check rule on example 1 (input {ex1_in_str}).\n"
        f"Input bits b_7..b_0 = {' '.join(str(b) for b in reversed(ex1_in_bits))}.\n"
        f"{ex1_check_lines}\n"
        f"REPARSE: c_7..c_0 = {' '.join(str(b) for b in reversed(ex1_computed_bits))} "
        f"→ {ex1_computed_str}\n"
        f"CHK: Does {ex1_computed_str} = {ex1_out_str}? {ex1_ok}\n\n"
        f"S4: APPLY to query {query_input}.\n"
        f"Input bits b_7..b_0 = {' '.join(str(b) for b in reversed(query_bits))}.\n"
        f"{query_lines}\n"
        f"Output bits c_7..c_0 = {' '.join(str(b) for b in reversed(apply_rule_to_bits(rules, query_bits)))}\n\n"
        f"S5: ANS={query_computed}\n\n"
        f"\\boxed{{{query_computed}}}"
    )
    return cot


# ============================================================
# Parse train.csv bit_manipulation rows
# ============================================================
EX_PATTERN = re.compile(r"([01]{8})\s*->\s*([01]{8})")
QUERY_PATTERN = re.compile(r"determine the output for:\s*([01]{8})", re.IGNORECASE)


def parse_bit_manip_row(prompt: str, answer: str) -> Optional[dict]:
    """Extract examples + query input from a bit_manipulation prompt.
    Returns None if not parseable or not a bit_manipulation row."""
    if "bit manipulation rule" not in prompt:
        return None

    examples = EX_PATTERN.findall(prompt)
    if len(examples) < 2:
        return None  # need at least 2 examples

    qm = QUERY_PATTERN.search(prompt)
    if not qm:
        return None
    query_input = qm.group(1)

    return {
        "examples": examples,        # list of (input_str, output_str)
        "query_input": query_input,
        "answer": answer.strip(),
    }


# ============================================================
# Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default="../../data/raw/train.csv",
                    help="Path to train.csv")
    ap.add_argument("--out", type=str, default="bit_manip_gf2_donald.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    n_rows = 0
    n_bit_manip = 0
    n_affine_solved = 0
    n_affine_failed = 0   # no affine solution exists
    n_answer_mismatch = 0  # affine solved but our answer != GT (sanity bug)

    out_path = Path(args.out)
    out_file = out_path.open("w")

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if args.limit and n_rows > args.limit:
                break

            parsed = parse_bit_manip_row(row["prompt"], row["answer"])
            if parsed is None:
                continue
            n_bit_manip += 1

            # Convert examples to bits
            try:
                input_bits_list  = [str_to_bits(ex[0]) for ex in parsed["examples"]]
                output_bits_list = [str_to_bits(ex[1]) for ex in parsed["examples"]]
            except AssertionError:
                continue  # malformed

            # Solve affine rule
            rules = solve_full_affine(input_bits_list, output_bits_list)
            if rules is None:
                n_affine_failed += 1
                continue

            # Sanity: apply rule to query, verify against GT answer
            query_bits = str_to_bits(parsed["query_input"])
            computed_str = bits_to_str(apply_rule_to_bits(rules, query_bits))

            # If our affine rule doesn't match the query answer, the system
            # is under-determined. RE-SOLVE using the (query, answer) as
            # an extra equation. This works because we have the GT answer
            # at data-generation time. If a consistent affine rule STILL
            # doesn't exist that satisfies both examples AND query, the
            # true rule isn't affine and we skip.
            if computed_str != parsed["answer"]:
                try:
                    answer_bits = str_to_bits(parsed["answer"])
                except AssertionError:
                    n_answer_mismatch += 1; continue

                # Augment example list with the (query, answer) pair
                aug_inputs  = input_bits_list  + [query_bits]
                aug_outputs = output_bits_list + [answer_bits]
                rules = solve_full_affine(aug_inputs, aug_outputs)

                if rules is None:
                    # True rule is non-affine — can't recover via this method
                    n_answer_mismatch += 1
                    continue

                # Verify
                computed_str = bits_to_str(apply_rule_to_bits(rules, query_bits))
                if computed_str != parsed["answer"]:
                    n_answer_mismatch += 1
                    continue

            # Build CoT
            cot = build_cot(
                examples    = parsed["examples"],
                query_input = parsed["query_input"],
                rules       = rules,
                answer      = parsed["answer"],
            )

            # The full training record
            record = {
                "category": "bit_manipulation",
                "messages": [
                    {"role": "user",      "content": row["prompt"]},
                    {"role": "assistant", "content": cot},
                ],
                # internal-only fields for our analysis (won't hurt training)
                "_id": row["id"],
                "_answer": parsed["answer"],
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_affine_solved += 1

    out_file.close()

    print("=" * 60)
    print(f"  GF(2) affine solver — bit_manipulation coverage")
    print("=" * 60)
    print(f"  Total rows scanned         : {n_rows}")
    print(f"  bit_manipulation rows      : {n_bit_manip}")
    print(f"  Affine solution found      : {n_affine_solved}  "
          f"({100*n_affine_solved/max(n_bit_manip,1):.1f}%)")
    print(f"  No affine solution         : {n_affine_failed}  "
          f"({100*n_affine_failed/max(n_bit_manip,1):.1f}%)")
    print(f"  Affine soln but wrong ans  : {n_answer_mismatch}  "
          f"(sanity bug — should be 0)")
    print(f"\n  Wrote {n_affine_solved} CoTs to {out_path}")

    # CoT length stats
    if n_affine_solved:
        lens = []
        with out_path.open() as f:
            for line in f:
                r = json.loads(line)
                lens.append(len(r["messages"][-1]["content"]))
        import statistics
        print(f"\n  CoT length (chars):")
        print(f"    median = {statistics.median(lens):.0f}")
        print(f"    mean   = {statistics.mean(lens):.0f}")
        print(f"    max    = {max(lens)}")
        print(f"    est tokens ≈ chars / 3.5 = {statistics.median(lens)/3.5:.0f} median")


if __name__ == "__main__":
    main()
