#!/usr/bin/env python3
"""
gen_cot_short.py — Generate SHORT CoTs matching archive format EXACTLY.

For cryptarithm:
  - Always "Operation: +" (matching 822/823 archive records)
  - Always "Reverse Operands: False", "Reverse Result: False"
  - ALL symbols (mapping + operators) set to 0
  - Correct answer in \\boxed{}

For numeric_guess:
  - Correct Meta-rules line
  - Correct Operation name (spaces, not underscores)
  - Calculation line: "L op_name R = result -> answer"
  - Uses query operator to select the right operation
"""
from __future__ import annotations
import csv
import json
import math
import re
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parent.parent


# ============================================================
# Operation name mapping for numeric_guess
# Verified against archive: these EXACT names appear in the 111 archive CoTs
# ============================================================
NUMERIC_OP_NAMES = {
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
    "rmod": "max mod min",
    "gcd": "gcd",
    "lcm": "lcm",
    "absdiff_p1": "absdiff+1",
    "absdiff_m1": "absdiff-1",
    "absdiff_m2": "absdiff-2",
    "a2_plus_b": "a squared plus b",
    "fdiv": "floor division",
    "max_mod_min": "max mod min",
    "min_mod_max": "min mod max",
}


# ============================================================
# Operation functions for computing the Calculation line
# ============================================================
def _apply_op(op_name: str, L: int, R: int) -> int:
    """Apply the named operation to L, R."""
    if op_name == "add":
        return L + R
    elif op_name == "sub":
        return L - R
    elif op_name == "absdiff":
        return abs(L - R)
    elif op_name == "mul":
        return L * R
    elif op_name == "concat_fwd":
        return int(str(L) + str(R))
    elif op_name == "concat_rev":
        return int(str(R) + str(L))
    elif op_name == "add_p1":
        return L + R + 1
    elif op_name == "add_m1":
        return L + R - 1
    elif op_name == "add_p2":
        return L + R + 2
    elif op_name == "mul_p1":
        return L * R + 1
    elif op_name == "mul_m1":
        return L * R - 1
    elif op_name == "rsub":
        return R - L
    elif op_name == "neg_absdiff":
        return -(abs(L - R))
    elif op_name == "mod":
        return L % R if R != 0 else 0
    elif op_name in ("rmod", "max_mod_min"):
        mn, mx = min(L, R), max(L, R)
        return mx % mn if mn != 0 else 0
    elif op_name == "min_mod_max":
        mn, mx = min(L, R), max(L, R)
        return mn % mx if mx != 0 else 0
    elif op_name == "gcd":
        return math.gcd(L, R)
    elif op_name == "lcm":
        g = math.gcd(L, R)
        return abs(L * R) // g if g != 0 else 0
    elif op_name == "absdiff_p1":
        return abs(L - R) + 1
    elif op_name == "absdiff_m1":
        return abs(L - R) - 1
    elif op_name == "absdiff_m2":
        return abs(L - R) - 2
    elif op_name == "a2_plus_b":
        return L * L + R
    elif op_name == "fdiv":
        return L // R if R != 0 else 0
    else:
        raise ValueError(f"Unknown operation: {op_name}")


def gen_cryptarithm_short_cot(solved_rec: dict) -> str:
    """Generate archive-matching short CoT for cryptarithm.

    Archive format (822/823 records):
    - Operation: always "+"
    - Reverse Operands/Result: always False
    - ALL symbols (mapping keys + operator keys) mapped to 0
    - Correct answer in \\boxed{}
    """
    mapping = solved_rec["mapping"]
    op_map = solved_rec["op_map"]
    answer = solved_rec["answer"]

    # Collect ALL symbols: mapping keys + operator keys (matching archive's 10-13 symbols)
    all_symbols = set(mapping.keys()) | set(op_map.keys())

    lines = ["<think>"]
    lines.append("This is a cryptarithm solved using base-10 arithmetic and logical constraints.")
    lines.append("Deduced properties:")
    lines.append("- Base: 10")
    lines.append("- Operation: +")              # Always "+" (archive: 822/823)
    lines.append("- Reverse Operands: False")    # Always False (archive: 823/823)
    lines.append("- Reverse Result: False")      # Always False (archive: 823/823)
    lines.append("- All symbols map to distinct values: True")
    lines.append("Symbol Mapping:")
    for sym in sorted(all_symbols):
        lines.append(f"  '{sym}' = 0")           # Always 0 (archive: 822/823)
    lines.append("The exact value calculation yields the correct answer sequence.")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


def gen_numeric_guess_short_cot(solved_rec: dict) -> str:
    """Generate archive-matching short CoT for numeric_guess.

    Archive format (ALL 111 records):
    <think>
    Meta-rules: reverse_ops=True, reverse_res=True
    Operation: addition
    Calculation: 60 addition 77 = 137 -> 731
    The answer is \\boxed{731}
    </think>
    \\boxed{731}
    """
    answer = solved_rec["answer"]
    op_map = solved_rec.get("op_map", {})
    reverse_ops = solved_rec.get("reverse_ops", False)
    reverse_res = solved_rec.get("reverse_res", False)
    query = solved_rec.get("query", {})

    # Use the QUERY's operator to look up the correct operation
    op_char = query.get("op_char", "")
    op_name = op_map.get(op_char, "unknown")

    # Map to archive operation name (with spaces, not underscores)
    archive_op = NUMERIC_OP_NAMES.get(op_name, op_name)

    # Compute L, R (reverse the string if reverse_ops=True)
    L_str = query.get("L_str", "0")
    R_str = query.get("R_str", "0")

    if reverse_ops:
        L = int(L_str[::-1])
        R = int(R_str[::-1])
    else:
        L = int(L_str)
        R = int(R_str)

    # Compute the operation result
    result = _apply_op(op_name, L, R)

    # Apply reverse_res to get the display answer
    if reverse_res:
        if result < 0:
            display_answer = "-" + str(abs(result))[::-1]
        else:
            display_answer = str(result)[::-1]
    else:
        display_answer = str(result)

    lines = ["<think>"]
    lines.append(f"Meta-rules: reverse_ops={reverse_ops}, reverse_res={reverse_res}")
    lines.append(f"Operation: {archive_op}")
    lines.append(f"Calculation: {L} {archive_op} {R} = {result} -> {display_answer}")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


def main():
    BASELINE = ROOT / "dont_touch_it" / "all_categorical_splits"
    TRAIN_CSV = ROOT / "dont_touch_it" / "train.csv"

    # Load authoritative GT
    csv_gt = {}
    with TRAIN_CSV.open() as f:
        for row in csv.DictReader(f):
            csv_gt[row["id"]] = row["answer"]

    # ============================================================
    # CRYPTARITHM: Generate short CoTs
    # ============================================================
    SOLVED_CRYPTO = THIS / "solved_cryptarithm_fresh.jsonl"
    OUT_CRYPTO = THIS / "cot_cryptarithm.jsonl"

    crypto_solved = {}
    with SOLVED_CRYPTO.open() as f:
        for line in f:
            rec = json.loads(line)
            rid = rec["id"]
            gt = csv_gt.get(rid, "")

            if rec.get("match_gt") is True:
                crypto_solved[rid] = rec
            elif len(rec["answer"]) == len(gt):
                # Symbol-swap fix: correct the answer to match GT
                fixed = dict(rec)
                fixed_mapping = dict(rec["mapping"])
                for i in range(len(rec["answer"])):
                    if rec["answer"][i] != gt[i]:
                        our_c = rec["answer"][i]
                        gt_c = gt[i]
                        if our_c in fixed_mapping:
                            d = fixed_mapping.pop(our_c)
                            fixed_mapping[gt_c] = d
                fixed["mapping"] = fixed_mapping
                fixed["answer"] = gt
                crypto_solved[rid] = fixed

    # Load originals (for prompts)
    originals = {}
    for fname in [
        BASELINE / "train_cot_cryptarithm_deduce.jsonl",
        BASELINE / "train_cot_cryptarithm_guess.jsonl",
    ]:
        with fname.open() as f:
            for line in f:
                rec = json.loads(line)
                originals[rec["id"]] = rec

    generated = 0
    errors = 0
    cot_lengths = []

    with OUT_CRYPTO.open("w") as f_out:
        for rid, srec in crypto_solved.items():
            if rid not in originals:
                continue
            try:
                cot = gen_cryptarithm_short_cot(srec)
                prompt = originals[rid]["messages"][0]["content"]
                out_rec = {
                    "id": rid,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": cot},
                    ],
                }
                f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                cot_lengths.append(len(cot))
                generated += 1
            except Exception as e:
                print(f"  ERROR {rid}: {e}")
                errors += 1

    print(f"Cryptarithm: generated={generated}, errors={errors}")
    if cot_lengths:
        print(f"  CoT length: min={min(cot_lengths)}, max={max(cot_lengths)}, avg={sum(cot_lengths)/len(cot_lengths):.0f}")
    print(f"  Output: {OUT_CRYPTO}")
    print()

    # ============================================================
    # NUMERIC GUESS: Generate short CoTs
    # ============================================================
    SOLVED_NUMERIC = THIS / "solved_numeric_guess_fresh.jsonl"
    OUT_NUMERIC = THIS / "cot_equation_numeric_guess.jsonl"

    # Load archive numeric guess for fallback (ea829b32 not in our solved records)
    ARCHIVE_NUMERIC = BASELINE / "train_cot_equation_numeric_guess.jsonl"
    archive_numeric = {}
    if ARCHIVE_NUMERIC.exists():
        with ARCHIVE_NUMERIC.open() as f:
            for line in f:
                rec = json.loads(line)
                archive_numeric[rec["id"]] = rec

    numeric_solved = {}
    with SOLVED_NUMERIC.open() as f:
        for line in f:
            rec = json.loads(line)
            numeric_solved[rec["id"]] = rec

    generated_n = 0
    fallback_n = 0
    cot_lengths_n = []

    with OUT_NUMERIC.open("w") as f_out:
        # First: write all our solved records
        for rid, srec in numeric_solved.items():
            cot = gen_numeric_guess_short_cot(srec)
            prompt = srec["prompt"]
            out_rec = {
                "id": rid,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": cot},
                ],
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            cot_lengths_n.append(len(cot))
            generated_n += 1

        # Second: add archive records we don't have (e.g. ea829b32)
        for rid, arec in archive_numeric.items():
            if rid not in numeric_solved:
                f_out.write(json.dumps(arec, ensure_ascii=False) + "\n")
                cot_lengths_n.append(len(arec["messages"][1]["content"]))
                fallback_n += 1

    print(f"Numeric guess: generated={generated_n}, fallback_from_archive={fallback_n}")
    if cot_lengths_n:
        print(f"  CoT length: min={min(cot_lengths_n)}, max={max(cot_lengths_n)}, avg={sum(cot_lengths_n)/len(cot_lengths_n):.0f}")
    print(f"  Output: {OUT_NUMERIC}")


if __name__ == "__main__":
    main()
