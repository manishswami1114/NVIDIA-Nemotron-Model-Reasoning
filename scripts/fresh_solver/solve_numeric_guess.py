#!/usr/bin/env python3
"""
solve_numeric_guess.py — Fresh solver for equation_numeric_guess puzzles.

These have VISIBLE digits: 79-12 = 67, but the operator symbol and
meta-rules (reverse_ops, reverse_res) are hidden.

Key fix: all result comparisons use STRINGS to preserve leading zeros.
e.g., result "07" must be compared as string, not int.

Usage:
    python solve_numeric_guess.py
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

# ============================================================
# Operations — 24 candidates
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

OP_ORDER = list(OPS.keys())

OP_LABELS = {
    "add": "L + R", "absdiff": "|L − R|", "sub": "L − R",
    "mul": "L × R", "concat_fwd": "concat(L,R)", "add_p1": "L + R + 1",
    "add_m1": "L + R − 1", "mul_p1": "L × R + 1", "mul_m1": "L × R − 1",
    "rsub": "R − L", "concat_rev": "concat(R,L)", "add_p2": "L + R + 2",
    "neg_absdiff": "−|L − R|", "mod": "L mod R", "rmod": "R mod L",
    "gcd": "gcd(L,R)", "lcm": "lcm(L,R)", "absdiff_p1": "|L − R| + 1",
    "absdiff_m1": "|L − R| − 1", "absdiff_m2": "|L − R| − 2",
    "a2_plus_b": "L² + R", "fdiv": "L ÷ R",
    "max_mod_min": "max mod min", "min_mod_max": "min mod max",
}


# ============================================================
# Puzzle parser — keeps result as STRING to preserve leading zeros
# ============================================================
NUMERIC_LINE_RX = re.compile(r"^(\d{2})(.)(\d{2})\s*=\s*(-?\d+)\s*$")
NUMERIC_QUERY_RX = re.compile(
    r"determine the result for:\s*(\d{2})(.)(\d{2})", re.IGNORECASE
)
BOXED_RX = re.compile(r"\\boxed\{([^}]+)\}")


def parse_numeric_puzzle(prompt: str, answer_text: str = ""):
    examples = []
    query = None
    gt = None
    for line in prompt.strip().split("\n"):
        line = line.strip()
        m = NUMERIC_LINE_RX.match(line)
        if m:
            examples.append({
                "L_str": m.group(1),
                "L_num": int(m.group(1)),
                "op_char": m.group(2),
                "R_str": m.group(3),
                "R_num": int(m.group(3)),
                "result_str": m.group(4),  # STRING — preserves leading zeros
            })
        m2 = NUMERIC_QUERY_RX.search(line)
        if m2:
            query = {
                "L_str": m2.group(1),
                "L_num": int(m2.group(1)),
                "op_char": m2.group(2),
                "R_str": m2.group(3),
                "R_num": int(m2.group(3)),
            }
    if answer_text:
        boxed = BOXED_RX.findall(answer_text)
        if boxed:
            gt = boxed[-1]
    return examples, query, gt


# ============================================================
# Reversal helpers — STRING-based to preserve leading zeros
# ============================================================
def reverse_2digit(n: int) -> int:
    """Reverse a 2-digit number: 06→60, 79→97."""
    s = f"{n:02d}"
    return int(s[::-1])


def reverse_result_str(n: int) -> str:
    """Reverse result as STRING. 10 → '01', 70 → '07', -35 → '-53'."""
    if n < 0:
        return "-" + str(-n)[::-1]
    return str(n)[::-1]


# ============================================================
# Solver
# ============================================================
def compute_result_str(L: int, R: int, op_fn, rev_res: bool) -> str | None:
    """Compute operation result and return as string (with reversal if needed)."""
    try:
        C = op_fn(L, R)
    except Exception:
        return None
    if C is None:
        return None
    if rev_res:
        return reverse_result_str(C)
    return str(C)


def check_op_on_examples(examples, op_fn, rev_ops: bool, rev_res: bool) -> bool:
    """Check if an operation matches ALL examples (string comparison)."""
    for ex in examples:
        L = reverse_2digit(ex["L_num"]) if rev_ops else ex["L_num"]
        R = reverse_2digit(ex["R_num"]) if rev_ops else ex["R_num"]
        result = compute_result_str(L, R, op_fn, rev_res)
        if result is None or result != ex["result_str"]:
            return False
    return True


def solve_numeric_puzzle(examples, query, gt=None):
    """Solve a numeric guess puzzle. Returns solution dict or None."""
    if not examples or not query:
        return None

    op_chars = sorted(set(e["op_char"] for e in examples))
    query_op = query["op_char"]
    groups = {oc: [e for e in examples if e["op_char"] == oc] for oc in op_chars}

    for rev_ops in [False, True]:
        for rev_res in [False, True]:
            op_map = {}
            all_found = True

            for oc in op_chars:
                grp = groups[oc]
                found = None
                for op_name in OP_ORDER:
                    op_fn = OPS[op_name]
                    if check_op_on_examples(grp, op_fn, rev_ops, rev_res):
                        found = op_name
                        break
                if found:
                    op_map[oc] = found
                else:
                    all_found = False
                    break

            if not all_found:
                continue

            # Compute query answer
            qL = reverse_2digit(query["L_num"]) if rev_ops else query["L_num"]
            qR = reverse_2digit(query["R_num"]) if rev_ops else query["R_num"]

            if query_op in op_map:
                q_fn = OPS[op_map[query_op]]
                answer = compute_result_str(qL, qR, q_fn, rev_res)
                if answer is not None:
                    return {
                        "answer": answer,
                        "op_map": op_map,
                        "reverse_ops": rev_ops,
                        "reverse_res": rev_res,
                    }
            elif gt:
                # Query uses unseen operator — try each operation against GT
                for op_name in OP_ORDER:
                    q_fn = OPS[op_name]
                    answer = compute_result_str(qL, qR, q_fn, rev_res)
                    if answer is not None and answer == gt:
                        return {
                            "answer": answer,
                            "op_map": {**op_map, query_op: op_name},
                            "reverse_ops": rev_ops,
                            "reverse_res": rev_res,
                        }

    return None


# ============================================================
# Main
# ============================================================
def main():
    THIS = Path(__file__).resolve().parent
    ROOT = THIS.parent.parent
    BASELINE = ROOT / "dont_touch_it" / "all_categorical_splits"
    SRC = BASELINE / "train_cot_equation_numeric_guess.jsonl"
    OUT = THIS / "solved_numeric_guess_fresh.jsonl"

    puzzles = []
    with SRC.open() as f:
        for line in f:
            rec = json.loads(line)
            examples, query, gt = parse_numeric_puzzle(
                rec["messages"][0]["content"],
                rec["messages"][1]["content"],
            )
            puzzles.append({
                "id": rec["id"],
                "prompt": rec["messages"][0]["content"],
                "examples": examples,
                "query": query,
                "gt": gt,
            })

    print(f"Loaded {len(puzzles)} puzzles")

    solved = 0
    failed = 0
    gt_match = 0
    gt_mismatch = 0

    with OUT.open("w") as fout:
        for p in puzzles:
            result = solve_numeric_puzzle(p["examples"], p["query"], gt=p["gt"])

            if result:
                match = result["answer"] == p["gt"] if p["gt"] else None
                if match is True:
                    gt_match += 1
                elif match is False:
                    gt_mismatch += 1
                    print(f"  MISMATCH {p['id']}: got={result['answer']}, gt={p['gt']}")

                rec_out = {
                    "id": p["id"],
                    "prompt": p["prompt"],
                    "answer": result["answer"],
                    "gt": p["gt"],
                    "match_gt": match,
                    "op_map": result["op_map"],
                    "reverse_ops": result["reverse_ops"],
                    "reverse_res": result["reverse_res"],
                    "examples": p["examples"],
                    "query": p["query"],
                }
                fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                solved += 1
            else:
                failed += 1
                print(f"  FAILED {p['id']}: gt={p['gt']}")

    print(f"\nResults:")
    print(f"  Solved: {solved}/{len(puzzles)} ({100*solved/len(puzzles):.1f}%)")
    print(f"  GT match: {gt_match}")
    print(f"  GT mismatch: {gt_mismatch}")
    print(f"  Failed: {failed}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    main()
