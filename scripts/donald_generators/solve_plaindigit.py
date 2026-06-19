"""
solve_plaindigit.py — solver for plain-digit cipher-equation puzzles.

These are the puzzles where digits 0-9 are PLAIN (not encrypted),
but the operator symbols ARE encrypted. Example:

    34/44 = 1
    41/32 = 9
    34|25 = 69
    87\\64 = 8853
    Query: 69/52

Three encrypted operators (`/`, `|`, `\\`). Need to figure out which
arithmetic operation each represents. No symbol→digit bijection needed.

Tries 30+ operation candidates per operator symbol, including:
  - Standard ops (mul, add, absdiff, sub_signed, ...)
  - With result reversed
  - With operand swap
  - Concatenation variants
  - Sign-based operations

Usage:
    python solve_plaindigit.py --csv ../../data/raw/train.csv \\
                                --out plain_solved.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
csv.field_size_limit(sys.maxsize)


# ============================================================
# Operations — base set
# ============================================================
def _safe_mod(L, R): return L % R if R else None
def _safe_div(L, R): return L // R if R else None

def _rev(n): return int(str(n)[::-1].lstrip('0') or '0')

BASE_OPS = {
    "mul":         lambda L, R: L * R,
    "add":         lambda L, R: L + R,
    "absdiff":     lambda L, R: abs(L - R),
    "sub_signed":  lambda L, R: L - R,
    "rsub_signed": lambda L, R: R - L,
    "concat_fwd":  lambda L, R: int(f"{L:02d}{R:02d}"),
    "concat_rev":  lambda L, R: int(f"{R:02d}{L:02d}"),
    "add_p1":      lambda L, R: L + R + 1,
    "add_m1":      lambda L, R: L + R - 1,
    "mul_p1":      lambda L, R: L * R + 1,
    "mul_m1":      lambda L, R: L * R - 1,
    "absdiff_p1":  lambda L, R: abs(L - R) + 1,
    "absdiff_m1":  lambda L, R: abs(L - R) - 1,
    "absdiff_m2":  lambda L, R: abs(L - R) - 2,
    "neg_absdiff": lambda L, R: -abs(L - R),
    "mod":         _safe_mod,
    "rmod":        lambda L, R: _safe_mod(R, L),
    "gcd":         lambda L, R: math.gcd(L, R),
    "lcm":         lambda L, R: math.lcm(L, R) if (L or R) else 0,
    "a2_plus_b":   lambda L, R: L * L + R,
    "b2_plus_a":   lambda L, R: R * R + L,
    "a2_minus_b":  lambda L, R: L * L - R,
    "fdiv":        _safe_div,
    "add_p2":      lambda L, R: L + R + 2,
    "sum_squares": lambda L, R: L * L + R * R,
    "diff_squares":lambda L, R: L * L - R * R,
    # Digit-reverse variants
    "rev_L_add_R":     lambda L, R: _rev(L) + R,
    "rev_L_sub_R":     lambda L, R: _rev(L) - R,
    "rev_L_mul_R":     lambda L, R: _rev(L) * R,
    "rev_L_absdiff_R": lambda L, R: abs(_rev(L) - R),
    "L_add_rev_R":     lambda L, R: L + _rev(R),
    "L_sub_rev_R":     lambda L, R: L - _rev(R),
    "L_mul_rev_R":     lambda L, R: L * _rev(R),
    "L_absdiff_rev_R": lambda L, R: abs(L - _rev(R)),
    "rev_L_add_rev_R": lambda L, R: _rev(L) + _rev(R),
    "rev_L_sub_rev_R": lambda L, R: _rev(L) - _rev(R),
    "rev_L_mul_rev_R": lambda L, R: _rev(L) * _rev(R),
    "rev_L_absdiff_rev_R": lambda L, R: abs(_rev(L) - _rev(R)),
    # Concat with operations
    "concat_then_rev": lambda L, R: int(f"{L:02d}{R:02d}"[::-1]),
    "L_mul_R_then_rev": lambda L, R: _rev(L * R),
    "L_add_R_then_rev": lambda L, R: _rev(L + R),
    # Sum of digits
    "digit_sum":    lambda L, R: sum(int(c) for c in f"{L:02d}{R:02d}"),
    "digit_sum_L":  lambda L, R: sum(int(c) for c in f"{L:02d}"),
    "digit_sum_LR": lambda L, R: sum(int(c) for c in f"{L:02d}{R:02d}"),
    "digit_prod":   lambda L, R: int(str(L)[0])*int(str(L)[1])*int(str(R)[0])*int(str(R)[1])
                    if L >= 10 and R >= 10 else None,
    # Specific patterns observed
    "L_plus_R_rev_each":   lambda L, R: int(str(L)[::-1]) + int(str(R)[::-1]),
    "absdiff_rev_unsigned": lambda L, R: _rev(abs(L - R)),
    "L_mod10_plus_R_mod10": lambda L, R: (L % 10) + (R % 10),
    "L_div10_plus_R_div10": lambda L, R: (L // 10) + (R // 10),
}


# ============================================================
# Result encoders — different ways result can be written
# ============================================================
def fmt_signed(n):
    """Signed integer with possible negative sign."""
    return str(n)

def fmt_reversed(n):
    """Result with digits reversed (negative sign keeps in front)."""
    if n < 0:
        return "-" + str(-n)[::-1]
    return str(n)[::-1]

def fmt_abs(n):
    """Absolute value as string."""
    return str(abs(n))

def fmt_abs_reversed(n):
    return str(abs(n))[::-1]


def fmt_pad2(n):
    if n < 0: return None
    return f"{n:02d}"

def fmt_pad3(n):
    if n < 0: return None
    return f"{n:03d}"

def fmt_pad4(n):
    if n < 0: return None
    return f"{n:04d}"

def fmt_pad2_rev(n):
    s = fmt_pad2(n);  return s[::-1] if s is not None else None

def fmt_pad3_rev(n):
    s = fmt_pad3(n);  return s[::-1] if s is not None else None

def fmt_pad4_rev(n):
    s = fmt_pad4(n);  return s[::-1] if s is not None else None

RESULT_FMTS = {
    "signed":       fmt_signed,
    "reversed":     fmt_reversed,
    "abs":          fmt_abs,
    "abs_reversed": fmt_abs_reversed,
    "pad2":         fmt_pad2,
    "pad3":         fmt_pad3,
    "pad4":         fmt_pad4,
    "pad2_rev":     fmt_pad2_rev,
    "pad3_rev":     fmt_pad3_rev,
    "pad4_rev":     fmt_pad4_rev,
}


# ============================================================
# Puzzle parser
# ============================================================
LINE_RX  = re.compile(r"^(\S+)\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"determine the result for:\s*(\S+)\s*$", re.IGNORECASE)


def parse(prompt: str):
    """Extract examples + query from a plain-digit cipher-equation prompt.
    Returns None if it's not a plain-digit puzzle (some chars in LHS aren't digits)."""
    if "secret set of transformation rules" not in prompt:
        return None

    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.rstrip()
        if not line: continue
        m = LINE_RX.match(line)
        if m and len(m.group(1)) == 5:
            lhs = m.group(1)
            # Check: positions 0,1,3,4 must be plain digits; position 2 = encrypted op
            if all(lhs[i].isdigit() for i in (0, 1, 3, 4)) and not lhs[2].isdigit():
                examples.append({
                    "L": int(lhs[:2]),
                    "op": lhs[2],
                    "R": int(lhs[3:5]),
                    "C": m.group(2),  # keep as string for sign/reverse comparison
                })
                continue
        m = QUERY_RX.search(line)
        if m and len(m.group(1)) == 5:
            qlhs = m.group(1)
            if all(qlhs[i].isdigit() for i in (0, 1, 3, 4)) and not qlhs[2].isdigit():
                query = {"L": int(qlhs[:2]), "op": qlhs[2], "R": int(qlhs[3:5])}

    if not examples or query is None:
        return None
    return {"examples": examples, "query": query}


# ============================================================
# Solver — find operation for each operator symbol
# ============================================================
def find_op_for_symbol(examples_with_this_op):
    """Given examples that all use the same operator symbol, find the (op, fmt)
    that satisfies all of them. Returns (op_name, fmt_name) or None."""
    for op_name, op_fn in BASE_OPS.items():
        for fmt_name, fmt_fn in RESULT_FMTS.items():
            ok = True
            for ex in examples_with_this_op:
                try:
                    val = op_fn(ex["L"], ex["R"])
                except Exception:
                    ok = False; break
                if val is None:
                    ok = False; break
                try:
                    pred = fmt_fn(val)
                except Exception:
                    ok = False; break
                if pred != ex["C"]:
                    ok = False; break
            if ok:
                return op_name, fmt_name
    return None


def solve(parsed, true_answer):
    """Solve the puzzle. Returns dict with op_map, fmt_map, query_result, or None."""
    examples = parsed["examples"]
    query    = parsed["query"]

    # Group examples by operator symbol
    by_op = {}
    for e in examples:
        by_op.setdefault(e["op"], []).append(e)

    op_map = {}    # op_char → operation name
    fmt_map = {}   # op_char → result format name

    for op_char, group in by_op.items():
        result = find_op_for_symbol(group)
        if result is None:
            # Couldn't identify this operator
            return None
        op_name, fmt_name = result
        op_map[op_char] = op_name
        fmt_map[op_char] = fmt_name

    # Query op may be NEW (not in examples) → "guess" type
    if query["op"] not in op_map:
        # Try each (op, fmt) pair against the GT answer
        for op_name, op_fn in BASE_OPS.items():
            for fmt_name, fmt_fn in RESULT_FMTS.items():
                try:
                    val = op_fn(query["L"], query["R"])
                except Exception: continue
                if val is None: continue
                try:
                    pred = fmt_fn(val)
                except Exception: continue
                if pred == true_answer:
                    op_map[query["op"]]  = op_name
                    fmt_map[query["op"]] = fmt_name
                    return {
                        "op_map":  op_map,
                        "fmt_map": fmt_map,
                        "query_result": val,
                        "query_pred":   pred,
                    }
        return None

    # Query op is known — apply it
    op_name = op_map[query["op"]]
    fmt_name = fmt_map[query["op"]]
    try:
        val = BASE_OPS[op_name](query["L"], query["R"])
        if val is None: return None
        pred = RESULT_FMTS[fmt_name](val)
    except Exception:
        return None

    # Verify against GT
    if pred != true_answer:
        return None

    return {
        "op_map":  op_map,
        "fmt_map": fmt_map,
        "query_result": val,
        "query_pred":   pred,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../../data/raw/train.csv")
    ap.add_argument("--out", default="plain_solved.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    n_total = 0
    n_solved = 0
    n_failed = 0
    n_not_plain = 0
    t_start = time.time()

    with open(args.csv) as f, open(args.out, "w") as out:
        for row in csv.DictReader(f):
            parsed = parse(row["prompt"])
            if parsed is None:
                continue
            n_total += 1
            if args.limit and n_total > args.limit:
                break

            sol = solve(parsed, row["answer"])
            if sol:
                n_solved += 1
                out.write(json.dumps({
                    "id":           row["id"],
                    "prompt":       row["prompt"],
                    "answer":       row["answer"],
                    "examples":     parsed["examples"],
                    "query":        parsed["query"],
                    "op_map":       sol["op_map"],
                    "fmt_map":      sol["fmt_map"],
                    "query_result": sol["query_result"],
                    "query_pred":   sol["query_pred"],
                }, ensure_ascii=False) + "\n")
            else:
                n_failed += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Plain-digit puzzle solver")
    print(f"{'='*60}")
    print(f"  Plain-digit puzzles found     : {n_total}")
    print(f"  Solved                         : {n_solved}  "
          f"({100*n_solved/max(n_total,1):.1f}%)")
    print(f"  Failed                         : {n_failed}")
    print(f"  Wall time                      : {elapsed:.1f}s")
    print(f"  Output → {args.out}")


if __name__ == "__main__":
    main()
