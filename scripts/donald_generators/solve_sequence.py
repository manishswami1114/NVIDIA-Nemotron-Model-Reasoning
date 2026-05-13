"""
solve_sequence.py — Independent symbolic solver for cipher-digit puzzles.

Smart algorithm: iterate (L_val, R_val) on example 1, derive op from
satisfying-C constraint, then verify remaining examples + query against
the ground-truth answer.

Per-puzzle target time: <2 seconds.

Usage:
    python solve_sequence.py --csv ../../data/raw/train.csv \\
                             --out sequence_solved.jsonl
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
# Operations — 22 from parquet observations
# ============================================================
def _safe_div(L, R):
    return L // R if R else None
def _safe_mod(L, R):
    return L % R if R else None

OPS = {
    "mul":         lambda L, R: L * R,
    "add":         lambda L, R: L + R,
    "absdiff":     lambda L, R: abs(L - R),
    "sub_signed":  lambda L, R: L - R,
    "concat_fwd":  lambda L, R: int(f"{L}{R}"),
    "add_m1":      lambda L, R: L + R - 1,
    "mul_m1":      lambda L, R: L * R - 1,
    "mul_p1":      lambda L, R: L * R + 1,
    "add_p1":      lambda L, R: L + R + 1,
    "rsub_signed": lambda L, R: R - L,
    "neg_absdiff": lambda L, R: -abs(L - R),
    "concat_rev":  lambda L, R: int(f"{R}{L}"),
    "mod":         _safe_mod,
    "gcd":         lambda L, R: math.gcd(L, R),
    "rmod":        lambda L, R: _safe_mod(R, L),
    "lcm":         lambda L, R: math.lcm(L, R) if (L or R) else 0,
    "absdiff_p1":  lambda L, R: abs(L - R) + 1,
    "absdiff_m1":  lambda L, R: abs(L - R) - 1,
    "absdiff_m2":  lambda L, R: abs(L - R) - 2,
    "add_p2":      lambda L, R: L + R + 2,
    "a2_plus_b":   lambda L, R: L * L + R,
    "fdiv":        _safe_div,
}
# Order by parquet frequency (most common first → solve faster on common cases)
OP_ORDER = [
    "mul", "add", "absdiff", "sub_signed", "concat_fwd",
    "add_m1", "mul_m1", "mul_p1", "add_p1", "rsub_signed",
    "neg_absdiff", "concat_rev", "mod", "gcd", "rmod", "lcm",
    "absdiff_p1", "absdiff_m1", "absdiff_m2", "add_p2",
    "a2_plus_b", "fdiv",
]


# ============================================================
# Puzzle parser
# ============================================================
LINE_RX  = re.compile(r"^(\S{5})\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"determine the result for:\s*(\S{5})\s*$", re.IGNORECASE)


def parse_puzzle(prompt: str):
    if "secret set of transformation rules" not in prompt:
        return None

    examples = []
    query = None

    for line in prompt.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        m = LINE_RX.match(line)
        if m and len(m.group(1)) == 5:
            lhs = m.group(1); rhs = m.group(2)
            examples.append({"L_str": lhs[0:2], "op": lhs[2],
                             "R_str": lhs[3:5], "C_str": rhs})
            continue
        m = QUERY_RX.search(line)
        if m and len(m.group(1)) == 5:
            qlhs = m.group(1)
            query = {"L_str": qlhs[0:2], "op": qlhs[2], "R_str": qlhs[3:5]}

    if not examples or query is None:
        return None
    return {"examples": examples, "query": query}


# ============================================================
# Helpers
# ============================================================
def split_two_digit(val: int, mode: str):
    """Return (digit_at_position_0, digit_at_position_1) for a 2-digit value."""
    if mode == "standard":
        return val // 10, val % 10
    else:  # little_endian: low digit first
        return val % 10, val // 10


def split_n_digit(val: int, length: int, mode: str):
    """Return list of length `length` of digits at positions 0..length-1."""
    s = str(val).zfill(length)
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return digits


def merge_two_digit(d0: int, d1: int, mode: str):
    return 10 * d0 + d1 if mode == "standard" else 10 * d1 + d0


def merge_n_digit(digits, mode: str):
    if mode == "little_endian":
        digits = list(reversed(digits))
    n = 0
    for d in digits:
        n = n * 10 + d
    return n


def add_to_mapping(mapping: dict, char: str, digit: int) -> dict | None:
    """Add char→digit to mapping. Return new mapping, or None on conflict."""
    if char in mapping:
        return mapping if mapping[char] == digit else None
    if digit in mapping.values():
        return None
    m = dict(mapping)
    m[char] = digit
    return m


def add_string_to_mapping(mapping: dict, chars: str, digits) -> dict | None:
    """Add each (char, digit) pair to mapping. Return new mapping or None."""
    m = mapping
    for c, d in zip(chars, digits):
        m = add_to_mapping(m, c, d)
        if m is None:
            return None
    return m


# ============================================================
# Solver — smart per-example pruning
# ============================================================
def solve(puzzle, true_answer, time_budget_sec=5.0):
    examples = puzzle["examples"]
    query    = puzzle["query"]
    op_chars = sorted(set(e["op"] for e in examples) | {query["op"]})

    t_start = time.time()
    e1 = examples[0]
    rest = examples[1:]

    for mode in ["standard", "little_endian"]:
        # Iterate (L_val, R_val) on example 1 → assign chars to digits
        for L_val in range(100):
            if time.time() - t_start > time_budget_sec:
                return None

            # Set L_str chars
            L_d0, L_d1 = split_two_digit(L_val, mode)
            m0 = add_to_mapping({}, e1["L_str"][0], L_d0)
            if m0 is None: continue
            m1 = add_to_mapping(m0, e1["L_str"][1], L_d1)
            if m1 is None: continue

            for R_val in range(100):
                R_d0, R_d1 = split_two_digit(R_val, mode)
                m2 = add_to_mapping(m1, e1["R_str"][0], R_d0)
                if m2 is None: continue
                m3 = add_to_mapping(m2, e1["R_str"][1], R_d1)
                if m3 is None: continue

                # Now try each op for op1
                for op_name in OP_ORDER:
                    try:
                        C_val = OPS[op_name](L_val, R_val)
                    except Exception:
                        continue
                    if C_val is None or C_val < 0:
                        continue
                    if len(str(C_val)) != len(e1["C_str"]):
                        continue

                    # Assign C_str chars from C_val
                    C_digits = split_n_digit(C_val, len(e1["C_str"]), mode)
                    m4 = add_string_to_mapping(m3, e1["C_str"], C_digits)
                    if m4 is None: continue

                    # Recurse on remaining examples
                    initial_op_map = {e1["op"]: op_name}
                    sol = _try_remaining(
                        m4, initial_op_map, rest, query, true_answer, mode
                    )
                    if sol is not None:
                        return {
                            "mapping": sol["mapping"],
                            "op_map":  sol["op_map"],
                            "mode":    mode,
                            "query_numeric": sol["query_numeric"],
                        }
    return None


def _try_remaining(mapping, op_map, remaining_examples, query, true_answer, mode):
    """Recursively fit remaining examples, then verify query → true_answer."""
    if not remaining_examples:
        return _check_query(mapping, op_map, query, true_answer, mode)

    e = remaining_examples[0]
    rest = remaining_examples[1:]

    # Decode L, R if possible
    if e["L_str"][0] not in mapping or e["L_str"][1] not in mapping \
       or e["R_str"][0] not in mapping or e["R_str"][1] not in mapping:
        # Can't decode yet — try every (L, R, op) combination consistent
        # with the example. Iterate L, R values.
        # (This is rare in practice — most examples reuse chars.)
        return _try_with_unknown(mapping, op_map, e, rest, query, true_answer, mode)

    L = merge_two_digit(mapping[e["L_str"][0]], mapping[e["L_str"][1]], mode)
    R = merge_two_digit(mapping[e["R_str"][0]], mapping[e["R_str"][1]], mode)

    # If op already known: use it
    if e["op"] in op_map:
        op_name = op_map[e["op"]]
        try:
            C_val = OPS[op_name](L, R)
        except Exception:
            return None
        if C_val is None or C_val < 0:
            return None
        if len(str(C_val)) != len(e["C_str"]):
            return None
        C_digits = split_n_digit(C_val, len(e["C_str"]), mode)
        new_map = add_string_to_mapping(mapping, e["C_str"], C_digits)
        if new_map is None:
            return None
        return _try_remaining(new_map, op_map, rest, query, true_answer, mode)

    # New op_char: try each op
    for op_name in OP_ORDER:
        try:
            C_val = OPS[op_name](L, R)
        except Exception:
            continue
        if C_val is None or C_val < 0:
            continue
        if len(str(C_val)) != len(e["C_str"]):
            continue
        C_digits = split_n_digit(C_val, len(e["C_str"]), mode)
        new_map = add_string_to_mapping(mapping, e["C_str"], C_digits)
        if new_map is None:
            continue
        new_op_map = {**op_map, e["op"]: op_name}
        sol = _try_remaining(new_map, new_op_map, rest, query, true_answer, mode)
        if sol is not None:
            return sol
    return None


def _try_with_unknown(mapping, op_map, e, rest, query, true_answer, mode):
    """Try (L_val, R_val) for an example whose chars aren't all mapped."""
    for L_val in range(100):
        Ld = split_two_digit(L_val, mode)
        mL = add_to_mapping(mapping, e["L_str"][0], Ld[0])
        if mL is None: continue
        mL = add_to_mapping(mL, e["L_str"][1], Ld[1])
        if mL is None: continue
        for R_val in range(100):
            Rd = split_two_digit(R_val, mode)
            mR = add_to_mapping(mL, e["R_str"][0], Rd[0])
            if mR is None: continue
            mR = add_to_mapping(mR, e["R_str"][1], Rd[1])
            if mR is None: continue

            ops_to_try = [op_map[e["op"]]] if e["op"] in op_map else OP_ORDER
            for op_name in ops_to_try:
                try:
                    C_val = OPS[op_name](L_val, R_val)
                except Exception:
                    continue
                if C_val is None or C_val < 0: continue
                if len(str(C_val)) != len(e["C_str"]): continue
                C_digits = split_n_digit(C_val, len(e["C_str"]), mode)
                mC = add_string_to_mapping(mR, e["C_str"], C_digits)
                if mC is None: continue
                new_op_map = {**op_map, e["op"]: op_name} \
                             if e["op"] not in op_map else op_map
                sol = _try_remaining(mC, new_op_map, rest, query, true_answer, mode)
                if sol is not None:
                    return sol
    return None


def _check_query(mapping, op_map, query, true_answer, mode):
    """All examples passed; verify query → true_answer."""
    # Need query op assigned
    if query["op"] not in op_map:
        # Try each op (uncommon)
        for op_name in OP_ORDER:
            test_op_map = {**op_map, query["op"]: op_name}
            res = _check_query(mapping, test_op_map, query, true_answer, mode)
            if res is not None:
                return res
        return None

    # Need query L, R chars mapped
    for c in query["L_str"] + query["R_str"]:
        if c not in mapping:
            # Can't decode — try all values for missing chars
            return _check_query_with_unknown(mapping, op_map, query, true_answer, mode)

    qL = merge_two_digit(mapping[query["L_str"][0]], mapping[query["L_str"][1]], mode)
    qR = merge_two_digit(mapping[query["R_str"][0]], mapping[query["R_str"][1]], mode)
    op_name = op_map[query["op"]]
    try:
        qC_val = OPS[op_name](qL, qR)
    except Exception:
        return None
    if qC_val is None or qC_val < 0:
        return None
    if len(str(qC_val)) != len(true_answer):
        return None

    # Decode true_answer chars using mapping (must be consistent)
    qC_digits = split_n_digit(qC_val, len(true_answer), mode)
    final_map = add_string_to_mapping(mapping, true_answer, qC_digits)
    if final_map is None:
        return None
    return {"mapping": final_map, "op_map": op_map, "query_numeric": qC_val}


def _check_query_with_unknown(mapping, op_map, query, true_answer, mode):
    for L_val in range(100):
        Ld = split_two_digit(L_val, mode)
        mL = add_to_mapping(mapping, query["L_str"][0], Ld[0])
        if mL is None: continue
        mL = add_to_mapping(mL, query["L_str"][1], Ld[1])
        if mL is None: continue
        for R_val in range(100):
            Rd = split_two_digit(R_val, mode)
            mR = add_to_mapping(mL, query["R_str"][0], Rd[0])
            if mR is None: continue
            mR = add_to_mapping(mR, query["R_str"][1], Rd[1])
            if mR is None: continue
            res = _check_query(mR, op_map, query, true_answer, mode)
            if res is not None:
                return res
    return None


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str,
                    default="../../data/raw/train.csv")
    ap.add_argument("--out", type=str, default="sequence_solved.jsonl")
    ap.add_argument("--time-budget", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    n_seq, n_solved, n_failed, n_timeout = 0, 0, 0, 0
    t_total_start = time.time()

    with open(args.csv) as f, open(args.out, "w") as out:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = parse_puzzle(row["prompt"])
            if parsed is None:
                continue
            n_seq += 1
            if args.limit and n_seq > args.limit:
                break

            t0 = time.time()
            sol = solve(parsed, row["answer"], time_budget_sec=args.time_budget)
            t = time.time() - t0

            if sol is None:
                if t >= args.time_budget * 0.95:
                    n_timeout += 1; status = "TIMEOUT"
                else:
                    n_failed += 1; status = "no-soln"
            else:
                n_solved += 1
                record = {
                    "id":            row["id"],
                    "prompt":        row["prompt"],
                    "answer":        row["answer"],
                    "examples":      parsed["examples"],
                    "query":         parsed["query"],
                    "mapping":       sol["mapping"],
                    "op_map":        sol["op_map"],
                    "mode":          sol["mode"],
                    "query_numeric": sol["query_numeric"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                status = f"SOLVED ({sol['mode'][:3]})"

            if n_seq % 25 == 0 or n_seq <= 10:
                el = (time.time() - t_total_start) / 60
                print(f"  [{n_seq:4d}] {row['id']}  {t:5.2f}s  {status:<15}  "
                      f"solved={n_solved}/{n_seq} ({100*n_solved/n_seq:.1f}%)  "
                      f"total={el:.1f}m")

    el = (time.time() - t_total_start) / 60
    print(f"\n{'='*60}")
    print(f"  Sequence puzzle solver — final results")
    print(f"{'='*60}")
    print(f"  Total puzzles  : {n_seq}")
    print(f"  Solved         : {n_solved}  ({100*n_solved/max(n_seq,1):.1f}%)")
    print(f"  No solution    : {n_failed}")
    print(f"  Timed out      : {n_timeout}")
    print(f"  Wall time      : {el:.1f} min")
    print(f"  Output → {args.out}")


if __name__ == "__main__":
    main()
