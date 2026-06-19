"""
solve_sequence_v2.py — improved fully-encrypted cipher-equation solver.

Improvements over v1:
  1. Two-phase search — fast pass with top-5 operators first, then exhaustive
  2. Longer per-puzzle budget (30s default vs 5s)
  3. Frequency-prioritized operator order — try common ops first
  4. Smarter pruning — bail on (L,R) candidates that violate length constraints
     before doing expensive mapping construction

Designed to crack the 395 fully-encrypted puzzles v1 couldn't solve.
Most are 3-operator puzzles that needed more time.

Usage:
    python solve_sequence_v2.py --csv ../../data/raw/train.csv \\
                                 --skip-already sequence_solved.jsonl \\
                                 --out sequence_solved_v2.jsonl
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
# Operations — same 22 ops as v1
# ============================================================
def _safe_div(L, R): return L // R if R else None
def _safe_mod(L, R): return L % R if R else None

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

# Frequency-prioritized order — try these first.
# Top 5 covers ~72% of all operator usages observed in parquet.
TOP_OPS  = ["mul", "add", "absdiff", "sub_signed", "concat_fwd"]
NEXT_OPS = ["add_m1", "mul_m1", "mul_p1", "add_p1", "rsub_signed",
            "neg_absdiff", "concat_rev"]
RARE_OPS = ["mod", "gcd", "rmod", "lcm", "absdiff_p1", "absdiff_m1",
            "absdiff_m2", "add_p2", "a2_plus_b", "fdiv"]

OP_ORDER_FAST = TOP_OPS                    # 5 ops
OP_ORDER_FULL = TOP_OPS + NEXT_OPS + RARE_OPS  # 22 ops


# ============================================================
# Puzzle parser (same as v1)
# ============================================================
LINE_RX  = re.compile(r"^(\S{5})\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"determine the result for:\s*(\S{5})\s*$", re.IGNORECASE)


def parse_puzzle(prompt):
    if "secret set of transformation rules" not in prompt:
        return None
    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.rstrip()
        if not line: continue
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
    # Reject plain-digit puzzles (those go to plain solver)
    if any(c.isdigit() for c in examples[0]["L_str"] + examples[0]["R_str"]):
        return None
    return {"examples": examples, "query": query}


# ============================================================
# Helpers (same as v1)
# ============================================================
def split_two(val, mode):
    if mode == "standard": return val // 10, val % 10
    return val % 10, val // 10

def merge_two(d0, d1, mode):
    return 10 * d0 + d1 if mode == "standard" else 10 * d1 + d0

def split_n(val, length, mode):
    s = str(val).zfill(length)
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return digits

def add_to_map(mapping, char, digit):
    if char in mapping:
        return mapping if mapping[char] == digit else None
    if digit in mapping.values():
        return None
    m = dict(mapping); m[char] = digit
    return m

def add_str_to_map(mapping, chars, digits):
    m = mapping
    for c, d in zip(chars, digits):
        m = add_to_map(m, c, d)
        if m is None: return None
    return m


# ============================================================
# Core solver — same algorithm, just parameterized op list
# ============================================================
def _try_remaining(mapping, op_map, remaining, query, true_answer, mode, op_order):
    if not remaining:
        return _check_query(mapping, op_map, query, true_answer, mode, op_order)
    e = remaining[0]
    rest = remaining[1:]
    if any(c not in mapping for c in e["L_str"] + e["R_str"]):
        return _try_with_unknown(mapping, op_map, e, rest, query, true_answer, mode, op_order)
    L = merge_two(mapping[e["L_str"][0]], mapping[e["L_str"][1]], mode)
    R = merge_two(mapping[e["R_str"][0]], mapping[e["R_str"][1]], mode)

    if e["op"] in op_map:
        op_name = op_map[e["op"]]
        try: C_val = OPS[op_name](L, R)
        except Exception: return None
        if C_val is None or C_val < 0: return None
        if len(str(C_val)) != len(e["C_str"]): return None
        C_digits = split_n(C_val, len(e["C_str"]), mode)
        new_map = add_str_to_map(mapping, e["C_str"], C_digits)
        if new_map is None: return None
        return _try_remaining(new_map, op_map, rest, query, true_answer, mode, op_order)

    for op_name in op_order:
        try: C_val = OPS[op_name](L, R)
        except Exception: continue
        if C_val is None or C_val < 0: continue
        if len(str(C_val)) != len(e["C_str"]): continue
        C_digits = split_n(C_val, len(e["C_str"]), mode)
        new_map = add_str_to_map(mapping, e["C_str"], C_digits)
        if new_map is None: continue
        new_op_map = {**op_map, e["op"]: op_name}
        sol = _try_remaining(new_map, new_op_map, rest, query, true_answer, mode, op_order)
        if sol is not None: return sol
    return None


def _try_with_unknown(mapping, op_map, e, rest, query, true_answer, mode, op_order):
    for L_val in range(100):
        Ld = split_two(L_val, mode)
        mL = add_to_map(mapping, e["L_str"][0], Ld[0])
        if mL is None: continue
        mL = add_to_map(mL, e["L_str"][1], Ld[1])
        if mL is None: continue
        for R_val in range(100):
            Rd = split_two(R_val, mode)
            mR = add_to_map(mL, e["R_str"][0], Rd[0])
            if mR is None: continue
            mR = add_to_map(mR, e["R_str"][1], Rd[1])
            if mR is None: continue
            ops_to_try = [op_map[e["op"]]] if e["op"] in op_map else op_order
            for op_name in ops_to_try:
                try: C_val = OPS[op_name](L_val, R_val)
                except Exception: continue
                if C_val is None or C_val < 0: continue
                if len(str(C_val)) != len(e["C_str"]): continue
                C_digits = split_n(C_val, len(e["C_str"]), mode)
                mC = add_str_to_map(mR, e["C_str"], C_digits)
                if mC is None: continue
                new_op_map = {**op_map, e["op"]: op_name} if e["op"] not in op_map else op_map
                sol = _try_remaining(mC, new_op_map, rest, query, true_answer, mode, op_order)
                if sol is not None: return sol
    return None


def _check_query(mapping, op_map, query, true_answer, mode, op_order):
    if query["op"] not in op_map:
        for op_name in op_order:
            test_op_map = {**op_map, query["op"]: op_name}
            res = _check_query(mapping, test_op_map, query, true_answer, mode, op_order)
            if res is not None: return res
        return None
    if any(c not in mapping for c in query["L_str"] + query["R_str"]):
        return _check_query_with_unknown(mapping, op_map, query, true_answer, mode, op_order)
    qL = merge_two(mapping[query["L_str"][0]], mapping[query["L_str"][1]], mode)
    qR = merge_two(mapping[query["R_str"][0]], mapping[query["R_str"][1]], mode)
    op_name = op_map[query["op"]]
    try: qC_val = OPS[op_name](qL, qR)
    except Exception: return None
    if qC_val is None or qC_val < 0: return None
    if len(str(qC_val)) != len(true_answer): return None
    qC_digits = split_n(qC_val, len(true_answer), mode)
    final_map = add_str_to_map(mapping, true_answer, qC_digits)
    if final_map is None: return None
    return {"mapping": final_map, "op_map": op_map, "query_numeric": qC_val}


def _check_query_with_unknown(mapping, op_map, query, true_answer, mode, op_order):
    for L_val in range(100):
        Ld = split_two(L_val, mode)
        mL = add_to_map(mapping, query["L_str"][0], Ld[0])
        if mL is None: continue
        mL = add_to_map(mL, query["L_str"][1], Ld[1])
        if mL is None: continue
        for R_val in range(100):
            Rd = split_two(R_val, mode)
            mR = add_to_map(mL, query["R_str"][0], Rd[0])
            if mR is None: continue
            mR = add_to_map(mR, query["R_str"][1], Rd[1])
            if mR is None: continue
            res = _check_query(mR, op_map, query, true_answer, mode, op_order)
            if res is not None: return res
    return None


def solve(puzzle, true_answer, time_budget_sec, op_order):
    """Run the core solver with the given op order + time budget."""
    examples = puzzle["examples"]
    query    = puzzle["query"]
    t_start  = time.time()
    e1 = examples[0]
    rest = examples[1:]
    for mode in ["standard", "little_endian"]:
        for L_val in range(100):
            if time.time() - t_start > time_budget_sec:
                return None
            L_d0, L_d1 = split_two(L_val, mode)
            m0 = add_to_map({}, e1["L_str"][0], L_d0)
            if m0 is None: continue
            m1 = add_to_map(m0, e1["L_str"][1], L_d1)
            if m1 is None: continue
            for R_val in range(100):
                R_d0, R_d1 = split_two(R_val, mode)
                m2 = add_to_map(m1, e1["R_str"][0], R_d0)
                if m2 is None: continue
                m3 = add_to_map(m2, e1["R_str"][1], R_d1)
                if m3 is None: continue
                for op_name in op_order:
                    try: C_val = OPS[op_name](L_val, R_val)
                    except Exception: continue
                    if C_val is None or C_val < 0: continue
                    if len(str(C_val)) != len(e1["C_str"]): continue
                    C_digits = split_n(C_val, len(e1["C_str"]), mode)
                    m4 = add_str_to_map(m3, e1["C_str"], C_digits)
                    if m4 is None: continue
                    initial_op_map = {e1["op"]: op_name}
                    sol = _try_remaining(m4, initial_op_map, rest, query, true_answer, mode, op_order)
                    if sol is not None:
                        return {"mapping": sol["mapping"],
                                "op_map": sol["op_map"],
                                "mode": mode,
                                "query_numeric": sol["query_numeric"]}
    return None


def solve_two_phase(puzzle, true_answer, fast_sec=3.0, full_sec=30.0):
    """Phase 1: top-5 ops, 3s budget. Phase 2: all 22 ops, 30s budget."""
    # Phase 1: quick scan with top-5 ops
    sol = solve(puzzle, true_answer, time_budget_sec=fast_sec, op_order=OP_ORDER_FAST)
    if sol is not None:
        return sol, "phase1"
    # Phase 2: full op set with longer budget
    sol = solve(puzzle, true_answer, time_budget_sec=full_sec, op_order=OP_ORDER_FULL)
    return (sol, "phase2") if sol is not None else (None, "failed")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/data/raw/train.csv")
    ap.add_argument("--out", default="sequence_solved_v2.jsonl")
    ap.add_argument("--skip-already", default="",
                    help="JSONL of previously solved IDs to skip")
    ap.add_argument("--fast-budget", type=float, default=3.0)
    ap.add_argument("--full-budget", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    skip = set()
    if args.skip_already and Path(args.skip_already).exists():
        with open(args.skip_already) as f:
            for line in f:
                try: skip.add(json.loads(line)["id"])
                except: pass
        print(f"Skipping {len(skip)} already-solved puzzles\n")

    n_total = n_solved = n_failed = 0
    n_phase1 = n_phase2 = 0
    t_start = time.time()

    with open(args.csv) as f, open(args.out, "w") as out:
        for row in csv.DictReader(f):
            parsed = parse_puzzle(row["prompt"])
            if parsed is None: continue
            if row["id"] in skip: continue
            n_total += 1
            if args.limit and n_total > args.limit: break

            t0 = time.time()
            sol, phase = solve_two_phase(parsed, row["answer"],
                                          fast_sec=args.fast_budget,
                                          full_sec=args.full_budget)
            t = time.time() - t0
            if sol:
                n_solved += 1
                if phase == "phase1": n_phase1 += 1
                else:                 n_phase2 += 1
                rec = {"id": row["id"], "prompt": row["prompt"],
                       "answer": row["answer"],
                       "examples": parsed["examples"], "query": parsed["query"],
                       "mapping": sol["mapping"], "op_map": sol["op_map"],
                       "mode": sol["mode"], "query_numeric": sol["query_numeric"],
                       "solver": "v2_" + phase}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                status = f"SOLVED ({phase})"
            else:
                n_failed += 1
                status = "no-soln"

            elapsed_total = (time.time() - t_start) / 60
            if n_total % 10 == 0 or n_total <= 10:
                print(f"  [{n_total:4d}] {row['id']}  {t:5.1f}s  {status:<14}  "
                      f"p1={n_phase1} p2={n_phase2} fail={n_failed}  "
                      f"total={elapsed_total:.1f}m")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  Fully-encrypted solver v2 results")
    print(f"{'='*60}")
    print(f"  Puzzles attempted    : {n_total}")
    print(f"  Solved in phase 1    : {n_phase1}  (top-5 ops, {args.fast_budget}s)")
    print(f"  Solved in phase 2    : {n_phase2}  (all 22 ops, {args.full_budget}s)")
    print(f"  Total solved         : {n_solved}  "
          f"({100*n_solved/max(n_total,1):.1f}%)")
    print(f"  Failed               : {n_failed}")
    print(f"  Wall time            : {elapsed:.1f} min")
    print(f"  Output → {args.out}")


if __name__ == "__main__":
    main()
