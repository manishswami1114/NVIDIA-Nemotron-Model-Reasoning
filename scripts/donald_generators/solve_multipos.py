"""
solve_multipos.py — multi-position cipher-equation solver.

Cracks puzzles where the operator is NOT at position 2 (the v1/v2 assumption).
For each candidate operator position (0-4), parses the LHS into L_str + op +
R_str with appropriate operand widths, then runs the standard solver.

Position layouts for 5-char LHS:
  pos 0:  op L1 L2 R1 R2   (1-char op + 2-digit L + 2-digit R)
  pos 1:  L1 op R1 R2 R3   (1-digit L + op + 3-digit R)
  pos 2:  L1 L2 op R1 R2   (2-digit L + op + 2-digit R) — STANDARD
  pos 3:  L1 L2 L3 op R1   (3-digit L + op + 1-digit R)
  pos 4:  L1 L2 R1 R2 op   (2-digit L + 2-digit R + op at end)

Usage:
    python solve_multipos.py --csv ../../data/raw/train.csv \\
                              --skip-already sequence_solved.jsonl \\
                              --out sequence_solved_multipos.jsonl
"""
from __future__ import annotations
import argparse, csv, json, math, re, sys, time
from pathlib import Path

csv.field_size_limit(sys.maxsize)


# ============================================================
# Operations
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
    "add_p2":      lambda L, R: L + R + 2,
    "a2_plus_b":   lambda L, R: L * L + R,
    "fdiv":        _safe_div,
}
OP_ORDER = ["mul", "add", "absdiff", "sub_signed", "concat_fwd",
            "add_m1", "mul_m1", "mul_p1", "add_p1", "rsub_signed",
            "neg_absdiff", "concat_rev", "absdiff_p1", "absdiff_m1",
            "add_p2", "mod", "gcd", "rmod", "lcm", "a2_plus_b", "fdiv"]


# ============================================================
# Position configurations
# ============================================================
# Each config: (name, op_idx, L_slice, R_slice, L_range, R_range)
POSITION_CONFIGS = [
    # pos 2 (standard) — try first since it's most common
    ("pos2", 2, slice(0, 2), slice(3, 5), 100, 100),
    # pos 0 (op first) — 80 failing puzzles
    ("pos0", 0, slice(1, 3), slice(3, 5), 100, 100),
    # pos 4 (op last) — 10 failing puzzles
    ("pos4", 4, slice(0, 2), slice(2, 4), 100, 100),
    # pos 1 (1-digit L, 3-digit R) — 59 failing puzzles
    ("pos1", 1, slice(0, 1), slice(2, 5), 10, 1000),
    # pos 3 (3-digit L, 1-digit R) — 14 failing puzzles
    ("pos3", 3, slice(0, 3), slice(4, 5), 1000, 10),
]


# ============================================================
# Parsers
# ============================================================
LINE_RX  = re.compile(r"^(\S{5})\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"determine the result for:\s*(\S{5})\s*$", re.IGNORECASE)


def parse_puzzle_raw(prompt):
    """Just extract the raw 5-char LHS and result strings without assuming
    operator position. Returns dict with raw_examples and raw_query."""
    if "secret set of transformation rules" not in prompt:
        return None
    raw_examples = []
    raw_query = None
    for line in prompt.split("\n"):
        line = line.rstrip()
        if not line: continue
        m = LINE_RX.match(line)
        if m and len(m.group(1)) == 5:
            raw_examples.append({"lhs": m.group(1), "C_str": m.group(2)})
            continue
        m = QUERY_RX.search(line)
        if m and len(m.group(1)) == 5:
            raw_query = m.group(1)
    if not raw_examples or raw_query is None:
        return None
    # Reject plain-digit puzzles
    if any(c.isdigit() for c in raw_examples[0]["lhs"]):
        return None
    return {"raw_examples": raw_examples, "raw_query": raw_query}


def apply_position(raw, cfg):
    """Reparse raw puzzle using given position config. Returns examples + query."""
    name, op_idx, L_sl, R_sl, L_range, R_range = cfg
    examples = []
    for e in raw["raw_examples"]:
        lhs = e["lhs"]
        examples.append({
            "L_str": lhs[L_sl],
            "op":    lhs[op_idx],
            "R_str": lhs[R_sl],
            "C_str": e["C_str"],
        })
    query = {
        "L_str": raw["raw_query"][L_sl],
        "op":    raw["raw_query"][op_idx],
        "R_str": raw["raw_query"][R_sl],
    }
    return {"examples": examples, "query": query, "cfg": cfg}


# ============================================================
# Generalized helpers (variable-width operands)
# ============================================================
def merge_digits(digit_chars, mapping, mode):
    """Merge a string of digit chars into an integer using mapping + mode."""
    try:
        digits = [mapping[c] for c in digit_chars]
    except KeyError:
        return None
    if mode == "little_endian":
        digits = list(reversed(digits))
    n = 0
    for d in digits:
        n = n * 10 + d
    return n


def split_digits_for_str(val, str_len, mode):
    """Split integer into digits for a string of given length, with mode."""
    s = str(val).zfill(str_len)
    if len(s) != str_len: return None
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return digits


def split_value(val, width, mode):
    """Split val into per-char digits for given operand width (1, 2, 3)."""
    if width == 1: return [val]
    if width == 2:
        if mode == "standard": return [val // 10, val % 10]
        return [val % 10, val // 10]
    if width == 3:
        if mode == "standard":
            return [val // 100, (val // 10) % 10, val % 10]
        # little_endian
        return [val % 10, (val // 10) % 10, val // 100]
    raise ValueError(f"Unsupported width {width}")


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
# Core solver (generalized for any operand width)
# ============================================================
def _try_remaining(mapping, op_map, remaining, query, true_answer, mode):
    if not remaining:
        return _check_query(mapping, op_map, query, true_answer, mode)
    e = remaining[0]; rest = remaining[1:]

    # Try to decode L, R from current mapping
    if any(c not in mapping for c in e["L_str"] + e["R_str"]):
        return _try_with_unknown(mapping, op_map, e, rest, query, true_answer, mode)

    L = merge_digits(e["L_str"], mapping, mode)
    R = merge_digits(e["R_str"], mapping, mode)
    if L is None or R is None: return None

    if e["op"] in op_map:
        op_name = op_map[e["op"]]
        try: C_val = OPS[op_name](L, R)
        except Exception: return None
        if C_val is None or C_val < 0: return None
        if len(str(C_val)) != len(e["C_str"]): return None
        C_digits = split_digits_for_str(C_val, len(e["C_str"]), mode)
        if C_digits is None: return None
        new_map = add_str_to_map(mapping, e["C_str"], C_digits)
        if new_map is None: return None
        return _try_remaining(new_map, op_map, rest, query, true_answer, mode)

    for op_name in OP_ORDER:
        try: C_val = OPS[op_name](L, R)
        except Exception: continue
        if C_val is None or C_val < 0: continue
        if len(str(C_val)) != len(e["C_str"]): continue
        C_digits = split_digits_for_str(C_val, len(e["C_str"]), mode)
        if C_digits is None: continue
        new_map = add_str_to_map(mapping, e["C_str"], C_digits)
        if new_map is None: continue
        new_op_map = {**op_map, e["op"]: op_name}
        sol = _try_remaining(new_map, new_op_map, rest, query, true_answer, mode)
        if sol is not None: return sol
    return None


def _try_with_unknown(mapping, op_map, e, rest, query, true_answer, mode):
    L_width = len(e["L_str"]); R_width = len(e["R_str"])
    L_range = 10 ** L_width
    R_range = 10 ** R_width

    for L_val in range(L_range):
        Ld = split_value(L_val, L_width, mode)
        mL = mapping
        ok = True
        for c, d in zip(e["L_str"], Ld):
            mL = add_to_map(mL, c, d)
            if mL is None: ok = False; break
        if not ok: continue
        for R_val in range(R_range):
            Rd = split_value(R_val, R_width, mode)
            mR = mL
            ok = True
            for c, d in zip(e["R_str"], Rd):
                mR = add_to_map(mR, c, d)
                if mR is None: ok = False; break
            if not ok: continue

            ops_to_try = [op_map[e["op"]]] if e["op"] in op_map else OP_ORDER
            for op_name in ops_to_try:
                try: C_val = OPS[op_name](L_val, R_val)
                except Exception: continue
                if C_val is None or C_val < 0: continue
                if len(str(C_val)) != len(e["C_str"]): continue
                C_digits = split_digits_for_str(C_val, len(e["C_str"]), mode)
                if C_digits is None: continue
                mC = add_str_to_map(mR, e["C_str"], C_digits)
                if mC is None: continue
                new_op_map = {**op_map, e["op"]: op_name} if e["op"] not in op_map else op_map
                sol = _try_remaining(mC, new_op_map, rest, query, true_answer, mode)
                if sol is not None: return sol
    return None


def _check_query(mapping, op_map, query, true_answer, mode):
    if query["op"] not in op_map:
        for op_name in OP_ORDER:
            test_op_map = {**op_map, query["op"]: op_name}
            res = _check_query(mapping, test_op_map, query, true_answer, mode)
            if res is not None: return res
        return None

    if any(c not in mapping for c in query["L_str"] + query["R_str"]):
        return _check_query_with_unknown(mapping, op_map, query, true_answer, mode)

    qL = merge_digits(query["L_str"], mapping, mode)
    qR = merge_digits(query["R_str"], mapping, mode)
    if qL is None or qR is None: return None
    op_name = op_map[query["op"]]
    try: qC_val = OPS[op_name](qL, qR)
    except Exception: return None
    if qC_val is None or qC_val < 0: return None
    if len(str(qC_val)) != len(true_answer): return None
    qC_digits = split_digits_for_str(qC_val, len(true_answer), mode)
    if qC_digits is None: return None
    final_map = add_str_to_map(mapping, true_answer, qC_digits)
    if final_map is None: return None
    return {"mapping": final_map, "op_map": op_map, "query_numeric": qC_val}


def _check_query_with_unknown(mapping, op_map, query, true_answer, mode):
    L_width = len(query["L_str"]); R_width = len(query["R_str"])
    for L_val in range(10 ** L_width):
        Ld = split_value(L_val, L_width, mode)
        mL = mapping
        ok = True
        for c, d in zip(query["L_str"], Ld):
            mL = add_to_map(mL, c, d)
            if mL is None: ok = False; break
        if not ok: continue
        for R_val in range(10 ** R_width):
            Rd = split_value(R_val, R_width, mode)
            mR = mL
            ok = True
            for c, d in zip(query["R_str"], Rd):
                mR = add_to_map(mR, c, d)
                if mR is None: ok = False; break
            if not ok: continue
            res = _check_query(mR, op_map, query, true_answer, mode)
            if res is not None: return res
    return None


def solve_with_position(puzzle, true_answer, time_budget_sec):
    examples = puzzle["examples"]; query = puzzle["query"]
    t_start = time.time()
    e1 = examples[0]; rest = examples[1:]
    L_width = len(e1["L_str"]); R_width = len(e1["R_str"])

    for mode in ["standard", "little_endian"]:
        for L_val in range(10 ** L_width):
            if time.time() - t_start > time_budget_sec: return None
            Ld = split_value(L_val, L_width, mode)
            mL = {}; ok = True
            for c, d in zip(e1["L_str"], Ld):
                mL = add_to_map(mL, c, d)
                if mL is None: ok = False; break
            if not ok: continue
            for R_val in range(10 ** R_width):
                Rd = split_value(R_val, R_width, mode)
                mR = mL; ok = True
                for c, d in zip(e1["R_str"], Rd):
                    mR = add_to_map(mR, c, d)
                    if mR is None: ok = False; break
                if not ok: continue
                for op_name in OP_ORDER:
                    try: C_val = OPS[op_name](L_val, R_val)
                    except Exception: continue
                    if C_val is None or C_val < 0: continue
                    if len(str(C_val)) != len(e1["C_str"]): continue
                    C_digits = split_digits_for_str(C_val, len(e1["C_str"]), mode)
                    if C_digits is None: continue
                    mC = add_str_to_map(mR, e1["C_str"], C_digits)
                    if mC is None: continue
                    initial_op_map = {e1["op"]: op_name}
                    sol = _try_remaining(mC, initial_op_map, rest, query, true_answer, mode)
                    if sol is not None:
                        return {"mapping": sol["mapping"],
                                "op_map":  sol["op_map"],
                                "mode":    mode,
                                "query_numeric": sol["query_numeric"]}
    return None


def solve_all_positions(raw, true_answer, time_budget_per_pos=10.0):
    """Try each position config. Return first solution + its config."""
    for cfg in POSITION_CONFIGS:
        puzzle = apply_position(raw, cfg)
        sol = solve_with_position(puzzle, true_answer, time_budget_per_pos)
        if sol is not None:
            return sol, cfg, puzzle
    return None, None, None


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../../data/raw/train.csv")
    ap.add_argument("--out", default="sequence_solved_multipos.jsonl")
    ap.add_argument("--skip-already", default="sequence_solved.jsonl")
    ap.add_argument("--time-per-pos", type=float, default=10.0,
                    help="Time budget per position config in seconds")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    skip = set()
    if Path(args.skip_already).exists():
        with open(args.skip_already) as f:
            for line in f:
                try: skip.add(json.loads(line)["id"])
                except: pass
        print(f"Skipping {len(skip)} already-solved puzzles\n")

    n_total = 0; n_solved = 0; n_failed = 0
    by_position = {}
    t_start = time.time()

    with open(args.csv) as f, open(args.out, "w") as out:
        for row in csv.DictReader(f):
            raw = parse_puzzle_raw(row["prompt"])
            if raw is None: continue
            if row["id"] in skip: continue
            n_total += 1
            if args.limit and n_total > args.limit: break

            t0 = time.time()
            sol, cfg, puzzle = solve_all_positions(raw, row["answer"],
                                                   time_budget_per_pos=args.time_per_pos)
            t = time.time() - t0
            if sol:
                n_solved += 1
                by_position[cfg[0]] = by_position.get(cfg[0], 0) + 1
                rec = {
                    "id":            row["id"],
                    "prompt":        row["prompt"],
                    "answer":        row["answer"],
                    "examples":      puzzle["examples"],
                    "query":         puzzle["query"],
                    "mapping":       sol["mapping"],
                    "op_map":        sol["op_map"],
                    "mode":          sol["mode"],
                    "op_position":   cfg[1],
                    "position_name": cfg[0],
                    "query_numeric": sol["query_numeric"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                status = f"SOLVED ({cfg[0]}, {sol['mode'][:3]})"
            else:
                n_failed += 1
                status = "FAILED"

            if n_total % 10 == 0 or n_total <= 10:
                el = (time.time() - t_start) / 60
                print(f"  [{n_total:4d}] {row['id']}  {t:5.1f}s  {status:<20}  "
                      f"solved={n_solved}/{n_total}  total={el:.1f}m")

    el = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  Multi-position solver results")
    print(f"{'='*60}")
    print(f"  Puzzles attempted    : {n_total}")
    print(f"  Solved               : {n_solved}  ({100*n_solved/max(n_total,1):.1f}%)")
    print(f"  Failed               : {n_failed}")
    print(f"  Wall time            : {el:.1f} min")
    print(f"  By position config   :")
    for pos in ["pos2", "pos0", "pos4", "pos1", "pos3"]:
        if pos in by_position:
            print(f"    {pos}: {by_position[pos]} puzzles")
    print(f"  Output → {args.out}")


if __name__ == "__main__":
    main()
