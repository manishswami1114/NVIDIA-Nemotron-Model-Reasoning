#!/usr/bin/env python3
"""
solve_cryptarithm.py — Fresh solver for cryptarithm_deduce + cryptarithm_guess.

Key fixes over v1:
  - Handles NEGATIVE results (leading '-' is a literal sign, not a data symbol)
  - Assigns digits to operator chars when they appear in answers
  - Two-pass GT strategy: first prefer GT match, then accept any valid
  - Multiprocessing for speed

Algorithm:
  For each mode (standard / little_endian):
    Group examples by operator symbol.
    For each candidate operation on most-constrained group:
      Extend partial mapping through examples (CSP).
      For surviving mappings, extend through remaining op groups.
      Apply to query → encode answer.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ============================================================
# Operations — 22 candidates
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
}


# ============================================================
# Puzzle parser
# ============================================================
LINE_RX = re.compile(r"^(\S{5})\s*=\s*(\S+)\s*$")
QUERY_RX = re.compile(r"determine the result for:\s*(\S{5})", re.IGNORECASE)
BOXED_RX = re.compile(r"\\boxed\{([^}]+)\}")


def parse_puzzle(prompt: str, answer_text: str = ""):
    examples, query, gt = [], None, None
    for line in prompt.strip().split("\n"):
        line = line.strip()
        m = LINE_RX.match(line)
        if m:
            lhs, rhs = m.group(1), m.group(2)
            examples.append({
                "L_str": lhs[:2], "op_char": lhs[2],
                "R_str": lhs[3:5], "C_str": rhs,
            })
        m2 = QUERY_RX.search(line)
        if m2:
            q = m2.group(1)
            query = {"L_str": q[:2], "op_char": q[2], "R_str": q[3:5]}
    if answer_text:
        boxed = BOXED_RX.findall(answer_text)
        if boxed:
            gt = boxed[-1]
    return examples, query, gt


# ============================================================
# Core helpers
# ============================================================
def decode_2d(s: str, mapping: dict, mode: str) -> int:
    d0, d1 = mapping[s[0]], mapping[s[1]]
    return d0 * 10 + d1 if mode == "standard" else d1 * 10 + d0


def encode_num(n: int, inv_map: dict, mode: str, target_len: int | None = None):
    """Encode an integer into symbols using inv_map.

    Handles negative numbers: prepends literal '-' character.
    If target_len given, pads with leading zeros to match.
    Returns None if encoding fails.
    """
    neg = n < 0
    abs_n = abs(n)
    s = str(abs_n)

    if target_len is not None:
        if len(s) > target_len:
            return None
        s = s.zfill(target_len)

    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))

    out = []
    for d in digits:
        if d not in inv_map:
            return None
        out.append(inv_map[d])

    result = "".join(out)
    if neg:
        result = "-" + result
    return result


def _digit_range(sym: str, base_map: dict, used_digits: set) -> list[int]:
    if sym in base_map:
        return [base_map[sym]]
    return [d for d in range(10) if d not in used_digits]


def _parse_c_str(c_str: str):
    """Parse result string, handling negative sign prefix.
    Returns (is_negative, data_symbols).
    """
    if len(c_str) >= 2 and c_str[0] == '-':
        return True, c_str[1:]
    return False, c_str


# ============================================================
# Core extension: extend mapping with one example
# ============================================================
def extend_mapping_with_example(base_map: dict, ex: dict, op_fn, mode: str,
                                max_candidates: int = 300):
    L_str, R_str, C_str = ex["L_str"], ex["R_str"], ex["C_str"]
    c_neg, c_data = _parse_c_str(C_str)
    c_data_len = len(c_data)
    results = []
    base_used = set(base_map.values())

    if mode == "standard":
        L_tens_sym, L_units_sym = L_str[0], L_str[1]
        R_tens_sym, R_units_sym = R_str[0], R_str[1]
    else:
        L_tens_sym, L_units_sym = L_str[1], L_str[0]
        R_tens_sym, R_units_sym = R_str[1], R_str[0]

    L_tens_range = _digit_range(L_tens_sym, base_map, base_used)
    L_units_range = _digit_range(L_units_sym, base_map, base_used)

    for Lt in L_tens_range:
        for Lu in L_units_range:
            if L_tens_sym == L_units_sym:
                if Lt != Lu: continue
            elif L_tens_sym not in base_map and L_units_sym not in base_map:
                if Lt == Lu: continue
            elif L_tens_sym not in base_map:
                if Lt in base_used or Lt == Lu: continue
            elif L_units_sym not in base_map:
                if Lu in base_used or Lu == Lt: continue

            L_val = Lt * 10 + Lu

            pm = dict(base_map)
            pm_used = set(base_used)
            l_ok = True
            for sym, dig in [(L_tens_sym, Lt), (L_units_sym, Lu)]:
                if sym in pm:
                    if pm[sym] != dig: l_ok = False; break
                else:
                    if dig in pm_used: l_ok = False; break
                    pm[sym] = dig; pm_used.add(dig)
            if not l_ok: continue

            R_tens_range2 = _digit_range(R_tens_sym, pm, pm_used)
            R_units_range2 = _digit_range(R_units_sym, pm, pm_used)

            for Rt in R_tens_range2:
                for Ru in R_units_range2:
                    if R_tens_sym == R_units_sym:
                        if Rt != Ru: continue
                    elif R_tens_sym not in pm and R_units_sym not in pm:
                        if Rt == Ru: continue
                    elif R_tens_sym not in pm:
                        if Rt in pm_used or Rt == Ru: continue
                    elif R_units_sym not in pm:
                        if Ru in pm_used or Ru == Rt: continue

                    R_val = Rt * 10 + Ru

                    merged = dict(pm)
                    m_used = set(pm_used)
                    r_ok = True
                    for sym, dig in [(R_tens_sym, Rt), (R_units_sym, Ru)]:
                        if sym in merged:
                            if merged[sym] != dig: r_ok = False; break
                        else:
                            if dig in m_used: r_ok = False; break
                            merged[sym] = dig; m_used.add(dig)
                    if not r_ok: continue

                    try:
                        C_val = op_fn(L_val, R_val)
                    except Exception:
                        continue
                    if C_val is None:
                        continue

                    # Handle sign
                    if c_neg and C_val >= 0:
                        continue
                    if not c_neg and C_val < 0:
                        continue

                    abs_C = abs(C_val)
                    cs = str(abs_C)
                    if len(cs) > c_data_len:
                        continue
                    cs = cs.zfill(c_data_len)
                    if len(cs) != c_data_len:
                        continue

                    c_digits = [int(c) for c in cs]
                    if mode == "little_endian":
                        c_digits = list(reversed(c_digits))

                    full = dict(merged)
                    f_used = set(m_used)
                    c_ok = True
                    for i, sym in enumerate(c_data):
                        d = c_digits[i]
                        if sym in full:
                            if full[sym] != d: c_ok = False; break
                        else:
                            if d in f_used: c_ok = False; break
                            full[sym] = d; f_used.add(d)
                    if not c_ok:
                        continue

                    results.append(full)
                    if len(results) >= max_candidates:
                        return results

    return results


def deduplicate_mappings(mappings: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for m in mappings:
        key = tuple(sorted(m.items()))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


# ============================================================
# Main solver
# ============================================================
def solve_puzzle(examples, query, gt_answer=None, timeout=40.0):
    t0 = time.time()
    op_chars = sorted(set(e["op_char"] for e in examples))
    query_op = query["op_char"]
    groups = {oc: [e for e in examples if e["op_char"] == oc] for oc in op_chars}
    sorted_ops = sorted(op_chars, key=lambda oc: -len(groups[oc]))

    # Collect unmapped operator chars (potential data symbols for answer encoding)
    all_op_syms = set(op_chars) | {query_op}

    for mode in ["standard", "little_endian"]:
        first_oc = sorted_ops[0]
        first_group = groups[first_oc]

        for op_name in OP_ORDER:
            if time.time() - t0 > timeout:
                return None
            op_fn = OPS[op_name]

            cands = extend_mapping_with_example({}, first_group[0], op_fn, mode)
            if not cands:
                continue
            cands = deduplicate_mappings(cands)

            for ex in first_group[1:]:
                if time.time() - t0 > timeout:
                    break
                new_cands = []
                for m in cands:
                    ext = extend_mapping_with_example(m, ex, op_fn, mode)
                    new_cands.extend(ext)
                cands = deduplicate_mappings(new_cands)
                if not cands:
                    break
            if not cands:
                continue

            remaining = [oc for oc in sorted_ops if oc != first_oc]

            for mapping in cands:
                if time.time() - t0 > timeout:
                    return None

                solution_ops = {first_oc: op_name}
                ok = True
                current_maps = [mapping]

                for oc in remaining:
                    grp = groups[oc]
                    found_op = None
                    found_maps = []

                    for op2_name in OP_ORDER:
                        if time.time() - t0 > timeout:
                            return None
                        op2_fn = OPS[op2_name]
                        op2_maps = list(current_maps)

                        for ex in grp:
                            new_maps = []
                            for m in op2_maps:
                                ext = extend_mapping_with_example(m, ex, op2_fn, mode)
                                new_maps.extend(ext)
                            op2_maps = deduplicate_mappings(new_maps)
                            if not op2_maps:
                                break

                        if op2_maps:
                            found_op = op2_name
                            found_maps = op2_maps
                            break

                    if found_op:
                        solution_ops[oc] = found_op
                        current_maps = found_maps
                    else:
                        ok = False
                        break

                if not ok:
                    continue

                # Determine which operations to try for the query
                if query_op in solution_ops:
                    q_op_candidates = [solution_ops[query_op]]
                else:
                    q_op_candidates = list(OP_ORDER)

                for final_map in current_maps:
                    q_syms_ok = all(
                        s in final_map for s in query["L_str"] + query["R_str"]
                    )
                    if not q_syms_ok:
                        continue

                    qL = decode_2d(query["L_str"], final_map, mode)
                    qR = decode_2d(query["R_str"], final_map, mode)

                    inv_map = {v: k for k, v in final_map.items()}

                    # Extend inv_map with unmapped operator chars
                    ext_map = dict(final_map)
                    ext_inv = dict(inv_map)
                    used_digits = set(ext_map.values())
                    for oc_sym in all_op_syms:
                        if oc_sym not in ext_map:
                            for d in range(10):
                                if d not in used_digits:
                                    ext_inv[d] = oc_sym
                                    ext_map[oc_sym] = d
                                    used_digits.add(d)
                                    break

                    for q_op_name in q_op_candidates:
                        if time.time() - t0 > timeout:
                            return None
                        q_fn = OPS[q_op_name]
                        try:
                            qC = q_fn(qL, qR)
                        except Exception:
                            continue
                        if qC is None:
                            continue

                        for ans_len in range(1, 6):
                            encoded = encode_num(qC, ext_inv, mode, target_len=ans_len)
                            if encoded is not None:
                                if gt_answer and encoded != gt_answer:
                                    continue
                                final_ops = dict(solution_ops)
                                if query_op not in final_ops:
                                    final_ops[query_op] = q_op_name
                                return {
                                    "answer": encoded,
                                    "mapping": dict(ext_map),
                                    "op_map": final_ops,
                                    "mode": mode,
                                    "query_numeric": qC,
                                }

    return None


# ============================================================
# Brute-force solver (fallback for hard puzzles)
# ============================================================
def solve_puzzle_bruteforce(examples, query, gt_answer=None, timeout=50.0):
    """Brute-force all digit permutations. Slower but more complete."""
    import itertools
    t0 = time.time()

    op_chars = set(e["op_char"] for e in examples)
    query_op = query["op_char"]
    all_op_syms = op_chars | {query_op}

    # Collect data symbols (exclude op chars used only as separators)
    data_syms = set()
    for e in examples:
        data_syms.update(e["L_str"])
        data_syms.update(e["R_str"])
        c_str = e["C_str"]
        c_neg, c_data = _parse_c_str(c_str)
        data_syms.update(c_data)
    data_syms.update(query["L_str"])
    data_syms.update(query["R_str"])
    if gt_answer:
        gt_neg, gt_data = _parse_c_str(gt_answer)
        data_syms.update(gt_data)

    sym_list = sorted(data_syms)
    k = len(sym_list)
    if k > 10:
        return None

    groups = {}
    for e in examples:
        groups.setdefault(e["op_char"], []).append(e)

    for mode in ["standard", "little_endian"]:
        for perm in itertools.permutations(range(10), k):
            if time.time() - t0 > timeout:
                return None

            mapping = dict(zip(sym_list, perm))

            all_ok = True
            op_map = {}

            for oc, grp in groups.items():
                found_op = None
                for e in grp:
                    c_neg, c_data = _parse_c_str(e["C_str"])
                    if mode == "standard":
                        L = mapping[e["L_str"][0]] * 10 + mapping[e["L_str"][1]]
                        R = mapping[e["R_str"][0]] * 10 + mapping[e["R_str"][1]]
                        c_digits = [mapping[s] for s in c_data]
                    else:
                        L = mapping[e["L_str"][1]] * 10 + mapping[e["L_str"][0]]
                        R = mapping[e["R_str"][1]] * 10 + mapping[e["R_str"][0]]
                        c_digits = [mapping[s] for s in reversed(c_data)]

                    C = int("".join(str(d) for d in c_digits))
                    if c_neg:
                        C = -C

                    if found_op is None:
                        matched = None
                        for opname in OP_ORDER:
                            try:
                                if OPS[opname](L, R) == C:
                                    matched = opname
                                    break
                            except Exception:
                                pass
                        if not matched:
                            all_ok = False
                            break
                        found_op = matched
                    else:
                        try:
                            if OPS[found_op](L, R) != C:
                                all_ok = False
                                break
                        except Exception:
                            all_ok = False
                            break

                if not all_ok:
                    break
                op_map[oc] = found_op

            if not all_ok:
                continue

            # All examples match — try query
            if mode == "standard":
                qL = mapping[query["L_str"][0]] * 10 + mapping[query["L_str"][1]]
                qR = mapping[query["R_str"][0]] * 10 + mapping[query["R_str"][1]]
            else:
                qL = mapping[query["L_str"][1]] * 10 + mapping[query["L_str"][0]]
                qR = mapping[query["R_str"][1]] * 10 + mapping[query["R_str"][0]]

            inv_map = {v: kk for kk, v in mapping.items()}
            # Add op chars to inv_map if not already present
            used = set(mapping.values())
            for oc_sym in all_op_syms:
                if oc_sym not in mapping:
                    for d in range(10):
                        if d not in used:
                            inv_map[d] = oc_sym
                            mapping[oc_sym] = d
                            used.add(d)
                            break

            q_op_candidates = [op_map[query_op]] if query_op in op_map else list(OP_ORDER)

            for q_op_name in q_op_candidates:
                try:
                    qC = OPS[q_op_name](qL, qR)
                except Exception:
                    continue
                if qC is None:
                    continue

                for ans_len in range(1, 6):
                    encoded = encode_num(qC, inv_map, mode, target_len=ans_len)
                    if encoded is not None:
                        if gt_answer and encoded != gt_answer:
                            continue
                        final_ops = dict(op_map)
                        if query_op not in final_ops:
                            final_ops[query_op] = q_op_name
                        return {
                            "answer": encoded,
                            "mapping": dict(mapping),
                            "op_map": final_ops,
                            "mode": mode,
                            "query_numeric": qC,
                        }

    return None


# ============================================================
# Worker for multiprocessing
# ============================================================
def _solve_one(args):
    puzzle, timeout_gt, timeout_any = args
    examples = puzzle["examples"]
    query = puzzle["query"]
    gt = puzzle["gt_answer"]

    # Pass 1: CSP solver with GT guidance
    if gt:
        result = solve_puzzle(examples, query, gt_answer=gt, timeout=timeout_gt)
        if result:
            return puzzle["id"], result, True

    # Pass 2: CSP solver without GT
    result = solve_puzzle(examples, query, gt_answer=None, timeout=timeout_any)
    if result:
        return puzzle["id"], result, False

    # Pass 3: Brute-force fallback with GT
    if gt:
        result = solve_puzzle_bruteforce(examples, query, gt_answer=gt, timeout=50.0)
        if result:
            return puzzle["id"], result, True

    # Pass 4: Brute-force without GT
    result = solve_puzzle_bruteforce(examples, query, gt_answer=None, timeout=30.0)
    if result:
        return puzzle["id"], result, False

    return puzzle["id"], None, False


# ============================================================
# Main
# ============================================================
def main():
    THIS = Path(__file__).resolve().parent
    ROOT = THIS.parent.parent
    BASELINE = ROOT / "dont_touch_it" / "all_categorical_splits"
    OUT = THIS / "solved_cryptarithm_fresh.jsonl"

    # Load authoritative GT from train.csv (not JSONL \boxed{} which truncates answers containing })
    TRAIN_CSV = ROOT / "dont_touch_it" / "train.csv"
    csv_gt = {}
    with TRAIN_CSV.open() as f:
        for row in csv.DictReader(f):
            csv_gt[row["id"]] = row["answer"]
    print(f"Loaded {len(csv_gt)} GT answers from train.csv")

    puzzles = []
    puzzle_by_id = {}
    for fname in [
        BASELINE / "train_cot_cryptarithm_deduce.jsonl",
        BASELINE / "train_cot_cryptarithm_guess.jsonl",
    ]:
        kind = "deduce" if "deduce" in fname.name else "guess"
        with fname.open() as f:
            for line in f:
                rec = json.loads(line)
                examples, query, _unused_gt = parse_puzzle(
                    rec["messages"][0]["content"],
                    rec["messages"][1]["content"],
                )
                p = {
                    "id": rec["id"],
                    "kind": kind,
                    "prompt": rec["messages"][0]["content"],
                    "examples": examples,
                    "query": query,
                    "gt_answer": csv_gt.get(rec["id"]),
                }
                puzzles.append(p)
                puzzle_by_id[rec["id"]] = p

    print(f"Loaded {len(puzzles)} puzzles "
          f"({sum(1 for p in puzzles if p['kind']=='deduce')} deduce, "
          f"{sum(1 for p in puzzles if p['kind']=='guess')} guess)")

    # Solve with multiprocessing
    n_workers = min(cpu_count(), 8)
    print(f"Using {n_workers} workers")

    timeout_gt = 30.0
    timeout_any = 25.0

    tasks = [(p, timeout_gt, timeout_any) for p in puzzles]
    t0 = time.time()

    solved = 0
    failed = 0
    gt_match = 0
    gt_mismatch = 0
    failed_ids = []

    with OUT.open("w") as fout:
        with Pool(n_workers) as pool:
            for i, (pid, result, was_gt_pass) in enumerate(
                pool.imap_unordered(_solve_one, tasks, chunksize=4)
            ):
                p = puzzle_by_id[pid]

                if result:
                    match = result["answer"] == p["gt_answer"] if p["gt_answer"] else None
                    if match is True:
                        gt_match += 1
                    elif match is False:
                        gt_mismatch += 1

                    rec_out = {
                        "id": pid,
                        "kind": p["kind"],
                        "prompt": p["prompt"],
                        "answer": result["answer"],
                        "gt_answer": p["gt_answer"],
                        "match_gt": match,
                        "mapping": result["mapping"],
                        "op_map": result["op_map"],
                        "mode": result["mode"],
                        "query_numeric": result["query_numeric"],
                        "examples": p["examples"],
                        "query": p["query"],
                    }
                    fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                    fout.flush()
                    solved += 1
                else:
                    failed += 1
                    failed_ids.append(pid)

                done = solved + failed
                if done % 25 == 0 or done == len(puzzles):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  [{done}/{len(puzzles)}] solved={solved} failed={failed} "
                          f"gt_match={gt_match} gt_mismatch={gt_mismatch} "
                          f"rate={rate:.1f}/s elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Solved: {solved}/{len(puzzles)} ({100*solved/len(puzzles):.1f}%)")
    print(f"  GT match: {gt_match}")
    print(f"  GT mismatch: {gt_mismatch}")
    print(f"  Failed: {failed}")
    if failed_ids[:20]:
        print(f"  First 20 failed IDs: {failed_ids[:20]}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    main()
