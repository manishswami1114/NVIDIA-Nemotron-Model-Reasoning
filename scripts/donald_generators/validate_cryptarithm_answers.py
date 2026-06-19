"""
Validate encrypted-equation cryptarithm answers against the prompt.

This is answer-conditioned validation: it never trusts the assistant CoT.
It extracts the final boxed answer, then searches for a symbol->digit mapping
and operator mapping that satisfies:
  1. every provided example in the prompt
  2. the query result encoded exactly as the boxed answer

Rows that pass are mathematically consistent with the prompt and answer.
Rows that fail either need a broader operator family, a different parser, or
are not solvable under the sequence_v2 arithmetic model.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from solve_sequence_v2 import OPS, parse_puzzle, solve_two_phase  # noqa: E402


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed_answer(rec: dict) -> str | None:
    for msg in reversed(rec.get("messages", [])):
        if msg.get("role") != "assistant":
            continue
        boxes = BOX_RE.findall(msg.get("content", ""))
        if boxes:
            return boxes[-1]
    return None


def merge_two(d0: int, d1: int, mode: str) -> int:
    return 10 * d0 + d1 if mode == "standard" else 10 * d1 + d0


def split_n(val: int, length: int, mode: str) -> list[int] | None:
    s = str(val)
    if len(s) != length:
        return None
    digits = [int(c) for c in s]
    if mode == "little_endian":
        digits = list(reversed(digits))
    return digits


def verify_solution(puzzle: dict, answer: str, sol: dict) -> tuple[bool, str]:
    mapping = sol["mapping"]
    op_map = sol["op_map"]
    mode = sol["mode"]

    if len(set(mapping.values())) != len(mapping.values()):
        return False, "mapping is not injective"
    if any(d < 0 or d > 9 for d in mapping.values()):
        return False, "mapping contains non-digit values"

    for i, ex in enumerate(puzzle["examples"], 1):
        chars = ex["L_str"] + ex["R_str"] + ex["C_str"]
        if any(c not in mapping for c in chars):
            return False, f"example {i} has unmapped symbols"
        if ex["op"] not in op_map:
            return False, f"example {i} has unmapped operator"
        L = merge_two(mapping[ex["L_str"][0]], mapping[ex["L_str"][1]], mode)
        R = merge_two(mapping[ex["R_str"][0]], mapping[ex["R_str"][1]], mode)
        try:
            val = OPS[op_map[ex["op"]]](L, R)
        except Exception as exc:
            return False, f"example {i} operation error: {exc}"
        if val is None or val < 0:
            return False, f"example {i} produced invalid value {val}"
        digits = split_n(val, len(ex["C_str"]), mode)
        if digits is None:
            return False, f"example {i} output length mismatch"
        expected = [mapping[c] for c in ex["C_str"]]
        if digits != expected:
            return False, f"example {i} output mismatch"

    query = puzzle["query"]
    chars = query["L_str"] + query["R_str"] + answer
    if any(c not in mapping for c in chars):
        return False, "query or answer has unmapped symbols"
    if query["op"] not in op_map:
        return False, "query has unmapped operator"
    qL = merge_two(mapping[query["L_str"][0]], mapping[query["L_str"][1]], mode)
    qR = merge_two(mapping[query["R_str"][0]], mapping[query["R_str"][1]], mode)
    try:
        qval = OPS[op_map[query["op"]]](qL, qR)
    except Exception as exc:
        return False, f"query operation error: {exc}"
    if qval is None or qval < 0:
        return False, f"query produced invalid value {qval}"
    qdigits = split_n(qval, len(answer), mode)
    if qdigits is None:
        return False, "query answer length mismatch"
    expected = [mapping[c] for c in answer]
    if qdigits != expected:
        return False, "query answer mismatch"

    return True, "ok"


def validate_row(rec: dict, fast_budget: float, full_budget: float) -> dict:
    rec_id = rec.get("id")
    answer = extract_boxed_answer(rec)
    if answer is None:
        return {"id": rec_id, "status": "missing_answer"}
    prompt = rec["messages"][0]["content"]
    puzzle = parse_puzzle(prompt)
    if puzzle is None:
        return {"id": rec_id, "answer": answer, "status": "parse_failed"}

    t0 = time.time()
    sol, phase = solve_two_phase(
        puzzle,
        answer,
        fast_sec=fast_budget,
        full_sec=full_budget,
    )
    elapsed = time.time() - t0
    if sol is None:
        return {
            "id": rec_id,
            "answer": answer,
            "status": "no_solution",
            "elapsed_sec": round(elapsed, 3),
        }

    ok, reason = verify_solution(puzzle, answer, sol)
    if not ok:
        return {
            "id": rec_id,
            "answer": answer,
            "status": "invalid_solution",
            "reason": reason,
            "elapsed_sec": round(elapsed, 3),
            "solution": sol,
        }

    return {
        "id": rec_id,
        "answer": answer,
        "status": "validated",
        "phase": phase,
        "elapsed_sec": round(elapsed, 3),
        "mode": sol["mode"],
        "query_numeric": sol["query_numeric"],
        "mapping": sol["mapping"],
        "op_map": sol["op_map"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files",
        nargs="*",
        default=[
            "all_categorical_splits_sequence_v2_replaced/train_cot_cryptarithm_deduce.jsonl",
            "all_categorical_splits_sequence_v2_replaced/train_cot_cryptarithm_guess.jsonl",
        ],
    )
    ap.add_argument("--out", default="cryptarithm_full_validation_report.jsonl")
    ap.add_argument("--fast-budget", type=float, default=3.0)
    ap.add_argument("--full-budget", type=float, default=30.0)
    args = ap.parse_args()

    totals: dict[str, int] = {}
    n = 0
    with Path(args.out).open("w") as out:
        for file_name in args.files:
            path = Path(file_name)
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    n += 1
                    rec = json.loads(line)
                    result = validate_row(rec, args.fast_budget, args.full_budget)
                    result["file"] = path.name
                    totals[result["status"]] = totals.get(result["status"], 0) + 1
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    if n <= 10 or n % 25 == 0:
                        print(
                            f"[{n:04d}] {result.get('id')} "
                            f"{result['status']} "
                            f"{result.get('elapsed_sec', 0):.2f}s"
                        )

    print("\nSummary:")
    for status, count in sorted(totals.items()):
        print(f"  {status:18s} {count:5d}")
    print(f"  {'total':18s} {n:5d}")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
