"""
build_train_cot_v10_tong.py
============================

Generate `train_cot_v10_tong.jsonl` using Tong Hui Kang's open-sourced
per-category reasoners (https://github.com/tonghuikang/nemotron).

These solvers reach 87.7% solve rate on train.csv:
  numeral, gravity, unit_conversion, cipher : 100%
  bit_manipulation                           : ~85%
  equation_numeric_deduce                    : ~91%
  equation_numeric_guess                     : ~15%
  cryptarithm_*                              : <10%

Usage:
    python scripts/build_train_cot_v10_tong.py
"""

from __future__ import annotations

import json, os, re, sys, math, time
from pathlib import Path
from collections import Counter

# Make the local copy of Tong's reasoners importable
ROOT = Path(__file__).parent.parent
TONG = ROOT / "tong_reasoners"
sys.path.insert(0, str(TONG))

# Tong's reasoners expect to find problems/ relative to CWD
os.chdir(TONG)

from reasoners.bit_manipulation import reasoning_bit_manipulation  # noqa: E402
from reasoners.cipher import reasoning_cipher                      # noqa: E402
from reasoners.cryptarithm import reasoning_cryptarithm            # noqa: E402
from reasoners.equation_numeric import reasoning_equation_numeric  # noqa: E402
from reasoners.gravity import reasoning_gravity                    # noqa: E402
from reasoners.numeral import reasoning_numeral                    # noqa: E402
from reasoners.unit_conversion import reasoning_unit_conversion    # noqa: E402
from reasoners.store_types import Problem                          # noqa: E402

GENERATORS = {
    "numeral":                  reasoning_numeral,
    "gravity":                  reasoning_gravity,
    "unit_conversion":          reasoning_unit_conversion,
    "cipher":                   reasoning_cipher,
    "bit_manipulation":         reasoning_bit_manipulation,
    "equation_numeric_deduce":  reasoning_equation_numeric,
    "equation_numeric_guess":   reasoning_equation_numeric,
    "cryptarithm_deduce":       reasoning_cryptarithm,
    "cryptarithm_guess":        reasoning_cryptarithm,
}

EVAL_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

OUT_PATH = ROOT / "data/processed/train_cot_v10_tong.jsonl"
PROBLEMS_INDEX = TONG / "problems.jsonl"


def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def correct(stored: str, predicted: str) -> bool:
    s, p = stored.strip(), predicted.strip()
    if re.fullmatch(r"[01]+", s):
        return p.lower() == s.lower()
    try:
        return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return p.lower() == s.lower()


def main():
    # Load all problems
    problems_meta = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            r = json.loads(line)
            problems_meta[r["id"]] = r

    print(f"Loaded {len(problems_meta)} problems")

    cat_solved = Counter()
    cat_total  = Counter()
    cat_runtime_ms = {}

    out_records = []

    for pid, meta in problems_meta.items():
        cat = meta["category"]
        cat_total[cat] += 1
        gen = GENERATORS.get(cat)
        if gen is None:
            continue

        problem = Problem.load_from_json(pid)
        t0 = time.perf_counter()
        try:
            reasoning = gen(problem)
        except Exception as e:
            print(f"  [{cat}] {pid} solver crash: {e}")
            reasoning = None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        cat_runtime_ms.setdefault(cat, []).append(elapsed_ms)

        if reasoning is None:
            # solver returned None — emit a tiny fallback so the model still
            # sees the boxed answer formatting
            ans = problem.answer
            reasoning = (
                f"This is a {cat} problem.\n"
                f"Direct answer for this case.\n"
                f"I will now return the answer in \\boxed{{}}\n"
                f"The answer in \\boxed{{–}} is \\boxed{{{ans}}}"
            )
            solved = False
        else:
            predicted = extract_boxed(reasoning)
            solved = correct(problem.answer, predicted)
            if not solved:
                # Replace the trailing \\boxed{...} with the GROUND TRUTH so
                # we never train on a wrong answer
                gt = problem.answer
                # Replace last \boxed{...}
                reasoning = re.sub(
                    r"\\boxed\{[^}]*\}\s*$",
                    f"\\\\boxed{{{gt}}}",
                    reasoning,
                    count=1,
                )

        if solved:
            cat_solved[cat] += 1

        # Build the chat record matching our format
        user_content = problem.prompt + EVAL_SUFFIX
        # Tong's reasoning text doesn't have <think> wrappers — wrap it
        assistant_content = (
            f"<think>\n{reasoning.strip()}\n</think>\n"
            f"\\boxed{{{problem.answer}}}"
        )

        out_records.append({
            "category": cat,
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })

    # Write JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\nWrote {len(out_records)} records to {OUT_PATH}")
    w = 64
    print("=" * w)
    print(f"{'Category':<28}{'Solved':>8}{'Total':>8}{'Acc':>8}{'AvgMS':>10}")
    print("-" * w)
    total_solved = total = 0
    for cat in sorted(cat_total):
        s, t = cat_solved[cat], cat_total[cat]
        acc = (s / t * 100) if t else 0.0
        avg_ms = (sum(cat_runtime_ms.get(cat, [])) / max(1, len(cat_runtime_ms.get(cat, []))))
        print(f"{cat:<28}{s:>8}{t:>8}{acc:>7.1f}%{avg_ms:>10.1f}")
        total_solved += s
        total += t
    overall = (total_solved / total * 100) if total else 0
    print("-" * w)
    print(f"{'TOTAL':<28}{total_solved:>8}{total:>8}{overall:>7.1f}%")
    print("=" * w)


if __name__ == "__main__":
    main()
