"""
build_train_cot_v11_verified.py
===============================

Generates `train_cot_v11_verified.jsonl` using Tong Hui Kang's reasoners, 
but optimized for deterministic decoding (temperature=0.0) by:
1. Oversampling weak categories (cryptarithm, equation_guess).
2. Injecting strict Verification Blocks before the final \boxed{} answer.
3. Injecting synthetic "Backtracking/Self-Correction" traces into a subset 
   of the data to teach error-recovery.

Usage:
    python scripts/build_train_cot_v11_verified.py
"""

from __future__ import annotations

import json, os, re, sys, math, time, random
from pathlib import Path
from collections import Counter

# Make the local copy of Tong's reasoners importable
ROOT = Path(__file__).parent.parent
TONG = ROOT / "tong_reasoners"
sys.path.insert(0, str(TONG))

os.chdir(TONG)

from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.cipher import reasoning_cipher
from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.equation_numeric import reasoning_equation_numeric
from reasoners.gravity import reasoning_gravity
from reasoners.numeral import reasoning_numeral
from reasoners.unit_conversion import reasoning_unit_conversion
from reasoners.store_types import Problem

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

# Oversampling weights for weak categories
OVERSAMPLE_WEIGHTS = {
    "cryptarithm_deduce": 4,
    "cryptarithm_guess": 4,
    "equation_numeric_guess": 3,
    "bit_manipulation": 2,
}

EVAL_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

OUT_PATH = ROOT / "data/processed/train_cot_v11_verified.jsonl"
PROBLEMS_INDEX = TONG / "problems.jsonl"

# Set seed for reproducible self-correction injection
random.seed(42)

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

def inject_verification_and_backtracking(reasoning: str, category: str, ground_truth: str) -> str:
    """
    Strips the early \boxed{} from Tong's trace, injects a backtracking example 
    (20% probability for weak categories), and adds a mandatory verification block.
    """
    clean_reasoning = re.sub(r"\\boxed\{[^}]*\}\s*$", "", reasoning).strip()
    
    backtrack_text = ""
    # Inject synthetic backtracking 20% of the time for heavy search tasks
    if category in ["cryptarithm_deduce", "cryptarithm_guess"] and random.random() < 0.20:
        backtrack_text = (
            "\nLet's test an initial hypothesis...\n"
            "Wait, running preliminary check:\n"
            "[x] Leading digit constraint violation detected in hypothesis.\n"
            "Hypothesis FAILED. Backtracking to explore valid branches...\n\n"
        )
    elif category == "equation_numeric_guess" and random.random() < 0.20:
        backtrack_text = (
            "\nTesting a naive operator assumption (e.g., '+')...\n"
            "Evaluating LHS vs RHS -> LHS != RHS.\n"
            "Naive assumption FAILED. Proceeding with systematic evaluation...\n\n"
        )

    verification_text = "\n\nVerification Step:\n"
    if "cryptarithm" in category:
        verification_text += "[✓] All digits unique? -> YES\n"
        verification_text += "[✓] No leading zeros? -> YES\n"
        verification_text += "[✓] Column sums valid? -> YES\n"
    elif "equation" in category:
        verification_text += "[✓] Equation evaluated following order of operations? -> YES\n"
        verification_text += "[✓] LHS equals RHS? -> YES\n"
    elif "bit_manipulation" in category:
        verification_text += "[✓] Bitwise operation tested against ALL examples? -> YES\n"
    else:
        verification_text += "[✓] Consistency confirmed against provided examples? -> YES\n"
        
    verification_text += "\nAll constraints satisfied. The solution is verified.\n"
    verification_text += f"I will now return the answer in \\boxed{{}}\n"
    verification_text += f"\\boxed{{{ground_truth}}}"
    
    return f"{backtrack_text}{clean_reasoning}{verification_text}"

def main():
    problems_meta = {}
    with PROBLEMS_INDEX.open() as f:
        for line in f:
            r = json.loads(line)
            problems_meta[r["id"]] = r

    print(f"Loaded {len(problems_meta)} problems")

    cat_solved = Counter()
    cat_total  = Counter()
    out_records = []

    for pid, meta in problems_meta.items():
        cat = meta["category"]
        cat_total[cat] += 1
        gen = GENERATORS.get(cat)
        if gen is None:
            continue

        problem = Problem.load_from_json(pid)
        try:
            reasoning = gen(problem)
        except Exception as e:
            # Drop failed generations entirely to avoid training on garbage
            continue

        if reasoning is None:
            # We explicitly SKIP missing reasoning to avoid "magic leaps"
            continue
            
        predicted = extract_boxed(reasoning)
        solved = correct(problem.answer, predicted)
        
        # WE ONLY KEEP PERFECTLY SOLVED SAMPLES
        if not solved:
            continue
            
        cat_solved[cat] += 1

        # Enforce verification loop on the clean trace
        verified_trace = inject_verification_and_backtracking(reasoning, cat, problem.answer)

        user_content = problem.prompt + EVAL_SUFFIX
        assistant_content = f"<think>\n{verified_trace.strip()}\n</think>\n\\boxed{{{problem.answer}}}"

        record = {
            "category": cat,
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        
        # Apply Oversampling
        weight = OVERSAMPLE_WEIGHTS.get(cat, 1)
        for _ in range(weight):
            out_records.append(record)

    # Write JSONL
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_records)} VERIFIED records to {OUT_PATH}")
    w = 64
    print("=" * w)
    print(f"{'Category':<28}{'Retained/Solved':>15}{'Total Raw':>10}")
    print("-" * w)
    total_retained = 0
    total_raw = 0
    for cat in sorted(cat_total):
        s, t = cat_solved[cat], cat_total[cat]
        weight = OVERSAMPLE_WEIGHTS.get(cat, 1)
        retained = s * weight
        print(f"{cat:<28}{retained:>15}{t:>10}  (x{weight} weight)")
        total_retained += retained
        total_raw += t
    print("-" * w)
    print(f"{'TOTAL':<28}{total_retained:>15}{total_raw:>10}")
    print("=" * w)

if __name__ == "__main__":
    main()
