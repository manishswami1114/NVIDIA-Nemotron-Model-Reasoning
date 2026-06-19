#!/usr/bin/env python3
"""Final V19 audit.

Checks:
  1. boxed answer == train.csv answer per-prompt for regenerated files
  2. boxed answer ∈ train.csv answers for baseline-copied files
  3. NO V17/V18 artifacts: "Sequential search:", "apply locked",
     "Family elimination", "STRUCTURAL OBSERVATIONS"
  4. EVERY record in the 4 unified categories contains baseline signature
     substrings "Trying common operations" and "correct, actions: "
"""
import json, re, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all

BASE = Path(__file__).resolve().parent.parent
V19 = BASE / "all_categorical_splits_v19"

UNIFIED = {
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}

REGENERATED = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
}

FORBIDDEN_ARTIFACTS = [
    "Sequential search:",
    "apply locked",
    "Family elimination",
    "STRUCTURAL OBSERVATIONS",
]
REQUIRED_BASELINE = [
    "Trying common operations",
    "correct, actions: ",
]


def extract_final_answer(text):
    if text is None: return 'NOT_FOUND'
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    return 'NOT_FOUND'


def verify(stored_answer, predicted):
    stored_answer, predicted = str(stored_answer).strip(), str(predicted).strip()
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        return math.isclose(float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


def main():
    rows = load_all(BASE / 'data/raw/train.csv')
    prompt_to_answer = {r['prompt']: r['answer'] for r in rows}
    all_answers = set(r['answer'] for r in rows)

    print(f"Loaded {len(rows)} answers from train.csv\n")

    print(f"{'FILE':<45} {'pass':>6}/{'total':>6}  mode")
    print('-' * 75)

    total_strict = 0; ok_strict = 0
    total_loose = 0; ok_loose = 0

    artifact_offenders = []
    baseline_missing = []

    for path in sorted(V19.glob('train_cot_*.jsonl')):
        fc = 0; fok = 0
        is_regen = path.name in REGENERATED
        is_unified = path.name in UNIFIED
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                fc += 1
                user = rec['messages'][0]['content']
                asst = rec['messages'][-1]['content']
                boxed = extract_final_answer(asst)
                if is_regen:
                    expected = prompt_to_answer.get(user)
                    if expected is not None and verify(expected, boxed):
                        fok += 1
                else:
                    if boxed in all_answers:
                        fok += 1
                # Artifact check
                if is_unified:
                    for art in FORBIDDEN_ARTIFACTS:
                        if art in asst:
                            artifact_offenders.append((path.name, art))
                            break
                    for sig in REQUIRED_BASELINE:
                        if sig not in asst:
                            baseline_missing.append((path.name, sig))
                            break
        if is_regen:
            total_strict += fc; ok_strict += fok; mode = 'strict'
        else:
            total_loose += fc; ok_loose += fok; mode = 'loose'
        flag = '' if fok == fc else '  ← MISMATCH'
        print(f"  {path.name:<43} {fok:>6}/{fc:>6}  {mode}{flag}")

    print('-' * 75)
    print(f"\nStrict (regenerated): {ok_strict}/{total_strict}  (mismatches: {total_strict - ok_strict})")
    print(f"Loose  (baseline   ): {ok_loose}/{total_loose}  (mismatches: {total_loose - ok_loose})")

    print(f"\nUnified-template signature checks:")
    print(f"  Records WITH forbidden V17/V18 artifact: {len(artifact_offenders)}")
    print(f"  Records MISSING baseline signature:      {len(baseline_missing)}")
    if artifact_offenders[:3]:
        print(f"  Example artifacts found: {artifact_offenders[:3]}")
    if baseline_missing[:3]:
        print(f"  Example missing signatures: {baseline_missing[:3]}")

    all_ok = (
        total_strict - ok_strict == 0
        and total_loose - ok_loose == 0
        and len(artifact_offenders) == 0
        and len(baseline_missing) == 0
    )
    print(f"\nPASS: {all_ok}")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
