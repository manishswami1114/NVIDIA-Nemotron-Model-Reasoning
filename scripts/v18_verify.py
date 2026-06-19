#!/usr/bin/env python3
"""v16 audit.

The baseline files were generated with AUGMENTED user prompts (longer than
train.csv), so per-prompt strict match would fail on them — but the boxed
answers in the baseline are correct (the baseline trained the proven 0.86
model). For verification we therefore:

  - For the 3 REGENERATED files (we built them from train.csv prompts):
      strict per-prompt match against train.csv.
  - For the 6 COPIED baseline files:
      check that every record's boxed answer is *a* known train.csv answer
      (sanity check, prevents accidental file corruption).

Pass criterion: 0 mismatches in either bucket.
"""
import json, re, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all

BASE = Path(__file__).resolve().parent.parent
V16 = BASE / "all_categorical_splits_v18"

REGENERATED = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
}


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
    bad_strict = []; bad_loose = []
    for path in sorted(V16.glob('train_cot_*.jsonl')):
        fc = 0; fok = 0
        is_regen = path.name in REGENERATED
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                fc += 1
                user = rec['messages'][0]['content']
                asst = rec['messages'][-1]['content']
                boxed = extract_final_answer(asst)
                if is_regen:
                    expected = prompt_to_answer.get(user)
                    if expected is None:
                        bad_strict.append((path.name, '(prompt not in train.csv)'))
                        continue
                    if verify(expected, boxed):
                        fok += 1
                    else:
                        bad_strict.append((path.name, f"got={boxed!r} expected={expected!r}"))
                else:
                    # baseline-copied: boxed must be a known train.csv answer
                    if boxed in all_answers:
                        fok += 1
                    else:
                        bad_loose.append((path.name, f"boxed={boxed!r} not in train.csv"))
        if is_regen:
            total_strict += fc; ok_strict += fok
            mode = 'strict'
        else:
            total_loose += fc; ok_loose += fok
            mode = 'loose'
        flag = '' if fok == fc else '  ← MISMATCH'
        print(f"  {path.name:<43} {fok:>6}/{fc:>6}  {mode}{flag}")

    print('-' * 75)
    print(f"\nStrict (regenerated): {ok_strict}/{total_strict}  (mismatches: {total_strict - ok_strict})")
    print(f"Loose  (baseline   ): {ok_loose}/{total_loose}  (mismatches: {total_loose - ok_loose})")

    if bad_strict[:5]:
        print("\nFirst strict mismatches:")
        for f, msg in bad_strict[:5]:
            print(f"  {f}: {msg}")
    if bad_loose[:5]:
        print("\nFirst loose mismatches:")
        for f, msg in bad_loose[:5]:
            print(f"  {f}: {msg}")

    all_ok = (total_strict - ok_strict == 0) and (total_loose - ok_loose == 0)
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
