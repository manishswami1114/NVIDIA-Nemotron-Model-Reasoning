#!/usr/bin/env python3
"""v15 audit: for every record, confirm boxed answer matches train.csv answer.
This is the lightweight integrity check — full per-step re-verification would
require parsing the writers' output formats and is unnecessary since each
writer's logic was already validated on samples."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all

BASE = Path(__file__).resolve().parent.parent
V15 = BASE / "all_categorical_splits_v15"


def main():
    answers = {r['id']: r['answer'] for r in load_all(BASE/'data/raw/train.csv')}
    print(f"Loaded {len(answers)} answers from train.csv", flush=True)

    total = 0; ok = 0; bad = []
    for path in sorted(V15.glob('train_cot_*.jsonl')):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                total += 1
                user = rec['messages'][0]['content']
                asst = rec['messages'][-1]['content']
                # find boxed
                m = re.findall(r'\\boxed\{(.*?)\}\s*$', asst)
                boxed = m[-1] if m else None
                # find puzzle id by matching prompt against train.csv
                # Note: train.csv has each prompt; cheaper to compare via answer field directly.
                # Use a simple heuristic: scan all rows for matching prompt.
                # Instead: confirm the boxed value occurs as some real answer in train.csv
                # OR: parse the query line to identify the puzzle, then look it up.
                # The simplest robust check: the boxed answer must EXIST as a value in train.csv
                if boxed in answers.values():
                    ok += 1
                else:
                    bad.append((path.name, boxed))
        print(f"  {path.name}: {sum(1 for _ in open(path))} records", flush=True)

    print(f"\nBoxed answer is a known train.csv value: {ok}/{total}")
    # More precise per-record check: rebuild id→answer map and match by prompt
    # Build prompt→id map
    rows = load_all(BASE/'data/raw/train.csv')
    prompt_to_answer = {r['prompt']: r['answer'] for r in rows}
    mismatch = 0
    total2 = 0
    for path in sorted(V15.glob('train_cot_*.jsonl')):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                total2 += 1
                u = rec['messages'][0]['content']
                a = rec['messages'][-1]['content']
                expected = prompt_to_answer.get(u)
                if expected is None:
                    mismatch += 1
                    continue
                m = re.findall(r'\\boxed\{(.*?)\}\s*$', a)
                boxed = m[-1] if m else None
                if boxed != expected:
                    mismatch += 1
                    if mismatch <= 10:
                        print(f"  MISMATCH in {path.name}: got {boxed!r} expected {expected!r}")
    print(f"\nPer-prompt strict match: {total2 - mismatch}/{total2}  (mismatches: {mismatch})")
    return mismatch == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
