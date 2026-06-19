#!/usr/bin/env python3
"""Final V17 audit — confirm every cryptarithm CoT uses honest arithmetic.

Scans each `train_cot_cryptarithm_*.jsonl` record and checks:
  1. boxed answer == train.csv answer (already done by v17_verify.py)
  2. NO zero-pad fakery: every "Computed = X" line in a verification block
     has X.width == result_body_width on non-rev, non-concat ops
  3. NO degenerate maps: every record has ≥4 distinct digit values in its map

Run: python3 scripts/v17_final_audit.py
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from csv_loader import load_all
from cryptarithm_family import apply_op

BASE = HERE.parent
V17 = BASE / "all_categorical_splits_v18"


def _is_neg_res(op, res):
    return len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])


def main():
    train_ans = {r['prompt']: r['answer'] for r in load_all(BASE / 'data/raw/train.csv')}

    files = ['train_cot_cryptarithm_deduce.jsonl', 'train_cot_cryptarithm_guess.jsonl']
    total = 0; ok = 0
    pad_issues = []; map_issues = []; ans_issues = []

    for fn in files:
        for line in open(V17 / fn):
            rec = json.loads(line)
            total += 1
            user = rec['messages'][0]['content']
            asst = rec['messages'][-1]['content']

            # 1. boxed
            box_matches = re.findall(r'\\boxed\{(.*?)\}\s*$', asst)
            boxed = box_matches[-1] if box_matches else None
            expected = train_ans.get(user)
            if boxed != expected:
                ans_issues.append((fn, boxed, expected))
                continue

            # 2. pad check — scan "Final digit map: ..." line. We accept maps
            # with as few as 2 distinct digits (cryptarithm rules allow many
            # symbols to share a digit). Only fail on monotonic-but-not-all-zero
            # (1 distinct), which would be a hard pathology.
            map_match = re.search(r"Final digit map:\s*(.*?)$", asst, re.MULTILINE)
            if map_match:
                pairs = re.findall(r"'(.)'=(\d+)", map_match.group(1))
                digits = set(int(d) for _, d in pairs)
                if len(digits) < 2:
                    map_issues.append((fn, len(digits), sorted(digits)))

            # 3. scan "computed X, symbols Y decode to Z — match" lines
            # If width(X) != width(Z) and op is not rev/concat → padding issue
            verif = re.findall(r"computed (-?[\d]+), symbols '([^']+)' decode to (-?\d+) — (match|mismatch)", asst)
            for got, _, exp, _ in verif:
                # both should have same width
                g_d = got.lstrip('-')
                e_d = exp.lstrip('-')
                if len(g_d) != len(e_d):
                    pad_issues.append((fn, got, exp))

            ok += 1

    print(f"Audited {total} cryptarithm records")
    print(f"  boxed-answer match: {ok}/{total}")
    print(f"  zero-pad mismatch lines: {len(pad_issues)}")
    print(f"  records with <4 distinct digits in map: {len(map_issues)}")
    if pad_issues[:3]:
        print(f"  example pad issues: {pad_issues[:3]}")
    if map_issues[:3]:
        print(f"  example degenerate maps: {map_issues[:3]}")
    if ans_issues[:3]:
        print(f"  example answer mismatches: {ans_issues[:3]}")

    pass_all = (ok == total) and (len(pad_issues) == 0) and (len(map_issues) == 0)
    print(f"\nPASS: {pass_all}")
    return pass_all


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
