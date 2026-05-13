#!/usr/bin/env python3
"""Deep failure analysis: Other category puzzle types + bit-error patterns."""

import csv
import re
import json
from collections import defaultdict, Counter

CSV_PATH = "/Users/manishswami/developer/NVIDIA-Nemotron Model/evaluation_results_086.csv"

rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# ── 1. "Other" category: classify by puzzle type from raw_output ────────────
print("=" * 70)
print("=== 'Other' Category: Puzzle-Type Classification from raw_output ===")
print("=" * 70)

other_rows = [r for r in rows if r["category"] == "Other"]
other_wrong = [r for r in other_rows if r["is_correct"] == "False"]

# Try to classify puzzle type from the raw_output or ground_truth
def classify_other_puzzle(row):
    raw = row.get("raw_output", "").lower()
    gt = row.get("ground_truth", "")
    
    # Check for various puzzle types
    if "cipher" in raw or "caesar" in raw or "rot13" in raw or "vigenere" in raw or "encrypt" in raw or "decrypt" in raw:
        return "cipher"
    elif "gravity" in raw or "fall" in raw or "drop" in raw:
        return "gravity"
    elif "equation" in raw or "solve for" in raw or "find the value" in raw:
        return "equation"
    elif "convert" in raw or "unit" in raw:
        return "unit_conversion_misc"
    elif "binary" in raw or "hexadecimal" in raw or "octal" in raw:
        return "number_base"
    elif "matrix" in raw:
        return "matrix"
    elif "sequence" in raw or "pattern" in raw or "next number" in raw:
        return "sequence"
    elif "grid" in raw:
        return "grid"
    else:
        return "unclassified"

# Classify all Other rows
other_classified = defaultdict(lambda: {"total": 0, "wrong": 0, "ids": []})
for r in other_rows:
    ptype = classify_other_puzzle(r)
    other_classified[ptype]["total"] += 1
    if r["is_correct"] == "False":
        other_classified[ptype]["wrong"] += 1
        other_classified[ptype]["ids"].append(r["id"])

print(f"\nTotal 'Other' rows: {len(other_rows)}")
print(f"Total 'Other' wrong: {len(other_wrong)}")
print(f"\n{'Puzzle Type':<25s}  {'Total':>5s}  {'Wrong':>5s}  {'Accuracy':>8s}")
print(f"{'-'*25}  {'-'*5}  {'-'*5}  {'-'*8}")
for ptype in sorted(other_classified.keys(), key=lambda x: -other_classified[x]["wrong"]):
    stats = other_classified[ptype]
    acc = (stats["total"] - stats["wrong"]) / stats["total"] if stats["total"] > 0 else 0
    print(f"{ptype:<25s}  {stats['total']:5d}  {stats['wrong']:5d}  {acc:8.4f}")

# ── 2. List all "Other" incorrect puzzle IDs ────────────────────────────────
print(f"\n\n{'='*70}")
print("=== All 'Other' Category Incorrect Puzzle IDs ===")
print(f"{'='*70}\n")

other_wrong_details = []
for r in other_wrong:
    ptype = classify_other_puzzle(r)
    other_wrong_details.append({
        "id": r["id"],
        "puzzle_type": ptype,
        "ground_truth": r["ground_truth"][:50],
        "prediction": r["prediction"][:50] if r["prediction"] else "EMPTY",
        "has_prediction": bool(r["prediction"].strip()),
    })

# Sort by puzzle type
other_wrong_details.sort(key=lambda x: (x["puzzle_type"], x["id"]))

print(f"Total: {len(other_wrong_details)} incorrect 'Other' puzzles\n")
print(f"{'ID':<12s}  {'Type':<20s}  {'Has Pred':>8s}  {'Ground Truth':<30s}  {'Prediction':<30s}")
print(f"{'-'*12}  {'-'*20}  {'-'*8}  {'-'*30}  {'-'*30}")
for d in other_wrong_details:
    print(f"{d['id']:<12s}  {d['puzzle_type']:<20s}  {'Yes' if d['has_prediction'] else 'NO':>8s}  {d['ground_truth']:<30s}  {d['prediction']:<30s}")

# ── 3. Bit manipulation: bit-level error analysis ───────────────────────────
print(f"\n\n{'='*70}")
print("=== Bit Manipulation: Bit-Level Error Pattern Analysis ===")
print(f"{'='*70}\n")

bit_wrong = [r for r in rows if r["category"] == "Bit manipulation" and r["is_correct"] == "False"]
bit_wrong_with_pred = [r for r in bit_wrong if r["prediction"].strip() and len(r["prediction"].strip()) == 8]

print(f"Total bit-manip wrong: {len(bit_wrong)}")
print(f"Wrong with valid 8-bit prediction: {len(bit_wrong_with_pred)}")

# Analyze which bit positions are most commonly wrong
bit_position_errors = Counter()
error_types = Counter()  # which positions flip direction (0->1, 1->0)

for r in bit_wrong_with_pred:
    gt = r["ground_truth"].strip()
    pred = r["prediction"].strip()
    if len(gt) == 8 and len(pred) == 8:
        for i in range(8):
            if gt[i] != pred[i]:
                bit_position_errors[i] += 1
                if gt[i] == '0' and pred[i] == '1':
                    error_types[f"pos{i}_0->1"] += 1
                elif gt[i] == '1' and pred[i] == '0':
                    error_types[f"pos{i}_1->0"] += 1

print(f"\nBit Position Error Frequency (0=MSB, 7=LSB):")
print(f"{'Position':>8s}  {'Errors':>6s}  {'% of wrong':>10s}")
for pos in range(8):
    cnt = bit_position_errors[pos]
    pct = cnt / len(bit_wrong_with_pred) * 100 if bit_wrong_with_pred else 0
    print(f"  bit[{pos}]    {cnt:6d}  {pct:9.1f}%")

print(f"\nFlip Direction Analysis:")
print(f"{'Type':>15s}  {'Count':>6s}")
for key in sorted(error_types.keys()):
    print(f"  {key:>13s}  {error_types[key]:6d}")

# Number of bits wrong per prediction
bits_wrong_dist = Counter()
for r in bit_wrong_with_pred:
    gt = r["ground_truth"].strip()
    pred = r["prediction"].strip()
    n_wrong = sum(1 for i in range(8) if gt[i] != pred[i])
    bits_wrong_dist[n_wrong] += 1

print(f"\nNumber of Bits Wrong per Prediction:")
print(f"{'# Bits Wrong':>12s}  {'Count':>6s}  {'% of wrong':>10s}")
for n in sorted(bits_wrong_dist.keys()):
    cnt = bits_wrong_dist[n]
    pct = cnt / len(bit_wrong_with_pred) * 100
    print(f"  {n:>10d}  {cnt:6d}  {pct:9.1f}%")

# ── 4. "Other" wrong: failure mode (no prediction vs wrong) ────────────────
print(f"\n\n{'='*70}")
print("=== 'Other' Wrong: Failure Mode Breakdown ===")
print(f"{'='*70}\n")

other_no_pred = [r for r in other_wrong if not r["prediction"].strip()]
other_has_pred = [r for r in other_wrong if r["prediction"].strip()]

print(f"Empty prediction: {len(other_no_pred)}")
print(f"Has prediction but wrong: {len(other_has_pred)}")

# For empty predictions, check if boxed exists in raw
other_no_pred_boxed = [r for r in other_no_pred if r"\\boxed{" in r["raw_output"] or r"\boxed{" in r["raw_output"]]
print(f"Empty pred but has \\boxed in raw: {len(other_no_pred_boxed)}")
print(f"Truly no answer at all: {len(other_no_pred) - len(other_no_pred_boxed)}")

# ── 5. All incorrect IDs comprehensive export ──────────────────────────────
print(f"\n\n{'='*70}")
print("=== Complete Failing Puzzle List (exported to JSON) ===")
print(f"{'='*70}\n")

all_failures = []
for r in rows:
    if r["is_correct"] == "False":
        raw = r.get("raw_output", "")
        has_boxed = r"\boxed{" in raw
        pred_empty = not r["prediction"].strip()
        
        if r["category"] == "Bit manipulation":
            gt = r["ground_truth"].strip()
            pred = r["prediction"].strip()
            n_bits_wrong = -1
            if len(gt) == 8 and len(pred) == 8:
                n_bits_wrong = sum(1 for i in range(8) if gt[i] != pred[i])
            
            all_failures.append({
                "id": r["id"],
                "category": r["category"],
                "sub_category": r["sub_category"],
                "ground_truth": r["ground_truth"],
                "prediction": r["prediction"],
                "failure_mode": "no_prediction" if pred_empty else "wrong_answer",
                "has_boxed": has_boxed,
                "bits_wrong": n_bits_wrong,
            })
        else:
            ptype = classify_other_puzzle(r) if r["category"] == "Other" else r["category"]
            all_failures.append({
                "id": r["id"],
                "category": r["category"],
                "sub_category": r["sub_category"],
                "ground_truth": r["ground_truth"][:100],
                "prediction": r["prediction"][:100] if r["prediction"] else "",
                "failure_mode": "no_prediction" if pred_empty else "wrong_answer",
                "has_boxed": has_boxed,
                "puzzle_type_guess": ptype,
            })

json_path = "/Users/manishswami/developer/NVIDIA-Nemotron Model/all_failures_detailed.json"
with open(json_path, "w") as f:
    json.dump(all_failures, f, indent=2)

print(f"Exported {len(all_failures)} failures to {json_path}")
print(f"  - Bit manipulation: {sum(1 for x in all_failures if x['category'] == 'Bit manipulation')}")
print(f"  - Other: {sum(1 for x in all_failures if x['category'] == 'Other')}")
