#!/usr/bin/env python3
"""Analyze evaluation_results_086.csv for failure patterns."""

import csv
import re
import json
from collections import defaultdict, Counter

CSV_PATH = "/Users/manishswami/developer/NVIDIA-Nemotron Model/evaluation_results_086.csv"

# ── 1. Read all rows ────────────────────────────────────────────────────────
rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Total rows: {len(rows)}")

# ── 2. Overall stats ────────────────────────────────────────────────────────
correct = sum(1 for r in rows if r["is_correct"] == "True")
wrong   = sum(1 for r in rows if r["is_correct"] == "False")
print(f"Correct: {correct}  Wrong: {wrong}  Accuracy: {correct/len(rows):.4f}")

# ── 3. Per-category stats ───────────────────────────────────────────────────
cat_total   = Counter()
cat_correct = Counter()
for r in rows:
    cat_total[r["category"]] += 1
    if r["is_correct"] == "True":
        cat_correct[r["category"]] += 1

print("\n=== Per-Category Accuracy ===")
for cat in sorted(cat_total.keys()):
    total = cat_total[cat]
    corr  = cat_correct[cat]
    acc   = corr / total
    print(f"  {cat:25s}  {corr:5d}/{total:5d}  acc={acc:.4f}")

# ── 4. All incorrect rows ───────────────────────────────────────────────────
incorrect_rows = [r for r in rows if r["is_correct"] == "False"]
print(f"\nTotal incorrect: {len(incorrect_rows)}")

# ── 5. Per-category incorrect breakdown ─────────────────────────────────────
cat_incorrect = defaultdict(list)
for r in incorrect_rows:
    cat_incorrect[r["category"]].append(r)

print("\n=== Incorrect Count by Category ===")
for cat in sorted(cat_incorrect.keys()):
    print(f"  {cat:25s}  {len(cat_incorrect[cat]):5d} wrong")

# ── 6. Failure mode analysis: has \boxed{} or not ───────────────────────────
def classify_failure(row):
    raw = row.get("raw_output", "")
    pred = row.get("prediction", "").strip()
    
    if not pred:
        # Empty prediction
        if r"\\boxed{" in raw or r"\boxed{" in raw:
            return "boxed_but_extraction_failed"
        else:
            return "no_boxed_no_prediction"
    else:
        # Has prediction but wrong
        return "wrong_answer"

print("\n=== Failure Mode Distribution (ALL categories) ===")
failure_modes = Counter()
for r in incorrect_rows:
    fm = classify_failure(r)
    failure_modes[fm] += 1
for fm, cnt in failure_modes.most_common():
    print(f"  {fm:40s}  {cnt:5d}  ({cnt/len(incorrect_rows)*100:.1f}%)")

# Per-category failure modes
print("\n=== Failure Modes by Category ===")
for cat in sorted(cat_incorrect.keys()):
    print(f"\n  [{cat}]")
    fm_counts = Counter()
    for r in cat_incorrect[cat]:
        fm_counts[classify_failure(r)] += 1
    for fm, cnt in fm_counts.most_common():
        print(f"    {fm:40s}  {cnt:5d}")

# ── 7. List ALL sub_category families with is_correct == False ──────────────
print("\n\n========================================================")
print("=== ALL Sub-Category (Family) Puzzles with Failures ===")
print("========================================================\n")

# Group incorrect by (category, sub_category)
family_failures = defaultdict(lambda: {"wrong": 0, "total": 0})
for r in rows:
    key = (r["category"], r["sub_category"])
    family_failures[key]["total"] += 1
    if r["is_correct"] == "False":
        family_failures[key]["wrong"] += 1

# Filter only families that have at least 1 wrong
failing_families = {k: v for k, v in family_failures.items() if v["wrong"] > 0}

# Sort by category, then by accuracy (worst first)
sorted_families = sorted(
    failing_families.items(),
    key=lambda x: (x[0][0], x[1]["wrong"] / x[1]["total"]),
    reverse=True
)

# Print grouped by category
current_cat = None
cat_family_count = 0
for (cat, sub_cat), stats in sorted_families:
    if cat != current_cat:
        if current_cat is not None:
            print(f"  --- Total failing families in {current_cat}: {cat_family_count} ---\n")
        current_cat = cat
        cat_family_count = 0
        print(f"\n### {cat} ###")
        print(f"  {'Sub-Category':<20s}  {'Wrong':>5s}  {'Total':>5s}  {'Accuracy':>8s}  {'Failure%':>8s}")
        print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}")
    
    acc = (stats["total"] - stats["wrong"]) / stats["total"]
    fail_pct = stats["wrong"] / stats["total"] * 100
    print(f"  {sub_cat:<20s}  {stats['wrong']:5d}  {stats['total']:5d}  {acc:8.4f}  {fail_pct:7.1f}%")
    cat_family_count += 1

if current_cat is not None:
    print(f"  --- Total failing families in {current_cat}: {cat_family_count} ---\n")

# ── 8. Bit manipulation: families with 0% accuracy ─────────────────────────
print("\n========================================================")
print("=== Bit Manipulation: Families with 0% Accuracy ===")
print("========================================================\n")

bit_zero = []
for (cat, sub_cat), stats in family_failures.items():
    if cat == "Bit manipulation" and stats["wrong"] == stats["total"]:
        bit_zero.append((sub_cat, stats["total"]))

bit_zero.sort(key=lambda x: -x[1])
print(f"Total families with 0% accuracy: {len(bit_zero)}")
print(f"Total puzzles in those families: {sum(t for _, t in bit_zero)}")
print(f"\n  {'Sub-Category':<20s}  {'Count':>5s}")
print(f"  {'-'*20}  {'-'*5}")
for sc, cnt in bit_zero:
    print(f"  {sc:<20s}  {cnt:5d}")

# ── 9. "Other" category: detailed sub_category breakdown ───────────────────
print("\n\n========================================================")
print("=== 'Other' Category: Detailed Sub-Category Failures ===")
print("========================================================\n")

other_families = defaultdict(lambda: {"wrong": 0, "total": 0})
for r in rows:
    if r["category"] not in ("Bit manipulation", "Unit conversion"):
        key = r["sub_category"]
        other_families[key]["total"] += 1
        if r["is_correct"] == "False":
            other_families[key]["wrong"] += 1

other_failing = {k: v for k, v in other_families.items() if v["wrong"] > 0}
other_sorted = sorted(other_failing.items(), key=lambda x: x[1]["wrong"], reverse=True)

print(f"Total 'Other' families with failures: {len(other_sorted)}")
print(f"\n  {'Sub-Category':<30s}  {'Wrong':>5s}  {'Total':>5s}  {'Accuracy':>8s}")
print(f"  {'-'*30}  {'-'*5}  {'-'*5}  {'-'*8}")
for sc, stats in other_sorted:
    acc = (stats["total"] - stats["wrong"]) / stats["total"]
    print(f"  {sc:<30s}  {stats['wrong']:5d}  {stats['total']:5d}  {acc:8.4f}")

# ── 10. Sample incorrect bit manipulation rows (for manual inspection) ──────
print("\n\n========================================================")
print("=== Sample Incorrect Bit Manipulation Rows (first 30) ===")
print("========================================================\n")

bit_wrong = [r for r in incorrect_rows if r["category"] == "Bit manipulation"]
for i, r in enumerate(bit_wrong[:30]):
    raw_tail = r["raw_output"][-300:] if r["raw_output"] else ""
    has_boxed = r"\\boxed{" in r["raw_output"] or r"\boxed{" in r["raw_output"]
    print(f"\n--- [{i+1}] id={r['id']}  sub_cat={r['sub_category']} ---")
    print(f"  ground_truth: {r['ground_truth']}")
    print(f"  prediction:   '{r['prediction']}'")
    print(f"  has_boxed:    {has_boxed}")
    print(f"  raw_tail:     ...{raw_tail[-200:]}")

# ── 11. Export failing families to JSON for downstream use ──────────────────
export = []
for (cat, sub_cat), stats in sorted_families:
    acc = (stats["total"] - stats["wrong"]) / stats["total"]
    export.append({
        "category": cat,
        "sub_category": sub_cat,
        "wrong": stats["wrong"],
        "total": stats["total"],
        "accuracy": round(acc, 4)
    })

json_path = "/Users/manishswami/developer/NVIDIA-Nemotron Model/failing_families.json"
with open(json_path, "w") as f:
    json.dump(export, f, indent=2)
print(f"\n\nExported {len(export)} failing families to {json_path}")
