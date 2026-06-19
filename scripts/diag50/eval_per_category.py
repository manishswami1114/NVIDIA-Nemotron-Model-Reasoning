"""Per-category accuracy on train.csv for a Kaggle eval-results CSV.

Use this AFTER training the diagnostic adapter on the diag50 dataset.

Compare against the baseline 0.86 per-category accuracy:
    bit_manipulation         82.6%
    cipher                   99.4%
    cryptarithm_deduce        1.2%
    cryptarithm_guess         1.8%
    equation_numeric_deduce  90.8%
    equation_numeric_guess    5.1%
    gravity                  99.8%
    numeral                 100.0%
    unit_conversion         100.0%
    TOTAL                    86.5%

Interpretation:
    - cryptarithm_deduce ↑, others flat → diag50 is safe, scale up
    - cryptarithm_deduce ↑, cipher/gravity/numeral/unit_conversion ↓ → LoRA leaks
    - cryptarithm_deduce flat → CoTs not being learned at this size
    - everything ↓ → can't safely modify this dataset

Usage:
    python scripts/diag50/eval_per_category.py path/to/your_eval.csv
"""
from __future__ import annotations
import csv, re, sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# ── categorizer (same as the one used to build the baseline split) ──
_OPENER_MAP = [
    ("secret bit manipulation rule",                  "bit_manipulation"),
    ("secret encryption rules are used on text",      "cipher"),
    ("gravitational constant has been secretly",      "gravity"),
    ("numbers are secretly converted into a different numeral system", "numeral"),
    ("secret unit conversion is applied to measurements", "unit_conversion"),
]
_EQ = "secret set of transformation rules is applied to equations"
_LHS_RE = re.compile(r"^\s*(\S{5})\s*=\s*(\S+)\s*$")
_Q_RE   = re.compile(r"determine the result for:\s*(\S+)", re.IGNORECASE)

def parse(p):
    exs, q = [], None
    for line in p.splitlines():
        m = _Q_RE.search(line)
        if m:
            qs = m.group(1).strip()
            if len(qs) >= 5: q = qs[:5]
            continue
        m = _LHS_RE.match(line)
        if m and "determine" not in line.lower():
            exs.append(m.group(1))
    return exs, q

def eq_sub(p):
    exs, q = parse(p)
    is_num = bool(exs) and (exs[0][0]+exs[0][1]+exs[0][3]+exs[0][4]).isdigit()
    base = "equation_numeric" if is_num else "cryptarithm"
    suf = "deduce"
    if q and exs:
        ops = {e[2] for e in exs}
        suf = "deduce" if q[2] in ops else "guess"
    return f"{base}_{suf}"

def categorize(prompt):
    pl = (prompt or "").lower()
    for n, lbl in _OPENER_MAP:
        if n in pl: return lbl
    if _EQ in pl: return eq_sub(prompt)
    return "Other"


BASELINE = {
    "bit_manipulation":         (1602, 1324, 82.6),
    "cipher":                   (1576, 1566, 99.4),
    "cryptarithm_deduce":       ( 659,    8,  1.2),
    "cryptarithm_guess":        ( 164,    3,  1.8),
    "equation_numeric_deduce":  ( 596,  541, 90.8),
    "equation_numeric_guess":   ( 136,    7,  5.1),
    "gravity":                  (1597, 1594, 99.8),
    "numeral":                  (1576, 1576,100.0),
    "unit_conversion":          (1594, 1594,100.0),
}


def main(eval_csv: str, train_csv: str = "data/raw/train.csv"):
    if not Path(eval_csv).exists():
        raise FileNotFoundError(eval_csv)
    if not Path(train_csv).exists():
        raise FileNotFoundError(train_csv)

    id2cat = {}
    with open(train_csv) as f:
        for r in csv.DictReader(f):
            id2cat[r["id"]] = categorize(r["prompt"])

    tot = defaultdict(int); ok = defaultdict(int)
    with open(eval_csv) as f:
        for r in csv.DictReader(f):
            cat = id2cat.get(r["id"])
            if cat is None: continue
            tot[cat] += 1
            if r["is_correct"].lower() in ("true","1","yes"):
                ok[cat] += 1

    print(f"\n{'category':<26} {'N':>5} {'OK':>5} {'wrong':>6} {'acc%':>7}  "
          f"{'baseline':>9}  {'Δ':>7}")
    print("-"*78)
    T = W = 0
    for cat in sorted(BASELINE):
        n = tot[cat]; c = ok[cat]; w = n - c
        bn, bc, bp = BASELINE[cat]
        delta = (c/max(1,n)*100) - bp
        marker = "  ↑↑" if delta > 1 else ("  ↑" if delta > 0 else ("  →" if delta == 0 else ("  ↓" if delta > -1 else "  ↓↓")))
        print(f"{cat:<26} {n:>5} {c:>5} {w:>6} {c/max(1,n)*100:>6.1f}% "
              f"  {bp:>7.1f}% {delta:>+6.1f}{marker}")
        T += n; W += w
    print("-"*78)
    print(f"{'TOTAL':<26} {T:>5} {T-W:>5} {W:>6} {(T-W)/max(1,T)*100:>6.1f}%   86.5%  {((T-W)/max(1,T)*100 - 86.5):>+6.1f}")

    print("\nVerdict guide:")
    print("  cryptarithm_deduce ↑  AND  cipher/gravity/numeral/unit_conversion → unchanged: SAFE → scale up")
    print("  cryptarithm_deduce ↑  AND  any of those 4 dropped:                       LEAKS  → lower LR / freeze layers")
    print("  cryptarithm_deduce →                                                     INERT  → CoT format isn't being learned at 50 records; try 200")
    print("  multiple categories ↓                                                    STRUCTURAL → don't ship, redesign approach")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_per_category.py path/to/your_eval.csv")
        sys.exit(1)
    main(sys.argv[1])
