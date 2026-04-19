# -*- coding: utf-8 -*-
"""
CoT v4 — Filter fake training data + regenerate bit_manip with real traces.

The v3 data has ~4700 examples with FAKE templated CoT (just
'I studied the pattern... answer: X'). This teaches the model to hallucinate.

This notebook:
  1. Loads the raw training CSV
  2. For each row, runs the matching solver
  3. KEEPS the row only if the solver actually produces the ground-truth answer
  4. For kept rows, generates REAL step-by-step CoT
  5. Writes train_cot_v4_real.jsonl
"""

# ============================================================
# Cell 1: Imports and setup
# ============================================================
import json
import re
import pandas as pd
from pathlib import Path

from solvers import (
    solve_unit_conversion,
    solve_gravitational,
    solve_numeral_system,
    solve_bit_manipulation,
)
from cot_v4_generators import gen_bit_manipulation_cot, gen_cipher_cot

TRAIN_CSV = "train.csv"
OUTPUT    = "train_cot_v4_real.jsonl"
EVAL_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

df = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(df)} raw training rows")


# ============================================================
# Cell 2: Category classifier
# ============================================================
def classify(prompt: str) -> str:
    p = prompt.lower()
    if 'bit manipulation' in p:                  return 'bit'
    if 'gravitational constant' in p:            return 'gravity'
    if 'unit conversion' in p:                   return 'unit'
    if 'encryption rules are used on text' in p: return 'cipher'
    if 'numeral system' in p:                    return 'roman'
    if 'transformation rules' in p:              return 'transform'
    return 'unknown'


# ============================================================
# Cell 3: CoT generators for categories with good solvers
# These wrap the existing solvers and emit REAL reasoning traces.
# ============================================================

def gen_roman_cot(prompt: str):
    """Roman numeral conversion — straightforward step-by-step."""
    # Query format: "write the number 38 in the Wonderland numeral system"
    m = re.search(r'write the number\s+(\d+)', prompt)
    if not m:
        return None, None
    num = int(m.group(1))
    original = num
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    steps = []
    result = ''
    n = num
    for v, s in zip(val, syms):
        while n >= v:
            steps.append(f"  {n} ≥ {v} → append `{s}`, remainder {n - v}")
            result += s
            n -= v
    cot = (
        f"I need to convert {original} to a Roman numeral.\n\n"
        f"I'll use the standard greedy algorithm: repeatedly find the largest Roman "
        f"value that fits, append its symbol, and subtract.\n\n"
        + "\n".join(steps)
        + f"\n\nFinal Roman numeral: {result}"
    )
    return cot, result


def gen_cipher_cot(prompt: str):
    """Text encryption — show the mapping derivation."""
    lines = prompt.strip().split('\n')
    examples = []
    query_text = None
    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        elif 'decrypt the following text:' in line.lower():
            m = re.search(r'decrypt the following text:\s*(.*)', line, re.IGNORECASE)
            if m:
                query_text = m.group(1).strip()
    if not examples or not query_text:
        return None, None

    # Build char map and verify
    char_map = {}
    for enc, dec in examples:
        if len(enc) != len(dec):
            return None, None
        for e, d in zip(enc, dec):
            if e == ' ':
                continue
            if e in char_map and char_map[e] != d:
                return None, None
            char_map[e] = d

    # Apply to query and check the full pipeline ends up consistent
    try:
        result = ''.join(char_map.get(c, c) for c in query_text)
    except Exception:
        return None, None

    # Build a CoT trace that shows the mapping derivation
    mapping_str = ", ".join(f"{k}→{v}" for k, v in sorted(char_map.items()))
    cot = (
        "This is a substitution cipher. I'll build a character mapping from the examples.\n\n"
        "Examples:\n"
        + "\n".join(f"  {e} → {d}" for e, d in examples)
        + f"\n\nPairing each encrypted character with its decrypted counterpart:\n"
        f"  {mapping_str}\n\n"
        f"Applying the mapping to '{query_text}':\n"
        f"  '{query_text}' → '{result}'"
    )
    return cot, result


def gen_unit_cot(prompt: str):
    """Unit conversion — show factor derivation + multiply."""
    # Example format: "10.08 m becomes 6.69"
    examples = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not examples:
        return None, None
    # Query format: "convert the following measurement: 25.09 m"
    query_match = re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt, re.IGNORECASE)
    if not query_match:
        return None, None
    query_val = float(query_match.group(1))

    # Derive conversion factor = output / input
    factors = []
    factor_lines = []
    for x, y in examples:
        x_f, y_f = float(x), float(y)
        if x_f == 0:
            continue
        f = y_f / x_f
        factors.append(f)
        factor_lines.append(f"  {y_f} / {x_f} = {f:.6f}")
    if not factors:
        return None, None

    avg = sum(factors) / len(factors)
    result = query_val * avg
    result_str = f"{result:.2f}"

    cot = (
        f"I need to convert {query_val} m using the secret conversion rule.\n\n"
        "Let me find the conversion factor from the examples (factor = output / input):\n\n"
        + "\n".join(factor_lines)
        + f"\n\nAverage conversion factor ≈ {avg:.6f}\n\n"
        f"Applying to {query_val} m:\n"
        f"  {query_val} × {avg:.6f} = {result_str}"
    )
    return cot, result_str


def gen_gravity_cot(prompt: str):
    """Gravitational puzzle — derive g from d = 0.5 g t^2, then apply."""
    # Example format: "For t = 1.37s, distance = 14.92 m"
    examples = re.findall(r'For t = ([\d.]+)s,\s*distance = ([\d.]+)\s*m', prompt)
    # Query format: "determine the falling distance for t = 4.41s"
    query_match = re.search(r'falling distance for t = ([\d.]+)s', prompt, re.IGNORECASE)
    if not examples or not query_match:
        return None, None

    query_t = float(query_match.group(1))
    g_values = []
    g_lines = []
    for t_str, d_str in examples:
        t, d = float(t_str), float(d_str)
        if t == 0:
            continue
        g = 2 * d / (t ** 2)
        g_values.append(g)
        g_lines.append(f"  t = {t}s, d = {d}m: g = 2×{d}/{t}² = {g:.4f}")
    if not g_values:
        return None, None
    avg_g = sum(g_values) / len(g_values)
    d_result = 0.5 * avg_g * (query_t ** 2)
    d_str = f"{d_result:.2f}"

    cot = (
        f"I need to find the falling distance for t = {query_t}s using d = 0.5·g·t².\n\n"
        "First, I'll determine g from each example using g = 2d/t²:\n\n"
        + "\n".join(g_lines)
        + f"\n\nAverage g ≈ {avg_g:.4f}\n\n"
        f"For t = {query_t}s:\n"
        f"  d = 0.5 × {avg_g:.4f} × {query_t}² = 0.5 × {avg_g:.4f} × {query_t**2:.4f} = {d_str}"
    )
    return cot, d_str


# ============================================================
# Cell 4: The main filter — run solvers, keep verified rows only
# ============================================================

def _verify_bit(prompt, gt):
    """Try old single-op solver, then truth-table solver."""
    from cot_v4_generators import _solve_truth_table
    # old solver first (fast)
    pred = solve_bit_manipulation(prompt)
    if pred is not None:
        return pred
    # truth-table fallback
    examples = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    qm = re.search(r'determine the output for:\s*([01]{8})', prompt, re.IGNORECASE)
    if not examples or not qm:
        return None
    res = _solve_truth_table(examples, qm.group(1), max_deps=4)
    return res[0] if res else None

GENERATORS = {
    # (verify_fn, cot_gen_fn)
    'bit':     (_verify_bit,          gen_bit_manipulation_cot),
    'cipher':  (None,                 gen_cipher_cot),   # gen_cipher_cot verifies internally
    'unit':    (solve_unit_conversion, gen_unit_cot),
    'gravity': (solve_gravitational,   gen_gravity_cot),
    'roman':   (solve_numeral_system,  gen_roman_cot),
    # transform: 0% solvable — excluded
}

stats = {cat: {'kept': 0, 'wrong_answer': 0, 'no_solve': 0, 'no_cot': 0}
         for cat in GENERATORS}
stats['transform'] = {'dropped': 0}
stats['unknown']   = {'dropped': 0}

kept_rows = []

for _, row in df.iterrows():
    prompt = row['prompt']
    gt = str(row['answer']).strip()
    cat = classify(prompt)

    if cat not in GENERATORS:
        # transform / unknown → drop entirely
        stats.setdefault(cat, {'dropped': 0})
        stats[cat]['dropped'] = stats[cat].get('dropped', 0) + 1
        continue

    verify_fn, cot_gen = GENERATORS[cat]

    # For cipher: cot_gen handles both verification and CoT generation
    if verify_fn is None:
        try:
            cot, cot_answer = cot_gen(prompt)
        except Exception:
            cot, cot_answer = None, None
        if cot is None or str(cot_answer).strip() != gt:
            stats[cat]['no_cot'] += 1
            continue
    else:
        # Verify first
        try:
            pred = verify_fn(prompt, gt) if cat == 'bit' else verify_fn(prompt)
        except Exception:
            pred = None
        if pred is None:
            stats[cat]['no_solve'] += 1
            continue
        if str(pred).strip() != gt:
            stats[cat]['wrong_answer'] += 1
            continue
        # Generate reasoning trace
        try:
            cot, cot_answer = cot_gen(prompt)
        except Exception:
            cot, cot_answer = None, None
        if cot is None or str(cot_answer).strip() != gt:
            stats[cat]['no_cot'] += 1
            continue

    # Build the final message
    user_content = prompt + EVAL_SUFFIX
    assistant_content = f"<think>\n{cot}\n</think>\n\\boxed{{{gt}}}"
    kept_rows.append({
        'messages': [
            {'role': 'user',      'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ],
        'category': cat,
    })
    stats[cat]['kept'] += 1

print(f"\nTotal kept: {len(kept_rows)}")
print("\nPer-category stats:")
for cat, s in stats.items():
    print(f"  {cat:10s}: {s}")


# ============================================================
# Cell 5: Sanity-check samples from each category
# ============================================================
print("\n" + "="*60)
print("SAMPLE REAL CoT (one per category)")
print("="*60)
seen_cats = set()
for row in kept_rows:
    cat = row['category']
    if cat in seen_cats:
        continue
    seen_cats.add(cat)
    print(f"\n--- {cat.upper()} ---")
    print(row['messages'][1]['content'][:800])
    if len(seen_cats) >= 5:
        break


# ============================================================
# Cell 6: Write v4 JSONL
# ============================================================
with open(OUTPUT, 'w') as f:
    for row in kept_rows:
        f.write(json.dumps({'messages': row['messages']}) + '\n')

size_mb = Path(OUTPUT).stat().st_size / 1024 / 1024
print(f"\nWrote {len(kept_rows)} examples to {OUTPUT} ({size_mb:.2f} MB)")

# Verify
with open(OUTPUT) as f:
    first = json.loads(f.readline())
    print("\nFirst row structure OK:", list(first.keys()))
