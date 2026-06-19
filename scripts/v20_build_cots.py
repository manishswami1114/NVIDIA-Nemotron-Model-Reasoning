#!/usr/bin/env python3
"""V20 orchestrator.

Steps:
  1. Load train.csv.
  2. For each V19 record: extract boxed answer + query signature;
     find the matching train.csv prompt; swap user content.
     If matching fails / is ambiguous: keep V19 user prompt (no failure).
  3. Hard categories (cryptarithm_deduce, cryptarithm_guess,
     equation_numeric_guess): write each record THREE times.
  4. Other 6 categories: write each record ONCE.
  5. Output to all_categorical_splits_v20/.
"""
import json, re, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_loader import load_all

BASE = Path(__file__).resolve().parent.parent
V19 = BASE / "all_categorical_splits_v19"
V20 = BASE / "all_categorical_splits_v20"

HARD_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
}

ALL_FILES = [
    "train_cot_bit_manipulation.jsonl",
    "train_cot_cipher.jsonl",
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_numeral.jsonl",
    "train_cot_unit_conversion.jsonl",
]


# ============================================================
# Helpers
# ============================================================

def _extract_boxed(text):
    if text is None: return None
    matches = re.findall(r'\\boxed\{(.*?)\}\s*$', text)
    return matches[-1] if matches else None


def _category_of_prompt(prompt: str) -> str:
    if 'bit manipulation rule' in prompt: return 'bit_manipulation'
    if 'encryption rules' in prompt: return 'cipher'
    if 'gravitational constant' in prompt: return 'gravity'
    if 'numeral system' in prompt: return 'numeral'
    if 'unit conversion' in prompt: return 'unit_conversion'
    if 'secret set of transformation rules' in prompt:
        for line in prompt.split('\n'):
            line = line.strip()
            if ' = ' not in line or 'determine' in line.lower(): continue
            left = line.split(' = ', 1)[0].strip()
            if len(left) == 5 and not any(c.isdigit() for c in left): return 'cryptarithm'
            if any(c.isdigit() for c in left): return 'equation_numeric'
        return 'equation_numeric'
    return 'unknown'


def _query_signature(prompt: str, category: str) -> str:
    """Extract a string from the prompt that uniquely identifies the puzzle's
    query. Used to disambiguate when multiple train.csv rows have the same
    answer."""
    if category == 'bit_manipulation':
        m = re.search(r'determine the output for:\s*([01]{8})', prompt, re.IGNORECASE)
        if m: return m.group(1)
    if category == 'cipher':
        m = re.search(r'decrypt the following text:\s*([a-z ]+)', prompt, re.IGNORECASE)
        if m: return m.group(1).strip()
    if category == 'cryptarithm':
        m = re.search(r'determine the result for:\s*(\S{5})', prompt, re.IGNORECASE)
        if m: return m.group(1)
    if category == 'equation_numeric':
        m = re.search(r'determine the result for:\s*(\S{5})', prompt, re.IGNORECASE)
        if m: return m.group(1)
    if category == 'gravity':
        m = re.search(r'determine the distance.*?t\s*=\s*(-?\d+\.?\d*)', prompt, re.IGNORECASE | re.DOTALL)
        if m: return m.group(1)
        m = re.search(r't\s*=\s*(-?\d+\.?\d*)\s*\.\s*$', prompt, re.MULTILINE)
        if m: return m.group(1)
    if category == 'unit_conversion':
        lines = [ln.strip() for ln in prompt.split('\n') if ln.strip()]
        for ln in lines[::-1]:
            if 'convert' in ln.lower():
                m = re.search(r'(-?\d+\.?\d*)', ln)
                if m: return m.group(1)
    if category == 'numeral':
        lines = [ln.strip() for ln in prompt.split('\n') if ln.strip()]
        for ln in lines[::-1]:
            if 'convert' in ln.lower() or 'determine' in ln.lower():
                m = re.search(r'(-?\d+)', ln)
                if m: return m.group(1)
    return ''


# ============================================================
# Main
# ============================================================

def main():
    V20.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: load train.csv + index by (category, answer, query_sig)")
    print("=" * 70)
    train_rows = load_all(BASE / 'data/raw/train.csv')
    print(f"  loaded {len(train_rows)} train.csv rows")

    # Index: (category, answer, query_signature) -> [prompts]
    train_index = defaultdict(list)
    for r in train_rows:
        cat = _category_of_prompt(r['prompt'])
        sig = _query_signature(r['prompt'], cat)
        train_index[(cat, r['answer'], sig)].append(r['prompt'])
    # Also by (category, answer) for fallback when no signature available
    train_index_loose = defaultdict(list)
    for r in train_rows:
        cat = _category_of_prompt(r['prompt'])
        train_index_loose[(cat, r['answer'])].append(r['prompt'])
    print(f"  indexed {len(train_index)} unique (category, answer, signature) keys")

    print()
    print("=" * 70)
    print("STEP 2: rewrite each V19 record + upsample Hard 3x")
    print("=" * 70)

    cat_short = {
        'bit_manipulation': 'bit_manipulation',
        'cipher': 'cipher',
        'cryptarithm': 'cryptarithm',
        'equation_numeric': 'equation_numeric',
        'gravity': 'gravity',
        'numeral': 'numeral',
        'unit_conversion': 'unit_conversion',
    }

    summary = []
    swap_ok = swap_ambig = swap_none = 0
    for fn in ALL_FILES:
        src = V19 / fn
        dst = V20 / fn
        if not src.exists():
            print(f"  [!] missing {fn}")
            continue
        is_hard = fn in HARD_FILES
        # category key for matching
        if 'cryptarithm' in fn:
            cat_key = 'cryptarithm'
        elif 'equation_numeric' in fn:
            cat_key = 'equation_numeric'
        else:
            cat_key = fn.replace('train_cot_', '').replace('.jsonl', '')

        written = 0
        with open(dst, 'w') as out_f:
            for line in open(src):
                rec = json.loads(line)
                old_prompt = rec['messages'][0]['content']
                asst = rec['messages'][-1]['content']
                boxed = _extract_boxed(asst)
                new_prompt = old_prompt
                if boxed is not None:
                    sig = _query_signature(old_prompt, cat_key)
                    # Strict match first
                    candidates = train_index.get((cat_key, boxed, sig), [])
                    if len(candidates) == 1:
                        new_prompt = candidates[0]
                        swap_ok += 1
                    elif len(candidates) > 1:
                        # ambiguous, prefer prompt that contains the query signature substring
                        new_prompt = candidates[0]
                        swap_ambig += 1
                    else:
                        # Loose match by (cat, answer) only
                        loose = train_index_loose.get((cat_key, boxed), [])
                        # filter loose by signature substring presence
                        if sig:
                            filtered = [p for p in loose if sig in p]
                            if len(filtered) == 1:
                                new_prompt = filtered[0]; swap_ok += 1
                            elif len(filtered) > 1:
                                new_prompt = filtered[0]; swap_ambig += 1
                            else:
                                swap_none += 1  # keep V19 prompt
                        else:
                            if len(loose) == 1:
                                new_prompt = loose[0]; swap_ok += 1
                            else:
                                swap_none += 1
                else:
                    swap_none += 1
                rec['messages'][0]['content'] = new_prompt

                n_copies = 3 if is_hard else 1
                for _ in range(n_copies):
                    out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    written += 1
        summary.append((fn, written, is_hard))
        print(f"  {fn:<45} {written:>5} records {'(3x upsampled)' if is_hard else ''}")

    print()
    print("=" * 70)
    print("STEP 3: PROMPT SWAP STATS")
    print("=" * 70)
    total_records_before = sum(1 for fn in ALL_FILES for _ in open(V19 / fn))
    print(f"  total V19 records audited: {total_records_before}")
    print(f"  unique train.csv match:    {swap_ok}")
    print(f"  ambiguous match (took 1st):{swap_ambig}")
    print(f"  no match (kept V19 prompt):{swap_none}")
    pct = (swap_ok + swap_ambig) / max(total_records_before, 1) * 100
    print(f"  prompt swap success rate:  {pct:.1f}%")

    print()
    print("=" * 70)
    print("V20 SUMMARY")
    print("=" * 70)
    total = 0
    for fn, n, hard in summary:
        flag = '  (HARD 3x)' if hard else ''
        total += n
        print(f"  {fn:<45} {n:>5} records{flag}")
    print(f"  {'TOTAL':<45} {total:>5}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {time.time()-t0:.1f}s")
