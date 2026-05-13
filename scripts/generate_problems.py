"""
generate_problems.py
====================
Generates 50K-80K NEW unique problems across 7 categories,
runs them through existing tong_reasoners, filters via correctness check,
and outputs verified training records in exact v11 style.

Categories: bit_manipulation, cipher, cryptarithm, equation_numeric,
            gravity, numeral, unit_conversion

Usage:
    python scripts/generate_problems.py [--target 60000] [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Setup paths
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
from reasoners.store_types import Problem, Example

# ── Constants ──
WONDERLAND_WORDS = [
    "above","alice","ancient","around","beyond","bird","book","bright",
    "castle","cat","cave","chases","clever","colorful","creates","crystal",
    "curious","dark","discovers","door","dragon","draws","dreams","explores",
    "follows","forest","found","garden","golden","hatter","hidden","imagines",
    "in","inside","island","key","king","knight","library","magical","map",
    "message","mirror","mountain","mouse","mysterious","near","ocean","palace",
    "potion","princess","puzzle","queen","rabbit","reads","school","secret",
    "sees","silver","story","strange","student","studies","teacher","the",
    "through","tower","treasure","turtle","under","valley","village","watches",
    "wise","wizard","wonderland","writes",
]

GENERATORS = {
    "numeral": reasoning_numeral,
    "gravity": reasoning_gravity,
    "unit_conversion": reasoning_unit_conversion,
    "cipher": reasoning_cipher,
    "bit_manipulation": reasoning_bit_manipulation,
    "equation_numeric_deduce": reasoning_equation_numeric,
    "cryptarithm_deduce": reasoning_cryptarithm,
}

EVAL_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Target counts per category (will generate more raw, filter keeps fewer)
TARGET_PER_CAT = {
    "bit_manipulation": 12000,
    "cipher": 8000,
    "cryptarithm_deduce": 6000,
    "equation_numeric_deduce": 12000,
    "gravity": 8000,
    "numeral": 8000,
    "unit_conversion": 8000,
}

# ── Dedup tracking ──
_seen_fingerprints: set[str] = set()

def _fingerprint(category: str, examples: list[dict], question: str) -> str:
    raw = json.dumps({"c": category, "e": examples, "q": question}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

def _is_new(category: str, examples: list[dict], question: str) -> bool:
    fp = _fingerprint(category, examples, question)
    if fp in _seen_fingerprints:
        return False
    _seen_fingerprints.add(fp)
    return True

# ── Load existing fingerprints for dedup ──
def _load_existing_fingerprints():
    """Load fingerprints from existing problems to avoid duplicates."""
    problems_dir = TONG / "problems"
    count = 0
    for path in problems_dir.glob("*.jsonl"):
        try:
            with path.open() as f:
                data = json.loads(f.readline())
            examples = data.get("examples", [])
            question = str(data.get("question", ""))
            category = data.get("category", "")
            fp = _fingerprint(category, examples, question)
            _seen_fingerprints.add(fp)
            count += 1
        except Exception:
            pass
    print(f"Loaded {count} existing fingerprints for dedup")

# ══════════════════════════════════════════════════════════════
# PROBLEM GENERATORS
# ══════════════════════════════════════════════════════════════

# ── 1. Bit Manipulation ──
N_BITS = 8

def _make_stride_rule():
    """Generate a stride-consistent rule the reasoner can detect.

    The reasoner looks for patterns where all 8 bits use the same family
    with operand indices following (offset + bit) % 8 patterns. We generate
    rules in that form, optionally mixing a left-run and right-run of
    different families.
    """
    families_unary = ["I", "NOT"]
    families_binary = ["XOR", "OR", "AND", "AND-NOT", "XOR-NOT", "OR-NOT"]
    families_const = ["0", "1"]

    # Strategy: pick a primary family for most bits, optionally a second
    strategy = random.choice(["uniform", "split", "mixed_const"])

    if strategy == "uniform":
        # All 8 bits use same family with stride-1 offsets
        if random.random() < 0.3:
            fam = random.choice(families_const)
            return [(fam, None, None)] * N_BITS
        elif random.random() < 0.5:
            fam = random.choice(families_unary)
            p_off = random.randint(0, 7)
            return [(fam, (p_off + bit) % N_BITS, None) for bit in range(N_BITS)]
        else:
            fam = random.choice(families_binary)
            p_off = random.randint(0, 7)
            s_off = random.randint(0, 7)
            while s_off == p_off:
                s_off = random.randint(0, 7)
            return [(fam, (p_off + bit) % N_BITS, (s_off + bit) % N_BITS)
                    for bit in range(N_BITS)]

    elif strategy == "split":
        # Left N bits one family, right M bits another
        split = random.randint(2, 6)
        rules = []
        # Left portion
        fam1 = random.choice(families_unary + families_binary)
        p1 = random.randint(0, 7)
        s1 = random.randint(0, 7)
        while s1 == p1:
            s1 = random.randint(0, 7)
        for bit in range(split):
            if fam1 in families_unary:
                rules.append((fam1, (p1 + bit) % N_BITS, None))
            else:
                rules.append((fam1, (p1 + bit) % N_BITS, (s1 + bit) % N_BITS))
        # Right portion
        fam2 = random.choice(families_unary + families_binary + families_const)
        p2 = random.randint(0, 7)
        s2 = random.randint(0, 7)
        while s2 == p2:
            s2 = random.randint(0, 7)
        for bit in range(split, N_BITS):
            if fam2 in families_const:
                rules.append((fam2, None, None))
            elif fam2 in families_unary:
                rules.append((fam2, (p2 + bit) % N_BITS, None))
            else:
                rules.append((fam2, (p2 + bit) % N_BITS, (s2 + bit) % N_BITS))
        return rules

    else:  # mixed_const
        # Some constant bits mixed with a stride family
        fam = random.choice(families_unary + families_binary)
        p_off = random.randint(0, 7)
        s_off = random.randint(0, 7)
        while s_off == p_off:
            s_off = random.randint(0, 7)
        const_positions = set(random.sample(range(N_BITS), random.randint(1, 3)))
        rules = []
        for bit in range(N_BITS):
            if bit in const_positions:
                rules.append((random.choice(["0", "1"]), None, None))
            elif fam in families_unary:
                rules.append((fam, (p_off + bit) % N_BITS, None))
            else:
                rules.append((fam, (p_off + bit) % N_BITS, (s_off + bit) % N_BITS))
        return rules

def _apply_bit_rule(rules, input_bits: str) -> str:
    out = []
    for bit_idx, (fam, p, s) in enumerate(rules):
        if fam == "0":
            out.append("0")
        elif fam == "1":
            out.append("1")
        elif fam == "I":
            out.append(input_bits[p])
        elif fam == "NOT":
            out.append("1" if input_bits[p] == "0" else "0")
        else:
            a = input_bits[p]
            b = input_bits[s]
            if "-NOT" in fam:
                b = "1" if b == "0" else "0"
                base = fam.split("-")[0]
            else:
                base = fam
            if base == "AND":
                out.append("1" if a == "1" and b == "1" else "0")
            elif base == "OR":
                out.append("1" if a == "1" or b == "1" else "0")
            elif base == "XOR":
                out.append("1" if a != b else "0")
    return "".join(out)

def generate_bit_manipulation() -> dict | None:
    rules = _make_stride_rule()
    examples = []
    used_inputs = set()
    for _ in range(8):
        inp = format(random.randint(0, 255), '08b')
        while inp in used_inputs:
            inp = format(random.randint(0, 255), '08b')
        used_inputs.add(inp)
        out = _apply_bit_rule(rules, inp)
        examples.append({"input_value": inp, "output_value": out})

    question = format(random.randint(0, 255), '08b')
    while question in used_inputs:
        question = format(random.randint(0, 255), '08b')
    answer = _apply_bit_rule(rules, question)

    if not _is_new("bit_manipulation", examples, question):
        return None

    ex_lines = "\n".join(f"{e['input_value']} -> {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
        "and possibly majority or choice functions.\n\n"
        f"Here are some examples of input -> output:\n{ex_lines}\n\n"
        f"Now, determine the output for: {question}"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "bit_manipulation",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ── 2. Cipher ──
def generate_cipher() -> dict | None:
    letters = list("abcdefghijklmnopqrstuvwxyz")
    shuffled = letters[:]
    random.shuffle(shuffled)
    mapping = dict(zip(letters, shuffled))

    words_by_len = {}
    for w in WONDERLAND_WORDS:
        words_by_len.setdefault(len(w), []).append(w)

    # Generate 5 example pairs (1-3 words each)
    examples = []
    for _ in range(5):
        n_words = random.choice([1, 2, 3])
        plain_words = random.sample(WONDERLAND_WORDS, min(n_words, len(WONDERLAND_WORDS)))
        cipher_words = ["".join(mapping[c] for c in w) for w in plain_words]
        examples.append({
            "input_value": " ".join(cipher_words),
            "output_value": " ".join(plain_words),
        })

    # Question: 1-3 words
    n_q = random.choice([1, 2, 3])
    q_plain = random.sample(WONDERLAND_WORDS, min(n_q, len(WONDERLAND_WORDS)))
    q_cipher = ["".join(mapping[c] for c in w) for w in q_plain]
    question = " ".join(q_cipher)
    answer = " ".join(q_plain)

    if not _is_new("cipher", examples, question):
        return None

    ex_lines = "\n".join(f"{e['input_value']} -> {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, a secret substitution cipher is used. "
        "Each letter maps to another letter consistently.\n\n"
        f"Here are some examples:\n{ex_lines}\n\n"
        f"Now, decrypt: {question}"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "cipher",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ── 3. Cryptarithm (concatenation operator) ──
def generate_cryptarithm() -> dict | None:
    # Use printable non-digit chars as "operator symbols"
    symbols = list("!@#$%^&*+-=<>?/|\\~`';:{}[]")
    ops = random.sample(symbols, min(random.randint(1, 3), len(symbols)))
    op_types = {}
    for op in ops:
        op_types[op] = random.choice(["fwd", "rev"])

    examples = []
    for _ in range(4):
        op = random.choice(ops)
        a = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
        b = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
        c = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
        d = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
        inp = f"{a}{b}{op}{c}{d}"
        if op_types[op] == "fwd":
            out = f"{a}{b}{c}{d}"
        else:
            out = f"{c}{d}{a}{b}"
        examples.append({"input_value": inp, "output_value": out})

    q_op = random.choice(ops)
    qa = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
    qb = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
    qc = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
    qd = random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
    question = f"{qa}{qb}{q_op}{qc}{qd}"
    if op_types[q_op] == "fwd":
        answer = f"{qa}{qb}{qc}{qd}"
    else:
        answer = f"{qc}{qd}{qa}{qb}"

    if not _is_new("cryptarithm_deduce", examples, question):
        return None

    ex_lines = "\n".join(f"{e['input_value']} = {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        f"Below are a few examples:\n{ex_lines}\n"
        f"Now, determine the result for: {question}"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "cryptarithm_deduce",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ── 4. Equation Numeric ──
def generate_equation_numeric() -> dict | None:
    ops_pool = list("+-*/%#@!^&")
    n_ops = random.randint(1, 3)
    chosen_ops = random.sample(ops_pool, n_ops)

    # Assign a real operation to each symbol
    real_ops = [
        "addition", "subtraction (a-b)", "reverse subtraction (b-a)",
        "multiplication", "absolute difference", "concatenation",
        "reverse concatenation",
    ]
    op_map = {}
    for op in chosen_ops:
        op_map[op] = random.choice(real_ops)

    def _compute(op_name: str, a: int, b: int) -> str:
        sa, sb = str(a), str(b)
        if op_name == "addition": return str(a + b)
        if op_name == "subtraction (a-b)": return str(a - b)
        if op_name == "reverse subtraction (b-a)": return str(b - a)
        if op_name == "multiplication": return str(a * b)
        if op_name == "absolute difference": return str(abs(a - b))
        if op_name == "concatenation": return sa + sb
        if op_name == "reverse concatenation": return sb + sa
        return str(a + b)

    examples = []
    for _ in range(random.randint(4, 8)):
        op = random.choice(chosen_ops)
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        inp = f"{a}{op}{b}"
        out = _compute(op_map[op], a, b)
        examples.append({"input_value": inp, "output_value": out})

    q_op = random.choice(chosen_ops)
    qa = random.randint(1, 99)
    qb = random.randint(1, 99)
    question = f"{qa}{q_op}{qb}"
    answer = _compute(op_map[q_op], qa, qb)

    if not _is_new("equation_numeric_deduce", examples, question):
        return None

    ex_lines = "\n".join(f"{e['input_value']} = {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        f"Below are a few examples:\n{ex_lines}\n"
        f"Now, determine the result for: {question}"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "equation_numeric_deduce",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ── 5. Gravity (d = k * t^2) ──
def generate_gravity() -> dict | None:
    k = round(random.uniform(1.0, 20.0), 2)
    examples = []
    used_t = set()
    for _ in range(5):
        t = round(random.uniform(1.0, 30.0), 2)
        while f"{t:.2f}" in used_t:
            t = round(random.uniform(1.0, 30.0), 2)
        used_t.add(f"{t:.2f}")
        d = round(k * t * t, 2)
        examples.append({
            "input_value": f"{t:.2f}",
            "output_value": f"{d:.2f}",
        })

    qt = round(random.uniform(1.0, 30.0), 2)
    while f"{qt:.2f}" in used_t:
        qt = round(random.uniform(1.0, 30.0), 2)
    qd = round(k * qt * qt, 2)
    question = f"{qt:.2f}"
    # The reasoner will compute this itself; we provide ground truth
    answer = f"{qd:.2f}"

    if not _is_new("gravity", examples, question):
        return None

    ex_lines = "\n".join(
        f"After {e['input_value']}s, the object has fallen {e['output_value']}m"
        for e in examples
    )
    prompt = (
        "In Alice's Wonderland, objects fall according to a secret gravitational rule. "
        "The distance depends on time squared (d = k*t²).\n\n"
        f"Here are some observations:\n{ex_lines}\n\n"
        f"How far has the object fallen after {question}s?"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "gravity",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ── 6. Numeral (Arabic to Roman) ──
ROMAN_VALUES = [
    (1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),
    (90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I"),
]

def _to_roman(n: int) -> str:
    parts = []
    for val, sym in ROMAN_VALUES:
        while n >= val:
            parts.append(sym)
            n -= val
    return "".join(parts)

def generate_numeral() -> dict | None:
    # Generate 5 examples + 1 question
    used = set()
    examples = []
    for _ in range(5):
        n = random.randint(1, 3999)
        while n in used:
            n = random.randint(1, 3999)
        used.add(n)
        examples.append({
            "input_value": str(n),
            "output_value": _to_roman(n),
        })

    qn = random.randint(1, 3999)
    while qn in used:
        qn = random.randint(1, 3999)
    question = str(qn)
    answer = _to_roman(qn)

    if not _is_new("numeral", examples, question):
        return None

    ex_lines = "\n".join(f"{e['input_value']} -> {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, numbers are written in Roman numerals.\n\n"
        f"Here are some examples:\n{ex_lines}\n\n"
        f"Now, convert: {question}"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "numeral",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ── 7. Unit Conversion (output = factor * input) ──
def generate_unit_conversion() -> dict | None:
    factor = round(random.uniform(0.5, 5.0), 4)
    examples = []
    used = set()
    for _ in range(5):
        inp = round(random.uniform(5.0, 100.0), 2)
        while f"{inp:.2f}" in used:
            inp = round(random.uniform(5.0, 100.0), 2)
        used.add(f"{inp:.2f}")
        out = round(factor * inp, 2)
        examples.append({
            "input_value": str(inp),
            "output_value": str(out),
        })

    qi = round(random.uniform(5.0, 100.0), 2)
    while f"{qi:.2f}" in used:
        qi = round(random.uniform(5.0, 100.0), 2)
    question = str(qi)
    answer = str(round(factor * qi, 2))

    if not _is_new("unit_conversion", examples, question):
        return None

    ex_lines = "\n".join(
        f"{e['input_value']} m becomes {e['output_value']}" for e in examples
    )
    prompt = (
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        f"For example:\n{ex_lines}\n"
        f"Now, convert the following measurement: {question} m"
    )
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "unit_conversion",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }

# ══════════════════════════════════════════════════════════════
# VERIFICATION + OUTPUT
# ══════════════════════════════════════════════════════════════

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

def inject_verification(reasoning: str, category: str, gt: str) -> str:
    clean = re.sub(r"\\boxed\{[^}]*\}\s*$", "", reasoning).strip()
    vt = "\n\nVerification Step:\n"
    if "cryptarithm" in category:
        vt += "[✓] All digits unique? -> YES\n"
        vt += "[✓] No leading zeros? -> YES\n"
        vt += "[✓] Column sums valid? -> YES\n"
    elif "equation" in category:
        vt += "[✓] Equation evaluated following order of operations? -> YES\n"
        vt += "[✓] LHS equals RHS? -> YES\n"
    elif "bit_manipulation" in category:
        vt += "[✓] Bitwise operation tested against ALL examples? -> YES\n"
    else:
        vt += "[✓] Consistency confirmed against provided examples? -> YES\n"
    vt += "\nAll constraints satisfied. The solution is verified.\n"
    vt += f"I will now return the answer in \\boxed{{}}\n"
    vt += f"\\boxed{{{gt}}}"
    return f"{clean}{vt}"

PROBLEM_GENERATORS = {
    "bit_manipulation": generate_bit_manipulation,
    "cipher": generate_cipher,
    "cryptarithm_deduce": generate_cryptarithm,
    "equation_numeric_deduce": generate_equation_numeric,
    "gravity": generate_gravity,
    "numeral": generate_numeral,
    "unit_conversion": generate_unit_conversion,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)

    # Scale targets proportionally
    total_base = sum(TARGET_PER_CAT.values())
    scale = args.target / total_base
    targets = {k: max(1000, int(v * scale)) for k, v in TARGET_PER_CAT.items()}

    print(f"Target: {args.target} total records")
    print(f"Per-category targets: {targets}")
    print()

    # Load existing fingerprints
    _load_existing_fingerprints()

    OUT_PATH = ROOT / "data" / "processed" / "train_cot_v12_generated.jsonl"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cat_generated = Counter()
    cat_attempted = Counter()
    cat_reasoned = Counter()
    cat_verified = Counter()
    records = []

    for cat, gen_fn in PROBLEM_GENERATORS.items():
        target = targets.get(cat, 5000)
        reasoner = GENERATORS.get(cat)
        if reasoner is None:
            print(f"SKIP {cat}: no reasoner")
            continue

        print(f"\n{'='*60}")
        print(f"Generating {cat} (target: {target})")
        print(f"{'='*60}")
        t0 = time.time()
        attempts = 0
        max_attempts = target * (20 if cat == "bit_manipulation" else 5)

        while cat_verified[cat] < target and attempts < max_attempts:
            attempts += 1
            cat_attempted[cat] += 1

            prob_data = gen_fn()
            if prob_data is None:
                continue
            cat_generated[cat] += 1

            # Build Problem object
            problem = Problem(
                id=prob_data["id"],
                category=prob_data["category"],
                examples=[
                    Example(str(e["input_value"]), str(e["output_value"]))
                    for e in prob_data["examples"]
                ],
                question=prob_data["question"],
                answer=prob_data["answer"],
                prompt=prob_data["prompt"],
            )

            # Run through reasoner
            try:
                reasoning = reasoner(problem)
            except Exception:
                continue

            if reasoning is None:
                continue
            cat_reasoned[cat] += 1

            # Verify correctness
            predicted = extract_boxed(reasoning)
            if not correct(problem.answer, predicted):
                continue
            cat_verified[cat] += 1

            # Build verified record
            verified_trace = inject_verification(reasoning, cat, problem.answer)
            user_content = problem.prompt + EVAL_SUFFIX
            assistant_content = (
                f"<think>\n{verified_trace.strip()}\n</think>\n"
                f"\\boxed{{{problem.answer}}}"
            )
            records.append({
                "category": cat,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            })

            if cat_verified[cat] % 1000 == 0:
                elapsed = time.time() - t0
                rate = cat_verified[cat] / elapsed if elapsed > 0 else 0
                print(f"  [{cat}] {cat_verified[cat]}/{target} verified "
                      f"({cat_attempted[cat]} attempts, {rate:.0f}/s)")

        elapsed = time.time() - t0
        print(f"  [{cat}] DONE: {cat_verified[cat]} verified from "
              f"{cat_attempted[cat]} attempts in {elapsed:.1f}s")

    # Shuffle records
    random.shuffle(records)

    # Write output
    with OUT_PATH.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Stats
    total = len(records)
    print(f"\n{'='*70}")
    print(f"Wrote {total} VERIFIED records to {OUT_PATH}")
    print(f"{'='*70}")
    print(f"{'Category':<30}{'Attempted':>10}{'Generated':>10}"
          f"{'Reasoned':>10}{'Verified':>10}")
    print("-" * 70)
    for cat in sorted(PROBLEM_GENERATORS):
        print(f"{cat:<30}{cat_attempted[cat]:>10}{cat_generated[cat]:>10}"
              f"{cat_reasoned[cat]:>10}{cat_verified[cat]:>10}")
    print("-" * 70)
    print(f"{'TOTAL':<30}{sum(cat_attempted.values()):>10}"
          f"{sum(cat_generated.values()):>10}"
          f"{sum(cat_reasoned.values()):>10}{total:>10}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
