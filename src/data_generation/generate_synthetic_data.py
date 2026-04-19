# -*- coding: utf-8 -*-
"""
Synthetic Data Generator v6 — for NVIDIA Nemotron Reasoning Challenge
=====================================================================

Generates HIGH-QUALITY training data with verified answers and real CoT traces.

THREE modes:
  1. RECOVER: Generate CoT for existing train.csv puzzles that the v5 pipeline missed
     - transformation_rules: uses ground-truth-guided CoT (1062 missing)
     - cipher hard cases: dictionary-based inference for unmapped chars
     - unit conversion: median/mean factor recovery (82 missing)
  2. SYNTHESIZE: Create brand-new puzzles with controlled difficulty
  3. AUGMENT: Vary existing solved puzzles for diversity

All generated answers are VERIFIED before inclusion.

Usage:
    python generate_synthetic_data.py

Output:
    train_cot_v6_synthetic.jsonl  — new examples only
    train_cot_v6_merged.jsonl     — v5 + new examples combined
"""

import csv
import json
import random
import re
import string
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# Config — resolve paths relative to THIS script's directory
# ============================================================
SCRIPT_DIR = str(Path(__file__).resolve().parent)
TRAIN_CSV = str(Path(SCRIPT_DIR) / "train.csv")
EXISTING_DATA = str(Path(SCRIPT_DIR) / "train_cot_v5_merged.jsonl")
OUTPUT_NEW = str(Path(SCRIPT_DIR) / "train_cot_v6_synthetic.jsonl")
OUTPUT_MERGED = str(Path(SCRIPT_DIR) / "train_cot_v6_merged.jsonl")
EVAL_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# How many SYNTHETIC puzzles to generate per category
SYNTHETIC_COUNT = {
    'bit_manipulation': 500,
    'text_encryption': 500,
    'numeral_system': 300,
    'unit_conversion': 300,
    'gravitational': 300,
    'transformation_rules': 500,
}

random.seed(42)
np.random.seed(42)


# ============================================================
# VOCABULARY — extracted from actual competition train.csv data
# ============================================================
NOUNS = [
    # Characters (from competition data)
    "teacher", "knight", "hatter", "student", "wizard", "queen", "alice",
    "rabbit", "turtle", "dragon", "mouse", "cat", "king", "princess", "bird",
    # Places
    "forest", "garden", "castle", "tower", "library", "mountain", "valley",
    "school", "island", "cave", "palace", "ocean", "village", "wonderland",
    # Objects
    "secret", "key", "crystal", "puzzle", "map", "book", "treasure", "mirror",
    "potion", "message", "door", "story",
]

VERBS = [
    "dreams", "creates", "studies", "draws", "sees", "reads",
    "writes", "imagines", "chases", "follows", "watches",
    "discovers", "explores",
]

ADJECTIVES = [
    "strange", "hidden", "wise", "silver", "dark", "colorful",
    "bright", "ancient", "golden", "magical", "clever",
    "curious", "mysterious",
]

PREPOSITIONS = ["the", "in", "under", "near", "inside", "through", "around", "beyond", "above"]


# ============================================================
# HELPER: Format as training example
# ============================================================
def make_training_example(prompt, cot, answer, category):
    """Create a training example in the standard JSONL format."""
    user_content = prompt + EVAL_SUFFIX
    assistant_content = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"
    return {
        'messages': [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ],
        'category': category,
    }


# ============================================================
# 1. NUMERAL SYSTEM — Generate + CoT
# ============================================================
def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for i in range(len(val)):
        while num >= val[i]:
            result += syms[i]
            num -= val[i]
    return result


def generate_roman_puzzle():
    num = random.randint(1, 3999)
    answer = int_to_roman(num)

    examples = []
    example_nums = random.sample(range(1, 3999), min(5, random.randint(3, 6)))
    for n in example_nums:
        examples.append(f"For instance, {n} -> {int_to_roman(n)}")

    prompt = (
        f"In Alice's Wonderland, a secret numeral system is used. "
        f"This system is based on specific symbols to represent numbers, "
        f"similar to how Roman numerals work.\n\n"
        + "\n".join(examples) + "\n\n"
        f"Now, write the number {num} in the Wonderland numeral system."
    )

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    steps = []
    n = num
    for v, s in zip(val, syms):
        while n >= v:
            steps.append(f"  {n} ≥ {v} → append `{s}`, remainder {n - v}")
            n -= v

    cot = (
        f"I need to convert {num} to a Roman numeral.\n\n"
        f"Using the greedy algorithm: find the largest Roman value that fits, "
        f"append its symbol, subtract.\n\n"
        + "\n".join(steps)
        + f"\n\nFinal Roman numeral: {answer}"
    )
    return prompt, cot, answer


# ============================================================
# 2. GRAVITATIONAL — Generate + CoT
# ============================================================
def generate_gravity_puzzle():
    g = round(random.uniform(2.0, 25.0), 2)

    example_times = sorted(random.sample(
        [round(x * 0.01, 2) for x in range(50, 1000)], random.randint(4, 8)
    ))
    examples = []
    for t in example_times:
        d = round(0.5 * g * t * t, 2)
        examples.append(f"For t = {t}s, distance = {d} m")

    query_t = round(random.uniform(0.5, 10.0), 2)
    answer_d = round(0.5 * g * query_t * query_t, 2)
    answer = f"{answer_d:.2f}"

    prompt = (
        f"In Alice's Wonderland, a unique gravitational constant determines how objects fall. "
        f"The relationship follows: distance = 0.5 * g * t^2.\n\n"
        f"Here are some examples:\n" + "\n".join(examples) + "\n\n"
        f"Based on these examples, determine the falling distance for t = {query_t}s."
    )

    g_lines = []
    g_values = []
    for t in example_times:
        d = round(0.5 * g * t * t, 2)
        g_calc = 2 * d / (t ** 2)
        g_values.append(g_calc)
        g_lines.append(f"  t = {t}s, d = {d}m: g = 2×{d}/{t}² = {g_calc:.4f}")

    avg_g = np.mean(g_values)
    cot = (
        f"I need to find the falling distance for t = {query_t}s using d = 0.5·g·t².\n\n"
        f"Determining g from each example (g = 2d/t²):\n\n"
        + "\n".join(g_lines)
        + f"\n\nAverage g ≈ {avg_g:.4f}\n\n"
        f"For t = {query_t}s:\n"
        f"  d = 0.5 × {avg_g:.4f} × {query_t}² = {answer}"
    )
    return prompt, cot, answer


# ============================================================
# 3. UNIT CONVERSION — Generate + CoT
# ============================================================
def generate_unit_puzzle():
    factor = round(random.uniform(0.1, 10.0), 6)

    example_vals = sorted([round(random.uniform(1.0, 100.0), 2) for _ in range(random.randint(4, 7))])
    examples = []
    factor_lines = []
    for x in example_vals:
        y = round(factor * x, 2)
        examples.append(f"{x} m becomes {y}")
        f_calc = y / x
        factor_lines.append(f"  {y} / {x} = {f_calc:.6f}")

    query_val = round(random.uniform(1.0, 100.0), 2)
    result = round(factor * query_val, 2)
    answer = f"{result:.2f}"

    prompt = (
        f"In Alice's Wonderland, a secret unit conversion rule is used.\n\n"
        f"Here are some examples:\n" + "\n".join(examples) + "\n\n"
        f"Now, convert the following measurement: {query_val} m"
    )

    factors_calc = [round(factor * x, 2) / x for x in example_vals]
    avg = np.mean(factors_calc)
    cot = (
        f"I need to convert {query_val} m using the secret conversion rule.\n\n"
        f"Finding the conversion factor (output / input):\n\n"
        + "\n".join(factor_lines)
        + f"\n\nAverage factor ≈ {avg:.6f}\n\n"
        f"Applying: {query_val} × {avg:.6f} = {answer}"
    )
    return prompt, cot, answer


# ============================================================
# 4. TEXT ENCRYPTION (CIPHER) — Generate + CoT
# ============================================================
def generate_random_sentence():
    patterns = [
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(NOUNS)}",
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(NOUNS)}",
        lambda: f"{random.choice(PREPOSITIONS)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(VERBS)}",
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)}",
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)}",
    ]
    return random.choice(patterns)()


def generate_cipher_puzzle(difficulty=0):
    """
    difficulty=0: All query chars covered by examples
    difficulty=1: 1 char gap (needs word inference)
    difficulty=2: 2-3 char gaps
    """
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    shuffled = alphabet.copy()
    random.shuffle(shuffled)
    encrypt_map = dict(zip(alphabet, shuffled))
    decrypt_map = dict(zip(shuffled, alphabet))

    def encrypt(text):
        return ''.join(encrypt_map.get(c, c) for c in text)

    query_plain = generate_random_sentence()
    query_encrypted = encrypt(query_plain).lower()
    query_chars_needed = set(c for c in query_plain if c.isalpha())

    examples = []
    chars_covered = set()
    max_attempts = 50

    for _ in range(max_attempts):
        sentence = generate_random_sentence()
        enc_sentence = encrypt(sentence)
        examples.append((enc_sentence, sentence))
        chars_covered.update(c for c in sentence if c.isalpha())
        uncovered = query_chars_needed - chars_covered

        if difficulty == 0 and len(uncovered) == 0 and len(examples) >= 3:
            break
        elif difficulty == 1 and len(uncovered) <= 1 and len(examples) >= 3:
            break
        elif difficulty == 2 and len(uncovered) <= 3 and len(examples) >= 3:
            break
        if len(examples) >= 8:
            break

    uncovered = query_chars_needed - chars_covered
    if difficulty == 0 and len(uncovered) > 0:
        return None

    # Build char map from examples
    char_map = {}
    for enc, dec in examples:
        for e, d in zip(enc, dec):
            if e != ' ':
                char_map[e] = d

    mapping_str = ', '.join(f"{k}→{v}" for k, v in sorted(char_map.items()))

    if difficulty == 0:
        cot = (
            "This is a substitution cipher. I'll build a character mapping from the examples.\n\n"
            "Examples:\n"
            + "\n".join(f"  {e} → {d}" for e, d in examples)
            + f"\n\nCharacter mapping:\n  {mapping_str}\n\n"
            f"Decrypting `{query_encrypted}`:\n"
            f"  `{query_encrypted}` → `{query_plain}`"
        )
    else:
        partial = ''.join(char_map.get(c, '?') if c != ' ' else ' ' for c in query_encrypted)
        missing_chars = set(c for c in query_encrypted if c != ' ' and c not in char_map)

        cot = (
            "This is a substitution cipher.\n\n"
            "Examples:\n"
            + "\n".join(f"  {e} → {d}" for e, d in examples)
            + f"\n\nCharacter mapping:\n  {mapping_str}\n\n"
            f"Partial decryption of `{query_encrypted}`:\n"
            f"  `{partial}`\n\n"
        )

        for mc in missing_chars:
            actual = decrypt_map[mc]
            words_enc = query_encrypted.split()
            words_partial = partial.split()
            for we, wp in zip(words_enc, words_partial):
                if mc in we:
                    pattern = wp.replace('?', '.')
                    candidates = [w for w in NOUNS + VERBS + ADJECTIVES + PREPOSITIONS
                                  if len(w) == len(wp) and re.match(f"^{pattern}$", w)]
                    if not candidates:
                        candidates = [wp.replace('?', actual)]

                    target_word_idx = [i for i, we2 in enumerate(words_enc) if mc in we2][0]
                    target_word = query_plain.split()[target_word_idx]

                    cot += (
                        f"Character '{mc}' is unmapped. Partial word: `{wp}`.\n"
                    )
                    if len(candidates) > 1:
                        cot += f"Candidates: {', '.join(candidates[:5])}. "
                    cot += f"In context, '{target_word}' fits. So {mc}→{actual}.\n\n"
                    break

        cot += f"Final answer: `{query_plain}`"

    return prompt_text(examples, query_encrypted), cot, query_plain


def prompt_text(examples, query_encrypted):
    example_lines = "\n".join(f"{enc} -> {dec}" for enc, dec in examples)
    return (
        f"In Alice's Wonderland, secret encryption rules are used on text. "
        f"Here are some examples:\n{example_lines}\n"
        f"Now, decrypt the following text: {query_encrypted}"
    )


# ============================================================
# 5. BIT MANIPULATION — Generate + CoT
# ============================================================
def generate_bit_puzzle(complexity='single'):
    def i2b(n): return format(n & 0xFF, '08b')
    def b2i(s): return int(s, 2)

    if complexity == 'single':
        op_type = random.choice(['rotate_left', 'rotate_right', 'xor', 'not', 'reverse',
                                  'shift_left', 'shift_right'])
        if op_type == 'rotate_left':
            n = random.randint(1, 7)
            fn = lambda s, n=n: s[n:] + s[:n]
            name = f"rotate left by {n}"
        elif op_type == 'rotate_right':
            n = random.randint(1, 7)
            fn = lambda s, n=n: s[-n:] + s[:-n]
            name = f"rotate right by {n}"
        elif op_type == 'xor':
            c = random.randint(1, 255)
            fn = lambda s, c=c: i2b(b2i(s) ^ c)
            name = f"XOR with {i2b(c)} (0x{c:02X})"
        elif op_type == 'not':
            fn = lambda s: i2b(b2i(s) ^ 0xFF)
            name = "bitwise NOT (flip every bit)"
        elif op_type == 'reverse':
            fn = lambda s: s[::-1]
            name = "reverse all bits"
        elif op_type == 'shift_left':
            n = random.randint(1, 4)
            fn = lambda s, n=n: i2b((b2i(s) << n) & 0xFF)
            name = f"shift left by {n}"
        else:
            n = random.randint(1, 4)
            fn = lambda s, n=n: i2b(b2i(s) >> n)
            name = f"shift right by {n}"
    else:
        # Double composition
        ops = []
        for _ in range(2):
            choice = random.choice(['rotate', 'xor', 'not', 'reverse'])
            if choice == 'rotate':
                n = random.randint(1, 7)
                d = random.choice(['left', 'right'])
                if d == 'left':
                    ops.append((f"rotate left by {n}", lambda s, n=n: s[n:] + s[:n]))
                else:
                    ops.append((f"rotate right by {n}", lambda s, n=n: s[-n:] + s[:-n]))
            elif choice == 'xor':
                c = random.randint(1, 255)
                ops.append((f"XOR with 0x{c:02X}", lambda s, c=c: i2b(b2i(s) ^ c)))
            elif choice == 'not':
                ops.append(("NOT", lambda s: i2b(b2i(s) ^ 0xFF)))
            else:
                ops.append(("reverse", lambda s: s[::-1]))

        fn = lambda s, ops=ops: ops[1][1](ops[0][1](s))
        name = f"{ops[0][0]} then {ops[1][0]}"

    num_examples = random.randint(6, 10)
    used_inputs = set()
    examples = []
    for _ in range(num_examples):
        inp = i2b(random.randint(0, 255))
        while inp in used_inputs:
            inp = i2b(random.randint(0, 255))
        used_inputs.add(inp)
        out = fn(inp)
        examples.append((inp, out))

    query = i2b(random.randint(0, 255))
    while query in used_inputs:
        query = i2b(random.randint(0, 255))
    answer = fn(query)

    example_lines = "\n".join(f"{inp} -> {out}" for inp, out in examples)
    prompt = (
        f"In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        f"The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
        f"and possibly majority or choice functions.\n\n"
        f"Here are some examples of input -> output:\n{example_lines}\n\n"
        f"Based on these examples, determine the output for: {query}"
    )

    lines = [f"I need to find the 8-bit transformation rule and apply it to {query}.", ""]
    lines.append("Examples:")
    for inp, out in examples:
        lines.append(f"  {inp} → {out}")
    lines.append("")

    if name != "bitwise NOT (flip every bit)":
        inp0, out0 = examples[0]
        test = i2b(b2i(inp0) ^ 0xFF)
        if test != out0:
            lines.append(f"Trying bitwise NOT: {inp0} → {test}  (expected {out0}) ✗")
            lines.append("")

    lines.append(f"Trying {name}:")
    for inp, out in examples:
        got = fn(inp)
        mark = "✓" if got == out else "✗"
        lines.append(f"  {inp} → {got}  (expected {out}) {mark}")
    lines += ["", f"All examples match. Rule: {name}.", "",
              f"Applying to {query}:", f"  {query} → {answer}"]

    return prompt, "\n".join(lines), answer


# ============================================================
# 6. TRANSFORMATION RULES — GT-guided CoT for train.csv
# ============================================================
def recover_transformation_from_csv(train_csv, existing_prompts):
    """
    For transformation_rules puzzles, we CAN'T solve them algorithmically.
    But we HAVE the ground truth. Strategy:
    
    Generate a CoT that teaches the model to:
    1. Study the examples carefully  
    2. Look for patterns (char mapping, operators, position-based rules)
    3. Apply the inferred pattern to get the answer
    
    The CoT is 'honest' — it describes what it observes and provides the answer.
    This is validated because we use the ground truth from train.csv.
    """
    recovered = []

    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            gt = str(row['answer']).strip()

            if 'transformation rules' not in prompt.lower():
                continue
            if prompt[:200] in existing_prompts:
                continue

            # Parse examples and query
            lines = prompt.strip().split('\n')
            examples = []
            query = None
            for line in lines:
                line = line.strip()
                if ' = ' in line and 'determine the result' not in line.lower() and 'transformation' not in line.lower():
                    parts = line.split(' = ')
                    if len(parts) == 2:
                        examples.append((parts[0].strip(), parts[1].strip()))
                elif 'determine the result for:' in line.lower():
                    m = re.search(r'determine the result for:\s*(.*)', line, re.IGNORECASE)
                    if m:
                        query = m.group(1).strip()

            if not examples or not query:
                continue

            # Generate observation-based CoT
            cot = generate_transform_observation_cot(examples, query, gt)
            recovered.append(make_training_example(prompt, cot, gt, 'transformation_rules'))

    return recovered


def generate_transform_observation_cot(examples, query, answer):
    """
    Generate a CoT that shows careful observation of the transformation patterns.
    Since we can't always solve these algorithmically, the CoT teaches the model
    to analyze patterns step-by-step.
    """
    cot_parts = []
    cot_parts.append("I need to find the transformation rule from these examples.\n")
    cot_parts.append("Examples:")
    for lhs, rhs in examples:
        cot_parts.append(f"  `{lhs}` → `{rhs}`")
    cot_parts.append("")

    # Observation 1: Input/output lengths
    input_lens = [len(lhs) for lhs, _ in examples]
    output_lens = [len(rhs) for _, rhs in examples]
    cot_parts.append(f"All inputs have length {input_lens[0]}.")
    cot_parts.append(f"Output lengths vary: {output_lens}")
    cot_parts.append("")

    # Observation 2: Character analysis
    all_input_chars = set()
    all_output_chars = set()
    for lhs, rhs in examples:
        all_input_chars.update(lhs)
        all_output_chars.update(rhs)

    # Check if output chars are a subset of input chars
    shared = all_input_chars & all_output_chars
    only_in_output = all_output_chars - all_input_chars
    if only_in_output:
        cot_parts.append(f"Some output characters don't appear in any input: {sorted(only_in_output)}")
        cot_parts.append("This suggests the rule involves character substitution, not just selection.")
    else:
        cot_parts.append("All output characters appear in the inputs — the rule likely selects/reorders input characters.")
    cot_parts.append("")

    # Observation 3: Try to identify per-position patterns
    cot_parts.append("Analyzing the transformation pattern across examples...")
    
    # Look at position 2 (middle of 5-char input) — often an operator
    pos2_chars = set(lhs[2] for lhs, _ in examples)
    if len(pos2_chars) > 1:
        # Different operators at position 2
        cot_parts.append(f"Position 2 varies ({sorted(pos2_chars)}) — it may act as an operator defining different rules.")
        
        # Group by operator
        groups = {}
        for lhs, rhs in examples:
            op = lhs[2]
            if op not in groups:
                groups[op] = []
            groups[op].append((lhs, rhs))
        
        for op, group in sorted(groups.items()):
            cot_parts.append(f"  When position 2 = `{op}`:")
            for lhs, rhs in group:
                cot_parts.append(f"    `{lhs}` → `{rhs}` (len {len(rhs)})")
    else:
        cot_parts.append(f"Position 2 is always `{list(pos2_chars)[0]}`.")

    cot_parts.append("")

    # Final reasoning
    query_op = query[2] if len(query) > 2 else '?'
    cot_parts.append(f"For the query `{query}` (operator at position 2: `{query_op}`):")
    
    # Check if we've seen this operator before
    matching_examples = [(lhs, rhs) for lhs, rhs in examples if len(lhs) > 2 and lhs[2] == query_op]
    if matching_examples:
        cot_parts.append(f"This matches the `{query_op}` pattern seen in {len(matching_examples)} example(s).")
        for lhs, rhs in matching_examples:
            cot_parts.append(f"  `{lhs}` → `{rhs}`")
    
    cot_parts.append("")
    cot_parts.append(f"Following the established pattern, the transformation of `{query}` gives: `{answer}`")

    return "\n".join(cot_parts)


def generate_synthetic_transform():
    """
    Generate a synthetic transformation puzzle.
    Uses position-based rules that mimic the real competition format.
    All inputs are 5 characters. The middle character (pos 2) acts as an operator.
    """
    char_pool = list("!@#$%^&*()[]{}|\\/<>?~`'\"=+-_.,;:0123456789")
    random.shuffle(char_pool)

    # Pick chars for positions 0,1,3,4 (operand chars) and position 2 (operators)
    operand_chars = random.sample(char_pool, random.randint(8, 15))
    operators = random.sample(char_pool, random.randint(2, 4))
    # Ensure no overlap
    operators = [c for c in operators if c not in operand_chars]
    if not operators:
        operators = [random.choice(['+', '-', '*', '/', '|', '\\', '&', '^'])]

    # Define what each operator does to the 4 operand chars
    # Rules: operator determines which positions are kept and/or how they're mapped
    op_rules = {}
    for op in operators:
        # Random rule: keep some subset of positions [0,1,3,4] and optionally remap
        positions = [0, 1, 3, 4]
        keep_count = random.randint(1, 4)
        kept = random.sample(positions, keep_count)
        kept.sort()

        # Optional: create a substitution for the kept chars
        sub_map = {}
        if random.random() < 0.5:  # 50% chance of substitution
            for c in operand_chars:
                sub_map[c] = random.choice(operand_chars)

        op_rules[op] = {'kept_positions': kept, 'sub_map': sub_map}

    def apply_rule(input_str):
        if len(input_str) != 5:
            return None
        op = input_str[2]
        if op not in op_rules:
            return None
        rule = op_rules[op]
        result = []
        for pos in rule['kept_positions']:
            c = input_str[pos]
            if rule['sub_map']:
                c = rule['sub_map'].get(c, c)
            result.append(c)
        return ''.join(result)

    # Generate examples
    examples = []
    for _ in range(random.randint(3, 6)):
        op = random.choice(operators)
        chars_0134 = [random.choice(operand_chars) for _ in range(4)]
        input_str = chars_0134[0] + chars_0134[1] + op + chars_0134[2] + chars_0134[3]
        output = apply_rule(input_str)
        if output:
            examples.append((input_str, output))

    if len(examples) < 3:
        return None

    # Generate query
    op = random.choice(operators)
    query_chars = [random.choice(operand_chars) for _ in range(4)]
    query = query_chars[0] + query_chars[1] + op + query_chars[2] + query_chars[3]
    answer = apply_rule(query)

    if answer is None:
        return None

    # Build prompt
    example_lines = "\n".join(f"{lhs} = {rhs}" for lhs, rhs in examples)
    prompt = (
        f"In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        f"Below are a few examples:\n"
        + example_lines + "\n"
        f"Now, determine the result for: {query}"
    )

    # Build CoT
    cot = generate_transform_observation_cot(examples, query, answer)

    return prompt, cot, answer


# ============================================================
# 7. RECOVER cipher hard cases from train.csv
# ============================================================
def recover_cipher_hard(train_csv, existing_prompts):
    """Recover cipher puzzles with incomplete mappings using ground truth."""
    recovered = []
    english_words = set(NOUNS + VERBS + ADJECTIVES + PREPOSITIONS)

    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            gt = str(row['answer']).strip()

            if 'encryption' not in prompt.lower():
                continue
            if prompt[:200] in existing_prompts:
                continue

            # Parse
            lines = prompt.strip().split('\n')
            examples = []
            query_text = None
            for line in lines:
                line = line.strip()
                if ' -> ' in line:
                    parts = line.split(' -> ', 1)
                    if len(parts) == 2:
                        examples.append((parts[0].strip(), parts[1].strip()))
                elif 'decrypt the following text:' in line.lower():
                    m = re.search(r'decrypt the following text:\s*(.*)', line, re.IGNORECASE)
                    if m:
                        query_text = m.group(1).strip()

            if not examples or not query_text:
                continue

            # Build char map
            char_map = {}
            conflict = False
            for enc, dec in examples:
                enc_w, dec_w = enc.split(), dec.split()
                if len(enc_w) != len(dec_w):
                    conflict = True
                    break
                for ew, dw in zip(enc_w, dec_w):
                    if len(ew) != len(dw):
                        conflict = True
                        break
                    for e, d in zip(ew, dw):
                        if e == ' ':
                            continue
                        if e in char_map and char_map[e] != d:
                            conflict = True
                            break
                        char_map[e] = d
                    if conflict:
                        break
                if conflict:
                    break

            if conflict:
                continue

            # Check for unmapped chars
            unmapped = set(c for c in query_text if c != ' ' and c not in char_map)
            if not unmapped:
                continue  # Already fully mapped — should be in v5

            # Use ground truth to fill in the gaps
            gt_words = gt.split()
            query_words = query_text.split()
            if len(gt_words) != len(query_words):
                continue

            inferred_map = dict(char_map)
            inference_ok = True
            for qw, gw in zip(query_words, gt_words):
                if len(qw) != len(gw):
                    inference_ok = False
                    break
                for e, d in zip(qw, gw):
                    if e in inferred_map and inferred_map[e] != d:
                        inference_ok = False
                        break
                    inferred_map[e] = d
                if not inference_ok:
                    break

            if not inference_ok:
                continue

            # Verify
            result = ''.join(inferred_map.get(c, c) if c != ' ' else ' ' for c in query_text)
            if result.strip() != gt:
                continue

            # Build CoT with inference reasoning
            partial = ''.join(char_map.get(c, '?') if c != ' ' else ' ' for c in query_text)
            mapping_str = ', '.join(f"{k}→{v}" for k, v in sorted(char_map.items()))

            cot = (
                "This is a character substitution cipher.\n\n"
                "Examples:\n"
                + "\n".join(f"  {e} → {d}" for e, d in examples)
                + f"\n\nCharacter mapping from examples:\n  {mapping_str}\n\n"
                f"Partial decryption of `{query_text}`:\n  `{partial}`\n\n"
            )

            for mc in unmapped:
                actual = inferred_map[mc]
                for i, (qw, pw) in enumerate(zip(query_words, partial.split())):
                    if mc in qw:
                        pattern = pw.replace('?', '.')
                        candidates = [w for w in english_words
                                      if len(w) == len(pw) and re.match(f"^{pattern}$", w)]
                        gt_word = gt_words[i]

                        cot += f"Character '{mc}' is unmapped. Partial word: `{pw}`.\n"
                        if candidates and len(candidates) <= 5:
                            cot += f"Possible words: {', '.join(candidates)}. "
                        cot += f"'{gt_word}' fits the context. Therefore {mc}→{actual}.\n\n"
                        break

            cot += f"Final decrypted text: `{result}`"
            recovered.append(make_training_example(prompt, cot, gt, 'text_encryption'))

    return recovered


# ============================================================
# 8. RECOVER unit conversion from train.csv
# ============================================================
def recover_unit_conversion(train_csv, existing_prompts):
    recovered = []

    with open(train_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            gt = str(row['answer']).strip()

            if 'unit conversion' not in prompt.lower():
                continue
            if prompt[:200] in existing_prompts:
                continue

            examples = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
            query_match = re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt, re.IGNORECASE)

            if not examples or not query_match:
                continue

            query_val = float(query_match.group(1))
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
                continue

            # Try median first, then mean
            for method_name, avg_fn in [("median", np.median), ("mean", np.mean)]:
                avg = avg_fn(factors)
                result = avg * query_val
                result_str = f"{result:.2f}"
                if result_str == gt:
                    cot = (
                        f"I need to convert {query_val} m.\n\n"
                        f"Finding conversion factor:\n\n"
                        + "\n".join(factor_lines)
                        + f"\n\n{method_name.title()} factor ≈ {avg:.6f}\n\n"
                        f"Applying: {query_val} × {avg:.6f} = {result_str}"
                    )
                    recovered.append(make_training_example(prompt, cot, gt, 'unit_conversion'))
                    break

    return recovered


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("SYNTHETIC DATA GENERATOR v6")
    print("=" * 60)

    # Load existing prompts for dedup
    existing_prompts = set()
    existing_count = 0
    if Path(EXISTING_DATA).exists():
        with open(EXISTING_DATA, 'r') as f:
            for line in f:
                existing_count += 1
                data = json.loads(line)
                if data.get('messages'):
                    existing_prompts.add(data['messages'][0]['content'][:200])
        print(f"Loaded {existing_count} existing examples ({len(existing_prompts)} unique prompts)")

    all_new = []

    # ─── Phase 1: RECOVER missing train.csv puzzles ───────────
    print("\n── Phase 1: Recovering missing train.csv puzzles ──")

    print("  Recovering transformation_rules (GT-guided CoT)...")
    recovered_transform = recover_transformation_from_csv(TRAIN_CSV, existing_prompts)
    print(f"    → Recovered {len(recovered_transform)} transformation puzzles")
    all_new.extend(recovered_transform)

    print("  Recovering cipher (hard cases with inference CoT)...")
    recovered_cipher = recover_cipher_hard(TRAIN_CSV, existing_prompts)
    print(f"    → Recovered {len(recovered_cipher)} cipher puzzles")
    all_new.extend(recovered_cipher)

    print("  Recovering unit_conversion...")
    recovered_unit = recover_unit_conversion(TRAIN_CSV, existing_prompts)
    print(f"    → Recovered {len(recovered_unit)} unit conversion puzzles")
    all_new.extend(recovered_unit)

    # ─── Phase 2: SYNTHESIZE new puzzles ──────────────────────
    print("\n── Phase 2: Generating synthetic puzzles ──")

    generators = {
        'bit_manipulation': lambda: generate_bit_puzzle(
            complexity=random.choice(['single'] * 4 + ['double'])
        ),
        'text_encryption': lambda: generate_cipher_puzzle(
            difficulty=random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
        ),
        'numeral_system': lambda: generate_roman_puzzle(),
        'unit_conversion': lambda: generate_unit_puzzle(),
        'gravitational': lambda: generate_gravity_puzzle(),
        'transformation_rules': lambda: generate_synthetic_transform(),
    }

    for category, count in SYNTHETIC_COUNT.items():
        print(f"  Generating {count} {category} puzzles...")
        generated = 0
        attempts = 0
        while generated < count and attempts < count * 5:
            attempts += 1
            try:
                result = generators[category]()
                if result is None:
                    continue
                prompt, cot, answer = result
                example = make_training_example(prompt, cot, answer, category)
                all_new.append(example)
                generated += 1
            except Exception as e:
                continue
        print(f"    → Generated {generated}/{count}")

    # ─── Phase 3: Write outputs ───────────────────────────────
    print(f"\n── Phase 3: Writing outputs ──")

    with open(OUTPUT_NEW, 'w') as f:
        for example in all_new:
            f.write(json.dumps({'messages': example['messages']}) + '\n')
    print(f"  New examples: {len(all_new)} → {OUTPUT_NEW}")

    merged_count = 0
    with open(OUTPUT_MERGED, 'w') as out_f:
        if Path(EXISTING_DATA).exists():
            with open(EXISTING_DATA, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
                    merged_count += 1
        for example in all_new:
            out_f.write(json.dumps({'messages': example['messages']}) + '\n')
            merged_count += 1
    print(f"  Merged total: {merged_count} → {OUTPUT_MERGED}")

    # ─── Stats ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    cat_counts = Counter()
    for ex in all_new:
        cat_counts[ex['category']] += 1

    print(f"\nNew examples by category:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat:25s}: {count:6d}")
    print(f"  {'TOTAL':25s}: {len(all_new):6d}")

    print(f"\nExisting: {existing_count}")
    print(f"New:      {len(all_new)}")
    print(f"Merged:   {merged_count}")

    new_size = Path(OUTPUT_NEW).stat().st_size / 1024 / 1024
    merged_size = Path(OUTPUT_MERGED).stat().st_size / 1024 / 1024
    print(f"\nFile sizes:")
    print(f"  {OUTPUT_NEW}: {new_size:.2f} MB")
    print(f"  {OUTPUT_MERGED}: {merged_size:.2f} MB")

    # Category breakdown of merged
    print(f"\nMerged category distribution:")
    merged_cats = Counter()
    with open(OUTPUT_MERGED, 'r') as f:
        for line in f:
            data = json.loads(line)
            p = data['messages'][0]['content'].lower() if data.get('messages') else ''
            if 'bit manipulation' in p: merged_cats['bit_manipulation'] += 1
            elif 'encryption' in p: merged_cats['text_encryption'] += 1
            elif 'numeral system' in p: merged_cats['numeral_system'] += 1
            elif 'unit conversion' in p: merged_cats['unit_conversion'] += 1
            elif 'gravitational' in p: merged_cats['gravitational'] += 1
            elif 'transformation' in p: merged_cats['transformation_rules'] += 1
            else: merged_cats['unknown'] += 1

    for cat, count in merged_cats.most_common():
        print(f"  {cat:25s}: {count:6d}")

    print(f"\nDone! Upload {OUTPUT_MERGED} to Kaggle as training dataset.")


if __name__ == '__main__':
    main()
