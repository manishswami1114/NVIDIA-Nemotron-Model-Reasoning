"""
Programmatic solvers for all 6 puzzle categories in the NVIDIA Nemotron Reasoning Challenge.
Each solver takes a puzzle prompt and returns the predicted answer.
"""

import csv
import re
import numpy as np
from collections import Counter
from itertools import product as iter_product


# =============================================================================
# Category 1: Numeral System (Decimal -> Roman Numerals)
# =============================================================================

def solve_numeral_system(prompt):
    """Extract the number to convert and return Roman numeral."""
    match = re.search(r'write the number (\d+)', prompt)
    if not match:
        return None
    num = int(match.group(1))
    return int_to_roman(num)


def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    result = ''
    for i in range(len(val)):
        while num >= val[i]:
            result += syms[i]
            num -= val[i]
    return result


# =============================================================================
# Category 2: Gravitational Constant (d = 0.5 * g * t^2)
# =============================================================================

def solve_gravitational(prompt):
    """Extract g from examples, then compute distance for the query time."""
    # Parse examples: "For t = X s, distance = Y m"
    examples = re.findall(r'For t = ([\d.]+)s, distance = ([\d.]+) m', prompt)
    if not examples:
        return None

    # Compute g from each example: g = 2*d / t^2
    g_values = []
    for t_str, d_str in examples:
        t = float(t_str)
        d = float(d_str)
        if t > 0:
            g = 2 * d / (t * t)
            g_values.append(g)

    if not g_values:
        return None

    # Use median g for robustness
    g = np.median(g_values)

    # Parse query time
    query_match = re.search(r'for t = ([\d.]+)s', prompt)
    if not query_match:
        return None

    t_query = float(query_match.group(1))
    d_result = 0.5 * g * t_query * t_query

    return f"{d_result:.2f}"


# =============================================================================
# Category 3: Unit Conversion (linear scaling: output = factor * input)
# =============================================================================

def solve_unit_conversion(prompt):
    """Extract scaling factor from examples, apply to query."""
    # Parse examples: "X m becomes Y"
    examples = re.findall(r'([\d.]+) m becomes ([\d.]+)', prompt)
    if not examples:
        return None

    # Compute scaling factors
    factors = []
    for x_str, y_str in examples:
        x = float(x_str)
        y = float(y_str)
        if x > 0:
            factors.append(y / x)

    if not factors:
        return None

    factor = np.median(factors)

    # Parse query
    query_match = re.search(r'convert the following measurement: ([\d.]+) m', prompt)
    if not query_match:
        return None

    x_query = float(query_match.group(1))
    result = factor * x_query

    return f"{result:.2f}"


# =============================================================================
# Category 4: Text Encryption (Substitution Cipher)
# =============================================================================

def solve_text_encryption(prompt):
    """Solve substitution cipher by building char mapping from examples."""
    # Split into examples section and query
    lines = prompt.strip().split('\n')

    # Find example lines (contain ' -> ')
    examples = []
    query_text = None

    for line in lines:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ')
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        elif 'decrypt the following text:' in line.lower():
            match = re.search(r'decrypt the following text:\s*(.*)', line, re.IGNORECASE)
            if match:
                query_text = match.group(1).strip()

    if not examples or not query_text:
        return None

    # Build character mapping from encrypted -> decrypted
    char_map = {}
    for encrypted, decrypted in examples:
        enc_chars = list(encrypted)
        dec_chars = list(decrypted)
        if len(enc_chars) == len(dec_chars):
            for e, d in zip(enc_chars, dec_chars):
                if e != ' ':  # spaces map to spaces
                    if e in char_map and char_map[e] != d:
                        pass  # conflict - skip
                    else:
                        char_map[e] = d

    # Apply mapping to query
    result = []
    for c in query_text:
        if c == ' ':
            result.append(' ')
        elif c in char_map:
            result.append(char_map[c])
        else:
            result.append(c)  # unknown mapping - keep original

    return ''.join(result)


# =============================================================================
# Category 5: Bit Manipulation
# =============================================================================

def solve_bit_manipulation(prompt):
    """
    Try common bit manipulation operations to find the rule.
    Each puzzle has its own unique rule - we must infer from examples.
    """
    # Parse examples
    examples = re.findall(r'([01]{8}) -> ([01]{8})', prompt)
    query_match = re.search(r'determine the output for:\s*([01]{8})', prompt)

    if not examples or not query_match:
        return None

    query = query_match.group(1)

    # Try various transformations
    # Strategy: try all common single-operation transforms, then compositions

    def bits_to_int(s):
        return int(s, 2)

    def int_to_bits(n):
        return format(n & 0xFF, '08b')

    # Single operations to try
    def rotate_left(s, n):
        n = n % 8
        return s[n:] + s[:n]

    def rotate_right(s, n):
        n = n % 8
        return s[8-n:] + s[:8-n]

    def xor_const(s, c):
        return int_to_bits(bits_to_int(s) ^ c)

    def reverse_bits(s):
        return s[::-1]

    def not_bits(s):
        return int_to_bits(bits_to_int(s) ^ 0xFF)

    def shift_left(s, n):
        return int_to_bits((bits_to_int(s) << n) & 0xFF)

    def shift_right(s, n):
        return int_to_bits(bits_to_int(s) >> n)

    # Try single operations
    candidates = []

    # Rotate left/right by 1-7
    for n in range(1, 8):
        candidates.append(('rot_l', n, lambda s, n=n: rotate_left(s, n)))
        candidates.append(('rot_r', n, lambda s, n=n: rotate_right(s, n)))

    # XOR with constants 1-255
    for c in range(1, 256):
        candidates.append(('xor', c, lambda s, c=c: xor_const(s, c)))

    # Reverse
    candidates.append(('reverse', 0, lambda s: reverse_bits(s)))

    # NOT
    candidates.append(('not', 0, lambda s: not_bits(s)))

    # Shift left/right 1-7
    for n in range(1, 8):
        candidates.append(('shl', n, lambda s, n=n: shift_left(s, n)))
        candidates.append(('shr', n, lambda s, n=n: shift_right(s, n)))

    # Try each single operation
    for name, param, func in candidates:
        match = True
        for inp, out in examples:
            if func(inp) != out:
                match = False
                break
        if match:
            return func(query)

    # Try two-operation compositions (most common pairs)
    simple_ops = []
    for n in range(1, 8):
        simple_ops.append(('rot_l', lambda s, n=n: rotate_left(s, n)))
        simple_ops.append(('rot_r', lambda s, n=n: rotate_right(s, n)))
    simple_ops.append(('reverse', lambda s: reverse_bits(s)))
    simple_ops.append(('not', lambda s: not_bits(s)))
    for c in range(1, 256):
        simple_ops.append(('xor', lambda s, c=c: xor_const(s, c)))

    # Two-op compositions
    for name1, op1 in simple_ops:
        for name2, op2 in simple_ops:
            match = True
            for inp, out in examples:
                if op2(op1(inp)) != out:
                    match = False
                    break
            if match:
                return op2(op1(query))

    return None  # Could not determine the rule


# =============================================================================
# Category 6: Transformation Rules (Symbolic Equations)
# =============================================================================

def solve_transformation_rules(prompt):
    """
    These puzzles apply character-level substitution to equation-like strings.
    Parse examples to build a char mapping, then apply to query.
    """
    lines = prompt.strip().split('\n')

    examples = []
    query = None

    for line in lines:
        line = line.strip()
        if ' = ' in line and 'determine the result' not in line.lower() and 'transformation rules' not in line.lower():
            parts = line.split(' = ')
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        elif 'determine the result for:' in line.lower():
            match = re.search(r'determine the result for:\s*(.*)', line, re.IGNORECASE)
            if match:
                query = match.group(1).strip()

    if not examples or not query:
        return None

    # These are tricky - could be char substitution, could be arithmetic
    # Let's first check if they look numeric

    # Try: character-level substitution cipher (like text encryption but with symbols)
    char_map = {}
    for lhs, rhs in examples:
        if len(lhs) == len(rhs):
            for a, b in zip(lhs, rhs):
                if a in char_map and char_map[a] != b:
                    char_map = {}  # conflicting mapping
                    break
                char_map[a] = b

    if char_map:
        result = ''.join(char_map.get(c, c) for c in query)
        # Verify against examples
        all_match = True
        for lhs, rhs in examples:
            if len(lhs) != len(rhs):
                all_match = False
                break
            predicted = ''.join(char_map.get(c, c) for c in lhs)
            if predicted != rhs:
                all_match = False
                break
        if all_match:
            return result

    # Try: these might be actual arithmetic with obfuscated operators
    # Check if inputs/outputs are numeric with operators
    # Pattern: numbers with operators like /, |, \, etc.

    # For now, try a more general approach: look at all chars in examples
    # and build mapping considering position independence

    # Fallback: try treating each example as showing a function applied to the input
    # This is the hardest category - may need more sophisticated analysis

    return None


# =============================================================================
# Main: Classify and Solve
# =============================================================================

def classify_puzzle(prompt):
    """Classify a puzzle by its category."""
    if 'bit manipulation' in prompt:
        return 'bit_manipulation'
    elif 'encryption' in prompt:
        return 'text_encryption'
    elif 'numeral system' in prompt:
        return 'numeral_system'
    elif 'unit conversion' in prompt:
        return 'unit_conversion'
    elif 'gravitational constant' in prompt:
        return 'gravitational'
    elif 'transformation rules' in prompt:
        return 'transformation_rules'
    return 'unknown'


def solve(prompt):
    """Solve a puzzle given its prompt."""
    category = classify_puzzle(prompt)

    solvers = {
        'numeral_system': solve_numeral_system,
        'gravitational': solve_gravitational,
        'unit_conversion': solve_unit_conversion,
        'text_encryption': solve_text_encryption,
        'bit_manipulation': solve_bit_manipulation,
        'transformation_rules': solve_transformation_rules,
    }

    solver = solvers.get(category)
    if solver:
        return solver(prompt)
    return None


# =============================================================================
# Validation
# =============================================================================

def validate_all():
    """Validate solvers against the full training set."""
    results = {
        'bit_manipulation': {'correct': 0, 'wrong': 0, 'unsolved': 0},
        'text_encryption': {'correct': 0, 'wrong': 0, 'unsolved': 0},
        'numeral_system': {'correct': 0, 'wrong': 0, 'unsolved': 0},
        'unit_conversion': {'correct': 0, 'wrong': 0, 'unsolved': 0},
        'gravitational': {'correct': 0, 'wrong': 0, 'unsolved': 0},
        'transformation_rules': {'correct': 0, 'wrong': 0, 'unsolved': 0},
    }

    wrong_examples = {cat: [] for cat in results}

    with open('train.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row['prompt']
            expected = row['answer'].strip()
            category = classify_puzzle(prompt)

            if category not in results:
                continue

            predicted = solve(prompt)

            if predicted is None:
                results[category]['unsolved'] += 1
                if len(wrong_examples[category]) < 2:
                    wrong_examples[category].append({
                        'id': row['id'],
                        'expected': expected,
                        'predicted': predicted,
                        'prompt_start': prompt[:200]
                    })
            elif str(predicted).strip() == expected:
                results[category]['correct'] += 1
            else:
                results[category]['wrong'] += 1
                if len(wrong_examples[category]) < 2:
                    wrong_examples[category].append({
                        'id': row['id'],
                        'expected': expected,
                        'predicted': predicted,
                        'prompt_start': prompt[:200]
                    })

    print("\n" + "=" * 70)
    print("SOLVER VALIDATION RESULTS")
    print("=" * 70)

    total_correct = 0
    total_all = 0

    for cat, res in results.items():
        total = res['correct'] + res['wrong'] + res['unsolved']
        total_all += total
        total_correct += res['correct']
        acc = res['correct'] / total * 100 if total > 0 else 0
        print(f"\n{cat}:")
        print(f"  Correct: {res['correct']}/{total} ({acc:.1f}%)")
        print(f"  Wrong: {res['wrong']}, Unsolved: {res['unsolved']}")

        if wrong_examples[cat]:
            print(f"  Sample failures:")
            for ex in wrong_examples[cat]:
                print(f"    ID: {ex['id']}, Expected: {ex['expected']}, Got: {ex['predicted']}")

    overall_acc = total_correct / total_all * 100 if total_all > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {total_correct}/{total_all} ({overall_acc:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    validate_all()
