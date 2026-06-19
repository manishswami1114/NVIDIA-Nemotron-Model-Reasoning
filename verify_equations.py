#!/usr/bin/env python3
"""Verify equations and identify solvable patterns"""

def rev(n):
    """Reverse a number's digits"""
    s = str(n).lstrip('0') or '0'
    return int(s[::-1])

# Test cases
tests = [
    # Line, a, op, b, expected_result, test_formulas
    (7, 6, '+', 67, 731, ['a+b', 'rev(a)+rev(b)', 'concatenation']),
    (8, 45, '+', 99, 451, ['a+b', 'b||a']),
    (99, 7, '-', 79, 72, ['b-a', 'abs(a-b)']),
    (182, 12, '+', 1, 13, ['a+b']),
    (185, 85, '+', 13, 98, ['a+b']),
    (194, 27, '-', 65, 61, ['rev(a)-rev(b)', 'abs(rev(a)-rev(b))']),
    (432, 14, '-', 4, 18, ['a+b']),  # Likely operator error
    (433, 99, '-', 7, 961, ['a*b', 'rev(a)*rev(b)']),
    (443, 9, '+', 3, 309, ['b||a']),
    (454, 42, '-', 1, 41, ['a-b']),
    (455, 86, '-', 52, 34, ['a-b']),
    (518, 6, '+', 41, 4106, ['b||a']),
    (519, 9, '+', 13, 1309, ['b||a']),
    (583, 15, '-', 77, 62, ['b-a']),
    (612, 97, '-', 87, 1, ['(a-b)//10']),
    (613, 2, '-', 73, 71, ['b-a']),
    (113, 63, '|', 8, 6308, ['a||b']),
    (114, 14, '|', 62, 1462, ['a||b']),
    (116, 26, '|', 16, 2616, ['a||b']),
]

print("="*80)
print("EQUATION VERIFICATION")
print("="*80)

for line, a, op, b, expected, formulas in tests:
    print(f"\nLine {line}: {a}{op}{b} = {expected}")

    matched = False
    for formula in formulas:
        try:
            if formula == 'a+b':
                result = a + b
            elif formula == 'a-b':
                result = a - b
            elif formula == 'a*b':
                result = a * b
            elif formula == 'b-a':
                result = b - a
            elif formula == 'abs(a-b)':
                result = abs(a - b)
            elif formula == 'rev(a)-rev(b)':
                result = rev(a) - rev(b)
            elif formula == 'abs(rev(a)-rev(b))':
                result = abs(rev(a) - rev(b))
            elif formula == 'a*b' and op == '*':
                result = a * b
            elif formula == 'rev(a)*rev(b)':
                result = rev(a) * rev(b)
            elif formula == '(a-b)//10':
                result = (a - b) // 10
            elif formula == 'a||b':  # concatenation
                result = int(str(a).zfill(2) + str(b).zfill(2))
            elif formula == 'b||a':
                result = int(str(b).zfill(2) + str(a).zfill(2))
            else:
                continue

            if result == expected:
                print(f"  ✓ MATCH: {formula}")
                matched = True
        except:
            pass

    if not matched:
        print(f"  ✗ No formula matched")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
