#!/usr/bin/env python3
import re

def rev(n):
    """Reverse digits of a number"""
    return int(str(n).lstrip('0') or '0'[::-1])

def parse_equation(line):
    """Parse a line from puzzles_guess_logic.md"""
    # Pattern: XX op YY = result (a op b) = formula
    match = re.match(r'(\d+)([@\+\-\*\!\.\_\|<>\&\$\?#\(\)\[\]\{\}~`"\:!/\\]|@|==|!=)(\d+)\s*=\s*(.+?)\s*\(a.?b\)\s*=\s*(.+)', line)
    if not match:
        return None

    a_str, op, b_str, result_str, formula = match.groups()
    a = int(a_str)
    b = int(b_str)
    result = int(result_str) if result_str.lstrip('-').isdigit() else result_str

    return {
        'a': a,
        'op': op,
        'b': b,
        'a_str': a_str,
        'b_str': b_str,
        'result': result,
        'formula': formula.strip(),
        'raw': line
    }

def evaluate_formula(a, b, formula):
    """Try to evaluate a formula and see if it matches result"""
    try:
        # Simple evaluation for basic formulas
        if formula == 'a-b':
            return a - b
        elif formula == 'a+b':
            return a + b
        elif formula == 'a*b':
            return a * b
        elif formula == 'abs(a-b)':
            return abs(a - b)
        elif formula == 'unknown logic':
            return None
        else:
            # Complex formula - would need more sophisticated parsing
            return None
    except:
        return None

# Read the file
with open('puzzles_guess_logic.md', 'r') as f:
    lines = f.readlines()

unknown_logic = []
mismatches = []
verified = []

for i, line in enumerate(lines, 1):
    line = line.strip()
    if not line or line.startswith('#'):
        continue

    parsed = parse_equation(line)
    if not parsed:
        continue

    a, b, result, formula = parsed['a'], parsed['b'], parsed['result'], parsed['formula']

    if formula == 'unknown logic':
        unknown_logic.append({
            'line': i,
            'equation': f"{a}{parsed['op']}{b} = {result}",
            'a': a,
            'b': b,
            'op': parsed['op'],
            'result': result,
            'raw': line
        })

# Analyze unknown logic equations by operator
print("=" * 80)
print("UNKNOWN LOGIC EQUATIONS - GROUPED BY OPERATOR")
print("=" * 80)

from collections import defaultdict
by_op = defaultdict(list)
for eq in unknown_logic:
    by_op[eq['op']].append(eq)

for op in sorted(by_op.keys()):
    print(f"\n{'='*60}")
    print(f"Operator: '{op}'")
    print(f"{'='*60}")

    eqs = by_op[op]
    for eq in eqs[:5]:  # Show first 5 of each
        print(f"Line {eq['line']:3d}: {eq['a']:2d}{eq['op']}{eq['b']:2d} = {eq['result']}")

    if len(eqs) > 5:
        print(f"... and {len(eqs) - 5} more")

# Print all unknown logic for analysis
print("\n\n" + "=" * 80)
print("ALL UNKNOWN LOGIC EQUATIONS")
print("=" * 80)
for eq in unknown_logic:
    print(f"Line {eq['line']:3d}: {eq['a']:2d}{eq['op']}{eq['b']:2d} = {eq['result']}")
