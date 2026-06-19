#!/usr/bin/env python3
import re

with open('perfect_solver_complete.md') as f:
    raw = [l.rstrip() for l in f]

eq_re = re.compile(r'^(?:Now, determine the result for:\s*)?\d+[^\d]+\d+\s*=\s*(.*)')

for i, line in enumerate(raw, 1):
    m = eq_re.match(line.strip())
    if not m: continue
    rest = m.group(1)
    parts = rest.split(' = ', 1)
    formula = parts[1].strip() if len(parts) == 2 else parts[0].strip()
    formula = re.sub(r'\s*\[op suffix\]', '', formula)
    nf = re.sub(r'\s+', '', formula)

    opens = nf.count('(')
    closes = nf.count(')')
    if opens != closes:
        print(f'L{i}: UNBALANCED PARENS ({opens}o, {closes}c): {formula}')
    if 'mi(' in nf and 'min(' not in nf:
        print(f'L{i}: TYPO mi(: {formula}')
    if '+,' in nf:
        print(f'L{i}: TYPO +,: {formula}')
    if 'rev(rev(' in nf and '))*' in nf:
        print(f'L{i}: UNUSUAL ))*: {formula}')
    if '|| ' in formula:
        print(f'L{i}: SPACE IN ||: {formula}')
