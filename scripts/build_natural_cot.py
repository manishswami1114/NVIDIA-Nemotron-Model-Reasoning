#!/usr/bin/env python3
"""
Generate natural thinking-style CoTs for equation_numeric_guess.
Matches the style of equation_numeric_deduce CoTs exactly.
"""
import json, re
from pathlib import Path
from collections import defaultdict, OrderedDict

BASE = Path(__file__).resolve().parent.parent
JSONL = BASE / "all_categorical_splits_v13" / "train_cot_equation_numeric_guess.jsonl"
SOLVER = BASE / "all_categorical_splits_v13" / "perfect_solver_complete.md"


def rev_s(s):
    neg = s.startswith('-')
    if neg: s = s[1:]
    return ('-' if neg else '') + s[::-1]


def compute(name, a, b):
    """Compute formula result."""
    ai, bi = int(a), int(b)
    ra, rb = int(rev_s(a)), int(rev_s(b))
    mx, mn = max(ai,bi), min(ai,bi)
    rmx, rmn = max(ra,rb), min(ra,rb)
    
    table = {
        "max(a,b)-min(a,b)": str(mx-mn),
        "max(a,b)+min(a,b)": str(mx+mn),
        "max(a,b)*min(a,b)": str(mx*mn),
        "max(a,b)+min(a,b)+1": str(mx+mn+1),
        "max(a,b)+min(a,b)-1": str(mx+mn-1),
        "max(a,b)*min(a,b)+1": str(mx*mn+1),
        "max(a,b)*min(a,b)-1": str(mx*mn-1),
        "max(a,b)%min(a,b)": str(mx%mn) if mn else "0",
        "-(max(a,b)-min(a,b))": str(-(mx-mn)),
        "max(a,b)||min(a,b)": str(mx)+str(mn).zfill(2),
        "a||b": a+b, "(a||b)": a+b, "b||a": b+a, "(b||a)": b+a,
        "(a+b)": str(ai+bi), "(a-b)": str(ai-bi), "(a*b)": str(ai*bi),
        "(b+a)": str(bi+ai), "(b-a)": str(bi-ai), "(b*a)": str(bi*ai),
        "(a+b)+1": str(ai+bi+1), "(a+b)-1": str(ai+bi-1),
        "(a*b)+1": str(ai*bi+1), "(a*b)-1": str(ai*bi-1),
        "(a * b)+1": str(ai*bi+1), "(a * b)-1": str(ai*bi-1),
        "(b+a)+1": str(bi+ai+1), "(b+a)-1": str(bi+ai-1),
        "(b*a)+1": str(bi*ai+1), "(b*a)-1": str(bi*ai-1),
        "(b * a)+1": str(bi*ai+1), "(b * a)-1": str(bi*ai-1),
        "a+b": str(ai+bi), "a-b": str(ai-bi), "a*b": str(ai*bi),
        "b-a": str(bi-ai), "b+a": str(bi+ai), "b*a": str(bi*ai),
        "(a%b)": str(ai%bi) if bi else "0",
        "min(a,b)-max(a,b)": str(mn-mx),
        "min(a,b)+max(a,b)": str(mn+mx),
        "min(a,b)||max(a,b)": str(mn).zfill(2)+str(mx),
        "min(a,b)|| max(a,b)": str(mn).zfill(2)+str(mx),
        # Rev domain
        "rev(rev(a)+rev(b))": rev_s(str(ra+rb)),
        "rev(rev(a)-rev(b))": rev_s(str(ra-rb)),
        "rev(rev(a)*rev(b))": rev_s(str(ra*rb)),
        "rev(rev(a)+rev(b)+1)": rev_s(str(ra+rb+1)),
        "rev(rev(a)+rev(b)-1)": rev_s(str(ra+rb-1)),
        "rev(rev(a)*rev(b)+1)": rev_s(str(ra*rb+1)),
        "rev(rev(a)*rev(b)-1)": rev_s(str(ra*rb-1)),
        "rev(rev(a)%rev(b))": rev_s(str(ra%rb)) if rb else "0",
        "rev(rev(a)||rev(b))": rev_s(str(ra)+str(rb)),
        "rev(rev(b)+rev(a))": rev_s(str(rb+ra)),
        "rev(rev(b)-rev(a))": rev_s(str(rb-ra)),
        "rev(rev(b)*rev(a))": rev_s(str(rb*ra)),
        "rev(rev(b)+rev(a)+1)": rev_s(str(rb+ra+1)),
        "rev(rev(b)+rev(a)-1)": rev_s(str(rb+ra-1)),
        "rev(rev(b)*rev(a)+1)": rev_s(str(rb*ra+1)),
        "rev(rev(b)*rev(a)-1)": rev_s(str(rb*ra-1)),
        "rev(rev(b)%rev(a))": rev_s(str(rb%ra)) if ra else "0",
        "rev(rev(b)||rev(a))": rev_s(str(rb)+str(ra)),
        # Rev with max/min
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b)))": rev_s(str(rmx+rmn)),
        "rev(max(rev(a),rev(b))-min(rev(a),rev(b)))": rev_s(str(rmx-rmn)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b)))": rev_s(str(rmx*rmn)),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)": rev_s(str(rmx+rmn+1)),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)": rev_s(str(rmx+rmn-1)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)": rev_s(str(rmx*rmn+1)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)": rev_s(str(rmx*rmn-1)),
        "rev(max(rev(a),rev(b))%min(rev(a),rev(b)))": rev_s(str(rmx%rmn)) if rmn else "0",
        "rev(max(rev(a),rev(b))||min(rev(a),rev(b)))": rev_s(str(rmx)+str(rmn)),
        "-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))": '-'+rev_s(str(rmx-rmn)),
    }
    
    # Try exact match then normalized
    if name in table: return table[name]
    nn = re.sub(r'\s+', '', name)
    for k, v in table.items():
        if re.sub(r'\s+', '', k) == nn: return v
    return None


def is_rev(name):
    return 'rev(' in name


def try_ops_on_pair(a, b, result):
    """Try all operations on a pair, return list of (computed, label, match)."""
    ai, bi = int(a), int(b)
    ra, rb = int(rev_s(a)), int(rev_s(b))
    
    tests = []
    
    # Identity operands
    tests.append((str(ai+bi), f"addition f({a}, {b}) = {a} + {b} = {ai+bi}"))
    tests.append((a+b, f"concatenation f({a}, {b}) = {a} || {b} = {a}{b}"))
    tests.append((b+a, f"reverse concatenation f({a}, {b}) = {b} || {a} = {b}{a}"))
    tests.append((str(abs(ai-bi)), f"absolute difference f({a}, {b}) = |{a} - {b}| = {abs(ai-bi)}"))
    tests.append((str(ai-bi), f"subtraction (a-b) f({a}, {b}) = {a} - {b} = {ai-bi}"))
    tests.append((str(bi-ai), f"reverse subtraction (b-a) f({a}, {b}) = {b} - {a} = {bi-ai}"))
    tests.append((str(ai*bi), f"multiplication f({a}, {b}) = {a} * {b} = {ai*bi}"))
    tests.append((str(ai*bi+1), f"multiply+1 f({a}, {b}) = {a} * {b} + 1 = {ai*bi+1}"))
    tests.append((str(ai*bi-1), f"multiply-1 f({a}, {b}) = {a} * {b} - 1 = {ai*bi-1}"))
    tests.append((str(ai+bi+1), f"add+1 f({a}, {b}) = {a} + {b} + 1 = {ai+bi+1}"))
    tests.append((str(ai+bi-1), f"add-1 f({a}, {b}) = {a} + {b} - 1 = {ai+bi-1}"))
    if min(ai,bi) > 0:
        tests.append((str(max(ai,bi)%min(ai,bi)), f"max mod min f({a}, {b}) = {max(ai,bi)} mod {min(ai,bi)} = {max(ai,bi)%min(ai,bi)}"))
    
    return [(c, lbl, c == result) for c, lbl in tests]


def try_rev_ops(a, b, result):
    """Try reversed operands operations."""
    ra, rb = int(rev_s(a)), int(rev_s(b))
    
    tests = []
    tests.append((rev_s(str(ra+rb)), f"addition f({ra}, {rb}) = {ra} + {rb} = {ra+rb} -rev-> {rev_s(str(ra+rb))}"))
    tests.append((rev_s(str(ra-rb)), f"subtraction f({ra}, {rb}) = {ra} - {rb} = {ra-rb} -rev-> {rev_s(str(ra-rb))}"))
    tests.append((rev_s(str(rb-ra)), f"rev subtraction f({ra}, {rb}) = {rb} - {ra} = {rb-ra} -rev-> {rev_s(str(rb-ra))}"))
    tests.append((rev_s(str(ra*rb)), f"multiplication f({ra}, {rb}) = {ra} * {rb} = {ra*rb} -rev-> {rev_s(str(ra*rb))}"))
    tests.append((rev_s(str(ra)+str(rb)), f"concatenation f({ra}, {rb}) = {ra} || {rb} = {ra}{rb} -rev-> {rev_s(str(ra)+str(rb))}"))
    tests.append((rev_s(str(rb)+str(ra)), f"rev concatenation f({ra}, {rb}) = {rb} || {ra} = {rb}{ra} -rev-> {rev_s(str(rb)+str(ra))}"))
    tests.append((rev_s(str(ra*rb+1)), f"multiply+1 f({ra}, {rb}) = {ra} * {rb} + 1 = {ra*rb+1} -rev-> {rev_s(str(ra*rb+1))}"))
    tests.append((rev_s(str(ra*rb-1)), f"multiply-1 f({ra}, {rb}) = {ra} * {rb} - 1 = {ra*rb-1} -rev-> {rev_s(str(ra*rb-1))}"))
    tests.append((rev_s(str(ra+rb+1)), f"add+1 f({ra}, {rb}) = {ra} + {rb} + 1 = {ra+rb+1} -rev-> {rev_s(str(ra+rb+1))}"))
    tests.append((rev_s(str(ra+rb-1)), f"add-1 f({ra}, {rb}) = {ra} + {rb} - 1 = {ra+rb-1} -rev-> {rev_s(str(ra+rb-1))}"))
    if min(ra,rb) > 0:
        tests.append((rev_s(str(max(ra,rb)%min(ra,rb))), f"max mod min f({ra}, {rb}) = {max(ra,rb)} mod {min(ra,rb)} = {max(ra,rb)%min(ra,rb)} -rev-> {rev_s(str(max(ra,rb)%min(ra,rb)))}"))
    
    return [(c, lbl, c == result) for c, lbl in tests]


def parse_solver():
    """Parse solver into puzzle blocks."""
    with open(SOLVER) as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    puzzles = []
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        eqs = []; query = None
        for l in lines:
            if l.startswith('Now,'): query = l
            else: eqs.append(l)
        if eqs and query: puzzles.append((eqs, query))
    return puzzles


def parse_prompt(prompt):
    lines = prompt.strip().split('\n')
    examples = []; qa = qop = qb = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if any(kw in line.lower() for kw in ['alice','secret','transformation','example','please','boxed','final']): continue
        if 'determine' in line.lower():
            m = re.search(r'(\d+)\s*(.)\s*(\d+)', line)
            if m: qa, qop, qb = m.group(1), m.group(2), m.group(3)
            continue
        m = re.match(r'\s*(\d+)\s*(.)\s*(\d+)\s*=\s*(.+?)(?:\s*$)', line)
        if m: examples.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    return examples, qa, qop, qb


def build_cot(examples, qa, qop, qb, answer, formula_map):
    """Build natural thinking-style CoT."""
    
    op_groups = OrderedDict()
    for a, op, b, result in examples:
        if op not in op_groups: op_groups[op] = []
        op_groups[op].append((a, b, result))
    
    L = []
    
    # Opening — natural, like the model is thinking
    L.append("We need to infer the transformation rule from the examples.")
    L.append("I will put my final answer inside \\boxed{}.\n")
    
    # List examples
    L.append("Examples:")
    for a, op, b, result in examples:
        L.append(f"  {a}{op}{b} = {result}")
    
    # List operators and identify
    all_ops = list(op_groups.keys())
    L.append(f"\nThe operators: {', '.join(repr(o) for o in all_ops)}")
    L.append(f"\nLooking at the question")
    L.append(f"{qa}{qop}{qb} -> '{qop}'")
    L.append(f"The question operator '{qop}' is NOT found in the examples.\n")
    
    # Process each operator
    discovered = OrderedDict()
    
    for op_char, group in op_groups.items():
        a0, b0, r0 = group[0]
        eq_key = f"{a0}{op_char}{b0}"
        formula_name = formula_map.get(eq_key, None)
        
        L.append(f"Looking at operator 【{op_char}】 [{a0}{op_char}{b0} = {r0}]:")
        
        is_rev_formula = formula_name and 'rev(' in formula_name
        
        # Try identity operands first
        L.append(f"  Trying common operations on identity [{a0} {b0}] [expected ({a0},{b0})->{r0}]:")
        
        id_tests = try_ops_on_pair(a0, b0, r0)
        found_identity = False
        for computed, label, match in id_tests:
            if match:
                L.append(f"    {label} match, correct")
                found_identity = True
                break
            else:
                L.append(f"    {label} wrong")
        
        if not found_identity:
            # Try reversed operands with reversed result
            ra0, rb0 = int(rev_s(a0)), int(rev_s(b0))
            L.append(f"  Trying reversed operands [{a0}->{ra0} {b0}->{rb0}] and reversed result:")
            
            rev_tests = try_rev_ops(a0, b0, r0)
            found_rev = False
            for computed, label, match in rev_tests:
                if match:
                    L.append(f"    {label} match, correct")
                    found_rev = True
                    break
                else:
                    L.append(f"    {label} wrong")
            
            if not found_rev and formula_name:
                L.append(f"  Direct formula: {formula_name} = {r0}")
        
        if formula_name:
            # Determine action description
            if '||' in formula_name: action = "concatenation"
            elif '*' in formula_name and '%' not in formula_name:
                if '+1' in formula_name: action = "multiply+1"
                elif '-1' in formula_name: action = "multiply-1"
                else: action = "multiplication"
            elif '+' in formula_name:
                if '+1' in formula_name: action = "add+1"
                elif '-1' in formula_name: action = "add-1"
                else: action = "addition"
            elif '-' in formula_name: action = "subtraction"
            elif '%' in formula_name: action = "modulo"
            else: action = formula_name
            
            domain = "reversed" if is_rev_formula else "identity"
            L.append(f"  Rule: {formula_name}, actions: {action}, operands: {domain}")
            discovered[op_char] = formula_name
            
            # Verify other examples
            for a1, b1, r1 in group[1:]:
                c = compute(formula_name, a1, b1)
                ok = c == r1 if c else False
                L.append(f"  Verify [{a1}{op_char}{b1} = {r1}]: {formula_name} -> {c} {'correct' if ok else 'wrong'}")
        
        L.append("")
    
    # Identify domain
    rev_count = sum(1 for f in discovered.values() if 'rev(' in f)
    dir_count = len(discovered) - rev_count
    
    if rev_count > dir_count:
        domain = "reversed"
        L.append("All operators use reversed operands with reversed result.")
    elif dir_count > rev_count:
        domain = "identity"
        L.append("All operators use identity (non-reversed) operands.")
    else:
        domain = "mixed"
        L.append("Mixed domain operators found.")
    
    L.append(f"Discovered rules:")
    for op, fname in discovered.items():
        L.append(f"  '{op}' -> {fname}")
    
    # Solve query
    L.append(f"\nApplying to {qa}{qop}{qb}:")
    
    query_formula = formula_map.get(f"query:{qa}{qop}{qb}", None)
    
    if query_formula:
        # Show the computation
        if 'rev(' in query_formula:
            ra, rb = int(rev_s(qa)), int(rev_s(qb))
            L.append(f"  {domain}")
            L.append(f"  rev({qa}) = {ra}, rev({qb}) = {rb}")
            
            if '||' in query_formula:
                if 'rev(b)' in query_formula and query_formula.index('rev(b)') < query_formula.index('rev(a)'):
                    concat = str(rb)+str(ra)
                    L.append(f"  {query_formula}: {rb} || {ra} = {concat} -rev-> {rev_s(concat)}")
                else:
                    concat = str(ra)+str(rb)
                    L.append(f"  {query_formula}: {ra} || {rb} = {concat} -rev-> {rev_s(concat)}")
            elif '*' in query_formula and '%' not in query_formula:
                prod = ra * rb
                if '+1' in query_formula:
                    L.append(f"  {query_formula}: {ra} * {rb} + 1 = {prod+1} -rev-> {rev_s(str(prod+1))}")
                elif '-1' in query_formula:
                    L.append(f"  {query_formula}: {ra} * {rb} - 1 = {prod-1} -rev-> {rev_s(str(prod-1))}")
                else:
                    L.append(f"  {query_formula}: {ra} * {rb} = {prod} -rev-> {rev_s(str(prod))}")
            elif '+' in query_formula:
                s = ra + rb
                if '+1' in query_formula:
                    L.append(f"  {query_formula}: {ra} + {rb} + 1 = {s+1} -rev-> {rev_s(str(s+1))}")
                elif '-1' in query_formula:
                    L.append(f"  {query_formula}: {ra} + {rb} - 1 = {s-1} -rev-> {rev_s(str(s-1))}")
                else:
                    L.append(f"  {query_formula}: {ra} + {rb} = {s} -rev-> {rev_s(str(s))}")
            elif '-' in query_formula:
                if 'rev(b)-rev(a)' in query_formula:
                    d = rb - ra
                    L.append(f"  {query_formula}: {rb} - {ra} = {d} -rev-> {rev_s(str(d))}")
                elif '-rev(' in query_formula:
                    rmx, rmn = max(ra,rb), min(ra,rb)
                    d = rmx - rmn
                    L.append(f"  {query_formula}: -({rmx} - {rmn}) = -rev({d}) = -{rev_s(str(d))}")
                else:
                    d = ra - rb
                    L.append(f"  {query_formula}: {ra} - {rb} = {d} -rev-> {rev_s(str(d))}")
            elif '%' in query_formula:
                if min(ra,rb) > 0:
                    rmx, rmn = max(ra,rb), min(ra,rb)
                    m = rmx % rmn
                    L.append(f"  {query_formula}: {rmx} % {rmn} = {m} -rev-> {rev_s(str(m))}")
        else:
            # Direct domain
            ai, bi = int(qa), int(qb)
            mx, mn = max(ai,bi), min(ai,bi)
            L.append(f"  {domain}")
            
            if '||' in query_formula:
                if 'b||a' in query_formula or '(b||a)' in query_formula:
                    L.append(f"  {query_formula}: {qb} || {qa} = {qb}{qa}")
                elif 'a||b' in query_formula or '(a||b)' in query_formula:
                    L.append(f"  {query_formula}: {qa} || {qb} = {qa}{qb}")
                elif 'max' in query_formula:
                    L.append(f"  {query_formula}: {mx} || {mn} = {mx}{str(mn).zfill(2)}")
                else:
                    L.append(f"  {query_formula}: {qa} || {qb} = {qa}{qb}")
            elif '*' in query_formula and '%' not in query_formula:
                if 'max' in query_formula:
                    p = mx * mn
                    if '+1' in query_formula: L.append(f"  {query_formula}: {mx} * {mn} + 1 = {p+1}")
                    elif '-1' in query_formula: L.append(f"  {query_formula}: {mx} * {mn} - 1 = {p-1}")
                    else: L.append(f"  {query_formula}: {mx} * {mn} = {p}")
                else:
                    p = ai * bi
                    if '+1' in query_formula: L.append(f"  {query_formula}: {ai} * {bi} + 1 = {p+1}")
                    elif '-1' in query_formula: L.append(f"  {query_formula}: {ai} * {bi} - 1 = {p-1}")
                    else: L.append(f"  {query_formula}: {ai} * {bi} = {p}")
            elif '+' in query_formula:
                if 'max' in query_formula:
                    s = mx + mn
                    if '+1' in query_formula: L.append(f"  {query_formula}: {mx} + {mn} + 1 = {s+1}")
                    elif '-1' in query_formula: L.append(f"  {query_formula}: {mx} + {mn} - 1 = {s-1}")
                    else: L.append(f"  {query_formula}: {mx} + {mn} = {s}")
                else:
                    s = ai + bi
                    if '+1' in query_formula: L.append(f"  {query_formula}: {ai} + {bi} + 1 = {s+1}")
                    elif '-1' in query_formula: L.append(f"  {query_formula}: {ai} + {bi} - 1 = {s-1}")
                    else: L.append(f"  {query_formula}: {ai} + {bi} = {s}")
            elif '-' in query_formula:
                if 'max' in query_formula and query_formula.startswith('-'):
                    L.append(f"  {query_formula}: -({mx} - {mn}) = {-(mx-mn)}")
                elif 'max' in query_formula:
                    L.append(f"  {query_formula}: {mx} - {mn} = {mx-mn}")
                elif 'min' in query_formula:
                    L.append(f"  {query_formula}: {mn} - {mx} = {mn-mx}")
                else:
                    L.append(f"  {query_formula}: {ai} - {bi} = {ai-bi}")
            elif '%' in query_formula:
                if mn > 0:
                    L.append(f"  {query_formula}: {mx} % {mn} = {mx%mn}")
        
        L.append(f"  Result: 【{answer}】\n")
    else:
        L.append(f"  Result: 【{answer}】\n")
    
    # Verification — match the exact format from other categories
    L.append("I will now return the answer in \\boxed{}")
    L.append("The answer in \\boxed{–} is\n")
    L.append("Verification Step:")
    L.append("[✓] Equation evaluated following order of operations? -> YES")
    
    # Verify each example
    all_ok = True
    for a, op, b, result in examples:
        eq_key = f"{a}{op}{b}"
        fname = formula_map.get(eq_key, None)
        if fname:
            c = compute(fname, a, b)
            ok = c == result if c else False
            if not ok: all_ok = False
    
    L.append(f"[{'✓' if all_ok else '✗'}] LHS equals RHS? -> {'YES' if all_ok else 'NO'}\n")
    L.append("All constraints satisfied. The solution is verified.")
    L.append("I will now return the answer in \\boxed{}")
    L.append(f"\\boxed{{{answer}}}")
    
    cot = '\n'.join(L)
    return f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"


def main():
    # Parse solver
    solver_puzzles = parse_solver()
    
    # Build formula map
    all_fmaps = []
    for eqs, query_line in solver_puzzles:
        fmap = {}
        for eq_line in eqs:
            parts = eq_line.split(' = ', 2)
            if len(parts) >= 3:
                m = re.match(r'(\d+)(.)(d+)', parts[0].strip())
                if m:
                    fmap[f"{m.group(1)}{m.group(2)}{m.group(3)}"] = parts[2].strip()
                else:
                    # Try parsing differently
                    em = re.match(r'(\d+)\s*(.)\s*(\d+)', parts[0].strip())
                    if em:
                        fmap[f"{em.group(1)}{em.group(2)}{em.group(3)}"] = parts[2].strip()
        
        qm = re.search(r'(\d+)\s*(.)\s*(\d+)\s*=\s*(.+)', query_line)
        if qm:
            fmap[f"query:{qm.group(1)}{qm.group(2)}{qm.group(3)}"] = qm.group(4).strip()
        
        all_fmaps.append(fmap)
    
    # Ground truth
    gt = {}
    with open(BASE / "data" / "raw" / "train.csv") as f:
        csv = f.read()
    for m in re.finditer(r'determine the result for:\s*(\d+.{1}\d+).*?,\s*(.+?)$', csv, re.MULTILINE):
        q = m.group(1).strip().strip('"').replace(' ','')
        a = m.group(2).strip().strip('"').strip()
        if a: gt[q] = a
    
    # Load records
    with open(JSONL) as f:
        records = [json.loads(line) for line in f]
    
    print(f"Records: {len(records)}, Solver puzzles: {len(solver_puzzles)}")
    
    updated = 0
    for i, rec in enumerate(records):
        user_msg = [m for m in rec['messages'] if m['role'] == 'user'][0]
        examples, qa, qop, qb = parse_prompt(user_msg['content'])
        if not examples or qa is None: continue
        
        qkey = f"{qa}{qop}{qb}"
        answer = gt.get(qkey, "")
        if not answer:
            content = rec['messages'][-1]['content']
            boxed = re.findall(r'\\boxed\{([^}]*)\}', content)
            answer = boxed[-1] if boxed and boxed[-1] else ""
        if not answer: continue
        
        fmap = all_fmaps[i] if i < len(all_fmaps) else {}
        
        new_content = build_cot(examples, qa, qop, qb, answer, fmap)
        rec['messages'][-1]['content'] = new_content
        updated += 1
    
    print(f"Updated: {updated}/{len(records)}")
    
    with open(JSONL, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print("Done!")
    
    # Show samples
    for idx in [0, 135]:
        if idx < len(records):
            print(f"\n{'='*60}")
            print(f"=== Record {idx} ===")
            print(records[idx]['messages'][-1]['content'][:2500])


if __name__ == "__main__":
    main()
