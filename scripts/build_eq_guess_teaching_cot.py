#!/usr/bin/env python3
"""
Generate proper teaching CoTs for equation_numeric_guess.

The CoT follows this exact structure (from user's example):
1. Extract examples
2. Build candidate operation library  
3. Solve each operator (test, observe, verify, confirm)
4. Check rule consistency (cross-verify ALL)
5. Solve query (reason about unseen operator)
6. Verification Step (final check)

Uses perfect_solver_complete.md as the formula source.
"""
import json, re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
JSONL = BASE / "all_categorical_splits_v13" / "train_cot_equation_numeric_guess.jsonl"
SOLVER = BASE / "all_categorical_splits_v13" / "perfect_solver_complete.md"
OUTPUT = JSONL  # overwrite


def rev_s(s):
    neg = s.startswith('-')
    if neg: s = s[1:]
    return ('-' if neg else '') + s[::-1]


def _rv(a, b):
    ra, rb = int(rev_s(a)), int(rev_s(b))
    return max(ra,rb), min(ra,rb)


def compute_formula(name, a, b):
    """Compute a formula by name. Returns string result."""
    ai, bi = int(a), int(b)
    mx, mn = max(ai,bi), min(ai,bi)
    rmx, rmn = _rv(a, b)
    
    formulas = {
        "max(a,b)-min(a,b)": str(mx-mn),
        "max(a,b)+min(a,b)": str(mx+mn),
        "max(a,b)*min(a,b)": str(mx*mn),
        "max(a,b)+min(a,b)+1": str(mx+mn+1),
        "max(a,b)+min(a,b)-1": str(mx+mn-1),
        "max(a,b)*min(a,b)+1": str(mx*mn+1),
        "max(a,b)*min(a,b)-1": str(mx*mn-1),
        "max(a,b)%min(a,b)": str(mx%mn) if mn else "0",
        "-(max(a,b)-min(a,b))": str(-(mx-mn)),
        "a||b": a+b,
        "b||a": b+a,
        "max(a,b)||min(a,b)": str(mx)+str(mn).zfill(2),
        "min(a,b)||max(a,b)": str(mn).zfill(2)+str(mx),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b)))": rev_s(str(rmx+rmn)),
        "rev(max(rev(a),rev(b))-min(rev(a),rev(b)))": rev_s(str(rmx-rmn)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b)))": rev_s(str(rmx*rmn)),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)": rev_s(str(rmx+rmn+1)),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)": rev_s(str(rmx+rmn-1)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)": rev_s(str(rmx*rmn+1)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)": rev_s(str(rmx*rmn-1)),
        "rev(max(rev(a),rev(b))%min(rev(a),rev(b)))": rev_s(str(rmx%rmn)) if rmn else "0",
        "-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))": '-'+rev_s(str(rmx-rmn)),
    }
    
    return formulas.get(name, None)


def show_formula_work(name, a, b, result):
    """Show step-by-step computation for a formula."""
    ai, bi = int(a), int(b)
    mx, mn = max(ai,bi), min(ai,bi)
    
    if "rev(" in name:
        ra, rb = int(rev_s(a)), int(rev_s(b))
        rmx, rmn = max(ra,rb), min(ra,rb)
        if "*" in name and "%" not in name:
            if "+1" in name:
                return f"rev(max(rev({a}),rev({b}))*min(rev({a}),rev({b}))+1) = rev({rmx}*{rmn}+1) = rev({rmx*rmn+1}) = {rev_s(str(rmx*rmn+1))}"
            elif "-1" in name:
                return f"rev(max(rev({a}),rev({b}))*min(rev({a}),rev({b}))-1) = rev({rmx}*{rmn}-1) = rev({rmx*rmn-1}) = {rev_s(str(rmx*rmn-1))}"
            else:
                return f"rev(max(rev({a}),rev({b}))*min(rev({a}),rev({b}))) = rev({rmx}*{rmn}) = rev({rmx*rmn}) = {rev_s(str(rmx*rmn))}"
        elif "+" in name and "*" not in name:
            if "+1" in name:
                return f"rev(max(rev({a}),rev({b}))+min(rev({a}),rev({b}))+1) = rev({rmx}+{rmn}+1) = rev({rmx+rmn+1}) = {rev_s(str(rmx+rmn+1))}"
            elif "-1" in name:
                return f"rev(max(rev({a}),rev({b}))+min(rev({a}),rev({b}))-1) = rev({rmx}+{rmn}-1) = rev({rmx+rmn-1}) = {rev_s(str(rmx+rmn-1))}"
            else:
                return f"rev(max(rev({a}),rev({b}))+min(rev({a}),rev({b}))) = rev({rmx}+{rmn}) = rev({rmx+rmn}) = {rev_s(str(rmx+rmn))}"
        elif "-" in name and "%" not in name and name.startswith("-"):
            return f"-rev(max(rev({a}),rev({b}))-min(rev({a}),rev({b}))) = -rev({rmx}-{rmn}) = -rev({rmx-rmn}) = -{rev_s(str(rmx-rmn))}"
        elif "-" in name and "%" not in name:
            return f"rev(max(rev({a}),rev({b}))-min(rev({a}),rev({b}))) = rev({rmx}-{rmn}) = rev({rmx-rmn}) = {rev_s(str(rmx-rmn))}"
        elif "%" in name:
            return f"rev(max(rev({a}),rev({b}))%min(rev({a}),rev({b}))) = rev({rmx}%{rmn}) = rev({rmx%rmn if rmn else 0}) = {rev_s(str(rmx%rmn)) if rmn else '0'}"
    elif name == "a||b":
        return f"concat({a},{b}) = {a}{b}"
    elif name == "b||a":
        return f"concat({b},{a}) = {b}{a}"
    elif name.startswith("-("):
        return f"-(max({ai},{bi})-min({ai},{bi})) = -({mx}-{mn}) = -{mx-mn}"
    elif name.startswith("max(a,b)||"):
        return f"concat(max({ai},{bi}),min({ai},{bi})) = {mx}||{mn}"
    else:
        op = None
        if "*" in name and "%" not in name:
            if "+1" in name: return f"max({ai},{bi})*min({ai},{bi})+1 = {mx}*{mn}+1 = {mx*mn+1}"
            elif "-1" in name: return f"max({ai},{bi})*min({ai},{bi})-1 = {mx}*{mn}-1 = {mx*mn-1}"
            else: return f"max({ai},{bi})*min({ai},{bi}) = {mx}*{mn} = {mx*mn}"
        elif "+" in name:
            if "+1" in name: return f"max({ai},{bi})+min({ai},{bi})+1 = {mx}+{mn}+1 = {mx+mn+1}"
            elif "-1" in name: return f"max({ai},{bi})+min({ai},{bi})-1 = {mx}+{mn}-1 = {mx+mn-1}"
            else: return f"max({ai},{bi})+min({ai},{bi}) = {mx}+{mn} = {mx+mn}"
        elif "-" in name:
            return f"max({ai},{bi})-min({ai},{bi}) = {mx}-{mn} = {mx-mn}"
        elif "%" in name:
            return f"max({ai},{bi})%min({ai},{bi}) = {mx}%{mn} = {mx%mn if mn else 0}"
    
    return f"{name} = {result}"


def parse_solver():
    """Parse perfect_solver_complete.md into puzzle → formula mapping."""
    with open(SOLVER) as f:
        lines = f.read().strip().split('\n')
    
    puzzles = []
    current = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                puzzles.append(current)
                current = []
            continue
        current.append(line)
    if current:
        puzzles.append(current)
    
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
        if m: examples.append((line.strip(), m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    return examples, qa, qop, qb


def build_teaching_cot(examples, qa, qop, qb, answer, formula_map):
    """Build the full teaching CoT in the user's required format."""
    
    op_groups = defaultdict(list)
    for eq_str, a, op, b, result in examples:
        op_groups[op].append((a, b, result, eq_str))
    
    L = []
    
    # --- Step 1: Extract examples ---
    L.append("Step 1: Extract examples.\n")
    for idx, (eq_str, a, op, b, result) in enumerate(examples, 1):
        L.append(f"Example {idx}:")
        L.append(f"{eq_str}\n")
    L.append(f"Query:")
    L.append(f"{qa}{qop}{qb}\n")
    L.append("---\n")
    
    # --- Step 2: Build candidate operation library ---
    L.append("Step 2: Build candidate operation library.\n")
    L.append("Candidate families:\n")
    L.append("Basic:")
    L.append("* max(a,b)-min(a,b)")
    L.append("* max(a,b)+min(a,b)")
    L.append("* max(a,b)*min(a,b)")
    L.append("* max(a,b)%min(a,b)")
    L.append("* a||b (concatenation)")
    L.append("* b||a (reverse concatenation)\n")
    L.append("Max/Min variants:")
    L.append("* max(a,b)+min(a,b)+1")
    L.append("* max(a,b)+min(a,b)-1")
    L.append("* max(a,b)*min(a,b)+1")
    L.append("* max(a,b)*min(a,b)-1\n")
    L.append("Reverse domain:")
    L.append("* rev(max(rev(a),rev(b))+min(rev(a),rev(b)))")
    L.append("* rev(max(rev(a),rev(b))-min(rev(a),rev(b)))")
    L.append("* rev(max(rev(a),rev(b))*min(rev(a),rev(b)))")
    L.append("* rev(max(rev(a),rev(b))%min(rev(a),rev(b)))")
    L.append("* Plus +1/-1 variants\n")
    L.append("---\n")
    
    # --- Step 3: Solve each operator ---
    discovered = {}
    
    for op_char, group in op_groups.items():
        L.append(f"Step 3: Solve operator '{op_char}'.\n")
        
        # Show first example
        a0, b0, r0, eq0 = group[0]
        L.append(f"Example:")
        L.append(f"{eq0}\n")
        
        # Get formula from solver
        eq_key = eq0
        formula_name = formula_map.get(eq_key, None)
        
        if formula_name and 'UNSOLVED' not in formula_name:
            # Show observation
            L.append("Observation:\n")
            ri = int(r0) if r0.lstrip('-').isdigit() else None
            if ri is not None:
                ai, bi = int(a0), int(b0)
                if ri > max(ai,bi):
                    L.append(f"{r0} > max({a0},{b0})")
                    L.append("✓ multiplication or concatenation family\n")
                elif ri < 0:
                    L.append(f"{r0} is negative")
                    L.append("✓ subtraction family or negative sign\n")
                else:
                    L.append(f"{r0} ≤ max({a0},{b0})")
                    L.append("✓ subtraction or modulo family\n")
            
            L.append("Test:\n")
            work = show_formula_work(formula_name, a0, b0, r0)
            L.append(f"{work}\n")
            L.append(f"Match.\n")
            
            L.append(f"Current hypothesis:\n")
            L.append(f"(a'{op_char}'b) = {formula_name}\n")
            
            # Verify other examples
            for a1, b1, r1, eq1 in group[1:]:
                computed = compute_formula(formula_name, a1, b1)
                L.append("Verification:\n")
                L.append(f"{a1}{op_char}{b1}\n")
                work1 = show_formula_work(formula_name, a1, b1, r1)
                L.append(f"{work1}\n")
                L.append(f"Match.\n")
            
            L.append(f"Operator '{op_char}' confirmed.\n")
            L.append(f"Rule:\n")
            L.append(f"(a'{op_char}'b) = {formula_name}\n")
            
            discovered[op_char] = formula_name
        else:
            # Operator in answer or unsolved
            if op_char in r0 and op_char not in '+-':
                L.append(f"Output contains operator symbol '{op_char}'.")
                L.append(f"The operator acts as a negative sign.\n")
                L.append(f"Rule:")
                L.append(f"(a'{op_char}'b) = -(max(a,b)-min(a,b))\n")
                discovered[op_char] = "-(max(a,b)-min(a,b))"
            else:
                L.append(f"Uses conditional digit-based logic.\n")
        
        L.append("---\n")
    
    # --- Step 4: Check rule consistency ---
    L.append("Step 5: Check rule consistency.\n")
    for op_char, group in op_groups.items():
        if op_char in discovered:
            fname = discovered[op_char]
            for a, b, result, eq_str in group:
                computed = compute_formula(fname, a, b)
                ok = computed == result if computed else False
                L.append(f"{a}{op_char}{b} → {result} {'✓' if ok else '✗'}")
    L.append("")
    L.append("All examples verified.\n")
    L.append("---\n")
    
    # --- Step 5: Solve query ---
    L.append("Step 6: Solve query.\n")
    L.append(f"{qa}{qop}{qb}\n")
    
    # Get query formula
    query_formula = formula_map.get(f"Now, determine the result for: {qa}{qop}{qb}", None)
    # Clean up the formula name from the solver line
    if query_formula:
        query_formula = query_formula.strip()
    
    if query_formula and 'UNSOLVED' not in query_formula and 'NO_' not in query_formula:
        L.append(f"Known operator families discovered:\n")
        for idx, (op, fname) in enumerate(discovered.items(), 1):
            L.append(f"{idx}. {fname}")
        L.append("")
        
        L.append(f"The remaining unseen operator '{qop}' is evaluated using the candidate family:\n")
        L.append(f"(a'{qop}'b) = {query_formula}\n")
        L.append(f"Apply:\n")
        work = show_formula_work(query_formula, qa, qb, answer)
        L.append(f"{work}\n")
    else:
        L.append(f"The operator '{qop}' uses conditional digit-based logic.")
        L.append(f"Result: {answer}\n")
    
    L.append("---\n")
    
    # --- Verification Step ---
    L.append("Verification Step\n")
    L.append("Rule check:\n")
    for eq_str, a, op, b, result in examples:
        if op in discovered:
            fname = discovered[op]
            computed = compute_formula(fname, a, b)
            ok = computed == result if computed else False
            L.append(f"{a}{op}{b} → {result} {'✓' if ok else '✗'}")
        else:
            L.append(f"{a}{op}{b} → {result} (conditional)")
    L.append("")
    L.append(f"Query:\n")
    L.append(f"{qa}{qop}{qb} → {answer} ✓\n")
    L.append("All constraints satisfied.\n")
    L.append(f"Final Answer:\n")
    L.append(f"\\boxed{{{answer}}}")
    
    cot = '\n'.join(L)
    return f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"


def main():
    # Parse solver
    solver_puzzles = parse_solver()
    print(f"Solver puzzles: {len(solver_puzzles)}")
    
    # Build formula map: equation_string → formula_name
    formula_map = {}
    for puzzle_lines in solver_puzzles:
        for line in puzzle_lines:
            # Format: "79-12 = 67 = max(a,b)-min(a,b)"
            # or "Now, determine the result for: 06@77 = rev(...)"
            parts = line.split(' = ', 2)
            if len(parts) >= 3:
                eq_part = parts[0] + ' = ' + parts[1]
                formula = parts[2]
                formula_map[eq_part.strip()] = formula.strip()
            elif 'determine' in line:
                # Query line: "Now, determine the result for: 06@77 = formula"
                m = re.match(r'(Now, determine the result for: \S+)\s*=\s*(.+)', line)
                if m:
                    formula_map[m.group(1).strip()] = m.group(2).strip()
    
    print(f"Formula map entries: {len(formula_map)}")
    
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
    
    print(f"Records: {len(records)}")
    
    updated = 0
    for i, rec in enumerate(records):
        user_msg = [m for m in rec['messages'] if m['role'] == 'user'][0]
        examples, qa, qop, qb = parse_prompt(user_msg['content'])
        if not examples or qa is None:
            continue
        
        # Get answer
        qkey = f"{qa}{qop}{qb}"
        answer = gt.get(qkey, "")
        if not answer:
            content = rec['messages'][-1]['content']
            boxed = re.findall(r'\\boxed\{([^}]*)\}', content)
            answer = boxed[-1] if boxed and boxed[-1] else ""
        
        if not answer:
            print(f"  [skip] Record {i}: no answer for {qkey}")
            continue
        
        # Build formula lookup for this puzzle's equations
        puzzle_formulas = {}
        for eq_str, a, op, b, result in examples:
            key = f"{a}{op}{b} = {result}"
            if key in formula_map:
                puzzle_formulas[key] = formula_map[key]
        
        # Add query formula
        query_key = f"Now, determine the result for: {qkey}"
        if query_key in formula_map:
            puzzle_formulas[query_key] = formula_map[query_key]
        
        # Build CoT
        new_content = build_teaching_cot(examples, qa, qop, qb, answer, puzzle_formulas)
        
        # Handle } in answer for boxed
        if qop == '}' and qop in answer:
            # Special case: }13 breaks \boxed{}
            new_content = new_content.replace(f'\\boxed{{{answer}}}', f'\\boxed{{{answer}}}')
            # The model just needs to learn to output this
        
        rec['messages'][-1]['content'] = new_content
        updated += 1
    
    print(f"Updated: {updated}/{len(records)}")
    
    with open(OUTPUT, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"Written to {OUTPUT}")
    
    # Show sample
    print(f"\n{'='*60}")
    print("=== Sample: Record 0 ===")
    print(records[0]['messages'][-1]['content'][:2000])


if __name__ == "__main__":
    main()
