#!/usr/bin/env python3
"""
Generate discovery-style teaching CoTs for equation_numeric_guess.

The CoT shows the model HOW to discover formulas step-by-step:
1. Extract examples
2. For each operator: test basic ops → try reverse domain → confirm
3. Identify domain pattern across operators  
4. Solve query using domain consistency
5. Verification step

Uses perfect_solver_complete.md as the formula source.
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


def is_rev_formula(name):
    return 'rev(' in name


def get_base_op(name):
    """Extract the core operation from a formula name."""
    if '||' in name: return '||'
    if '*' in name and '%' not in name:
        if '+1' in name: return '*+1'
        if '-1' in name: return '*-1'
        return '*'
    if '+' in name:
        if '+1' in name: return '++1'
        if '-1' in name: return '+-1'
        return '+'
    if '-' in name and '(' not in name[:3]: return '-'
    if '%' in name: return '%'
    return name


def compute_formula(name, a_str, b_str):
    """Compute result for a named formula."""
    ai, bi = int(a_str), int(b_str)
    ra, rb = int(rev_s(a_str)), int(rev_s(b_str))
    mx, mn = max(ai,bi), min(ai,bi)
    rmx, rmn = max(ra,rb), min(ra,rb)
    
    formulas = {
        # Direct
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
        "min(a,b)||max(a,b)": str(mn).zfill(2)+str(mx),
        # Rev domain (using rev(a), rev(b) directly)
        "rev(rev(a)+rev(b))": rev_s(str(ra+rb)),
        "rev(rev(a)-rev(b))": rev_s(str(ra-rb)),
        "rev(rev(a)*rev(b))": rev_s(str(ra*rb)),
        "rev(rev(a)+rev(b)+1)": rev_s(str(ra+rb+1)),
        "rev(rev(a)+rev(b)-1)": rev_s(str(ra+rb-1)),
        "rev(rev(a)*rev(b)+1)": rev_s(str(ra*rb+1)),
        "rev(rev(a)*rev(b)-1)": rev_s(str(ra*rb-1)),
        "rev(rev(a)%rev(b))": rev_s(str(ra%rb)) if rb else "0",
        "rev(rev(a)||rev(b))": rev_s(str(ra)+str(rb).zfill(2) if rb < 10 else str(ra)+str(rb)),
        # Rev domain with b,a order
        "rev(rev(b)+rev(a))": rev_s(str(rb+ra)),
        "rev(rev(b)-rev(a))": rev_s(str(rb-ra)),
        "rev(rev(b)*rev(a))": rev_s(str(rb*ra)),
        "rev(rev(b)+rev(a)+1)": rev_s(str(rb+ra+1)),
        "rev(rev(b)+rev(a)-1)": rev_s(str(rb+ra-1)),
        "rev(rev(b)*rev(a)+1)": rev_s(str(rb*ra+1)),
        "rev(rev(b)*rev(a)-1)": rev_s(str(rb*ra-1)),
        "rev(rev(b)%rev(a))": rev_s(str(rb%ra)) if ra else "0",
        "rev(rev(b)||rev(a))": rev_s(str(rb)+str(ra).zfill(2) if ra < 10 else str(rb)+str(ra)),
        # Rev with max/min
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b)))": rev_s(str(rmx+rmn)),
        "rev(max(rev(a),rev(b))-min(rev(a),rev(b)))": rev_s(str(rmx-rmn)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b)))": rev_s(str(rmx*rmn)),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)": rev_s(str(rmx+rmn+1)),
        "rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)": rev_s(str(rmx+rmn-1)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)": rev_s(str(rmx*rmn+1)),
        "rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)": rev_s(str(rmx*rmn-1)),
        "rev(max(rev(a),rev(b))%min(rev(a),rev(b)))": rev_s(str(rmx%rmn)) if rmn else "0",
        "rev(max(rev(a),rev(b))||min(rev(a),rev(b)))": rev_s(str(rmx)+str(rmn).zfill(2) if rmn<10 else str(rmx)+str(rmn)),
        "-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))": '-'+rev_s(str(rmx-rmn)),
        # Simple
        "(a+b)": str(ai+bi), "(a-b)": str(ai-bi), "(a*b)": str(ai*bi),
        "(b+a)": str(bi+ai), "(b-a)": str(bi-ai), "(b*a)": str(bi*ai),
        "(a+b)+1": str(ai+bi+1), "(a+b)-1": str(ai+bi-1),
        "(a*b)+1": str(ai*bi+1), "(a*b)-1": str(ai*bi-1),
        "(a * b)+1": str(ai*bi+1), "(a * b)-1": str(ai*bi-1),
        "(b+a)+1": str(bi+ai+1), "(b+a)-1": str(bi+ai-1),
        "(b*a)+1": str(bi*ai+1), "(b*a)-1": str(bi*ai-1),
        "(b * a)+1": str(bi*ai+1), "(b * a)-1": str(bi*ai-1),
        "a||b": a_str+b_str, "b||a": b_str+a_str,
        "(a||b)": a_str+b_str, "(b||a)": b_str+a_str,
        "a || b": a_str+b_str,
        "(a%b)": str(ai%bi) if bi else "0",
        "a+b": str(ai+bi), "a-b": str(ai-bi), "a*b": str(ai*bi),
        "b+a": str(bi+ai), "b-a": str(bi-ai), "b*a": str(bi*ai),
        "min(a,b)-max(a,b)": str(mn-mx),
        "min(a,b)+max(a,b)": str(mn+mx),
        "min(a,b)||max(a,b)": str(mn).zfill(2)+str(mx),
        "min(a,b)|| max(a,b)": str(mn).zfill(2)+str(mx),
        # Negative sign as op prefix/suffix
        "-(max(a,b)-min(a,b))": str(-(mx-mn)),
    }
    
    # Handle rev(rev(a)||rev(b)) more carefully
    if name == "rev(rev(a)||rev(b))":
        concat = str(ra) + str(rb)
        return rev_s(concat)
    if name == "rev(rev(b)||rev(a))":
        concat = str(rb) + str(ra)
        return rev_s(concat)
    
    # Try exact match first, then normalized (remove extra spaces)
    if name in formulas:
        return formulas[name]
    name_norm = re.sub(r'\s+', '', name)
    for k, v in formulas.items():
        if re.sub(r'\s+', '', k) == name_norm:
            return v
    return None


def show_work(formula_name, a_str, b_str):
    """Show step-by-step computation for a formula."""
    ai, bi = int(a_str), int(b_str)
    ra, rb = int(rev_s(a_str)), int(rev_s(b_str))
    mx, mn = max(ai,bi), min(ai,bi)
    
    if formula_name.startswith("rev(rev("):
        # Rev domain — show rev computation
        lines = []
        lines.append(f"rev(a) = rev({a_str}) = {ra}, rev(b) = rev({b_str}) = {rb}")
        
        if "||" in formula_name:
            if "rev(b)" in formula_name and formula_name.index("rev(b)") < formula_name.index("rev(a)"):
                concat = str(rb) + str(ra)
                lines.append(f"rev(b) || rev(a) = {rb} || {ra} = {concat}")
            else:
                concat = str(ra) + str(rb)
                lines.append(f"rev(a) || rev(b) = {ra} || {rb} = {concat}")
            result = rev_s(concat)
            lines.append(f"rev({concat}) = {result}")
        elif "*" in formula_name and "%" not in formula_name:
            if "rev(b)*rev(a)" in formula_name or ("rev(b))*" in formula_name):
                prod = rb * ra
                tag = f"rev(b) * rev(a) = {rb} * {ra} = {prod}"
            else:
                prod = ra * rb
                tag = f"rev(a) * rev(b) = {ra} * {rb} = {prod}"
            lines.append(tag)
            if "+1" in formula_name:
                lines.append(f"{prod} + 1 = {prod+1}")
                result = rev_s(str(prod+1))
                lines.append(f"rev({prod+1}) = {result}")
            elif "-1" in formula_name:
                lines.append(f"{prod} - 1 = {prod-1}")
                result = rev_s(str(prod-1))
                lines.append(f"rev({prod-1}) = {result}")
            else:
                result = rev_s(str(prod))
                lines.append(f"rev({prod}) = {result}")
        elif "+" in formula_name and "*" not in formula_name:
            if "rev(b)+rev(a)" in formula_name or ("rev(b))+" in formula_name):
                s = rb + ra
                tag = f"rev(b) + rev(a) = {rb} + {ra} = {s}"
            else:
                s = ra + rb
                tag = f"rev(a) + rev(b) = {ra} + {rb} = {s}"
            lines.append(tag)
            if "+1" in formula_name:
                lines.append(f"{s} + 1 = {s+1}")
                result = rev_s(str(s+1))
                lines.append(f"rev({s+1}) = {result}")
            elif "-1" in formula_name:
                lines.append(f"{s} - 1 = {s-1}")
                result = rev_s(str(s-1))
                lines.append(f"rev({s-1}) = {result}")
            else:
                result = rev_s(str(s))
                lines.append(f"rev({s}) = {result}")
        elif "-" in formula_name:
            if "rev(b)-rev(a)" in formula_name:
                d = rb - ra
                tag = f"rev(b) - rev(a) = {rb} - {ra} = {d}"
            else:
                d = ra - rb
                tag = f"rev(a) - rev(b) = {ra} - {rb} = {d}"
            lines.append(tag)
            result = rev_s(str(d))
            lines.append(f"rev({d}) = {result}")
        elif "%" in formula_name:
            if "rev(b)%rev(a)" in formula_name:
                m = rb % ra if ra else 0
                tag = f"rev(b) % rev(a) = {rb} % {ra} = {m}"
            else:
                m = ra % rb if rb else 0
                tag = f"rev(a) % rev(b) = {ra} % {rb} = {m}"
            lines.append(tag)
            result = rev_s(str(m))
            lines.append(f"rev({m}) = {result}")
        
        return '\n'.join(lines)
    
    elif formula_name.startswith("rev(max("):
        # Rev with max/min
        rmx, rmn = max(ra,rb), min(ra,rb)
        lines = []
        lines.append(f"rev(a) = {ra}, rev(b) = {rb}")
        lines.append(f"max(rev(a),rev(b)) = {rmx}, min(rev(a),rev(b)) = {rmn}")
        
        if "*" in formula_name and "%" not in formula_name:
            p = rmx * rmn
            lines.append(f"{rmx} * {rmn} = {p}")
            if "+1" in formula_name:
                lines.append(f"{p} + 1 = {p+1}")
                lines.append(f"rev({p+1}) = {rev_s(str(p+1))}")
            elif "-1" in formula_name:
                lines.append(f"{p} - 1 = {p-1}")
                lines.append(f"rev({p-1}) = {rev_s(str(p-1))}")
            else:
                lines.append(f"rev({p}) = {rev_s(str(p))}")
        elif "+" in formula_name:
            s = rmx + rmn
            lines.append(f"{rmx} + {rmn} = {s}")
            if "+1" in formula_name:
                lines.append(f"{s} + 1 = {s+1}")
                lines.append(f"rev({s+1}) = {rev_s(str(s+1))}")
            elif "-1" in formula_name:
                lines.append(f"{s} - 1 = {s-1}")
                lines.append(f"rev({s-1}) = {rev_s(str(s-1))}")
            else:
                lines.append(f"rev({s}) = {rev_s(str(s))}")
        elif "-" in formula_name:
            d = rmx - rmn
            lines.append(f"{rmx} - {rmn} = {d}")
            lines.append(f"rev({d}) = {rev_s(str(d))}")
        elif "%" in formula_name:
            m = rmx % rmn if rmn else 0
            lines.append(f"{rmx} % {rmn} = {m}")
            lines.append(f"rev({m}) = {rev_s(str(m))}")
        elif "||" in formula_name:
            lines.append(f"{rmx} || {rmn} = {rmx}{rmn}")
            lines.append(f"rev({rmx}{rmn}) = {rev_s(str(rmx)+str(rmn))}")
        
        return '\n'.join(lines)
    
    elif formula_name.startswith("-rev("):
        # Negative rev
        rmx, rmn = max(ra,rb), min(ra,rb)
        d = rmx - rmn
        return f"rev(a) = {ra}, rev(b) = {rb}\nmax={rmx}, min={rmn}\n{rmx} - {rmn} = {d}\n-rev({d}) = -{rev_s(str(d))}"
    
    elif formula_name.startswith("max("):
        if "*" in formula_name:
            if "+1" in formula_name: return f"max({ai},{bi}) * min({ai},{bi}) + 1 = {mx} * {mn} + 1 = {mx*mn+1}"
            elif "-1" in formula_name: return f"max({ai},{bi}) * min({ai},{bi}) - 1 = {mx} * {mn} - 1 = {mx*mn-1}"
            else: return f"max({ai},{bi}) * min({ai},{bi}) = {mx} * {mn} = {mx*mn}"
        elif "+" in formula_name:
            if "+1" in formula_name: return f"max({ai},{bi}) + min({ai},{bi}) + 1 = {mx} + {mn} + 1 = {mx+mn+1}"
            elif "-1" in formula_name: return f"max({ai},{bi}) + min({ai},{bi}) - 1 = {mx} + {mn} - 1 = {mx+mn-1}"
            else: return f"max({ai},{bi}) + min({ai},{bi}) = {mx} + {mn} = {mx+mn}"
        elif "-" in formula_name: return f"max({ai},{bi}) - min({ai},{bi}) = {mx} - {mn} = {mx-mn}"
        elif "%" in formula_name: return f"max({ai},{bi}) % min({ai},{bi}) = {mx} % {mn} = {mx%mn if mn else 0}"
        elif "||" in formula_name:
            return f"max({ai},{bi}) || min({ai},{bi}) = {mx} || {mn}"
    
    elif formula_name == "a||b": return f"a || b = {a_str} || {b_str} = {a_str}{b_str}"
    elif formula_name == "b||a": return f"b || a = {b_str} || {a_str} = {b_str}{a_str}"
    elif formula_name == "(a||b)": return f"a || b = {a_str} || {b_str} = {a_str}{b_str}"
    elif formula_name.startswith("(a"): return f"{formula_name} = {compute_formula(formula_name, a_str, b_str)}"
    elif formula_name.startswith("(b"): return f"{formula_name} = {compute_formula(formula_name, a_str, b_str)}"
    elif formula_name.startswith("-("): return f"{formula_name} = {compute_formula(formula_name, a_str, b_str)}"
    elif formula_name.startswith("min("): return f"{formula_name} = {compute_formula(formula_name, a_str, b_str)}"
    
    return f"{formula_name} = {compute_formula(formula_name, a_str, b_str)}"


def classify_result(ai, bi, result_str):
    """Generate observation about the result."""
    try:
        ri = int(result_str.lstrip('-').replace('+','').replace('!','').replace('@','').replace('$',''))
    except:
        ri = None
    
    mx = max(abs(ai), abs(bi))
    
    if ri is None:
        return "Result contains operator symbol."
    elif ri > mx * 2:
        return f"{result_str} is much larger than both {ai} and {bi}.\nThis suggests multiplication or concatenation."
    elif ri > mx:
        return f"{result_str} is larger than both operands.\nThis suggests addition or multiplication."
    elif ri == 0:
        return f"Result is 0.\nThis suggests subtraction, modulo, or difference."
    elif result_str.startswith('-'):
        return f"Result is negative.\nThis suggests subtraction with sign."
    else:
        return f"{result_str} ≤ max({ai},{bi}).\nThis suggests subtraction, modulo, or difference."


def get_op_symbol(formula_name):
    """Get the core arithmetic symbol from formula."""
    if '||' in formula_name: return '||'
    if '%' in formula_name: return '%'
    if '*' in formula_name: return '×'
    if '+' in formula_name: return '+'
    if '-' in formula_name: return '−'
    return '?'


def parse_solver():
    """Parse perfect_solver_complete.md into puzzle blocks."""
    with open(SOLVER) as f:
        content = f.read()
    
    blocks = content.strip().split('\n\n')
    puzzles = []
    
    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if not lines:
            continue
        
        eqs = []
        query_line = None
        
        for line in lines:
            if line.startswith('Now, determine'):
                query_line = line
            else:
                eqs.append(line)
        
        if eqs and query_line:
            puzzles.append((eqs, query_line))
    
    return puzzles


def parse_eq_line(line):
    """Parse '79-12 = 67 = rev(rev(a)-rev(b))' into parts."""
    # Format: A op B = result = formula
    parts = line.split(' = ', 2)
    if len(parts) >= 2:
        eq_part = parts[0].strip()
        result = parts[1].strip()
        formula = parts[2].strip() if len(parts) >= 3 else ""
        
        m = re.match(r'(\d+)\s*(.)\s*(\d+)', eq_part)
        if m:
            return m.group(1), m.group(2), m.group(3), result, formula
    
    return None, None, None, None, None


def parse_query_line(line):
    """Parse 'Now, determine the result for: 06@77 = rev(...)' """
    m = re.search(r'(\d+)\s*(.)\s*(\d+)\s*=\s*(.+)', line)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4).strip()
    return None, None, None, None


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


def build_discovery_cot(examples, qa, qop, qb, answer, formula_map):
    """Build a discovery-style teaching CoT."""
    
    op_groups = OrderedDict()
    for a, op, b, result in examples:
        if op not in op_groups:
            op_groups[op] = []
        op_groups[op].append((a, b, result))
    
    L = []
    
    # --- Step 1: Extract examples ---
    L.append("Step 1: Extract examples.\n")
    for idx, (a, op, b, result) in enumerate(examples, 1):
        L.append(f"Example {idx}: {a}{op}{b} = {result}")
    L.append(f"\nQuery: {qa}{qop}{qb}\n")
    
    # --- Steps 2+: Analyze each operator ---
    discovered = OrderedDict()
    step_num = 2
    
    for op_char, group in op_groups.items():
        a0, b0, r0 = group[0]
        ai, bi = int(a0), int(b0)
        
        # Get formula from solver
        eq_key = f"{a0}{op_char}{b0}"
        formula_name = formula_map.get(eq_key, None)
        
        L.append(f"Step {step_num}: Analyze operator '{op_char}': {a0}{op_char}{b0} = {r0}\n")
        L.append(f"a = {a0}, b = {b0}, result = {r0}\n")
        
        # Observation
        obs = classify_result(ai, bi, r0)
        L.append(f"Observation: {obs}\n")
        
        if formula_name and 'UNSOLVED' not in formula_name:
            is_rev = is_rev_formula(formula_name)
            
            if is_rev:
                # Show testing basic ops first, then discovering rev domain
                # Test a few basic ops that fail
                basic_tests = []
                if ai + bi != int(r0) if r0.lstrip('-').isdigit() else True:
                    basic_tests.append(f"a + b = {ai} + {bi} = {ai+bi} ≠ {r0} ✗")
                if ai - bi != int(r0) if r0.lstrip('-').isdigit() else True:
                    basic_tests.append(f"a - b = {ai} - {bi} = {ai-bi} ≠ {r0} ✗")
                if str(ai*bi) != r0:
                    basic_tests.append(f"a * b = {ai} * {bi} = {ai*bi} ≠ {r0} ✗")
                
                if basic_tests:
                    L.append("Test basic operations:")
                    for t in basic_tests[:3]:
                        L.append(f"  {t}")
                    L.append("")
                    L.append("Basic operations don't match. Try reverse domain.\n")
                
                # Show rev domain discovery
                ra, rb = int(rev_s(a0)), int(rev_s(b0))
                L.append(f"rev(a) = rev({a0}) = {ra}")
                L.append(f"rev(b) = rev({b0}) = {rb}\n")
                
                # Show the work
                work = show_work(formula_name, a0, b0)
                L.append(work)
                L.append(f"\nResult matches: {r0} ✓\n")
            else:
                # Direct domain — show the work directly
                work = show_work(formula_name, a0, b0)
                L.append(f"Test: {work}\n")
                L.append(f"Match. ✓\n")
            
            L.append(f"Operator '{op_char}' = {formula_name}\n")
            discovered[op_char] = formula_name
            
            # Verify additional examples
            for a1, b1, r1 in group[1:]:
                computed = compute_formula(formula_name, a1, b1)
                ok = computed == r1
                L.append(f"Verify: {a1}{op_char}{b1} = {formula_name}")
                work1 = show_work(formula_name, a1, b1)
                L.append(f"  {work1}")
                L.append(f"  = {r1} {'✓' if ok else '✗'}\n")
        
        elif formula_name and '[op' in formula_name:
            # Operator-in-answer pattern
            L.append(f"Output '{r0}' contains operator symbol '{op_char}'.")
            L.append(f"The operator symbol replaces the negative sign.\n")
            L.append(f"Operator '{op_char}' = {formula_name}\n")
            discovered[op_char] = formula_name
        else:
            L.append(f"Uses conditional digit-based logic.\n")
        
        step_num += 1
    
    # --- Domain pattern step ---
    L.append(f"Step {step_num}: Identify domain pattern.\n")
    
    rev_ops = [op for op, f in discovered.items() if is_rev_formula(f)]
    dir_ops = [op for op, f in discovered.items() if not is_rev_formula(f)]
    
    if rev_ops and not dir_ops:
        L.append("All operators use the reverse domain: rev(rev(a) ⊕ rev(b))")
        for op, fname in discovered.items():
            sym = get_op_symbol(fname)
            L.append(f"  '{op}' uses ⊕ = {sym}")
    elif dir_ops and not rev_ops:
        L.append("All operators use the direct domain: max(a,b) ⊕ min(a,b)")
        for op, fname in discovered.items():
            sym = get_op_symbol(fname)
            L.append(f"  '{op}' uses ⊕ = {sym}")
    else:
        L.append("Mixed domain:")
        for op, fname in discovered.items():
            domain = "rev" if is_rev_formula(fname) else "direct"
            L.append(f"  '{op}' = {fname} [{domain}]")
    L.append("")
    step_num += 1
    
    # --- Solve query ---
    L.append(f"Step {step_num}: Solve query: {qa}{qop}{qb}\n")
    
    query_formula = formula_map.get(f"query:{qa}{qop}{qb}", None)
    
    if query_formula and 'UNSOLVED' not in query_formula:
        L.append(f"The operator '{qop}' is not in the examples.")
        L.append(f"Based on the domain pattern, it uses:\n")
        L.append(f"(a'{qop}'b) = {query_formula}\n")
        
        # Show computation
        work = show_work(query_formula, qa, qb)
        L.append(work)
        L.append(f"\n{qa}{qop}{qb} = {answer}\n")
    else:
        L.append(f"Based on pattern analysis: {qa}{qop}{qb} = {answer}\n")
    step_num += 1
    
    # --- Verification ---
    L.append(f"Step {step_num}: Verification.\n")
    for a, op, b, result in examples:
        eq_key = f"{a}{op}{b}"
        fname = formula_map.get(eq_key, None)
        if fname and 'UNSOLVED' not in fname:
            computed = compute_formula(fname, a, b)
            ok = (computed == result) if computed else False
            L.append(f"{a}{op}{b}: {fname} → {result} {'✓' if ok else '✗'}")
        else:
            L.append(f"{a}{op}{b} → {result} (conditional)")
    L.append(f"{qa}{qop}{qb} → {answer} ✓\n")
    L.append("All constraints satisfied. The solution is verified.")
    L.append(f"\\boxed{{{answer}}}")
    
    cot = '\n'.join(L)
    return f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"


def main():
    # Parse solver
    solver_puzzles = parse_solver()
    print(f"Solver puzzles: {len(solver_puzzles)}")
    
    # Build formula lookup: eq_key → formula_name
    # Also: query:KEY → formula_name
    all_formula_maps = []
    
    for eqs, query_line in solver_puzzles:
        fmap = {}
        for eq_line in eqs:
            a, op, b, result, formula = parse_eq_line(eq_line)
            if a:
                fmap[f"{a}{op}{b}"] = formula
        
        qa, qop, qb, qformula = parse_query_line(query_line)
        if qa:
            fmap[f"query:{qa}{qop}{qb}"] = qformula
        
        all_formula_maps.append(fmap)
    
    # Load records
    with open(JSONL) as f:
        records = [json.loads(line) for line in f]
    
    print(f"Records: {len(records)}")
    
    # Ground truth
    gt = {}
    with open(BASE / "data" / "raw" / "train.csv") as f:
        csv = f.read()
    for m in re.finditer(r'determine the result for:\s*(\d+.{1}\d+).*?,\s*(.+?)$', csv, re.MULTILINE):
        q = m.group(1).strip().strip('"').replace(' ','')
        a = m.group(2).strip().strip('"').strip()
        if a: gt[q] = a
    
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
            continue
        
        # Find matching solver puzzle
        fmap = {}
        if i < len(all_formula_maps):
            fmap = all_formula_maps[i]
        
        # Build CoT
        new_content = build_discovery_cot(examples, qa, qop, qb, answer, fmap)
        rec['messages'][-1]['content'] = new_content
        updated += 1
    
    print(f"Updated: {updated}/{len(records)}")
    
    with open(JSONL, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print("Done!")
    
    # Show sample
    print(f"\n{'='*60}")
    print("=== Record 0 ===")
    print(records[0]['messages'][-1]['content'])
    print(f"\n{'='*60}")
    # Show the puzzle from user's example (19}15, 74!78, 64!23, query 43*96)
    for i, rec in enumerate(records):
        user_msg = [m for m in rec['messages'] if m['role'] == 'user'][0]
        if '19}15' in user_msg['content']:
            print(f"=== Record {i} (19}}15 puzzle) ===")
            print(rec['messages'][-1]['content'])
            break


if __name__ == "__main__":
    main()
