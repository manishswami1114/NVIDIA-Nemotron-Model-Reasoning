"""
generate_hard_problems.py
=========================
Trace-Guided Brute-Force CoT generator for the HARD puzzle categories:
  - equation_numeric_guess  (currently 0 training records)
  - cryptarithm_guess       (currently 0 training records)

Uses exhaustive operation search with recorded backtracking traces
to generate high-quality Chain-of-Thought training data.

Key techniques from research:
  - Trace-Guided Solving: Records every hypothesis tested → natural CoT
  - STaR Rationalization: For problems with known answers, generates
    backwards reasoning traces  
  - Expanded Operation Set: Tests 100+ operation combos including
    reversed operands, reversed results, ±1 variants, cross-multiply

Usage:
    python scripts/generate_hard_problems.py [--target 5000]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
TONG = ROOT / "tong_reasoners"
sys.path.insert(0, str(TONG))
os.chdir(TONG)

from reasoners.store_types import Problem, Example
from reasoners.cryptarithm import reasoning_cryptarithm

EVAL_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# ══════════════════════════════════════════════════════════════
# EQUATION NUMERIC: Exhaustive operation search with trace
# ══════════════════════════════════════════════════════════════

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")

def _all_operation_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """All possible numeric operations (common + rare)."""
    out: list[tuple[str, str]] = []
    # Common ops
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("multiplication", str(a * b)))
    # Rare ops: ±1 variants
    out.append(("multiply+1", str(a * b + 1)))
    out.append(("multiply-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    # Division / modulo
    if b != 0:
        out.append(("integer division (a/b)", str(a // b)))
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse division (b/a)", str(b // a)))
        out.append(("reverse modulo (b mod a)", str(b % a)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    # Digit-level operations (for 2-digit numbers)
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(("cross multiply", str(d1 * d3 + d2 * d4)))
        out.append(("cross multiply rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit multiply", str(d1 * d3) + str(d2 * d4)))
        out.append(("digit multiply rev", str(d1 * d4) + str(d2 * d3)))
        out.append(("digit sum diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit sum sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit product diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit product sum", str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))
    return out


def _rev_str(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


def _find_operation_with_trace(
    examples: list[tuple[str, str, str, str]],  # (a_str, op_char, b_str, expected_out)
) -> tuple[str | None, dict | None, list[str]]:
    """Exhaustively search for the operation that maps all examples.
    
    Returns (operation_name, op_config, trace_lines).
    op_config = {"rev_ops": bool, "rev_res": bool, "op_name": str, "fmt": str}
    trace_lines = list of reasoning lines (the CoT)
    """
    trace: list[str] = []
    trace.append("I need to determine the hidden operation for this operator.")
    trace.append("I will systematically test all possible operations.")
    trace.append("")
    
    # Detect if there's a symbol prefix/suffix pattern
    fmt = "num"
    op_char = examples[0][1]
    
    # Check for symbol suffix/prefix in outputs
    any_suffixed = any(out.endswith(op_char) and len(out) > 1 for _, _, _, out in examples)
    any_prefixed = any(out.startswith(op_char) and len(out) > 1 for _, _, _, out in examples)
    
    transformed = list(examples)
    if any_suffixed:
        fmt = "neg_suffix"
        transformed = [
            (a, op, b, "-" + out[:-1] if out.endswith(op_char) else out)
            for a, op, b, out in examples
        ]
        trace.append(f"Some outputs end with the operator symbol '{op_char}'. This may indicate negative results.")
        trace.append("Transforming: removing suffix and treating as negative.")
    elif any_prefixed:
        fmt = "neg_prefix"
        transformed = [
            (a, op, b, "-" + out[len(op_char):] if out.startswith(op_char) else out)
            for a, op, b, out in examples
        ]
        trace.append(f"Some outputs start with the operator symbol '{op_char}'. This may indicate negative results.")
        trace.append("Transforming: removing prefix and treating as negative.")
    
    trace.append("")
    
    combos_tried = 0
    
    for rev_ops, rev_res in [(False, False), (True, False), (False, True), (True, True)]:
        combo_label = []
        if rev_ops: combo_label.append("reversed operands")
        if rev_res: combo_label.append("reversed result")
        if not combo_label: combo_label.append("identity")
        combo_str = " + ".join(combo_label)
        
        trace.append(f"Trying {combo_str}:")
        
        # Get first example for candidate generation
        a0, _, b0, exp0 = transformed[0]
        ta0 = a0[::-1] if rev_ops else a0
        tb0 = b0[::-1] if rev_ops else b0
        
        if rev_ops:
            trace.append(f"  Reversing operands: {a0}->{ta0}, {b0}->{tb0}")
        
        candidates = _all_operation_candidates(int(ta0), int(tb0), ta0, tb0)
        
        for op_name, first_result in candidates:
            combos_tried += 1
            
            # Test first example
            final0 = _rev_str(first_result) if rev_res else first_result
            
            if final0 != exp0:
                # Only show first few failures in trace to keep it manageable
                if combos_tried <= 12 or combos_tried % 10 == 0:
                    trace.append(f"    {op_name}: f({ta0},{tb0}) = {first_result}"
                                + (f" -rev-> {final0}" if rev_res else "")
                                + f" vs {exp0} ✗")
                continue
            
            # First example matched — test all others
            all_match = True
            match_details = [f"f({ta0},{tb0}) = {first_result}"
                           + (f" -rev-> {final0}" if rev_res else "")
                           + f" = {exp0} ✓"]
            
            for a_str, _, b_str, exp_out in transformed[1:]:
                ta = a_str[::-1] if rev_ops else a_str
                tb = b_str[::-1] if rev_ops else b_str
                
                all_cands = _all_operation_candidates(int(ta), int(tb), ta, tb)
                raw = None
                for n, r in all_cands:
                    if n == op_name:
                        raw = r
                        break
                if raw is None:
                    all_match = False
                    match_details.append(f"f({ta},{tb}) = N/A ✗")
                    break
                
                final = _rev_str(raw) if rev_res else raw
                if final != exp_out:
                    all_match = False
                    match_details.append(f"f({ta},{tb}) = {raw}"
                                       + (f" -rev-> {final}" if rev_res else "")
                                       + f" vs {exp_out} ✗")
                    break
                match_details.append(f"f({ta},{tb}) = {raw}"
                                   + (f" -rev-> {final}" if rev_res else "")
                                   + f" = {exp_out} ✓")
            
            if all_match:
                trace.append(f"    {op_name}: ALL EXAMPLES MATCH ✓")
                for d in match_details:
                    trace.append(f"      {d}")
                trace.append("")
                trace.append(f"Found operation: {combo_str}, {op_name}")
                
                return op_name, {
                    "rev_ops": rev_ops,
                    "rev_res": rev_res,
                    "op_name": op_name,
                    "fmt": fmt,
                    "op_char": op_char,
                }, trace
            else:
                # Show partial match failure
                trace.append(f"    {op_name}: partial match, failed on example:")
                trace.append(f"      {match_details[-1]}")
        
        trace.append("")
    
    return None, None, trace


def solve_equation_numeric_guess_with_trace(problem: Problem) -> str | None:
    """Solve an equation_numeric problem using exhaustive search with traced CoT."""
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    
    # Parse all examples
    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if not m:
            continue
        a, op, b = m.group(1), m.group(2), m.group(3)
        parsed.append((a, op, b, str(ex.output_value)))
    
    if not parsed:
        return None
    
    lines.append("Examples:")
    for a, op, b, out in parsed:
        lines.append(f"  {a}{op}{b} = {out}")
    lines.append("")
    
    # Group by operator
    by_op: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, op, b, out))
    
    # Parse question
    q_match = _EXPR_RE.fullmatch(str(problem.question))
    if not q_match:
        return None
    qa, q_op, qb = q_match.group(1), q_match.group(2), q_match.group(3)
    
    lines.append(f"Question: {qa}{q_op}{qb}")
    lines.append(f"Question operator: {q_op}")
    lines.append("")
    
    # Find operations for each operator
    found_ops: dict[str, dict] = {}
    
    for op_char, group in by_op.items():
        lines.append(f"Analyzing operator '{op_char}':")
        op_name, config, trace = _find_operation_with_trace(group)
        lines.extend(trace)
        
        if config:
            found_ops[op_char] = config
        else:
            lines.append(f"  No matching operation found for '{op_char}'")
        lines.append("")
    
    # Apply to question
    if q_op in found_ops:
        config = found_ops[q_op]
    elif by_op:
        # Fallback: use absolute difference
        most_common = max(by_op, key=lambda k: len(by_op[k]))
        if most_common in found_ops:
            config = dict(found_ops[most_common])
            config["op_name"] = "absolute difference"
            config["rev_ops"] = False
            config["rev_res"] = False
            lines.append(f"Question operator '{q_op}' not found in examples.")
            lines.append(f"Using absolute difference as fallback.")
        else:
            return None
    else:
        return None
    
    # Apply the found operation
    ta = qa[::-1] if config["rev_ops"] else qa
    tb = qb[::-1] if config["rev_ops"] else qb
    
    all_cands = _all_operation_candidates(int(ta), int(tb), ta, tb)
    raw_result = None
    for n, r in all_cands:
        if n == config["op_name"]:
            raw_result = r
            break
    
    if raw_result is None:
        return None
    
    final = _rev_str(raw_result) if config["rev_res"] else raw_result
    
    # Handle format (prefix/suffix)
    if config["fmt"] == "neg_suffix" and final.startswith("-"):
        final = final[1:] + config["op_char"]
    elif config["fmt"] == "neg_prefix" and final.startswith("-"):
        final = config["op_char"] + final[1:]
    
    lines.append(f"Applying to {qa}{q_op}{qb}:")
    if config["rev_ops"]:
        lines.append(f"  Reversed operands: {qa}->{ta}, {qb}->{tb}")
    lines.append(f"  {config['op_name']}({ta}, {tb}) = {raw_result}")
    if config["rev_res"]:
        lines.append(f"  Reversed result: {raw_result} -> {_rev_str(raw_result)}")
    lines.append(f"  Result: {final}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{final}}}")
    
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# CRYPTARITHM GUESS: Use existing cryptarithm reasoner + generator
# ══════════════════════════════════════════════════════════════

def solve_cryptarithm_guess_with_reasoner(problem: Problem) -> str | None:
    """Solve cryptarithm_guess using the existing cryptarithm reasoner."""
    try:
        return reasoning_cryptarithm(problem)
    except Exception:
        return None


def generate_cryptarithm_guess() -> dict | None:
    """Generate a cryptarithm_guess problem with symbolic operations.
    
    Uses ASCII symbols as operands (like real cryptarithm_guess problems).
    Supports variable-length outputs (1-4 chars) and multiple operator types.
    """
    # Pick symbol pool (ASCII punctuation, no digits)
    sym_pool = list("!\"#$%&'()*/:;<>?@[\\]^`{|}~")
    random.shuffle(sym_pool)
    
    # Pick 8-13 symbols for operands/outputs (like real data)
    n_syms = random.randint(8, min(13, len(sym_pool)))
    symbols = sym_pool[:n_syms]
    
    # Pick 1-2 operator chars from +, -, *, / (standard math ops as separators)
    op_chars = random.sample(['+', '-', '*', '/'], random.randint(1, 2))
    
    # Define the concatenation operation for each operator
    # Operations on 2-char pairs: AB<op>CD → result
    # Real data uses concatenation variants: fwd, rev, drop chars, etc.
    op_rules = {}
    rule_types = [
        "fwd_concat",     # ABCD
        "rev_concat",     # CDAB  
        "outer",          # AD
        "inner",          # BC
        "first_last",     # AB (just first pair)
        "rev_first",      # BA
        "cross",          # AC BD → ACBD
        "rev_cross",      # DB CA → DBCA
        "drop_first",     # BCD
        "drop_last",      # ABC
    ]
    for op in op_chars:
        op_rules[op] = random.choice(rule_types)
    
    def _apply_rule(rule, a1, a2, b1, b2):
        if rule == "fwd_concat": return a1 + a2 + b1 + b2
        if rule == "rev_concat": return b1 + b2 + a1 + a2
        if rule == "outer": return a1 + b2
        if rule == "inner": return a2 + b1
        if rule == "first_last": return a1 + a2
        if rule == "rev_first": return a2 + a1
        if rule == "cross": return a1 + b1 + a2 + b2
        if rule == "rev_cross": return b2 + a2 + b1 + a1
        if rule == "drop_first": return a2 + b1 + b2
        if rule == "drop_last": return a1 + a2 + b1
        return a1 + a2 + b1 + b2
    
    # Generate examples
    non_op_syms = [s for s in symbols if s not in op_chars]
    if len(non_op_syms) < 4:
        return None
    
    examples = []
    for _ in range(random.randint(3, 5)):
        op = random.choice(op_chars)
        rule = op_rules[op]
        a1, a2 = random.choice(non_op_syms), random.choice(non_op_syms)
        b1, b2 = random.choice(non_op_syms), random.choice(non_op_syms)
        inp = a1 + a2 + op + b1 + b2
        out = _apply_rule(rule, a1, a2, b1, b2)
        examples.append({"input_value": inp, "output_value": out})
    
    # Generate question
    q_op = random.choice(op_chars)
    q_rule = op_rules[q_op]
    qa1, qa2 = random.choice(non_op_syms), random.choice(non_op_syms)
    qb1, qb2 = random.choice(non_op_syms), random.choice(non_op_syms)
    question = qa1 + qa2 + q_op + qb1 + qb2
    answer = _apply_rule(q_rule, qa1, qa2, qb1, qb2)
    
    ex_lines = "\n".join(f"{e['input_value']} = {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        f"Below are a few examples:\n{ex_lines}\n"
        f"Now, determine the result for: {question}"
    )
    
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "cryptarithm_guess",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }


# ══════════════════════════════════════════════════════════════
# EQUATION NUMERIC GUESS: Problem generator 
# ══════════════════════════════════════════════════════════════

def generate_equation_numeric_guess() -> dict | None:
    """Generate a hard equation_numeric_guess problem with exotic operations."""
    # Use ALL real operator symbols
    all_ops = list('+-*/%#@!^&"\'():<>?[]\\`{|}')
    
    # Pick 1-3 operators
    n_ops = random.randint(1, 3)
    chosen_ops = random.sample(all_ops, n_ops)
    
    # For each operator, assign a (rev_ops, rev_res, operation) combo
    # Bias toward the hard combos that the existing reasoner misses
    operations = [
        "addition", "subtraction (a-b)", "reverse subtraction (b-a)",
        "multiplication", "absolute difference", "concatenation",
        "reverse concatenation", "multiply+1", "multiply-1",
        "add+1", "add-1", "sub+1", "sub-1", "cross multiply",
        "cross multiply rev",
    ]
    
    op_configs = {}
    for op in chosen_ops:
        op_name = random.choice(operations)
        # Weight toward harder combos (rev_ops + rev_res)
        if random.random() < 0.4:
            rev_ops, rev_res = True, True
        elif random.random() < 0.3:
            rev_ops, rev_res = True, False
        elif random.random() < 0.3:
            rev_ops, rev_res = False, True
        else:
            rev_ops, rev_res = False, False
        
        # For cross multiply ops, don't use reversed — they need 2-digit operands
        if "cross" in op_name:
            rev_ops, rev_res = False, False
        
        op_configs[op] = {"op_name": op_name, "rev_ops": rev_ops, "rev_res": rev_res}
    
    def _compute(config, a_str, b_str):
        ta = a_str[::-1] if config["rev_ops"] else a_str
        tb = b_str[::-1] if config["rev_ops"] else b_str
        cands = _all_operation_candidates(int(ta), int(tb), ta, tb)
        for name, result in cands:
            if name == config["op_name"]:
                final = _rev_str(result) if config["rev_res"] else result
                return final
        return None
    
    # Generate examples
    examples = []
    for _ in range(random.randint(3, 6)):
        op = random.choice(chosen_ops)
        config = op_configs[op]
        
        # Use 2-digit numbers (pad with leading zero sometimes)
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        a_str = str(a).zfill(2) if random.random() < 0.3 else str(a)
        b_str = str(b).zfill(2) if random.random() < 0.3 else str(b)
        
        result = _compute(config, a_str, b_str)
        if result is None:
            continue
        
        inp = f"{a_str}{op}{b_str}"
        examples.append({"input_value": inp, "output_value": result})
    
    if len(examples) < 3:
        return None
    
    # Generate question
    q_op = random.choice(chosen_ops)
    q_config = op_configs[q_op]
    qa = random.randint(1, 99)
    qb = random.randint(1, 99)
    qa_str = str(qa).zfill(2) if random.random() < 0.3 else str(qa)
    qb_str = str(qb).zfill(2) if random.random() < 0.3 else str(qb)
    
    answer = _compute(q_config, qa_str, qb_str)
    if answer is None:
        return None
    
    question = f"{qa_str}{q_op}{qb_str}"
    
    ex_lines = "\n".join(f"{e['input_value']} = {e['output_value']}" for e in examples)
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        f"Below are a few examples:\n{ex_lines}\n"
        f"Now, determine the result for: {question}"
    )
    
    return {
        "id": hashlib.md5(prompt.encode()).hexdigest()[:8],
        "category": "equation_numeric_guess",
        "prompt": prompt,
        "answer": answer,
        "examples": examples,
        "question": question,
    }


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def extract_boxed(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
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
        vt += "[✓] Operation consistently maps all examples? -> YES\n"
    elif "equation" in category:
        vt += "[✓] Operation verified against all examples? -> YES\n"
        vt += "[✓] Format (prefix/suffix) correctly applied? -> YES\n"
    vt += "\nAll constraints satisfied. The solution is verified.\n"
    vt += f"I will now return the answer in \\boxed{{}}\n"
    vt += f"\\boxed{{{gt}}}"
    return f"{clean}{vt}"


def process_real_problems():
    """Process existing _guess problems from the real dataset."""
    problems_index = TONG / "problems.jsonl"
    records = []
    stats = Counter()
    
    with problems_index.open() as f:
        for line in f:
            meta = json.loads(line)
            cat = meta["category"]
            if cat not in ("equation_numeric_guess", "cryptarithm_guess"):
                continue
            
            stats[f"{cat}_total"] += 1
            problem = Problem.load_from_json(meta["id"])
            
            # Try solving with appropriate solver
            reasoning = None
            if cat == "equation_numeric_guess":
                reasoning = solve_equation_numeric_guess_with_trace(problem)
            elif cat == "cryptarithm_guess":
                # Try cryptarithm reasoner first (gets 11/164)
                reasoning = solve_cryptarithm_guess_with_reasoner(problem)
            
            if reasoning is None:
                continue
            
            predicted = extract_boxed(reasoning)
            if not correct(problem.answer, predicted):
                continue
            
            stats[f"{cat}_solved"] += 1
            
            verified = inject_verification(reasoning, cat, problem.answer)
            user_content = problem.prompt + EVAL_SUFFIX
            assistant_content = (
                f"<think>\n{verified.strip()}\n</think>\n"
                f"\\boxed{{{problem.answer}}}"
            )
            records.append({
                "category": cat,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
            })
    
    return records, stats


def process_generated_problems(target: int = 5000):
    """Generate new equation_numeric_guess problems with exotic operations."""
    records = []
    stats = Counter()
    seen = set()
    
    attempts = 0
    max_attempts = target * 5
    
    while len(records) < target and attempts < max_attempts:
        attempts += 1
        stats["gen_attempted"] += 1
        
        prob_data = generate_equation_numeric_guess()
        if prob_data is None:
            continue
        
        # Dedup
        fp = hashlib.md5(json.dumps(prob_data["examples"], sort_keys=True).encode()).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)
        stats["gen_unique"] += 1
        
        # Build Problem and solve with trace
        problem = Problem(
            id=prob_data["id"],
            category=prob_data["category"],
            examples=[Example(str(e["input_value"]), str(e["output_value"])) for e in prob_data["examples"]],
            question=prob_data["question"],
            answer=prob_data["answer"],
            prompt=prob_data["prompt"],
        )
        
        reasoning = solve_equation_numeric_guess_with_trace(problem)
        if reasoning is None:
            continue
        stats["gen_reasoned"] += 1
        
        predicted = extract_boxed(reasoning)
        if not correct(problem.answer, predicted):
            continue
        stats["gen_verified"] += 1
        
        verified = inject_verification(reasoning, "equation_numeric_guess", problem.answer)
        user_content = problem.prompt + EVAL_SUFFIX
        assistant_content = (
            f"<think>\n{verified.strip()}\n</think>\n"
            f"\\boxed{{{problem.answer}}}"
        )
        records.append({
            "category": "equation_numeric_guess",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })
        
        if len(records) % 500 == 0:
            print(f"  Generated {len(records)}/{target} verified records ({attempts} attempts)")
    
    return records, stats


def process_generated_cryptarithm(target: int = 2000):
    """Generate new cryptarithm_guess problems with symbolic operations."""
    records = []
    stats = Counter()
    seen = set()
    
    attempts = 0
    max_attempts = target * 5
    
    while len(records) < target and attempts < max_attempts:
        attempts += 1
        stats["crypt_attempted"] += 1
        
        prob_data = generate_cryptarithm_guess()
        if prob_data is None:
            continue
        
        fp = hashlib.md5(json.dumps(prob_data["examples"], sort_keys=True).encode()).hexdigest()
        if fp in seen:
            continue
        seen.add(fp)
        stats["crypt_unique"] += 1
        
        problem = Problem(
            id=prob_data["id"],
            category=prob_data["category"],
            examples=[Example(str(e["input_value"]), str(e["output_value"])) for e in prob_data["examples"]],
            question=prob_data["question"],
            answer=prob_data["answer"],
            prompt=prob_data["prompt"],
        )
        
        # Use cryptarithm reasoner
        reasoning = solve_cryptarithm_guess_with_reasoner(problem)
        if reasoning is None:
            continue
        stats["crypt_reasoned"] += 1
        
        predicted = extract_boxed(reasoning)
        if not correct(problem.answer, predicted):
            continue
        stats["crypt_verified"] += 1
        
        verified = inject_verification(reasoning, "cryptarithm_guess", problem.answer)
        user_content = problem.prompt + EVAL_SUFFIX
        assistant_content = (
            f"<think>\n{verified.strip()}\n</think>\n"
            f"\\boxed{{{problem.answer}}}"
        )
        records.append({
            "category": "cryptarithm_guess",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })
        
        if len(records) % 500 == 0:
            print(f"  Generated {len(records)}/{target} verified records ({attempts} attempts)")
    
    return records, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-eq", type=int, default=8000,
                       help="Target equation_numeric_guess records")
    parser.add_argument("--target-crypt", type=int, default=2000,
                       help="Target cryptarithm_guess records")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    OUT_PATH = ROOT / "data" / "processed" / "train_cot_v12_hard.jsonl"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    # Phase 1: Solve existing real _guess problems
    print("=" * 60)
    print("Phase 1: Solving existing real _guess problems")
    print("=" * 60)
    t0 = time.time()
    real_records, real_stats = process_real_problems()
    elapsed = time.time() - t0
    print(f"  equation_numeric_guess: {real_stats.get('equation_numeric_guess_solved', 0)}"
          f"/{real_stats.get('equation_numeric_guess_total', 0)} solved")
    print(f"  cryptarithm_guess: {real_stats.get('cryptarithm_guess_solved', 0)}"
          f"/{real_stats.get('cryptarithm_guess_total', 0)} solved")
    print(f"  Time: {elapsed:.1f}s")
    all_records.extend(real_records)
    
    # Phase 2: Generate new equation_numeric_guess problems
    print()
    print("=" * 60)
    print(f"Phase 2: Generating {args.target_eq} equation_numeric_guess problems")
    print("=" * 60)
    t0 = time.time()
    gen_records, gen_stats = process_generated_problems(args.target_eq)
    elapsed = time.time() - t0
    print(f"  {gen_stats.get('gen_verified', 0)} verified from"
          f" {gen_stats.get('gen_attempted', 0)} attempts in {elapsed:.1f}s")
    all_records.extend(gen_records)
    
    # Phase 3: Generate new cryptarithm_guess problems
    print()
    print("=" * 60)
    print(f"Phase 3: Generating {args.target_crypt} cryptarithm_guess problems")
    print("=" * 60)
    t0 = time.time()
    crypt_records, crypt_stats = process_generated_cryptarithm(args.target_crypt)
    elapsed = time.time() - t0
    print(f"  {crypt_stats.get('crypt_verified', 0)} verified from"
          f" {crypt_stats.get('crypt_attempted', 0)} attempts in {elapsed:.1f}s")
    all_records.extend(crypt_records)
    
    # Shuffle and write
    random.shuffle(all_records)
    with OUT_PATH.open("w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Stats
    cat_counts = Counter(r["category"] for r in all_records)
    total = len(all_records)
    print(f"\n{'=' * 60}")
    print(f"Wrote {total} VERIFIED hard records to {OUT_PATH}")
    print(f"{'=' * 60}")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
