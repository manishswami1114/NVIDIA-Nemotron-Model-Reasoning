"""
solve_and_split_weak_categories.py
==================================

Final robust reasoner for structural puzzles (cryptarithm).
"""

import json, os, re, sys, math, random, itertools
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/Users/manishswami/developer/NVIDIA-Nemotron Model")
TONG = ROOT / "tong_reasoners"
sys.path.insert(0, str(TONG))

from reasoners.store_types import Problem
from reasoners.equation_numeric import _EXPR_RE, _all_candidates

def solve_structural_cryptarithm(problem: Problem) -> str | None:
    """
    Deduces which input indices are kept and in what order for each operator.
    """
    examples = problem.examples
    question = str(problem.question)
    answer = str(problem.answer)
    
    # Group examples by operator (assuming operator is at index 2 if len=5)
    by_op = defaultdict(list)
    for ex in examples:
        inp, out = str(ex.input_value), str(ex.output_value)
        op = inp[2] if len(inp) == 5 else "unknown"
        by_op[op].append((inp, out))
        
    op_rules = {}
    
    # For each operator, find the index transformation
    for op, group in by_op.items():
        found_rule = False
        # Try all permutations of all lengths
        for length in range(1, 6):
            for indices in itertools.permutations(range(5), length):
                # Check if this index rule works for all in group
                all_match = True
                for inp, out in group:
                    if len(inp) != 5: all_match = False; break
                    try:
                        reconstructed = "".join(inp[i] for i in indices)
                        if reconstructed != out:
                            all_match = False; break
                    except: all_match = False; break
                if all_match:
                    op_rules[op] = indices
                    found_rule = True; break
            if found_rule: break
            
    # Determine rule for question
    q_op = question[2] if len(question) == 5 else "unknown"
    rule = op_rules.get(q_op)
    
    # If q_op not found, try the most common rule
    if not rule and op_rules:
        rule = Counter(op_rules.values()).most_common(1)[0][0]
        
    if rule:
        reconstructed = "".join(question[i] for i in rule)
        if reconstructed == answer:
            lines = ["The transformation rule depends on the operator and selects specific indices from the input string."]
            lines.append("Deduced Rules:")
            for op, indices in op_rules.items():
                lines.append(f"  Operator '{op}' -> Indices {indices}")
            lines.append(f"\nApplying rule to question '{question}' (operator '{q_op}'):")
            lines.append(f"  Indices used: {rule}")
            lines.append(f"  Result: {''.join(question[i] for i in rule)}")
            lines.append(f"The answer is \\boxed{{{answer}}}")
            return "\n".join(lines)
            
    return None

def improved_equation_numeric_guess(problem: Problem) -> str | None:
    # (Kept same, already high)
    parsed = []
    by_op = defaultdict(list)
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if m:
            a, op, b = m.group(1), m.group(2), m.group(3)
            parsed.append((a, op, b, str(ex.output_value)))
            by_op[op].append((a, b, str(ex.output_value)))
            
    q_match = _EXPR_RE.fullmatch(str(problem.question))
    if not q_match: return None
    qa, q_op, qb = q_match.group(1), q_match.group(2), q_match.group(3)
    
    potential_metas = []
    for op, group in by_op.items():
        for rev_ops, rev_res in [(True, True), (False, False), (True, False), (False, True)]:
            all_match = True
            for a_ex, b_ex, out_ex in group:
                ta = a_ex[::-1] if rev_ops else a_ex
                tb = b_ex[::-1] if rev_ops else b_ex
                cands = dict(_all_candidates(int(ta), int(tb), ta, tb))
                match = False
                for name, res in cands.items():
                    if (res[::-1] if rev_res else res) == out_ex:
                        match = True; break
                if not match: all_match = False; break
            if all_match: potential_metas.append((rev_ops, rev_res)); break

    if not potential_metas: potential_metas = [(False, False)]
    for rev_ops, rev_res in potential_metas:
        ta, tb = qa[::-1] if rev_ops else qa, qb[::-1] if rev_ops else qb
        for name, res in dict(_all_candidates(int(ta), int(tb), ta, tb)).items():
            final = res[::-1] if rev_res else res
            if final == str(problem.answer):
                lines = [f"Meta-rules: reverse_ops={rev_ops}, reverse_res={rev_res}"]
                lines.append(f"Operation: {name}")
                lines.append(f"Calculation: {ta} {name} {tb} = {res} -> {final}")
                lines.append(f"The answer is \\boxed{{{final}}}")
                return "\n".join(lines)
    return None

def main():
    problems_index = ROOT / "tong_reasoners/problems.jsonl"
    problems_meta = []
    with problems_index.open() as f:
        for line in f: problems_meta.append(json.loads(line))

    targets = ["equation_numeric_guess", "cryptarithm_deduce", "cryptarithm_guess"]
    os.chdir(ROOT / "tong_reasoners")
    
    for target in targets:
        records = []
        total = 0
        for meta in problems_meta:
            if meta["category"] != target: continue
            total += 1
            problem = Problem.load_from_json(meta["id"])
            res = improved_equation_numeric_guess(problem) if target == "equation_numeric_guess" else solve_structural_cryptarithm(problem)
            if res:
                records.append({
                    "id": meta["id"],
                    "messages": [
                        {"role": "user", "content": problem.prompt + "\nPlease put your final answer inside `\\boxed{}`."},
                        {"role": "assistant", "content": f"<think>\n{res}\n</think>\n\\boxed{{{problem.answer}}}"}
                    ]
                })
        print(f"Category: {target} | Solved: {len(records)}/{total} ({len(records)/total*100 if total > 0 else 0:.1f}%)")
        if records:
            out_file = ROOT / "data/processed" / f"train_cot_{target}.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w") as f:
                for rec in records: f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
