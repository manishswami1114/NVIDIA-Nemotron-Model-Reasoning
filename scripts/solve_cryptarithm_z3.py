"""
solve_cryptarithm_z3.py
=======================

Uses Z3 SMT solver to find mapping constraints for Wonderland cryptarithms.
Handles varying bases, reversals, and operations.
"""

import json, os, re, sys
from pathlib import Path
import z3

ROOT = Path("/Users/manishswami/developer/NVIDIA-Nemotron Model")
TONG = ROOT / "tong_reasoners"
sys.path.insert(0, str(TONG))

from reasoners.store_types import Problem

def str_to_z3(s, z3_vars, base=10):
    val = 0
    for i, char in enumerate(reversed(s)):
        if char in z3_vars:
            val += z3_vars[char] * (base ** i)
    return val

def solve_with_z3(problem: Problem) -> str | None:
    examples = problem.examples
    question = str(problem.question)
    answer = str(problem.answer)
    
    all_text = "".join(str(ex.input_value) + str(ex.output_value) for ex in examples) + question + answer
    symbols = list(set(re.findall(r"[^\s0-9a-zA-Z]", all_text)))
    if not symbols: return None
    
    # To prevent Z3 from taking too long on overly complex ones
    if len(symbols) > 16: return None

    bases = [10, 2, 16]
    ops = ['+', '-', '*', 'concat']
    
    # split indices: we heuristically assume the operator is a single character.
    # usually index 1, 2, or 3.
    possible_splits = [1, 2, 3]

    for base in bases:
        for require_distinct in [True, False]:
            for rev_ops in [False, True]:
                for rev_res in [False, True]:
                    for split_idx in possible_splits:
                        for op in ops:
                            solver = z3.Solver()
                            z3_vars = {s: z3.Int(s) for s in symbols}
                            
                            # Add domain constraints
                            for v in z3_vars.values():
                                solver.add(v >= 0, v < base)
                            
                            if require_distinct and len(symbols) <= base:
                                solver.add(z3.Distinct(*list(z3_vars.values())))
                                
                            valid_format = True
                            for ex in examples:
                                inp, out = str(ex.input_value), str(ex.output_value)
                                if len(inp) <= split_idx: valid_format = False; break
                                
                                a_str = inp[:split_idx]
                                b_str = inp[split_idx+1:]
                                out_str = out
                                
                                if rev_ops:
                                    a_str = a_str[::-1]
                                    b_str = b_str[::-1]
                                if rev_res:
                                    out_str = out_str[::-1]
                                    
                                a_val = str_to_z3(a_str, z3_vars, base)
                                b_val = str_to_z3(b_str, z3_vars, base)
                                out_val = str_to_z3(out_str, z3_vars, base)
                                
                                if op == '+': solver.add(a_val + b_val == out_val)
                                elif op == '-': solver.add(a_val - b_val == out_val)
                                elif op == '*': solver.add(a_val * b_val == out_val)
                                elif op == 'concat':
                                    solver.add(a_val * (base**len(b_str)) + b_val == out_val)
                                    
                            if not valid_format: continue
                            
                            # Add constraint that question matches answer
                            if len(question) > split_idx:
                                qa = question[:split_idx]
                                qb = question[split_idx+1:]
                                q_out = answer
                                
                                if rev_ops:
                                    qa = qa[::-1]
                                    qb = qb[::-1]
                                if rev_res:
                                    q_out = q_out[::-1]
                                    
                                qa_v = str_to_z3(qa, z3_vars, base)
                                qb_v = str_to_z3(qb, z3_vars, base)
                                qout_v = str_to_z3(q_out, z3_vars, base)
                                
                                if op == '+': solver.add(qa_v + qb_v == qout_v)
                                elif op == '-': solver.add(qa_v - qb_v == qout_v)
                                elif op == '*': solver.add(qa_v * qb_v == qout_v)
                                elif op == 'concat':
                                    solver.add(qa_v * (base**len(qb)) + qb_v == qout_v)
                            else:
                                continue

                            # Quick check with a timeout so we don't hang
                            solver.set("timeout", 100) # 100ms timeout per configuration
                            if solver.check() == z3.sat:
                                m = solver.model()
                                mapping = {s: m[z3_vars[s]].as_long() for s in symbols}
                                
                                # Generate CoT
                                lines = [f"This is a cryptarithm solved using base-{base} arithmetic and logical constraints."]
                                lines.append(f"Deduced properties:")
                                lines.append(f"- Base: {base}")
                                lines.append(f"- Operation: {op}")
                                lines.append(f"- Reverse Operands: {rev_ops}")
                                lines.append(f"- Reverse Result: {rev_res}")
                                lines.append(f"- All symbols map to distinct values: {require_distinct}")
                                lines.append(f"Symbol Mapping:")
                                for k, v in mapping.items():
                                    lines.append(f"  '{k}' = {v}")
                                lines.append(f"The exact value calculation yields the correct answer sequence.")
                                lines.append(f"The answer is \\boxed{{{answer}}}")
                                return "\n".join(lines)
    
    # 2. Positional Substitution cipher (Length invariant)
    # Sometimes it's just a 1-to-1 character mapping without arithmetic
    mapping = {}
    consistent = True
    for ex in examples:
        inp, out = str(ex.input_value), str(ex.output_value)
        if len(inp) == len(out):
            for i in range(len(inp)):
                if inp[i] in mapping and mapping[inp[i]] != out[i]:
                    consistent = False; break
                mapping[inp[i]] = out[i]
        else:
            consistent = False; break
            
    if consistent and mapping:
        q_mapped = "".join(mapping.get(c, c) for c in question)
        if q_mapped == answer:
            lines = ["This puzzle uses a simple positional character substitution cipher."]
            lines.append("Mapping deduced from examples:")
            for c in sorted(mapping.keys()): lines.append(f"  '{c}' -> '{mapping[c]}'")
            lines.append(f"Applying to question: {question} -> {q_mapped}")
            lines.append(f"The answer is \\boxed{{{answer}}}")
            return "\n".join(lines)

    return None

def main():
    problems_index = ROOT / "tong_reasoners/problems.jsonl"
    problems_meta = []
    with problems_index.open() as f:
        for line in f: problems_meta.append(json.loads(line))

    targets = ["cryptarithm_deduce", "cryptarithm_guess"]
    os.chdir(ROOT / "tong_reasoners")
    
    for target in targets:
        records = []
        total = 0
        for meta in problems_meta:
            if meta["category"] != target: continue
            total += 1
            problem = Problem.load_from_json(meta["id"])
            res = solve_with_z3(problem)
            
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
            out_file = ROOT / "data/processed" / f"train_cot_{target}_z3.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w") as f:
                for rec in records: f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
