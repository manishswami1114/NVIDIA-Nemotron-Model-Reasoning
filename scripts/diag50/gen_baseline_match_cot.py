"""Generate baseline-template-matching CoT with REAL digit mappings.

Mirrors the byte-level structure of dont_touch_it/.../train_cot_cryptarithm_deduce.jsonl:

    <think>
    This is a cryptarithm solved using base-10 arithmetic and logical constraints.
    Deduced properties:
    - Base: 10
    - Operation: <op_char>
    - Reverse Operands: False
    - Reverse Result: False
    - All symbols map to distinct values: True
    Symbol Mapping:
      '<c>' = <d>
      ...
    The exact value calculation yields the correct answer sequence.
    The answer is \boxed{<ans>}
    </think>
    \boxed{<ans>}

Only differences from the baseline poison:
  1. "Operation:" carries the actual operator character (still 1 char, same line length)
  2. Each symbol mapping uses its real distinct digit (0-9), not all-zeros

Length stays in the 450-490 range. Same template, same vocabulary, real math.
"""
from __future__ import annotations


def build_baseline_match_cot(details: dict) -> str:
    """details from AliceEquationSolver.solve() — required keys:
        - mapping: {symbol: digit}
        - q_op: the operator character used in the query
        - answer: final answer string
    """
    mapping = details["mapping"]
    q_op    = details["q_op"]
    answer  = details["answer"]

    # Symbol order: same as baseline — appears to be a stable order across the
    # symbols. Baseline orders by listing each content_symbol; we sort by digit
    # to make the mapping easy to read (sorted by digit ascending: 0,1,2,...).
    # Either order is structurally identical for the model — both are just
    # "Symbol Mapping:" with N lines like "  'c' = d".
    items = sorted(mapping.items(), key=lambda kv: kv[1])

    lines = []
    lines.append("<think>")
    lines.append("This is a cryptarithm solved using base-10 arithmetic and logical constraints.")
    lines.append("Deduced properties:")
    lines.append("- Base: 10")
    lines.append(f"- Operation: {q_op}")
    lines.append("- Reverse Operands: False")
    lines.append("- Reverse Result: False")
    lines.append("- All symbols map to distinct values: True")
    lines.append("Symbol Mapping:")
    for c, d in items:
        lines.append(f"  '{c}' = {d}")
    lines.append("The exact value calculation yields the correct answer sequence.")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)
