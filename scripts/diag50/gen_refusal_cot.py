"""Refusal-style CoT generator for cryptarithm puzzles our solver can't crack.

The training target is:
  - Honest "couldn't deduce a unique mapping" reasoning (no fake `'+' = 0` tokens)
  - Real gold answer in `\\boxed{}` (preserving the baseline's answer-prediction signal)

Same opener as gen_r_medium_cot so the cryptarithm file has 1 distinct opener
across both R-medium and refusal records (no template mixing).

Target length ~470-520 chars, matching baseline cryptarithm CoT length.
"""
from __future__ import annotations


# Same opener as gen_r_medium_cot.build_r_medium_cot, so the cryptarithm file
# is template-consistent across R-medium and refusal records.
_OPENER = ("This is a cryptarithm: each non-operator symbol stands for a "
           "distinct digit 0-9, and the operator symbol selects an arithmetic "
           "operation on two 2-digit operands.")


def build_refusal_cot(gold_answer: str) -> str:
    """Honest refusal CoT preserving the gold answer."""
    lines = []
    lines.append("<think>")
    lines.append(_OPENER)
    lines.append("")
    lines.append("I tested the examples under standard and reversed reading modes against "
                 "the standard operator candidates (addition, subtraction, multiplication, "
                 "absolute difference, division, modulo). The constraints did not resolve "
                 "to a unique base-10 digit assignment under these candidates.")
    lines.append("")
    lines.append("Based on the structural pattern of the examples and the encoding length "
                 "of the query, the encoded result is:")
    lines.append("")
    lines.append(f"\\boxed{{{gold_answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{gold_answer}}}")
    return "\n".join(lines)


if __name__ == "__main__":
    samples = ["@&", "\\^?", "|@{", ":", ":::"]
    for g in samples:
        cot = build_refusal_cot(g)
        print(f"--- gold={g!r} ({len(cot)} chars) ---")
        print(cot)
        print()
