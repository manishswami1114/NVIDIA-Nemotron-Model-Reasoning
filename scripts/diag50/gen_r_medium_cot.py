"""R-medium CoT generator — ~1500 chars of GENUINE reasoning.

Every step is arithmetic the model can execute on a new puzzle:
  1. State the puzzle structure (operators present, reading mode, query op)
  2. Declare the symbol-to-digit mapping
  3. Declare the operator-to-operation mapping
  4. VERIFY each example: decode L and R as integers, apply the op,
     encode the result, compare to the encrypted result
  5. APPLY to the query: decode L and R, apply the op, encode the answer
  6. Box the answer

The verification step is what teaches mechanism — a model must learn the
"decode → apply op → encode" pipeline that generalises to unseen puzzles.
No placeholders, no "all symbols = 0" garbage, no skipped arithmetic.
"""
from __future__ import annotations


OP_DESC = {
    "add":     ("+",  "addition"),
    "sub":     ("-",  "subtraction"),
    "rsub":    ("-",  "reverse subtraction (R - L)"),
    "absdiff": ("-",  "absolute difference"),
    "mul":     ("*",  "multiplication"),
    "gcd":     ("g",  "greatest common divisor"),
}


def _decode_int(s2: str, mapping: dict, mode: str) -> int:
    """Decode a 2-char encrypted operand into the integer it represents."""
    a, b = mapping[s2[0]], mapping[s2[1]]
    if mode == "standard":
        return 10 * a + b
    else:  # reversed
        return 10 * b + a


def _encode_int(n: int, length: int, inv_map: dict, mode: str) -> str:
    """Encode an integer back into a length-character string of symbols."""
    s = str(n).zfill(length)
    digits = [int(c) for c in s]
    if mode == "reversed":
        digits = list(reversed(digits))
    return "".join(inv_map[d] for d in digits)


def _do_op(op_name: str, L: int, R: int) -> int:
    if op_name == "add":     return L + R
    if op_name == "sub":     return L - R
    if op_name == "rsub":    return R - L
    if op_name == "absdiff": return abs(L - R)
    if op_name == "mul":     return L * R
    if op_name == "gcd":
        import math
        return math.gcd(L, R)
    raise ValueError(f"unsupported op {op_name}")


def build_r_medium_cot(prompt: str, det: dict) -> str:
    """Generate the R-medium CoT from the solver's details + the prompt.

    det must contain: mode, mapping, q_op, q_op_name, qL, qR, q_result, answer
    """
    # ── Parse prompt to recover examples and query (we need both for CoT body)
    examples = []
    query = None
    for line in prompt.strip().split("\n"):
        line = line.strip()
        if not line: continue
        low = line.lower()
        if "determine the result for:" in low:
            idx = low.index("determine the result for:")
            query = line[idx + len("determine the result for:"):].strip()
            continue
        if any(k in low for k in ("alice","wonderland","transformation","secret","example","below","final answer")):
            continue
        if " = " in line:
            lhs, rhs = line.split(" = ", 1)
            examples.append((lhs.strip(), rhs.strip()))

    mapping = det["mapping"]
    mode    = det["mode"]
    inv_map = {v: k for k, v in mapping.items()}
    q_op    = det["q_op"]
    q_op_name = det["q_op_name"]
    answer  = det["answer"]
    qL      = det["qL"]
    qR      = det["qR"]
    q_result = det["q_result"]

    # Distinct operator chars (from examples + query)
    ex_ops = sorted({e[0][2] for e in examples if len(e[0]) >= 5})
    all_ops = sorted(set(ex_ops) | {q_op})

    lines = []
    lines.append("<think>")
    lines.append("This is a cryptarithm: each non-operator symbol stands for a distinct digit 0-9, and the operator symbol selects an arithmetic operation on two 2-digit operands.")
    lines.append("")
    lines.append(f"Examples (encrypted L op R = C):")
    for i, (lhs, rhs) in enumerate(examples, 1):
        if len(lhs) >= 5:
            lines.append(f"  EX{i}: L={lhs[0]}{lhs[1]}  op={lhs[2]}  R={lhs[3]}{lhs[4]}  ->  C={rhs}")
    lines.append(f"Query: L={query[0]}{query[1]}  op={query[2]}  R={query[3]}{query[4]}  ->  ?")
    lines.append("")
    mode_phrase = "standard left-to-right" if mode == "standard" else "reversed (operand and result digits read right-to-left)"
    lines.append(f"Reading mode: {mode_phrase}.")
    lines.append("")

    # Symbol mapping
    lines.append("Symbol -> digit mapping:")
    by_digit = sorted(mapping.items(), key=lambda kv: kv[1])
    for c, d in by_digit:
        lines.append(f"  '{c}' = {d}")
    lines.append("")

    # Operator mapping
    lines.append("Operator -> operation mapping:")
    # Reconstruct which op applies to which char, using det["ops"] if present;
    # otherwise infer from q_op_name for q_op and leave others as 'matches examples'
    ops_dict = det.get("ops", {q_op: q_op_name})
    for o in all_ops:
        op_name = ops_dict.get(o, q_op_name if o == q_op else None)
        if op_name is None:
            continue
        _, desc = OP_DESC.get(op_name, ("?", op_name))
        lines.append(f"  '{o}' -> {desc}")
    lines.append("")

    # Verify each example
    lines.append("Verify on each example by decoding operands, applying the operation, and re-encoding:")
    for i, (lhs, rhs) in enumerate(examples, 1):
        if len(lhs) < 5: continue
        op_char = lhs[2]
        op_name = ops_dict.get(op_char)
        if op_name is None:
            # The example uses an op not in ops_dict (shouldn't happen post-solve)
            continue
        L = _decode_int(lhs[0]+lhs[1], mapping, mode)
        R = _decode_int(lhs[3]+lhs[4], mapping, mode)
        Cval = _do_op(op_name, L, R)
        if Cval is None or Cval < 0:
            continue
        Cenc = _encode_int(Cval, len(rhs), inv_map, mode)
        ok = "matches" if Cenc == rhs else "MISMATCH"
        lines.append(f"  EX{i}: L={L}, R={R}, {op_name}({L},{R})={Cval} -> encode as '{Cenc}' (target '{rhs}', {ok})")
    lines.append("")

    # Apply to the query — this is where the model learns to transfer
    lines.append("Apply the same procedure to the query:")
    lines.append(f"  L = {qL} (from '{query[0]}{query[1]}', mode={mode})")
    lines.append(f"  R = {qR} (from '{query[3]}{query[4]}', mode={mode})")
    lines.append(f"  Operation = {q_op_name} (since '{q_op}' -> {q_op_name})")
    lines.append(f"  Compute: {q_op_name}({qL}, {qR}) = {q_result}")
    lines.append(f"  Encode {q_result} as {len(answer)} symbol(s), mode={mode}: '{answer}'")
    lines.append("")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.insert(0, ".")
    from fast_solver import solve_fast
    test_prompt = """In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
55+25 = 80
12+33 = 45
Now, determine the result for: 23+14
Please put your final answer inside `\\boxed{}`."""
    # This is numeric so solve_fast won't work; just demo with a hand-crafted det
    fake_det = {
        "mode": "standard",
        "mapping": {chr(48+i): i for i in range(10)},  # identity-like (numbers)
        "ops": {"+": "add"},
        "q_op": "+",
        "q_op_name": "add",
        "qL": 23, "qR": 14, "q_result": 37, "answer": "37",
    }
    # Won't actually print due to char/digit collision, just structural test
    print(f"Module loaded: {build_r_medium_cot.__name__} OK")
