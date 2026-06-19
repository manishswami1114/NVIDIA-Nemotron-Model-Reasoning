"""THK-style algorithmic CoT generator for cryptarithm puzzles.

Following Mark Cooper's playbook from the public forum:
 - Operator candidates are filtered by output-length signature (THK's bitsum analog)
 - Per-operator-character arithmetic identification is shown
 - Reading mode declared explicitly
 - Symbol mapping declared as "the unique consistent assignment"
 - Full per-example verification with explicit decode → operate → encode
 - Query computed step-by-step

This is INCREMENTAL improvement over R-medium: adds derivation reasoning that
the model can transfer to unseen puzzles (operator filtering by output length,
per-character op identification).

Length target: 2000-4000 chars per record.
"""

from __future__ import annotations
import numpy as np


# Same op set as fast_solver (6 ops, expand later if needed)
OP_DESCRIPTIONS = {
    "add":     ("L + R",                        "2-3 digits when L,R are 2-digit"),
    "sub":     ("L - R (when L >= R)",          "1-2 digits when non-negative"),
    "rsub":    ("R - L (when R >= L)",          "1-2 digits when non-negative"),
    "absdiff": ("|L - R|",                      "1-2 digits"),
    "mul":     ("L * R",                        "1-4 digits (3-4 when L,R both >= 10)"),
    "gcd":     ("greatest common divisor(L,R)", "1-2 digits"),
}

OP_FNS = {
    "add":     lambda a, b: a + b,
    "sub":     lambda a, b: a - b if a >= b else -1,
    "rsub":    lambda a, b: b - a if b >= a else -1,
    "absdiff": lambda a, b: abs(a - b),
    "mul":     lambda a, b: a * b,
    "gcd":     lambda a, b: _gcd(a, b),
}


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _parse_prompt(prompt):
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
    return examples, query


def _decode_two(s, mapping, mode):
    a, b = mapping[s[0]], mapping[s[1]]
    if mode == "standard":
        return 10 * a + b
    else:  # reversed
        return 10 * b + a


def _encode_int(n, length, inv_map, mode):
    s = str(n).zfill(length)
    if mode == "reversed":
        s = s[::-1]
    return "".join(inv_map[int(c)] for c in s)


def _identify_op_per_char(mapping, examples, mode):
    """For each operator character in the examples, find which arithmetic op fits."""
    inv = {v: k for k, v in mapping.items()}
    result = {}
    op_chars = {lhs[2] for lhs, _ in examples if len(lhs) >= 5}
    for oc in op_chars:
        oc_examples = [(lhs, rhs) for (lhs, rhs) in examples if len(lhs) >= 5 and lhs[2] == oc]
        for op_name, fn in OP_FNS.items():
            all_match = True
            for lhs, rhs in oc_examples:
                L = _decode_two(lhs[0]+lhs[1], mapping, mode)
                R = _decode_two(lhs[3]+lhs[4], mapping, mode)
                v = fn(L, R)
                if v is None or v < 0:
                    all_match = False; break
                if v >= 10**len(rhs):
                    all_match = False; break
                if _encode_int(v, len(rhs), inv, mode) != rhs:
                    all_match = False; break
            if all_match:
                result[oc] = op_name
                break
    return result


def _length_compatible_ops(observed_lengths):
    """Given a list of observed output lengths, return ops that can produce all of them.

    For 2-digit operands:
      add:     produces 2-3 digits
      sub/rsub/absdiff/gcd: produces 1-2 digits (or fewer)
      mul:     produces 1-4 digits
    """
    max_len = max(observed_lengths)
    min_len = min(observed_lengths)
    compatible = []
    for op in OP_DESCRIPTIONS:
        if op == "add":
            if max_len <= 3:
                compatible.append(op)
        elif op == "mul":
            if max_len <= 4:
                compatible.append(op)
        elif op in ("sub", "rsub", "absdiff", "gcd"):
            if max_len <= 2:
                compatible.append(op)
    return compatible


def build_thk_medium_cot(prompt: str, det: dict) -> str:
    """Generate THK-style cryptarithm CoT.

    Required keys in det: mode, mapping, q_op, q_op_name, qL, qR, q_result, answer
    """
    examples, query = _parse_prompt(prompt)
    if not query or not examples:
        raise ValueError("Could not parse prompt for examples/query")

    mapping  = det["mapping"]
    mode     = det["mode"]
    q_op     = det["q_op"]
    q_op_name = det["q_op_name"]
    answer   = det["answer"]
    qL       = det["qL"]
    qR       = det["qR"]
    q_result = det["q_result"]
    inv_map  = {v: k for k, v in mapping.items()}

    # Identify operator semantics per char
    ops_per_char = _identify_op_per_char(mapping, examples, mode)
    ops_per_char.setdefault(q_op, q_op_name)

    # Distinct op chars (from examples + query)
    ex_op_chars = sorted({lhs[2] for lhs, _ in examples if len(lhs) >= 5})
    all_op_chars = sorted(set(ex_op_chars) | {q_op})

    lines = []
    lines.append("<think>")
    lines.append("This is a cryptarithm: each non-operator symbol stands for a distinct digit 0-9, and the operator symbol selects an arithmetic operation on two 2-digit operands.")
    lines.append("")
    lines.append("Encrypted equations (L op R = C):")
    for i, (lhs, rhs) in enumerate(examples, 1):
        if len(lhs) < 5: continue
        lines.append(f"  EX{i}: L='{lhs[0]}{lhs[1]}'  op='{lhs[2]}'  R='{lhs[3]}{lhs[4]}'  ->  C='{rhs}' (|C|={len(rhs)})")
    lines.append(f"Query: L='{query[0]}{query[1]}'  op='{query[2]}'  R='{query[3]}{query[4]}'  ->  ? (expected |C|={len(answer)})")
    lines.append("")

    # ── Step 1: operator-length filter ───────────────────────────
    lines.append("Step 1: Filter operator candidates by output length.")
    lines.append("For 2-digit operands L, R, each arithmetic operation produces a result of bounded length:")
    for op_name in OP_DESCRIPTIONS:
        formula, rng = OP_DESCRIPTIONS[op_name]
        lines.append(f"  - {op_name}: {formula}  ({rng})")
    lines.append("")
    lines.append("Apply this filter per operator-character based on observed output lengths:")
    for oc in all_op_chars:
        observed = [len(rhs) for lhs, rhs in examples if len(lhs) >= 5 and lhs[2] == oc]
        if not observed:
            observed = [len(answer)]
            scope = "query only"
        else:
            scope = f"examples observed: {sorted(set(observed))}"
        compatible = _length_compatible_ops(observed)
        lines.append(f"  '{oc}' ({scope}): length-compatible ops = {compatible}")
    lines.append("")

    # ── Step 2: reading mode ────────────────────────────────────
    lines.append("Step 2: Reading mode.")
    if mode == "standard":
        lines.append("  Operands are read left-to-right (standard): 'AB' represents 10*digit(A) + digit(B).")
        lines.append("  Result digits are encoded in the same left-to-right order.")
    else:
        lines.append("  Operands are read right-to-left (reversed): 'AB' represents 10*digit(B) + digit(A).")
        lines.append("  Result digits are encoded right-to-left as well.")
    lines.append("")

    # ── Step 3: distinct-digit mapping ──────────────────────────
    lines.append("Step 3: Symbol-to-digit mapping (each symbol gets a distinct digit 0-9).")
    lines.append("The unique distinct-digit assignment consistent with every example equation under the mode and length-filtered operators above is:")
    for c, d in sorted(mapping.items(), key=lambda kv: kv[1]):
        lines.append(f"  '{c}' = {d}")
    lines.append("")

    # ── Step 4: operator → arithmetic op identification ─────────
    lines.append("Step 4: Operator-character to arithmetic-op identification.")
    lines.append("Given the mapping, the only arithmetic op consistent with every example using each operator character is:")
    for oc in all_op_chars:
        op_name = ops_per_char.get(oc)
        if op_name is not None:
            lines.append(f"  '{oc}' -> {op_name} ({OP_DESCRIPTIONS[op_name][0]})")
    lines.append("")

    # ── Step 5: verify every example ────────────────────────────
    lines.append("Step 5: Verify mapping + ops on each example.")
    for i, (lhs, rhs) in enumerate(examples, 1):
        if len(lhs) < 5: continue
        oc = lhs[2]
        op_name = ops_per_char.get(oc)
        if op_name is None: continue
        L = _decode_two(lhs[0]+lhs[1], mapping, mode)
        R = _decode_two(lhs[3]+lhs[4], mapping, mode)
        v = OP_FNS[op_name](L, R)
        enc = _encode_int(v, len(rhs), inv_map, mode) if v is not None and v >= 0 else None
        match = "OK" if enc == rhs else "MISMATCH"
        lines.append(f"  EX{i}: L={L} (from '{lhs[0]}{lhs[1]}', {mode}), "
                     f"R={R} (from '{lhs[3]}{lhs[4]}', {mode})")
        lines.append(f"        op '{oc}' = {op_name}, compute: {op_name}({L},{R}) = {v}")
        lines.append(f"        encode {v} as {len(rhs)} digits ({mode}): '{enc}'  target '{rhs}'  {match}")
    lines.append("")

    # ── Step 6: apply to query ──────────────────────────────────
    lines.append("Step 6: Apply to query.")
    lines.append(f"  Query L = {qL} (from '{query[0]}{query[1]}', {mode})")
    lines.append(f"  Query R = {qR} (from '{query[3]}{query[4]}', {mode})")
    lines.append(f"  Query op '{q_op}' = {q_op_name}")
    lines.append(f"  Compute: {q_op_name}({qL}, {qR}) = {q_result}")
    lines.append(f"  Encode {q_result} as {len(answer)} digits ({mode}): '{answer}'")
    lines.append("")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    lines.append("</think>")
    lines.append(f"\\boxed{{{answer}}}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    import sys, csv, json
    csv.field_size_limit(sys.maxsize)
    sys.path.insert(0, ".")
    from fast_solver import solve_fast
    with open("data/raw/train.csv") as f:
        for r in csv.DictReader(f):
            if r["id"] == "00c032a8":
                det = solve_fast(r["prompt"], r["answer"])
                cot = build_thk_medium_cot(r["prompt"], det)
                print(f"=== THK-medium CoT ({len(cot)} chars) ===\n")
                print(cot)
                break
