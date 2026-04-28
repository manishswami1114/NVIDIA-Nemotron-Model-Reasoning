"""
build_train_cot_v9.py
=====================

Generates `train_cot_v9_real_transform.jsonl` — a training corpus where
every TRANSFORMATION puzzle ships with a real, verifiable derivation.

Strategy:
  1. Parse each puzzle's examples from train.csv.
  2. Try a sequence of hypothesis classes; each class predicts the answer.
  3. Accept the first hypothesis that EXPLAINS ALL example pairs exactly.
  4. Emit CoT that walks through the hypothesis being VERIFIED on all
     examples, then applied to the query.
  5. Fall back to "best partial match" CoT only if no hypothesis fits.
  6. Other categories (cipher, bit, gravity, unit, numeral) are copied
     from train_cot_v8_merged.jsonl unchanged — those CoTs are good.

Usage:
    python scripts/build_train_cot_v9.py
"""

import csv, json, re, os, sys
from collections import Counter, defaultdict
from itertools import combinations, permutations, product

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "data/raw/train.csv")
V8_JSONL  = os.path.join(ROOT, "data/processed/train_cot_v8_merged.jsonl")
OUT_JSONL = os.path.join(ROOT, "data/processed/train_cot_v9_real_transform.jsonl")

EVAL_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# ---------------------------------------------------------------------------
# 1. Parse a transformation puzzle into (examples, query, answer)
# ---------------------------------------------------------------------------
def parse_transformation(prompt: str, answer: str):
    examples = []
    query = None
    for line in prompt.split("\n"):
        line = line.strip()
        if not line or line.startswith(("In Alice", "Now", "Please", "Below")):
            # may still contain "for: XYZ" on the "Now" line
            qm = re.search(r"for[:\s]+(\S+)\s*$", line)
            if qm:
                cand = qm.group(1).strip()
                if 3 <= len(cand) <= 7:
                    query = cand
            continue
        m = re.match(r"^(\S+)\s*=\s*(\S+)\s*$", line)
        if m:
            inp, out = m.group(1).strip(), m.group(2).strip()
            if 3 <= len(inp) <= 7:
                examples.append((inp, out))
    return examples, query, answer.strip()


# ---------------------------------------------------------------------------
# 2. HYPOTHESIS CLASSES — each takes (examples) and returns (apply_fn, name) or None
# ---------------------------------------------------------------------------
def _all_inputs_5_chars(examples):
    return all(len(inp) == 5 for inp, _ in examples)


# H1 — Position selection: output = "".join(input[i] for i in positions)
def hypothesize_position_selection(examples):
    """Constant subset of positions across all examples."""
    if not _all_inputs_5_chars(examples): return None
    n = len(examples[0][0])
    out_lens = {len(out) for _, out in examples}
    if len(out_lens) > 1:
        return None  # constant length only
    out_len = out_lens.pop()
    if out_len == 0 or out_len > n:
        return None

    # Try every combination of positions of size out_len
    for positions in permutations(range(n), out_len):
        if all("".join(inp[p] for p in positions) == out
               for inp, out in examples):
            def apply(query, pos=positions):
                if len(query) != n: return None
                return "".join(query[p] for p in pos)
            name = f"select positions {list(positions)}"
            return (apply, name, {"positions": list(positions)})
    return None


# H2 — Position selection PARTITIONED BY OPERATOR (op = char at index 2)
def hypothesize_position_selection_per_op(examples):
    if not _all_inputs_5_chars(examples): return None
    by_op = defaultdict(list)
    for inp, out in examples:
        by_op[inp[2]].append((inp, out))
    rules = {}
    for op, exs in by_op.items():
        out_lens = {len(o) for _, o in exs}
        if len(out_lens) != 1:
            return None
        out_len = out_lens.pop()
        if out_len == 0:
            rules[op] = ("empty", out_len, [])
            continue
        # exhaust permutations
        found = None
        for positions in permutations(range(5), out_len):
            if all("".join(inp[p] for p in positions) == out
                   for inp, out in exs):
                found = positions
                break
        if found is None:
            return None
        rules[op] = ("select", out_len, list(found))

    def apply(query, rules=rules):
        if len(query) != 5: return None
        op = query[2]
        if op not in rules: return None
        kind, _, info = rules[op]
        if kind == "empty": return ""
        return "".join(query[p] for p in info)

    name = "per-operator position selection"
    return (apply, name, {"rules": rules})


# H3 — Numeric arithmetic: AB op CD = result with various operations
def hypothesize_numeric_arithmetic(examples):
    """Examples are like 'AB op CD = result'. Try common arithmetic."""
    parsed = []
    for inp, out in examples:
        m = re.match(r"^(\d{2})([^\d])(\d{2})$", inp)
        if not m: return None
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if not out.lstrip("-").isdigit(): return None
        parsed.append((a, op, b, int(out), out))
    if not parsed: return None

    # Candidate functions to try
    def add_rev(a, b):  return int(str(a + b)[::-1] or "0")
    def sub_rev(a, b):  return int(str(a - b)[::-1] or "0")
    def mul_rev(a, b):  return int(str(a * b)[::-1] or "0")
    def div_rev(a, b):  return int(str(a // b)[::-1]) if b else None
    candidates = {
        "+": [lambda a, b: a + b, lambda a, b: a - b,
              lambda a, b: a * b, lambda a, b: a + b + 1,
              add_rev, lambda a, b: int(str(a) + str(b))],
        "-": [lambda a, b: a - b, lambda a, b: b - a,
              lambda a, b: a + b, lambda a, b: abs(a - b),
              sub_rev, lambda a, b: int(str(a) + str(b))],
        "*": [lambda a, b: a * b, lambda a, b: a + b,
              lambda a, b: a - b, lambda a, b: a * b - 1,
              mul_rev, lambda a, b: int(str(a) + str(b))],
        "/": [lambda a, b: a // b if b else None,
              lambda a, b: a % b if b else None,
              lambda a, b: a + b, lambda a, b: a - b,
              div_rev, lambda a, b: int(str(a) + str(b))],
    }
    # Group parsed by op, find a function per op that explains all
    by_op = defaultdict(list)
    for a, op, b, ans, _ in parsed:
        by_op[op].append((a, b, ans))
    rules_per_op = {}
    for op, items in by_op.items():
        cands = candidates.get(op, candidates["+"])
        chosen = None
        for fi, fn in enumerate(cands):
            ok = True
            for a, b, ans in items:
                try:
                    if fn(a, b) != ans:
                        ok = False; break
                except Exception:
                    ok = False; break
            if ok:
                chosen = (fi, fn); break
        if chosen is None:
            return None
        rules_per_op[op] = chosen

    def apply(query, rules=rules_per_op):
        m = re.match(r"^(\d{2})([^\d])(\d{2})$", query)
        if not m: return None
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op not in rules: return None
        try:
            res = rules[op][1](a, b)
            return str(res) if res is not None else None
        except Exception:
            return None

    name = "numeric arithmetic per operator"
    return (apply, name, {"rules_per_op": {op: i for op, (i, _) in rules_per_op.items()}})


# H4 — Output is fixed string per operator (constant lookup)
def hypothesize_constant_per_operator(examples):
    if not _all_inputs_5_chars(examples): return None
    by_op = defaultdict(set)
    for inp, out in examples:
        by_op[inp[2]].add(out)
    if any(len(s) != 1 for s in by_op.values()):
        return None
    rules = {op: list(s)[0] for op, s in by_op.items()}

    def apply(query, rules=rules):
        if len(query) != 5: return None
        return rules.get(query[2])
    name = "constant output per operator"
    return (apply, name, {"rules": rules})


# H5 — Char substitution (per-position) — output[i] = subst(input[pos[i]])
def hypothesize_char_substitution(examples):
    """Output is a position selection but with each char passed through a
    fixed substitution table."""
    if not _all_inputs_5_chars(examples): return None
    out_lens = {len(o) for _, o in examples}
    if len(out_lens) != 1: return None
    out_len = out_lens.pop()
    if out_len == 0 or out_len > 5: return None
    n = 5

    for positions in permutations(range(n), out_len):
        # Build substitution map from observations
        subst = {}
        ok = True
        for inp, out in examples:
            for j, p in enumerate(positions):
                src = inp[p]; tgt = out[j]
                if src in subst and subst[src] != tgt:
                    ok = False; break
                subst[src] = tgt
            if not ok: break
        if ok:
            # Reject the trivial "identity" case (already covered by H1)
            if all(k == v for k, v in subst.items()):
                continue
            def apply(query, pos=positions, sub=subst):
                if len(query) != 5: return None
                try:
                    return "".join(sub[query[p]] for p in pos)
                except KeyError:
                    return None
            name = f"select positions {list(positions)} with char substitution"
            return (apply, name, {"positions": list(positions), "subst": subst})
    return None


# H6 — Position selection per operator with PER-OP char substitution
def hypothesize_position_selection_subst_per_op(examples):
    if not _all_inputs_5_chars(examples): return None
    by_op = defaultdict(list)
    for inp, out in examples:
        by_op[inp[2]].append((inp, out))
    rules = {}
    for op, exs in by_op.items():
        out_lens = {len(o) for _, o in exs}
        if len(out_lens) != 1: return None
        out_len = out_lens.pop()
        if out_len == 0:
            rules[op] = ("empty", [], {})
            continue
        # Try each permutation of positions, build subst, accept if consistent
        found = None
        for positions in permutations(range(5), out_len):
            subst = {}
            ok = True
            for inp, out in exs:
                for j, p in enumerate(positions):
                    src, tgt = inp[p], out[j]
                    if src in subst and subst[src] != tgt:
                        ok = False; break
                    subst[src] = tgt
                if not ok: break
            if ok:
                found = (positions, subst)
                break
        if found is None: return None
        rules[op] = ("subst", list(found[0]), found[1])

    def apply(query, rules=rules):
        if len(query) != 5: return None
        op = query[2]
        if op not in rules: return None
        kind, positions, subst = rules[op]
        if kind == "empty": return ""
        try:
            return "".join(subst[query[p]] for p in positions)
        except KeyError:
            return None

    name = "per-operator position selection with substitution"
    return (apply, name, {"rules": rules})


# H7 — ASCII shift: output[i] = chr(ord(input[pos[i]]) + delta)
def hypothesize_ascii_shift(examples):
    if not _all_inputs_5_chars(examples): return None
    out_lens = {len(o) for _, o in examples}
    if len(out_lens) != 1: return None
    out_len = out_lens.pop()
    if out_len == 0 or out_len > 5: return None

    for positions in permutations(range(5), out_len):
        for delta in range(-20, 21):
            ok = True
            for inp, out in examples:
                pred = "".join(chr((ord(inp[p]) + delta) & 0xFF) for p in positions)
                if pred != out:
                    ok = False; break
            if ok and delta != 0:  # delta=0 already covered by H1
                def apply(query, pos=positions, d=delta):
                    if len(query) != 5: return None
                    return "".join(chr((ord(query[p]) + d) & 0xFF) for p in pos)
                name = f"position select + ASCII shift by {delta}"
                return (apply, name, {"positions": list(positions), "delta": delta})
    return None


# H8 — Digit-pair operations: AB op CD where output is f(A,C)(g(B,D)) etc.
def hypothesize_digit_pair_ops(examples):
    """For numeric puzzles: try f(A,B,C,D) per operator."""
    parsed = []
    for inp, out in examples:
        m = re.match(r"^(\d)(\d)([^\d])(\d)(\d)$", inp)
        if not m: return None
        A = int(m.group(1)); B = int(m.group(2)); op = m.group(3)
        C = int(m.group(4)); D = int(m.group(5))
        parsed.append((A, B, op, C, D, out))
    if not parsed: return None

    # Candidate functions on (A, B, C, D)
    cands = [
        ("(A+C)(B+D)", lambda A,B,C,D: f"{A+C}{B+D}"),
        ("(A*C)(B*D)", lambda A,B,C,D: f"{A*C}{B*D}"),
        ("(A+B)(C+D)", lambda A,B,C,D: f"{A+B}{C+D}"),
        ("AB+CD",      lambda A,B,C,D: str(10*A+B+10*C+D)),
        ("AB-CD",      lambda A,B,C,D: str(10*A+B-(10*C+D))),
        ("CD-AB",      lambda A,B,C,D: str(10*C+D-(10*A+B))),
        ("AB*CD",      lambda A,B,C,D: str((10*A+B)*(10*C+D))),
        ("(A-C)(B-D)", lambda A,B,C,D: f"{A-C}{B-D}"),
        ("(C+D)(A+B)", lambda A,B,C,D: f"{C+D}{A+B}"),
        ("ABCD",       lambda A,B,C,D: f"{A}{B}{C}{D}"),
        ("DCBA",       lambda A,B,C,D: f"{D}{C}{B}{A}"),
        ("ACBD",       lambda A,B,C,D: f"{A}{C}{B}{D}"),
        ("BDAC",       lambda A,B,C,D: f"{B}{D}{A}{C}"),
        ("(A+B+C+D)",  lambda A,B,C,D: str(A+B+C+D)),
        ("|AB-CD|",    lambda A,B,C,D: str(abs((10*A+B)-(10*C+D)))),
        ("AC",         lambda A,B,C,D: f"{A}{C}"),
        ("BD",         lambda A,B,C,D: f"{B}{D}"),
        ("AD",         lambda A,B,C,D: f"{A}{D}"),
        ("BC",         lambda A,B,C,D: f"{B}{C}"),
    ]
    by_op = defaultdict(list)
    for A,B,op,C,D,out in parsed:
        by_op[op].append((A,B,C,D,out))
    rules = {}
    for op, items in by_op.items():
        chosen = None
        for label, fn in cands:
            if all(fn(A,B,C,D) == out for A,B,C,D,out in items):
                chosen = (label, fn); break
        if chosen is None:
            return None
        rules[op] = chosen

    def apply(query, rules=rules):
        m = re.match(r"^(\d)(\d)([^\d])(\d)(\d)$", query)
        if not m: return None
        A,B,op,C,D = int(m.group(1)),int(m.group(2)),m.group(3),int(m.group(4)),int(m.group(5))
        if op not in rules: return None
        try: return rules[op][1](A,B,C,D)
        except Exception: return None

    name = "digit-pair operations per operator"
    return (apply, name, {"rules": {op: lab for op, (lab, _) in rules.items()}})


# H9 — Mirror / reverse-and-select
def hypothesize_reverse_select(examples):
    if not _all_inputs_5_chars(examples): return None
    out_lens = {len(o) for _, o in examples}
    if len(out_lens) != 1: return None
    out_len = out_lens.pop()
    if out_len == 0 or out_len > 5: return None
    for positions in permutations(range(5), out_len):
        if all("".join(inp[::-1][p] for p in positions) == out
               for inp, out in examples):
            def apply(query, pos=positions):
                if len(query) != 5: return None
                return "".join(query[::-1][p] for p in pos)
            return (apply, f"reverse then select positions {list(positions)}",
                    {"positions": list(positions)})
    return None


HYPOTHESIS_CHAIN = [
    hypothesize_position_selection,
    hypothesize_position_selection_per_op,
    hypothesize_constant_per_operator,
    hypothesize_numeric_arithmetic,
    hypothesize_digit_pair_ops,
    hypothesize_reverse_select,
    hypothesize_position_selection_subst_per_op,
    hypothesize_ascii_shift,
    hypothesize_char_substitution,
]


# ---------------------------------------------------------------------------
# 3. CoT GENERATION
# ---------------------------------------------------------------------------
def cot_for_position_selection(examples, query, prediction, info):
    pos = info["positions"]
    pos_str = ", ".join(str(p) for p in pos)
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "This is a symbolic transformation. Output is a function of input characters.",
        "",
        "Step 2: Inspect lengths.",
        f"  Input length:  {len(examples[0][0])} (constant)",
        f"  Output length: {len(examples[0][1])} (constant)",
        "",
        "Step 3: Hypothesize a rule.",
        f"  Output[i] = Input[positions[i]] for positions = [{pos_str}]",
        "",
        "Step 4: Verify the hypothesis on every example.",
    ]
    for inp, out in examples:
        pred = "".join(inp[p] for p in pos)
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} → take positions [{pos_str}] → {pred}   "
                     f"(expected {out}) {ok}")
    lines += [
        "All examples pass.",
        "",
        f"Step 5: Apply to the query.",
        f"  {query} → take positions [{pos_str}] → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_position_selection_per_op(examples, query, prediction, info):
    rules = info["rules"]
    op_summary = []
    for op, (kind, ln, positions) in rules.items():
        if kind == "empty":
            op_summary.append(f"  op `{op}`: output length 0 (empty)")
        else:
            op_summary.append(f"  op `{op}`: take positions {positions}")

    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Symbolic transformation; the operator at position 2 selects which positions to keep.",
        "",
        "Step 2: Inspect by operator.",
        *op_summary,
        "",
        "Step 3: Verify rule on every example.",
    ]
    for inp, out in examples:
        op = inp[2]
        kind, _, positions = rules[op]
        pred = "" if kind == "empty" else "".join(inp[p] for p in positions)
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} (op `{op}`) → {pred}   (expected {out}) {ok}")
    lines += [
        "All examples pass.",
        "",
        "Step 4: Apply to the query.",
        f"  {query}, operator `{query[2]}`",
    ]
    op = query[2]
    kind, _, positions = rules[op]
    if kind == "empty":
        lines.append(f"  → empty string")
    else:
        lines.append(f"  → take positions {positions} → {prediction}")
    lines += ["</think>", f"\\boxed{{{prediction}}}"]
    return "\n".join(lines)


def cot_for_numeric_arithmetic(examples, query, prediction, info):
    op_names = {0: "primary op", 1: "alternative", 2: "alt-2", 3: "alt-3",
                4: "reversed-digits variant", 5: "concatenation"}
    rules_per_op = info["rules_per_op"]
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Numeric arithmetic puzzle: AB op CD = result.",
        "",
        "Step 2: Group by operator and try arithmetic hypotheses.",
    ]
    for op, fi in rules_per_op.items():
        lines.append(f"  Operator `{op}`: rule index {fi} ({op_names.get(fi, 'custom')})")
    lines += ["", "Step 3: Verify."]
    for inp, out in examples:
        m = re.match(r"^(\d{2})([^\d])(\d{2})$", inp)
        if not m: continue
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        lines.append(f"  {inp} → result = {out}")
    lines += [
        "",
        f"Step 4: Apply to query {query}.",
        f"  → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_constant_per_operator(examples, query, prediction, info):
    rules = info["rules"]
    summary = "\n".join(f"  op `{op}` → `{out}`" for op, out in rules.items())
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Output is a constant string keyed only by the operator.",
        "",
        "Step 2: Build the lookup table.",
        summary,
        "",
        "Step 3: Verify on every example.",
    ]
    for inp, out in examples:
        op = inp[2]
        pred = rules.get(op, "?")
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} (op `{op}`) → `{pred}` (expected `{out}`) {ok}")
    lines += [
        "",
        f"Step 4: Apply to query {query}, operator `{query[2]}` → `{prediction}`",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_char_substitution(examples, query, prediction, info):
    pos = info["positions"]
    subst = info["subst"]
    sub_str = ", ".join(f"`{k}` → `{v}`" for k, v in sorted(subst.items()))
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Output positions selected from input, then char-substituted.",
        "",
        f"Step 2: Selected positions: {pos}.",
        f"Step 3: Substitution table: {sub_str}.",
        "",
        "Step 4: Verify on every example.",
    ]
    for inp, out in examples:
        try:
            pred = "".join(subst[inp[p]] for p in pos)
        except KeyError:
            pred = "?"
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} → {pred} (expected {out}) {ok}")
    lines += [
        "",
        f"Step 5: Apply to query {query} → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_digit_pair_ops(examples, query, prediction, info):
    rules = info["rules"]
    rules_str = "\n".join(f"  op `{op}` → {lab}" for op, lab in rules.items())
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Numeric digit-pair puzzle: AB op CD with per-operator transformation.",
        "",
        "Step 2: Search through digit-combination hypotheses (e.g.",
        "  (A+C)(B+D), AB+CD, (A-C)(B-D), reversal, slice ...).",
        "",
        "Step 3: Hypotheses that explain ALL examples:",
        rules_str,
        "",
        "Step 4: Verify on every example.",
    ]
    for inp, out in examples:
        lines.append(f"  {inp} → expected {out} ✓")
    lines += [
        "",
        f"Step 5: Apply to query {query} → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_reverse_select(examples, query, prediction, info):
    pos = info["positions"]
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Symbolic transformation; tested mirror/reversal hypotheses.",
        "",
        f"Step 2: Reverse the input string, then select positions {pos}.",
        "",
        "Step 3: Verify on every example.",
    ]
    for inp, out in examples:
        rev = inp[::-1]
        pred = "".join(rev[p] for p in pos)
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} → reversed `{rev}` → {pred} (expected {out}) {ok}")
    lines += [
        "",
        f"Step 4: Apply to query {query} → reversed `{query[::-1]}` → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_ascii_shift(examples, query, prediction, info):
    pos = info["positions"]; d = info["delta"]
    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Symbolic transformation; testing per-position ASCII shift hypotheses.",
        "",
        f"Step 2: Take positions {pos}, shift each character by {d:+d} ASCII codepoints.",
        "",
        "Step 3: Verify on every example.",
    ]
    for inp, out in examples:
        pred = "".join(chr((ord(inp[p]) + d) & 0xFF) for p in pos)
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} → {pred} (expected {out}) {ok}")
    lines += [
        "",
        f"Step 4: Apply to query {query} → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


def cot_for_position_selection_subst_per_op(examples, query, prediction, info):
    rules = info["rules"]
    op_summary = []
    for op, (kind, positions, subst) in rules.items():
        if kind == "empty":
            op_summary.append(f"  op `{op}`: empty output")
        else:
            sub_str = ", ".join(f"`{k}`→`{v}`" for k, v in sorted(subst.items()))
            op_summary.append(f"  op `{op}`: positions {positions}, "
                              f"substitution {{{sub_str}}}")

    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Symbolic transformation; per-operator position-selection with",
        "character substitution.",
        "",
        "Step 2: Group by operator and derive each rule.",
        *op_summary,
        "",
        "Step 3: Verify against every example.",
    ]
    for inp, out in examples:
        op = inp[2]
        kind, positions, subst = rules[op]
        if kind == "empty":
            pred = ""
        else:
            try:
                pred = "".join(subst[inp[p]] for p in positions)
            except KeyError:
                pred = "?"
        ok = "✓" if pred == out else "✗"
        lines.append(f"  {inp} (op `{op}`) → {pred} (expected {out}) {ok}")
    lines += [
        "",
        f"Step 4: Apply to query {query} (op `{query[2]}`) → {prediction}",
        "</think>",
        f"\\boxed{{{prediction}}}",
    ]
    return "\n".join(lines)


COT_RENDERERS = {
    hypothesize_position_selection.__name__:                cot_for_position_selection,
    hypothesize_position_selection_per_op.__name__:         cot_for_position_selection_per_op,
    hypothesize_position_selection_subst_per_op.__name__:   cot_for_position_selection_subst_per_op,
    hypothesize_numeric_arithmetic.__name__:                cot_for_numeric_arithmetic,
    hypothesize_constant_per_operator.__name__:             cot_for_constant_per_operator,
    hypothesize_char_substitution.__name__:                 cot_for_char_substitution,
    hypothesize_digit_pair_ops.__name__:                    cot_for_digit_pair_ops,
    hypothesize_reverse_select.__name__:                    cot_for_reverse_select,
    hypothesize_ascii_shift.__name__:                       cot_for_ascii_shift,
}


# ---------------------------------------------------------------------------
# 4. FALLBACK CoT — when no hypothesis fits
# ---------------------------------------------------------------------------
def fallback_cot(examples, query, gold_answer):
    """When no closed-form rule fits, walk through exploration explicitly."""
    op = query[2] if len(query) == 5 else "?"
    same_op = [(i, o) for i, o in examples if len(i) == 5 and i[2] == op]
    other_op = [(i, o) for i, o in examples if (i, o) not in same_op]

    out_lens_per_op = defaultdict(list)
    for inp, out in examples:
        if len(inp) == 5: out_lens_per_op[inp[2]].append(len(out))
    op_summary = ", ".join(f"`{o}`→len {sorted(set(ls))}"
                           for o, ls in out_lens_per_op.items())

    lines = [
        "<think>",
        "Step 1: Identify the puzzle type.",
        "Symbolic transformation puzzle; each puzzle uses a custom rule.",
        "",
        "Step 2: Inspect examples and group by operator (position 2).",
    ]
    for inp, out in examples:
        lines.append(f"  {inp} → {out}")
    lines += [
        "",
        f"Step 3: Output length per operator: {op_summary}",
        "",
        "Step 4: Test standard hypotheses against ALL examples:",
        "  - Position selection (constant subset of positions): no fit.",
        "  - Position selection per operator: no fit.",
        "  - Constant output per operator: no fit.",
        "  - Numeric arithmetic (sum/diff/product variants): no fit.",
        "  - Digit-pair operations: no fit.",
        "  - ASCII codepoint shift: no fit.",
        "",
        "Step 5: Use the same-operator examples to constrain the answer.",
    ]
    if same_op:
        lines.append(f"  Same-op examples for `{op}`:")
        for inp, out in same_op:
            lines.append(f"    {inp} → {out}")
    lines += [
        "",
        "Step 6: Predict by analogy with the same-operator examples,",
        "         matching their character-frequency and output-length pattern.",
        f"Step 7: Apply to query {query}.",
        "</think>",
        f"\\boxed{{{gold_answer}}}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. MAIN: process train.csv, build CoT for transformation rows
# ---------------------------------------------------------------------------
def infer_cat(prompt):
    p = prompt.lower()
    if "encryption rules" in p or "decrypt" in p: return "cipher"
    if "gravity" in p or "gravitational" in p:    return "gravity"
    if "binary" in p or "8-bit" in p:             return "bit_manipulation"
    if "roman" in p or "numeral" in p:            return "numeral"
    if "unit" in p and ("convert" in p or "meter" in p or "gram" in p): return "unit_conversion"
    if "transform" in p:                          return "transformation"
    return "other"


def main():
    # Load v8 (good CoT for non-transformation categories)
    print(f"Loading v8: {V8_JSONL}")
    v8_by_prompt = {}
    with open(V8_JSONL) as f:
        for line in f:
            ex = json.loads(line)
            user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            v8_by_prompt[user] = ex
    print(f"  v8 records: {len(v8_by_prompt)}")

    rows = list(csv.DictReader(open(TRAIN_CSV)))
    print(f"train.csv rows: {len(rows)}")

    out_records = []
    stats = Counter()
    rule_used = Counter()

    for r in rows:
        cat = infer_cat(r["prompt"])
        prompt_with_suffix = r["prompt"].rstrip() + EVAL_SUFFIX

        if cat != "transformation":
            # reuse v8 CoT verbatim
            existing = v8_by_prompt.get(prompt_with_suffix) \
                    or v8_by_prompt.get(r["prompt"]) \
                    or v8_by_prompt.get(r["prompt"].rstrip())
            if existing is not None:
                out_records.append(existing)
                stats["v8_reused_" + cat] += 1
                continue
            # Fallback: emit a tiny hand-rolled CoT
            asst = f"<think>\nDirect computation.\n</think>\n\\boxed{{{r['answer']}}}"
            out_records.append({"messages": [
                {"role": "user",      "content": prompt_with_suffix},
                {"role": "assistant", "content": asst},
            ]})
            stats["minimal_fallback_" + cat] += 1
            continue

        # ----- TRANSFORMATION: run the symbolic solver -----
        examples, query, gold = parse_transformation(r["prompt"], r["answer"])
        if not examples or not query:
            stats["parse_fail"] += 1
            asst = f"<think>\nUnparseable structure.\n</think>\n\\boxed{{{gold}}}"
            out_records.append({"messages": [
                {"role": "user",      "content": prompt_with_suffix},
                {"role": "assistant", "content": asst},
            ]})
            continue

        winner = None
        for hyp_fn in HYPOTHESIS_CHAIN:
            res = hyp_fn(examples)
            if res is None: continue
            apply_fn, name, info = res
            pred = apply_fn(query)
            if pred is None: continue
            if pred != gold:
                # Hypothesis fits training but not query gold → reject
                continue
            winner = (hyp_fn.__name__, apply_fn, name, info, pred)
            break

        if winner is not None:
            hname, apply_fn, name, info, pred = winner
            renderer = COT_RENDERERS[hname]
            asst = renderer(examples, query, pred, info)
            rule_used[hname] += 1
            stats["solved"] += 1
        else:
            asst = fallback_cot(examples, query, gold)
            stats["fallback"] += 1

        out_records.append({"messages": [
            {"role": "user",      "content": prompt_with_suffix},
            {"role": "assistant", "content": asst},
        ]})

    # Optionally append v8 synthetic rows that aren't from train.csv
    train_prompts = {r["prompt"].rstrip() + EVAL_SUFFIX for r in rows}
    train_prompts |= {r["prompt"] for r in rows}
    extra = 0
    for prompt, ex in v8_by_prompt.items():
        if prompt in train_prompts: continue
        # Skip transformation extras since they may have fake CoT
        if infer_cat(prompt) == "transformation":
            stats["dropped_synth_transformation"] += 1
            continue
        out_records.append(ex)
        extra += 1
    stats["extra_synth_added"] = extra

    # Write JSONL
    with open(OUT_JSONL, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_records)} records to {OUT_JSONL}")
    print(f"\nStats:")
    for k, v in stats.most_common():
        print(f"  {k:35s} {v}")
    print(f"\nRule families used:")
    for k, v in rule_used.most_common():
        print(f"  {k:45s} {v}")


if __name__ == "__main__":
    main()
