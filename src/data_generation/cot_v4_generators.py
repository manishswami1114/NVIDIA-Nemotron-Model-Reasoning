"""
CoT v4 — REAL reasoning trace generators.

Each function returns (cot_text, answer) or (None, None) to SKIP.
We only emit CoT when the solver can VERIFY the answer — no templates.
"""
import re
from itertools import combinations
from typing import Optional, Tuple


# =============================================================================
# BIT MANIPULATION — truth table + linear solver with step-by-step trace
# =============================================================================

def _b2v(s):  return [int(c) for c in s]
def _v2b(v):  return ''.join(str(b) for b in v)
def _b2i(s):  return int(s, 2)
def _i2b(n):  return format(n & 0xFF, '08b')

# ── Fast single-op hypotheses ────────────────────────────────────────────────
def _check_single_ops(examples):
    """Return (name, apply_fn) for the first single-op that fits all examples."""
    inps = [e[0] for e in examples]
    outs = [e[1] for e in examples]

    def check(name, fn):
        if all(fn(i) == o for i, o in zip(inps, outs)):
            return name, fn
        return None

    # NOT
    r = check("bitwise NOT (flip every bit)", lambda s: _i2b(_b2i(s) ^ 0xFF))
    if r: return r

    # Reverse
    r = check("reverse all bits", lambda s: s[::-1])
    if r: return r

    # Rotations
    for n in range(1, 8):
        r = check(f"rotate left by {n}",
                  lambda s, n=n: s[n:] + s[:n])
        if r: return r
        r = check(f"rotate right by {n}",
                  lambda s, n=n: s[-n:] + s[:-n])
        if r: return r

    # XOR with constant
    for c in range(1, 256):
        r = check(f"XOR with {_i2b(c)} (0x{c:02X})",
                  lambda s, c=c: _i2b(_b2i(s) ^ c))
        if r: return r

    # Shifts
    for n in range(1, 8):
        r = check(f"shift left by {n}",
                  lambda s, n=n: _i2b((_b2i(s) << n) & 0xFF))
        if r: return r
        r = check(f"shift right by {n}",
                  lambda s, n=n: _i2b(_b2i(s) >> n))
        if r: return r

    return None


# ── Truth-table solver (handles any Boolean function) ────────────────────────
def _solve_truth_table(examples, query, max_deps=4):
    """
    For each output bit j, find the smallest set S of input positions (|S| ≤ max_deps)
    such that output[j] is a consistent AND FULLY DETERMINED function of input[S].

    "Fully determined" = we've seen all 2^|S| possible patterns of input[S] in the examples.
    This prevents picking a rule that accidentally fits the examples but generalises wrong.

    Returns (answer_str, [(S_j, truth_table_j), ...]) or None.
    """
    inputs  = [_b2v(e[0]) for e in examples]
    outputs = [_b2v(e[1]) for e in examples]

    soln = []
    for j in range(8):
        target = [o[j] for o in outputs]
        found  = None
        for size in range(0, max_deps + 1):
            full_patterns = 2 ** size  # need to see all patterns
            for S in combinations(range(8), size):
                tt   = {}
                ok   = True
                for inp, t in zip(inputs, target):
                    key = tuple(inp[i] for i in S)
                    if key in tt and tt[key] != t:
                        ok = False; break
                    tt[key] = t
                # Accept only if ALL input patterns for S appear in examples
                if ok and len(tt) == full_patterns:
                    found = (list(S), tt); break
            if found: break
        if found is None:
            return None
        soln.append(found)

    # Apply to query
    q   = _b2v(query)
    out = []
    for S, tt in soln:
        key = tuple(q[i] for i in S)
        if key not in tt:
            return None   # query pattern unseen in examples
        out.append(tt[key])
    return _v2b(out), soln


def _describe_rule(soln):
    """Human-readable description of a truth-table solution."""
    lines = []
    for j, (S, tt) in enumerate(soln):
        if not S:
            val = list(tt.values())[0]
            lines.append(f"  output[{j}] = {val}  (constant)")
        elif len(S) == 1:
            # Could be identity or NOT
            k = S[0]
            entries = sorted(tt.items())
            if entries == [((0,), 0), ((1,), 1)]:
                lines.append(f"  output[{j}] = input[{k}]")
            elif entries == [((0,), 1), ((1,), 0)]:
                lines.append(f"  output[{j}] = NOT input[{k}]")
            else:
                lines.append(f"  output[{j}] = f(input[{k}])  table={dict(entries)}")
        else:
            # Describe XOR / AND / OR if it fits
            xor_val = None
            if all(sum(k) % 2 == v for k, v in tt.items()):
                xor_val = f"XOR input[{', '.join(map(str, S))}]"
            if xor_val:
                lines.append(f"  output[{j}] = {xor_val}")
            else:
                lines.append(f"  output[{j}] = f(input[{list(S)}])  table={tt}")
    return "\n".join(lines)


def gen_bit_manipulation_cot(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    examples    = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    query_match = re.search(r'determine the output for:\s*([01]{8})', prompt, re.IGNORECASE)
    if not examples or not query_match:
        return None, None
    query = query_match.group(1)

    lines = []
    lines.append(f"I need to find the 8-bit transformation rule from these examples and apply it to {query}.")
    lines.append("")
    lines.append("Examples:")
    for inp, out in examples:
        lines.append(f"  {inp} → {out}")
    lines.append("")

    # ── Try single-op first (fast path, clean CoT) ──────────────────────────
    single = _check_single_ops(examples)
    if single:
        name, fn = single
        # Show 1 failed attempt for color
        for test_name, test_fn in [
            ("bitwise NOT",     lambda s: _i2b(_b2i(s) ^ 0xFF)),
            ("reverse bits",    lambda s: s[::-1]),
        ]:
            if test_name != name.split()[0]:
                inp0, out0 = examples[0]
                got = test_fn(inp0)
                if got != out0:
                    lines.append(f"Trying {test_name}: {inp0} → {got}  (expected {out0}) ✗")
                    lines.append("")
                    break

        lines.append(f"Trying {name}:")
        all_ok = True
        for inp, out in examples:
            got = fn(inp)
            mark = "✓" if got == out else "✗"
            lines.append(f"  {inp} → {got}  (expected {out}) {mark}")
            if got != out:
                all_ok = False
        lines.append("")
        if all_ok:
            answer = fn(query)
            lines.append(f"All examples match. Rule: {name}.")
            lines.append("")
            lines.append(f"Applying to {query}:")
            lines.append(f"  {query} → {fn(query)}")
            return "\n".join(lines), answer

    # ── Truth-table solver (handles complex / non-linear rules) ─────────────
    lines.append("Simple single-operation rules don't fit. I'll look for a per-bit Boolean function.")
    lines.append("")
    result = _solve_truth_table(examples, query, max_deps=4)
    if result is None:
        return None, None
    answer, soln = result

    lines.append("For each output bit, I'll find which input bits determine it:")
    lines.append("")
    lines.append(_describe_rule(soln))
    lines.append("")
    lines.append(f"Applying the per-bit rules to {query}:")
    q   = _b2v(query)
    out = []
    for j, (S, tt) in enumerate(soln):
        key = tuple(q[i] for i in S)
        bit = tt[key]
        out.append(bit)
        if S:
            src = ', '.join(f"input[{i}]={q[i]}" for i in S)
            lines.append(f"  output[{j}]: {src} → {bit}")
        else:
            lines.append(f"  output[{j}]: constant → {bit}")
    lines.append(f"\nResult: {''.join(map(str, out))}")
    return "\n".join(lines), answer


# =============================================================================
# CIPHER — substitution with inference for missing chars
# =============================================================================

def gen_cipher_cot(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    lines_in    = prompt.strip().split('\n')
    examples    = []
    query_text  = None

    for line in lines_in:
        line = line.strip()
        if ' -> ' in line:
            parts = line.split(' -> ', 1)
            if len(parts) == 2:
                examples.append((parts[0].strip(), parts[1].strip()))
        elif 'decrypt the following text:' in line.lower():
            m = re.search(r'decrypt the following text:\s*(.*)', line, re.IGNORECASE)
            if m:
                query_text = m.group(1).strip()

    if not examples or not query_text:
        return None, None

    # Build char map from all examples
    char_map = {}
    for enc, dec in examples:
        enc_w = enc.split()
        dec_w = dec.split()
        if len(enc_w) != len(dec_w):
            return None, None
        for ew, dw in zip(enc_w, dec_w):
            if len(ew) != len(dw):
                return None, None
            for e, d in zip(ew, dw):
                if e == ' ':
                    continue
                if e in char_map and char_map[e] != d:
                    return None, None  # conflict
                char_map[e] = d

    # Infer missing query chars via bijection:
    # chars used as output so far vs all lowercase letters
    used_outputs = set(char_map.values())
    all_alpha    = set('abcdefghijklmnopqrstuvwxyz')
    unused_in    = [c for c in query_text if c not in ' ' and c not in char_map]
    unused_out   = list(all_alpha - used_outputs)

    # If exactly one unmapped input char and one unused output → deterministic
    if len(set(unused_in)) == 1 and len(unused_out) == 1:
        char_map[unused_in[0]] = unused_out[0]

    # Apply mapping
    result_chars = []
    for c in query_text:
        if c == ' ':
            result_chars.append(' ')
        elif c in char_map:
            result_chars.append(char_map[c])
        else:
            return None, None  # still unknown — skip
    result = ''.join(result_chars)

    # Build CoT
    mapping_str = ', '.join(f"{k}→{v}" for k, v in sorted(char_map.items()))
    cot = (
        "This is a character substitution cipher.\n\n"
        "Examples:\n"
        + "\n".join(f"  {e} → {d}" for e, d in examples)
        + "\n\nPairing each encrypted character with its plaintext counterpart:\n"
        f"  {mapping_str}\n"
    )
    if set(unused_in):
        cot += (
            f"\nThe character `{''.join(set(unused_in))}` doesn't appear in any example. "
            f"Since this must be a bijection and the only unused output letter is "
            f"`{''.join(unused_out)}`, we can infer the mapping.\n"
        )
    cot += f"\nDecrypting `{query_text}`:\n  `{query_text}` → `{result}`"
    return cot, result
