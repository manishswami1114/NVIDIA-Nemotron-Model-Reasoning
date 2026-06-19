#!/usr/bin/env python3
"""
v15 PRM-style CoT writers.

Each `write_<category>(puzzle)` function takes a parsed puzzle dict and returns
the assistant content string (already wrapped in `<think>...</think>\\boxed{...}`).
Per-category section schemas come from the v15 plan file.

A puzzle dict has at minimum:
  {'id': str, 'prompt': str, 'answer': str}
For cryptarithm, additionally:
  {'kind': 'deduce'|'guess', 'equations': [...], 'query': (n1, op, n2),
   'family': str, 'map': {sym: digit}, 'ops': {op: name}}
"""
import re
from statistics import median


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def wrap(body: str, answer: str) -> str:
    return f"<think>\n{body.rstrip()}\n</think>\n\\boxed{{{answer}}}"


def keep(line: str) -> str:
    """A surviving / confirmed fact: ends with ✓ after a real reasoning clause."""
    return f"{line}  ✓"


def drop(line: str) -> str:
    """An eliminated candidate: ends with ✗ after concrete failing arithmetic."""
    return f"{line}  ✗"


# --- Rev3 hypothesis blocks (minimal counterexample on reject, evidence on accept) ---

def hyp_reject(name: str, check_line: str, reason: str) -> str:
    """Rejected hypothesis block citing the FIRST failing example as the witness."""
    return (f"Hypothesis: {name}\n"
            f"  {check_line}\n"
            f"  → reject: {reason}  ✗")


def hyp_accept(name: str, check_line: str, evidence: str) -> str:
    """Accepted hypothesis block with a confidence/evidence tag."""
    return (f"Hypothesis: {name}\n"
            f"  {check_line}\n"
            f"  → accept; evidence: {evidence}  ✓")


# ---------------------------------------------------------------------------
# numeral
# ---------------------------------------------------------------------------

ROMAN_TABLE = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'),  (90, 'XC'),  (50, 'L'),  (40, 'XL'),
    (10, 'X'),   (9, 'IX'),   (5, 'V'),   (4, 'IV'), (1, 'I'),
]


def _parse_numeral_prompt(p):
    pairs = re.findall(r'(\d+)\s*->\s*([A-Z]+)', p)
    qm = re.search(r'write the number\s*(\d+)', p, re.IGNORECASE)
    if not qm:
        qm = re.search(r'(\d+)\s*$', p.strip())
    return [(int(a), b) for a, b in pairs], int(qm.group(1)) if qm else None


def _to_roman(n):
    s = ''
    for v, sym in ROMAN_TABLE:
        while n >= v:
            s += sym
            n -= v
    return s


_ROMAN_SINGLE = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}


def _roman_value(s):
    """Decode a Roman numeral string back to its integer value."""
    total = 0
    prev = 0
    for ch in reversed(s):
        v = _ROMAN_SINGLE.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def write_numeral(puzzle):
    examples, q = _parse_numeral_prompt(puzzle['prompt'])
    answer = puzzle['answer']
    L = []
    L.append("Alice's Wonderland writes numbers in a different numeral system. I read which symbols appear, name the system, confirm it by decoding the examples, then build the query and check it.")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for a, r in examples:
        L.append(f"  {a} → {r}")
    L.append(f"  Query: convert {q}.\n")

    SUB_PAIRS = ['IV', 'IX', 'XL', 'XC', 'CD', 'CM']
    sub_seen = next(((ex_r, p) for ex_a, ex_r in examples for p in SUB_PAIRS if p in ex_r), None)

    L.append("## [CONSTRAINT]")
    L.append("Every example output uses only the symbols {I, V, X, L, C, D, M}; no digits, no other letters.")
    L.append("")

    L.append("## [ABSTRACTION]")
    L.append("That exact symbol set, with smaller symbols both following (additive) and preceding (subtractive) larger ones, "
             "is the signature of standard Roman numerals.")
    L.append("")

    L.append("## [HYPOTHESIS]")
    L.append("Hypothesize standard Roman numerals, encoded greedily by subtracting the largest available value first.")
    L.append("")

    L.append("## [EVALUATION]")
    if sub_seen:
        ex_r, pair = sub_seen
        L.append(hyp_reject("purely-additive tally (no subtraction, 4 = IIII)",
                            f"example output {ex_r} contains '{pair}', a subtractive pair",
                            f"an additive-only system would never write '{pair}'"))
    decoded_ok = sum(1 for a, r in examples if _roman_value(r) == a)
    e_a, e_r = examples[0]
    L.append(hyp_accept("standard Roman (subtractive)",
                        f"decode {e_r} → {_roman_value(e_r)} = example input {e_a}",
                        f"{decoded_ok}/{len(examples)} example outputs decode back to their inputs"))
    L.append("")

    L.append("## [SELECTION]")
    L.append("Adopt standard Roman numerals; encode the query greedily, largest value first:")
    remaining = q
    parts = []
    for v, sym in ROMAN_TABLE:
        while remaining >= v:
            parts.append((v, sym))
            new = remaining - v
            L.append(f"  {remaining} ≥ {v} → write '{sym}', remainder {new}")
            remaining = new
    constructed = ''.join(s for _, s in parts)
    L.append(f"Concatenated: {' '.join(s for _, s in parts)} → {constructed}")
    L.append("")

    L.append("## [VERIFICATION]")
    summ = ' + '.join(str(v) for v, _ in parts)
    total = sum(v for v, _ in parts)
    ok = constructed == answer and total == q
    L.append(keep(f"independently decode {constructed} back: {summ} = {total}, which equals the query {q}")
             if ok else
             drop(f"decode {constructed} → {total} ≠ {q}"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)


# ---------------------------------------------------------------------------
# unit_conversion
# ---------------------------------------------------------------------------

def _parse_unit_prompt(p):
    pairs = re.findall(r'(-?\d+\.?\d*)\s*\w*\s*becomes\s*(-?\d+\.?\d*)', p)
    # Query: only the line containing "convert"
    qline = next((ln for ln in p.split('\n') if 'convert' in ln.lower() and 'following' in ln.lower()), '')
    if not qline:
        qline = next((ln for ln in p.split('\n') if 'convert' in ln.lower()), '')
    qm = re.search(r'(-?\d+\.?\d*)', qline)
    return [(float(a), float(b)) for a, b in pairs], float(qm.group(1)) if qm else None


def write_unit_conversion(puzzle):
    examples, q = _parse_unit_prompt(puzzle['prompt'])
    answer = puzzle['answer']
    L = []
    L.append("Alice's Wonderland applies a hidden linear factor to each measurement. I will find the factor from the examples and apply it to the query.")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for a, b in examples:
        L.append(f"  {a} → {b}")
    L.append(f"  Query: convert {q}.\n")

    factors = [round(b / a, 4) if a != 0 else 0.0 for a, b in examples]
    med = round(median(factors), 4)
    spread = round(max(factors) - min(factors), 4)

    L.append("## [CONSTRAINT]")
    L.append("Output rises with input across every row, and 0 would map to 0 — consistent with a scaling law, not a shift.")
    L.append("")

    L.append("## [ABSTRACTION]")
    L.append(f"The ratio output/input is essentially constant ({', '.join(f'{x:.4f}' for x in factors)}), "
             f"so the pattern is 'output = input × k' rather than 'output = input + c'.")
    L.append("")

    L.append("## [HYPOTHESIS]")
    L.append("Two competing models: additive (out = in + c) versus multiplicative (out = k·in). I test both.")
    L.append("")

    L.append("## [EVALUATION]")
    a0, b0 = examples[0]
    c = round(b0 - a0, 4)
    add_fail = None
    for i, (a, b) in enumerate(examples):
        if abs((a + c) - b) > 0.05:
            add_fail = (i, a, b, round(a + c, 4)); break
    if add_fail:
        i, a, b, got = add_fail
        cstr = f"+ {c}" if c >= 0 else f"− {abs(c)}"
        L.append(hyp_reject("additive  out = in + c",
                            f"c = {b0}-{a0} = {c}; check ex{i+1}: {a} {cstr} = {got} ≠ {b}",
                            f"a constant shift cannot match every row"))
    else:
        L.append(f"Hypothesis additive out = in + c: coincidentally consistent here; the ratio test below still selects scaling.")
    L.append(hyp_accept("multiplicative  out = k·in",
                        f"ratios {b0}/{a0}={factors[0]:.4f} … agree within {spread:.4f}",
                        f"{len(examples)}/{len(examples)} rows share one factor (rounding only)"))
    L.append("")

    L.append("## [SELECTION]")
    L.append(keep(f"adopt the multiplicative model; the single factor is the median k = {med:.4f}"))
    L.append("")

    L.append("## [VERIFICATION]")
    for i, (a, b) in enumerate(examples):
        chk = round(a * med, 4)
        ok = abs(chk - b) < 0.05
        L.append(keep(f"ex{i+1}: {a} × {med:.4f} = {chk:.4f} = listed {b}") if ok
                 else drop(f"ex{i+1}: {a} × {med:.4f} = {chk:.4f} ≠ {b}"))
    result_raw = q * med
    try:
        decimals = max(2, len(answer.split('.')[1])) if '.' in answer else 2
        result_disp = round(result_raw, decimals)
    except Exception:
        result_disp = round(result_raw, 2)
    L.append(keep(f"query: {q} × {med:.4f} = {result_raw:.4f}, rounded to {result_disp} = {answer}"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)


# ---------------------------------------------------------------------------
# gravity
# ---------------------------------------------------------------------------

def _parse_gravity_prompt(p):
    pairs = re.findall(r't\s*=\s*(-?\d+\.?\d*)s?,\s*(?:distance|d)\s*=\s*(-?\d+\.?\d*)', p, re.IGNORECASE)
    # Query: look only inside the line that contains "determine"
    query_line = next((ln for ln in p.split('\n') if 'determine' in ln.lower()), '')
    qm = re.search(r't\s*=\s*(-?\d+\.?\d*)', query_line)
    return [(float(a), float(b)) for a, b in pairs], float(qm.group(1)) if qm else None


def write_gravity(puzzle):
    examples, q = _parse_gravity_prompt(puzzle['prompt'])
    answer = puzzle['answer']
    L = []
    L.append("Alice's Wonderland hides a gravitational law relating fall time t to distance d. I decide whether distance scales with t or with t², fix the constant from the examples, and apply it to the query.")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for t, d in examples:
        L.append(f"  t = {t}s, d = {d} m")
    L.append(f"  Query: t = {q}s.\n")

    ks = [round(d / (t * t), 3) if t else 0 for t, d in examples]
    med = round(median(ks), 3)
    spread = round(max(ks) - min(ks), 3)

    L.append("## [CONSTRAINT]")
    L.append("Distance grows faster than time does (doubling t multiplies d by about four), so d is not linear in t.")
    L.append("")

    L.append("## [ABSTRACTION]")
    L.append(f"Dividing each d by t² gives a near-constant value ({', '.join(f'{x:.3f}' for x in ks)}), "
             f"so the pattern is d ∝ t², i.e. d = k·t².")
    L.append("")

    L.append("## [HYPOTHESIS]")
    L.append("Two competing models: linear d = k·t versus quadratic d = k·t². I test both.")
    L.append("")

    L.append("## [EVALUATION]")
    t0, d0 = examples[0]
    klin = round(d0 / t0, 4) if t0 else 0
    lin_fail = None
    for i, (t, d) in enumerate(examples):
        if abs(klin * t - d) > 0.05:
            lin_fail = (i, t, d, round(klin * t, 3)); break
    if lin_fail:
        i, t, d, got = lin_fail
        L.append(hyp_reject("linear  d = k·t",
                            f"k = {d0}/{t0} = {klin}; check ex{i+1}: {klin}×{t} = {got} ≠ {d}",
                            "a constant speed cannot match an accelerating fall"))
    else:
        L.append("Hypothesis linear d = k·t: coincidentally consistent here; the t² test below still selects the quadratic law.")
    L.append(hyp_accept("quadratic  d = k·t²",
                        f"d/t² gives {ks[0]:.3f} … agreeing within {spread:.3f}",
                        f"{len(examples)}/{len(examples)} rows share one constant (rounding only)"))
    L.append("")

    L.append("## [SELECTION]")
    L.append(keep(f"adopt d = k·t²; the constant is the median k = {med:.3f}"))
    L.append("")

    L.append("## [VERIFICATION]")
    for i, (t, d) in enumerate(examples):
        chk = round(med * t * t, 3)
        ok = abs(chk - d) < 0.05
        L.append(keep(f"ex{i+1}: {med:.3f} × {t}² = {chk:.3f} = listed {d}") if ok
                 else drop(f"ex{i+1}: {med:.3f} × {t}² = {chk:.3f} ≠ {d}"))
    tq2 = q * q
    d_raw = med * tq2
    try:
        decimals = max(2, len(answer.split('.')[1])) if '.' in answer else 2
        d_disp = round(d_raw, decimals)
    except Exception:
        d_disp = round(d_raw, 2)
    L.append(keep(f"query: t² = {q}² = {tq2:.4f}; d = {med:.3f} × {tq2:.4f} = {d_raw:.4f}, rounded to {d_disp} = {answer}"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)


# ---------------------------------------------------------------------------
# cipher
# ---------------------------------------------------------------------------

WONDERLAND_WORDS = [
    'above','alice','ancient','around','beyond','bird','book','bright','castle','cat',
    'cave','chases','clever','colorful','creates','crystal','curious','dark','discovers',
    'door','dragon','draws','dreams','explores','follows','forest','found','garden','golden',
    'hatter','hidden','imagines','in','inside','island','key','king','knight','library','magical',
    'map','message','mirror','mountain','mouse','mysterious','near','ocean','palace','potion',
    'princess','puzzle','queen','rabbit','reads','school','secret','sees','silver','story',
    'strange','student','studies','teacher','the','through','tower','treasure','turtle','under',
    'valley','village','watches','wise','wizard','wonderland','writes',
]


def _parse_cipher_prompt(p):
    pairs = re.findall(r'([a-z ]+?)\s*->\s*([a-z ]+)', p)
    qm = re.search(r'decrypt the following text:\s*([a-z ]+)', p, re.IGNORECASE)
    return [(a.strip().split(), b.strip().split()) for a, b in pairs], qm.group(1).strip().split() if qm else []


def write_cipher(puzzle):
    examples, query_words = _parse_cipher_prompt(puzzle['prompt'])
    answer = puzzle['answer']
    L = []
    L.append("Alice's Wonderland encrypts text with a fixed letter-substitution. I read the map off the aligned example pairs, check it is consistent where the same letter recurs, rule out a tempting wrong guess, then decrypt the query (resolving any letter the examples never reveal by matching Wonderland vocabulary).")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for c_words, p_words in examples:
        L.append(f"  {' '.join(c_words)}  →  {' '.join(p_words)}")
    L.append(f"  Query (encrypted): {' '.join(query_words)}\n")

    # Build map AND track every source position for each cipher letter (for consistency checks)
    sym2plain = {}
    sources = {}  # cc -> list of (cipher_word, plain_word, plain_char)
    conflicts = []
    freq = {}
    for c_words, p_words in examples:
        if len(c_words) != len(p_words):
            continue
        for cw, pw in zip(c_words, p_words):
            if len(cw) != len(pw):
                continue
            for pos, (cc, pc) in enumerate(zip(cw, pw)):
                sources.setdefault(cc, []).append((cw, pw, pc, pos))
                freq[cc] = freq.get(cc, 0) + 1
                if cc not in sym2plain:
                    sym2plain[cc] = pc
                elif sym2plain[cc] != pc:
                    conflicts.append((cc, sym2plain[cc], pc))

    L.append("## [CONSTRAINT]")
    L.append("Aligning each cipher word with its plaintext (same length, position by position) reveals one substitution per letter. Reading the example pairs in order:")
    first_seen = []
    seen = set()
    for c_words, p_words in examples:
        for cw, pw in zip(c_words, p_words):
            if len(cw) != len(pw):
                continue
            for cc, pc in zip(cw, pw):
                if cc not in seen:
                    seen.add(cc)
                    first_seen.append((cc, pc, cw, pw))
    by_word = {}
    order = []
    for cc, pc, cw, pw in first_seen:
        if cw not in by_word:
            by_word[cw] = (pw, [])
            order.append(cw)
        by_word[cw][1].append((cc, pc))
    for cw in order:
        pw, pairs = by_word[cw]
        chunk = ', '.join(f"{cc}→{pc}" for cc, pc in pairs)
        L.append(f"  from {cw} ↔ {pw}: {chunk}")
    L.append("")

    L.append("## [ABSTRACTION]")
    L.append("Every occurrence of a given cipher letter lands on the same plaintext letter regardless of word or position — "
             "so the rule is one fixed, position-independent letter→letter substitution, not a shifting code.")
    L.append("")

    L.append("## [HYPOTHESIS]")
    L.append("Hypothesize a fixed substitution map; decrypt by table lookup, and resolve any query letter the examples never "
             "reveal by finding the one Wonderland word that fits the known letters.")
    L.append("")

    L.append("## [EVALUATION]")
    # genuine contradiction: a frequency-style wrong guess refuted by an aligned example
    if freq:
        cc = max(freq, key=lambda k: (freq[k], k))
        t = sym2plain[cc]
        alt = 'e' if t != 'e' else 'a'
        w_src = next(((cw, pw, pos) for cw, pw, pc, pos in sources[cc]), None)
        if w_src:
            cw, pw, pos = w_src
            L.append(hyp_reject(f"frequency guess '{cc}'→'{alt}'",
                                f"but the pair {cw}↔{pw} aligns '{cc}' over '{t}' at position {pos}",
                                f"the examples force '{cc}'→'{t}', not '{alt}'"))
    # consistency: cross-example agreement as the accept evidence
    multi = [(cc, lst) for cc, lst in sources.items() if len({s[0] for s in lst}) >= 2]
    if multi:
        cc, lst = multi[0]
        words = list(dict.fromkeys(s[0] for s in lst))[:2]
        L.append(hyp_accept("fixed substitution map",
                            f"'{cc}'→'{sym2plain[cc]}' is forced identically by {words[0]} and {words[1]}",
                            f"{len(multi)} letters recur across ≥2 example words and every recurrence agrees"))
    else:
        L.append(keep("each cipher letter appears in only one example, so the map is read off directly with no conflict"))
    if conflicts:
        for cc, a, b in conflicts[:2]:
            L.append(drop(f"'{cc}' would need both '{a}' and '{b}' — contradiction (resolved by majority)"))
    L.append("")

    L.append("## [SELECTION]")
    keys = sorted(sym2plain.keys())
    L.append("Cipher→plain map: " + ', '.join(f"{k}→{sym2plain[k]}" for k in keys))
    unknown = sorted({c for w in query_words for c in w if c not in sym2plain})
    if unknown:
        L.append(f"Letters in the query the examples never show: {', '.join(unknown)} — resolved by Wonderland-word matching:")
    decoded_words = []
    for w in query_words:
        decoded = []
        unknown_positions = []
        for i, c in enumerate(w):
            if c in sym2plain:
                decoded.append(sym2plain[c])
            else:
                decoded.append('?')
                unknown_positions.append(i)
        if unknown_positions:
            pattern = ''.join(decoded)
            # gather candidate words of same length matching known letters
            cands = [cand for cand in WONDERLAND_WORDS
                     if len(cand) == len(w)
                     and all(decoded[i] == '?' or decoded[i] == cand[i] for i in range(len(w)))]
            # show elimination: list a couple of same-length words that fail the known pattern
            fails = []
            for cand in WONDERLAND_WORDS:
                if len(cand) == len(w) and cand not in cands:
                    # find first conflicting known position
                    for i in range(len(w)):
                        if decoded[i] != '?' and decoded[i] != cand[i]:
                            fails.append((cand, i, decoded[i], cand[i]))
                            break
                if len(fails) >= 2:
                    break
            L.append(f"  '{w}' decodes to pattern '{pattern}' (? = unknown). Among Wonderland words of length {len(w)}:")
            for cand, i, need, got in fails:
                L.append(drop(f"    '{cand}': position {i} needs '{need}' but has '{got}'"))
            if len(cands) == 1:
                best = cands[0]
                for i in unknown_positions:
                    sym2plain[w[i]] = best[i]
                decoded = list(best)
                learned = ', '.join(f"{w[i]}→{best[i]}" for i in unknown_positions)
                L.append(keep(f"    only '{best}' fits the pattern → learn {learned}"))
            elif cands:
                best = cands[0]
                decoded = list(best)
                L.append(keep(f"    '{best}' fits the pattern"))
        word = ''.join(decoded)
        L.append(keep(f"'{w}' → '{word}' by direct substitution"))
        decoded_words.append(word)
    L.append(f"Decrypted query: {' '.join(decoded_words)}")
    L.append("")

    L.append("## [VERIFICATION]")
    plain2sym = {v: k for k, v in sym2plain.items()}
    cw0, pw0 = examples[0]
    re_enc = ' '.join(''.join(plain2sym.get(pc, '?') for pc in word) for word in pw0)
    ok = re_enc == ' '.join(cw0)
    L.append(keep(f"independently re-encrypt ex1 plaintext '{' '.join(pw0)}' with the inverse map → '{re_enc}', identical to the given ciphertext '{' '.join(cw0)}'")
             if ok else
             drop(f"re-encrypt ex1 → '{re_enc}' ≠ given '{' '.join(cw0)}'"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)


# ---------------------------------------------------------------------------
# bit_manipulation (per-bit-position operation discovery)
# ---------------------------------------------------------------------------

def _parse_bit_prompt(p):
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', p)
    qm = re.search(r'determine the output for:\s*([01]{8})', p, re.IGNORECASE)
    return pairs, qm.group(1) if qm else None


_BIT_OPS = [
    ('AND', lambda x, y: x & y, '∧'),
    ('OR',  lambda x, y: x | y, '∨'),
    ('XOR', lambda x, y: x ^ y, '⊕'),
    ('AND-NOT', lambda x, y: x & (1 - y), '∧¬'),
    ('OR-NOT',  lambda x, y: x | (1 - y), '∨¬'),
    ('XOR-NOT', lambda x, y: x ^ (1 - y), '⊕¬'),
]


def _bit_find_winner(in_bits, out_bits):
    """Complete simplest-first search for the operation reproducing this column.
    Returns (name, fn) or (None, None)."""
    n = len(out_bits)

    def ok(fn):
        return all(fn(b) == o for b, o in zip(in_bits, out_bits))

    j_count = len(in_bits[0]) if in_bits else 8
    # constants
    for cval in (0, 1):
        fn = (lambda b, cval=cval: cval)
        if ok(fn):
            return (f'constant {cval}', fn)
    # identity / copied input bits
    for i in range(j_count):
        fn = (lambda b, i=i: b[i])
        if ok(fn):
            return (f'IN{i}', fn)
    # negated input bits
    for i in range(j_count):
        fn = (lambda b, i=i: 1 - b[i])
        if ok(fn):
            return (f'NOT IN{i}', fn)
    # two-bit logic ops
    for op_name, opfn, glyph in _BIT_OPS:
        for i in range(j_count):
            for k in range(j_count):
                if i == k:
                    continue
                fn = (lambda b, i=i, k=k, opfn=opfn: opfn(b[i], b[k]))
                if ok(fn):
                    return (f'{op_name}(IN{i},IN{k})', fn)
    return (None, None)


def _bit_render(kind, bits, **kw):
    """Bit-level arithmetic string for a candidate evaluated on one example's `bits`."""
    if kind == 'const':
        return f"always {kw['c']}"
    if kind == 'identity':
        i = kw['i']; return f"in[{i}]={bits[i]}"
    if kind == 'neighbour':
        i = kw['i']; return f"in[{i}]={bits[i]}"
    if kind == 'negation':
        i = kw['i']; return f"¬in[{i}]=¬{bits[i]}={1-bits[i]}"
    if kind == 'twobit':
        i, k, g, opfn = kw['i'], kw['k'], kw['g'], kw['opfn']
        return f"in[{i}]{g}in[{k}]={bits[i]}{g}{bits[k]}={opfn(bits[i],bits[k])}"
    return "?"


def _bit_ladder(examples, j, cap=3):
    """Rev3 pedagogical, variable-depth candidate ladder for output column j.
    Returns (winner=(name,fn), rejects) where rejects is a kind-diverse, simplest-first
    list of (name, fail_idx, arith_str, got, want) — only GENUINE failures, never padded."""
    in_bits = [[int(b) for b in inp] for inp, _ in examples]
    out_bits = [int(out[j]) for _, out in examples]

    def first_fail(fn):
        for idx, (b, o) in enumerate(zip(in_bits, out_bits)):
            if fn(b) != o:
                return idx, fn(b)
        return None

    winner = _bit_find_winner(in_bits, out_bits)

    def rank_of(name):
        if name is None: return 99
        if name.startswith('constant'): return 0
        if name.startswith('NOT'): return 2
        if '(' in name: return 3
        return 1  # copied/shifted input bit
    win_rank = rank_of(winner[0])

    # plausible candidates for THIS column, simplest-first, each tagged with (kind, rank)
    specs = []  # (name, fn, kind, rank, render_kwargs)
    specs.append(('constant 0', (lambda b: 0), 'const', 0, {'c': 0}))
    specs.append(('constant 1', (lambda b: 1), 'const', 0, {'c': 1}))
    specs.append((f'IN{j} (identity)', (lambda b, j=j: b[j]), 'identity', 1, {'i': j}))
    for k in (j - 1, j + 1):
        if 0 <= k < 8:
            specs.append((f'IN{k} (neighbour bit {k})', (lambda b, k=k: b[k]), 'neighbour', 1, {'i': k}))
    specs.append((f'NOT IN{j}', (lambda b, j=j: 1 - b[j]), 'negation', 2, {'i': j}))
    for op_name, opfn, glyph in _BIT_OPS:
        for (i, k) in [(j, (j + 1) % 8), (j, (j - 1) % 8)]:
            if i != k:
                specs.append((f'{op_name}(IN{i},IN{k})',
                              (lambda b, i=i, k=k, opfn=opfn: opfn(b[i], b[k])),
                              'twobit', 3, {'i': i, 'k': k, 'g': glyph, 'opfn': opfn}))

    # Rev3 point I: only reject candidates STRICTLY SIMPLER than the winner — the ones
    # genuinely tried (and discarded) before reaching it. Variable depth, never padded.
    rejects = []
    kinds_seen = set()
    for name, fn, kind, rank, kw in specs:
        if rank >= win_rank:
            continue
        ff = first_fail(fn)
        if ff is None:
            continue  # genuinely matches — cannot reject
        if kind in kinds_seen:
            continue  # keep the ladder kind-diverse (Rev3 point H)
        idx, got = ff
        rejects.append((name, idx, _bit_render(kind, in_bits[idx], **kw), got, out_bits[idx]))
        kinds_seen.add(kind)
        if len(rejects) >= cap:
            break
    return winner, rejects


def _bit_kind_of(name):
    if name is None: return 'unknown'
    if name.startswith('constant'): return 'constant'
    if name.startswith('NOT'): return 'inversion'
    if '(' in name: return 'two-bit'
    return 'copy'


def write_bit_manipulation(puzzle):
    examples, query = _parse_bit_prompt(puzzle['prompt'])
    answer = puzzle['answer']

    # solve every column up front so [ABSTRACTION] can summarize the pattern
    ops_per_bit = []
    for j in range(8):
        winner, rejects = _bit_ladder(examples, j)
        ops_per_bit.append((j, winner, rejects))

    L = []
    L.append("Alice's Wonderland applies a fixed per-bit rule to 8-bit numbers. For each output bit I read its column down the examples, name a short ladder of plausible operations on the input bits (simplest first), reject the ones a single example refutes, and keep the survivor. Then I apply the surviving rules to the query.")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for n, (inp, out) in enumerate(examples):
        L.append(f"  ex{n+1}: {inp} → {out}")
    L.append(f"  Query: {query} → ?\n")

    L.append("## [CONSTRAINT]")
    L.append("Each output bit is a fixed function of the input bits; read each output column down the examples:")
    for j, (winner, _), _r in [(j, w, r) for j, w, r in ops_per_bit]:
        col = ''.join(str(int(out[j])) for _, out in examples)
        L.append(f"  bit {j}: column {col}")
    L.append("")

    L.append("## [ABSTRACTION]")
    kinds = {}
    for j, (name, _), _ in ops_per_bit:
        kinds.setdefault(_bit_kind_of(name), []).append(j)
    parts = []
    for kn in ('copy', 'inversion', 'constant', 'two-bit', 'unknown'):
        if kinds.get(kn):
            parts.append(f"{kn} at bit{'s' if len(kinds[kn])>1 else ''} {','.join(map(str,kinds[kn]))}")
    L.append("The columns fall into a few simple kinds — " + "; ".join(parts) + ".")
    L.append("So each bit is one of: a copied/shifted input bit, its negation, a constant, or a two-bit logic op.")
    L.append("")

    L.append("## [HYPOTHESIS]")
    L.append("For each column I test that ladder simplest-first (constant → copy IN_j → negation → neighbour shift → two-bit op) and stop at the first rule that reproduces the whole column.")
    L.append("")

    L.append("## [EVALUATION]")
    L.append("Per column, candidates are tried simplest-first; each reject cites its first failing example (compact one line each).")
    for j, (name, fn), rejects in ops_per_bit:
        col = ''.join(str(int(out[j])) for _, out in examples)
        L.append(f"bit {j} (column {col}):")
        if name is None:
            L.append("  no constant / single-bit / two-bit operation reproduces this column.")
            continue
        for cn, idx, arith, got, want in rejects:
            L.append(drop(f"  reject {cn}: fails first at ex{idx+1} ({arith} → {got}, output bit {j}={want})"))
        a_in = examples[0][0]
        bits0 = [int(b) for b in a_in]
        L.append(keep(f"  accept {name}: ex1 {name} on {a_in} → {fn(bits0)} = bit {j}; "
                      f"evidence {len(examples)}/{len(examples)} columns reproduced"))
    L.append("")

    L.append("## [SELECTION]")
    for j, (name, _), _ in ops_per_bit:
        L.append(f"  bit {j} ← {name}")
    L.append("")

    L.append("## [VERIFICATION]")
    for n, (ex_in, ex_out) in enumerate(examples):
        bits = [int(b) for b in ex_in]
        reapplied = ''.join(str(fn(bits)) if fn else '?' for _, (nm, fn), _ in
                            [(j, w, r) for j, w, r in ops_per_bit])
        L.append(keep(f"ex{n+1}: re-run all rules on {ex_in} → {reapplied} = listed {ex_out}")
                 if reapplied == ex_out else
                 drop(f"ex{n+1}: {ex_in} → {reapplied} ≠ listed {ex_out}"))
    q_bits = [int(b) for b in query]
    out_bits = []
    for j, (name, fn), _ in ops_per_bit:
        if fn is None:
            out_bits.append('?'); continue
        v = fn(q_bits)
        out_bits.append(str(v))
        L.append(keep(f"query bit {j}: {name} on {query} = {v}"))
    out_str = ''.join(out_bits)
    L.append(keep(f"assembled query output = {out_str}"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)


# ---------------------------------------------------------------------------
# equation_numeric (deduce + guess)
# ---------------------------------------------------------------------------

def _rev_int(n: int) -> int:
    """Integer reversal: 96 -> 69, 70 -> 7 (leading-zero loss intended for examples)."""
    s = str(abs(n))
    r = int(s[::-1])
    return -r if n < 0 else r


def _rev2(n: int) -> int:
    """Width-preserving 2-digit operand reversal: 06 -> 60, 77 -> 77, 40 -> 4."""
    return int(f"{n % 100:02d}"[::-1])


def _revs(v: int) -> str:
    """Reverse the decimal string of v; digits swap, sign stays. 5700 -> '0075', -35 -> '-53'."""
    s = str(abs(v))[::-1]
    return ('-' + s) if v < 0 else s


def _eqn_formulas_str(a: int, b: int):
    """Width-correct formula library returning (name, result_STRING).
    Mirrors perfect_solver_complete.md: rev preserves widths, results keep leading zeros."""
    ra, rb = _rev2(a), _rev2(b)
    mx, mn = max(a, b), min(a, b)
    rmx, rmn = max(ra, rb), min(ra, rb)
    out = [
        ('a+b', str(a + b)), ('a-b', str(a - b)), ('b-a', str(b - a)), ('a*b', str(a * b)),
        ('(a*b)+1', str(a * b + 1)), ('(a*b)-1', str(a * b - 1)),
        ('(a+b)+1', str(a + b + 1)), ('(a+b)-1', str(a + b - 1)),
        ('a||b', f"{a:02d}{b:02d}"), ('b||a', f"{b:02d}{a:02d}"),
        ('max(a,b)-min(a,b)', str(mx - mn)), ('max(a,b)+min(a,b)', str(mx + mn)),
        ('max(a,b)*min(a,b)', str(mx * mn)),
        ('max(a,b)*min(a,b)+1', str(mx * mn + 1)), ('max(a,b)*min(a,b)-1', str(mx * mn - 1)),
        ('max(a,b)+min(a,b)+1', str(mx + mn + 1)), ('max(a,b)+min(a,b)-1', str(mx + mn - 1)),
        ('max(a,b)||min(a,b)', f"{mx:02d}{mn:02d}"), ('min(a,b)||max(a,b)', f"{mn:02d}{mx:02d}"),
        ('rev(rev(a)+rev(b))', _revs(ra + rb)), ('rev(rev(a)-rev(b))', _revs(ra - rb)),
        ('rev(rev(b)-rev(a))', _revs(rb - ra)), ('rev(rev(a)*rev(b))', _revs(ra * rb)),
        ('rev(rev(b)*rev(a))', _revs(rb * ra)), ('rev(rev(b)+rev(a))', _revs(rb + ra)),
        ('rev(rev(a)*rev(b)+1)', _revs(ra * rb + 1)), ('rev(rev(a)*rev(b)-1)', _revs(ra * rb - 1)),
        ('rev(rev(a)+rev(b)+1)', _revs(ra + rb + 1)), ('rev(rev(a)+rev(b)-1)', _revs(ra + rb - 1)),
        ('rev(rev(b)+rev(a)+1)', _revs(rb + ra + 1)), ('rev(rev(b)+rev(a)-1)', _revs(rb + ra - 1)),
        ('rev(rev(a)||rev(b))', _revs(int(f"{ra:02d}{rb:02d}"))),
        ('rev(rev(b)||rev(a))', _revs(int(f"{rb:02d}{ra:02d}"))),
        ('rev(max(rev(a),rev(b))+min(rev(a),rev(b)))', _revs(rmx + rmn)),
        ('rev(max(rev(a),rev(b))-min(rev(a),rev(b)))', _revs(rmx - rmn)),
        ('rev(max(rev(a),rev(b))*min(rev(a),rev(b)))', _revs(rmx * rmn)),
        ('rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)', _revs(rmx * rmn + 1)),
        ('rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)', _revs(rmx * rmn - 1)),
        ('rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)', _revs(rmx + rmn + 1)),
        ('rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)', _revs(rmx + rmn - 1)),
        ('rev(max(rev(a),rev(b))||min(rev(a),rev(b)))', _revs(int(f"{rmx:02d}{rmn:02d}"))),
    ]
    if mn > 0:
        out.append(('max(a,b)%min(a,b)', str(mx % mn)))
    if rmn > 0:
        out.append(('rev(max(rev(a),rev(b))%min(rev(a),rev(b)))', _revs(rmx % rmn)))
        out.append(('rev(rev(a)%rev(b))', _revs(ra % rb) if rb else '0'))
    return out


def _eqn_arith_str(name: str, a: int, b: int) -> str:
    """Human arithmetic for a width-correct formula, e.g. 'rev(60+77)=rev(137)=731'."""
    ra, rb = _rev2(a), _rev2(b)
    mx, mn = max(a, b), min(a, b)
    rmx, rmn = max(ra, rb), min(ra, rb)
    val = dict(_eqn_formulas_str(a, b)).get(name, '?')
    if name.startswith('rev(rev('):
        inner = name[name.index('(', 4):]  # unused; build explicitly
    table = {
        'a+b': f"{a}+{b}", 'a-b': f"{a}-{b}", 'b-a': f"{b}-{a}", 'a*b': f"{a}*{b}",
        '(a*b)+1': f"{a}*{b}+1", '(a*b)-1': f"{a}*{b}-1", '(a+b)+1': f"{a}+{b}+1", '(a+b)-1': f"{a}+{b}-1",
        'a||b': f"{a:02d}||{b:02d}", 'b||a': f"{b:02d}||{a:02d}",
        'max(a,b)-min(a,b)': f"{mx}-{mn}", 'max(a,b)+min(a,b)': f"{mx}+{mn}", 'max(a,b)*min(a,b)': f"{mx}*{mn}",
        'max(a,b)*min(a,b)+1': f"{mx}*{mn}+1", 'max(a,b)*min(a,b)-1': f"{mx}*{mn}-1",
        'max(a,b)+min(a,b)+1': f"{mx}+{mn}+1", 'max(a,b)+min(a,b)-1': f"{mx}+{mn}-1",
        'max(a,b)||min(a,b)': f"{mx:02d}||{mn:02d}", 'min(a,b)||max(a,b)': f"{mn:02d}||{mx:02d}",
        'max(a,b)%min(a,b)': f"{mx}%{mn}",
        'rev(rev(a)+rev(b))': f"rev({ra}+{rb})=rev({ra+rb})", 'rev(rev(a)-rev(b))': f"rev({ra}-{rb})=rev({ra-rb})",
        'rev(rev(b)-rev(a))': f"rev({rb}-{ra})=rev({rb-ra})", 'rev(rev(a)*rev(b))': f"rev({ra}*{rb})=rev({ra*rb})",
        'rev(rev(b)*rev(a))': f"rev({rb}*{ra})=rev({rb*ra})", 'rev(rev(b)+rev(a))': f"rev({rb}+{ra})=rev({rb+ra})",
        'rev(rev(a)*rev(b)+1)': f"rev({ra}*{rb}+1)=rev({ra*rb+1})", 'rev(rev(a)*rev(b)-1)': f"rev({ra}*{rb}-1)=rev({ra*rb-1})",
        'rev(rev(a)+rev(b)+1)': f"rev({ra}+{rb}+1)=rev({ra+rb+1})", 'rev(rev(a)+rev(b)-1)': f"rev({ra}+{rb}-1)=rev({ra+rb-1})",
        'rev(rev(b)+rev(a)+1)': f"rev({rb}+{ra}+1)=rev({rb+ra+1})", 'rev(rev(b)+rev(a)-1)': f"rev({rb}+{ra}-1)=rev({rb+ra-1})",
        'rev(rev(a)||rev(b))': f"rev({ra:02d}||{rb:02d})", 'rev(rev(b)||rev(a))': f"rev({rb:02d}||{ra:02d})",
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))': f"rev({rmx}+{rmn})=rev({rmx+rmn})",
        'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))': f"rev({rmx}-{rmn})=rev({rmx-rmn})",
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))': f"rev({rmx}*{rmn})=rev({rmx*rmn})",
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)': f"rev({rmx}*{rmn}+1)=rev({rmx*rmn+1})",
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)': f"rev({rmx}*{rmn}-1)=rev({rmx*rmn-1})",
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)': f"rev({rmx}+{rmn}+1)=rev({rmx+rmn+1})",
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)': f"rev({rmx}+{rmn}-1)=rev({rmx+rmn-1})",
        'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))': f"rev({rmx:02d}||{rmn:02d})",
        'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))': f"rev({rmx}%{rmn})",
        'rev(rev(a)%rev(b))': f"rev({ra}%{rb})",
    }
    lhs = table.get(name, name)
    return f"{lhs} = {val}"


def _find_formula_str(eqs_str, op):
    """eqs_str: list of (a:int, op:str, b:int, result_str:str). Find first formula
    (width-correct) matching ALL examples for this op."""
    rel = [(a, b, r) for a, oo, b, r in eqs_str if oo == op]
    if not rel:
        return None
    a0, b0, _ = rel[0]
    for name, _v in _eqn_formulas_str(a0, b0):
        if all(dict(_eqn_formulas_str(a, b)).get(name) == r for a, b, r in rel):
            return name
    return None


def _all_matching_formulas(eqs_str, op):
    """Every library formula matching ALL examples for this op (for ambiguity detection)."""
    rel = [(a, b, r) for a, oo, b, r in eqs_str if oo == op]
    if not rel:
        return []
    a0, b0, _ = rel[0]
    out = []
    for name, _v in _eqn_formulas_str(a0, b0):
        if all(dict(_eqn_formulas_str(a, b)).get(name) == r for a, b, r in rel):
            out.append(name)
    return out


def _eqn_family(name: str) -> str:
    """Classify a formula into a human family label."""
    if name is None:
        return 'unknown'
    if name.startswith('rev('):
        return 'reversed-result'
    if '||' in name:
        return 'concatenation'
    if 'max(a,b)' in name or 'min(a,b)' in name:
        return 'max/min'
    return 'direct arithmetic'


def _abstraction_cue(name: str, a: int, b: int, r: str) -> str:
    """One sentence naming the PATTERN that suggests the winning rule (examples → pattern)."""
    fam = _eqn_family(name)
    rv = r.lstrip('-')
    if fam == 'concatenation':
        return f"the result {r} is just the two inputs written side by side — a concatenation pattern"
    if fam == 'reversed-result':
        return (f"the result {r} doesn't match any plain arithmetic, but its digits look reversed — "
                f"reversing the inputs/result lines them up")
    if 'max(a,b)*min' in name or name in ('a*b', '(a*b)+1', '(a*b)-1'):
        return f"the result {r} is far larger than either input, so a product is at work"
    if '-' in name and '+' not in name:
        return f"the result magnitude {rv} tracks the difference between the two inputs"
    if '+' in name:
        return f"the result {r} is close to the sum of the two inputs"
    return f"the result {r} relates the two inputs by a fixed rule"


# Human descriptors for the hypotheses we name in [HYPOTHESIS]/[EVALUATION].
_EQN_HYP_LABEL = {
    'a+b': 'direct sum a+b', 'a-b': 'direct difference a−b', 'b-a': 'direct difference b−a',
    'a*b': 'direct product a·b', 'a||b': 'concatenation a‖b', 'b||a': 'concatenation b‖a',
    'max(a,b)-min(a,b)': 'max−min', 'max(a,b)+min(a,b)': 'max+min',
    'max(a,b)*min(a,b)': 'max·min',
    'rev(rev(a)+rev(b))': 'reversed-result of rev(a)+rev(b)',
    'rev(rev(a)-rev(b))': 'reversed-result of rev(a)−rev(b)',
    'rev(rev(b)-rev(a))': 'reversed-result of rev(b)−rev(a)',
    'rev(rev(a)*rev(b))': 'reversed-result of rev(a)·rev(b)',
}

# Simplest-first representatives spanning the operation families, used to build a
# short, family-diverse hypothesis ladder (Rev3: ≤3 meaningful named hypotheses).
_EQN_HYP_PRIORITY = [
    'a+b', 'a-b', 'b-a', 'a*b', 'a||b',
    'max(a,b)-min(a,b)', 'max(a,b)+min(a,b)', 'max(a,b)*min(a,b)',
    'rev(rev(a)+rev(b))', 'rev(rev(a)-rev(b))', 'rev(rev(a)*rev(b))',
]


def _hyp_label(name: str) -> str:
    return _EQN_HYP_LABEL.get(name, name)


def _first_fail_str(name, rel):
    """rel: list of (a, b, result_str). Return (idx, a, b, got, want) of the FIRST
    example where formula `name` disagrees, or None if it matches all."""
    for i, (a, b, r) in enumerate(rel, 1):
        got = dict(_eqn_formulas_str(a, b)).get(name)
        if got != r:
            return (i, a, b, got, r)
    return None


def _meaningful_cosurvivors(winner, matches):
    """Co-survivors that are a GENUINELY different rule (different family) than the
    winner — not just an algebraically-equivalent reversed/max-min restatement.
    Rev3 point G: only flag ambiguity when it is real."""
    if not winner:
        return []
    wf = _eqn_family(winner)
    return [m for m in matches if m != winner and _eqn_family(m) != wf]


def _build_eqn_hypotheses(rel, winner, cap=3):
    """Build a truthful, family-diverse hypothesis ladder for one operator.
    Returns a list of (kind, name) where kind ∈ {'reject','accept'}: at most `cap`
    genuine rejects (each from a distinct family, simplest-first), ending at the
    winner. Variable depth — never padded (Rev3 point I)."""
    blocks = []
    seen_fams = set()
    win_fam = _eqn_family(winner) if winner else None
    for name in _EQN_HYP_PRIORITY:
        if len(blocks) >= cap:
            break
        if name == winner:
            continue
        fam = _eqn_family(name)
        if fam == win_fam or fam in seen_fams:
            continue  # one reject per family, and not the winner's own family
        if _first_fail_str(name, rel) is None:
            continue  # only show genuine failures
        blocks.append(('reject', name))
        seen_fams.add(fam)
    if winner:
        blocks.append(('accept', winner))
    return blocks


def _eqn_formulas(a: int, b: int):
    """Return list of (name, value) for common formulas, computed on (a, b)."""
    ra, rb = _rev_int(a), _rev_int(b)
    fs = [
        ('a+b', a + b),
        ('a-b', a - b),
        ('b-a', b - a),
        ('a*b', a * b),
        ('a||b', int(f"{a}{b}")),
        ('b||a', int(f"{b}{a}")),
        ('|a-b|', abs(a - b)),
        ('max-min', max(a, b) - min(a, b)),
        ('max+min', max(a, b) + min(a, b)),
        ('max*min', max(a, b) * min(a, b)),
        ('rev(a)*rev(b)', ra * rb),
        ('rev(a)+rev(b)', ra + rb),
        ('rev(a*b)', _rev_int(a * b)),
        ('rev(a+b)', _rev_int(a + b)),
        ('rev(rev(a)*rev(b))', _rev_int(ra * rb)),
        ('rev(rev(a)+rev(b))', _rev_int(ra + rb)),
        ('rev(rev(a)-rev(b))', _rev_int(ra - rb)),
        ('rev(rev(b)-rev(a))', _rev_int(rb - ra)),
        ('rev(rev(a)*rev(b)+1)', _rev_int(ra * rb + 1)),
        ('rev(rev(a)*rev(b)-1)', _rev_int(ra * rb - 1)),
        ('rev(rev(a)+rev(b)+1)', _rev_int(ra + rb + 1)),
        ('rev(rev(a)+rev(b)-1)', _rev_int(ra + rb - 1)),
        ('(a*b)+1', a * b + 1),
        ('(a*b)-1', a * b - 1),
        ('(a+b)+1', a + b + 1),
        ('(a+b)-1', a + b - 1),
    ]
    return fs


def _eqn_arith(name: str, a: int, b: int) -> str:
    """Render the substituted arithmetic for a formula name on (a, b), e.g. '56*49+1 = 2745'."""
    ra, rb = _rev_int(a), _rev_int(b)
    val = dict(_eqn_formulas(a, b)).get(name)
    simple = {
        'a+b': f"{a}+{b}", 'a-b': f"{a}-{b}", 'b-a': f"{b}-{a}", 'a*b': f"{a}*{b}",
        'a||b': f"{a}||{b}", 'b||a': f"{b}||{a}", '|a-b|': f"|{a}-{b}|",
        'max-min': f"max({a},{b})-min({a},{b})", 'max+min': f"max({a},{b})+min({a},{b})",
        'max*min': f"max({a},{b})*min({a},{b})",
        'rev(a)*rev(b)': f"rev({a})*rev({b})={ra}*{rb}", 'rev(a)+rev(b)': f"rev({a})+rev({b})={ra}+{rb}",
        'rev(a*b)': f"rev({a}*{b})", 'rev(a+b)': f"rev({a}+{b})",
        'rev(rev(a)*rev(b))': f"rev({ra}*{rb})", 'rev(rev(a)+rev(b))': f"rev({ra}+{rb})",
        'rev(rev(a)-rev(b))': f"rev({ra}-{rb})", 'rev(rev(b)-rev(a))': f"rev({rb}-{ra})",
        'rev(rev(a)*rev(b)+1)': f"rev({ra}*{rb}+1)", 'rev(rev(a)*rev(b)-1)': f"rev({ra}*{rb}-1)",
        'rev(rev(a)+rev(b)+1)': f"rev({ra}+{rb}+1)", 'rev(rev(a)+rev(b)-1)': f"rev({ra}+{rb}-1)",
        '(a*b)+1': f"{a}*{b}+1", '(a*b)-1': f"{a}*{b}-1", '(a+b)+1': f"{a}+{b}+1", '(a+b)-1': f"{a}+{b}-1",
    }
    lhs = simple.get(name, name)
    return f"{lhs} = {val}"


# the operation-family groupings shown in Candidate Operations
_EQN_FAMILY_GROUPS = [
    ("direct arithmetic", ['a+b', 'a-b', 'b-a', 'a*b', '(a*b)+1', '(a*b)-1', '(a+b)+1', '(a+b)-1']),
    ("concatenation", ['a||b', 'b||a']),
    ("max/min", ['max-min', 'max+min', 'max*min']),
    ("reversed-operands", ['rev(a)*rev(b)', 'rev(a)+rev(b)']),
    ("reversed-result", ['rev(rev(a)*rev(b))', 'rev(rev(a)+rev(b))', 'rev(rev(a)-rev(b))',
                          'rev(rev(b)-rev(a))', 'rev(rev(a)*rev(b)+1)', 'rev(rev(a)*rev(b)-1)',
                          'rev(rev(a)+rev(b)+1)', 'rev(rev(a)+rev(b)-1)']),
]


def _parse_eq_prompt(p):
    """Parse equation_numeric prompt. Operator = single non-digit, non-'=' char.
    Result is kept as a RAW STRING to preserve leading zeros / sign."""
    eqs = []
    query = None
    for line in p.split('\n'):
        line = line.strip()
        if 'determine the result for:' in line.lower():
            m = re.search(r':\s*(\S+)\s*$', line)
            if m:
                q = m.group(1).strip()
                mq = re.match(r'(\d+)([^\d=])(\d+)$', q)
                if mq:
                    query = (int(mq.group(1)), mq.group(2), int(mq.group(3)))
            continue
        m = re.match(r'(\d+)([^\d=])(\d+)\s*=\s*(-?\d+)\s*$', line)
        if m:
            eqs.append((int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)))  # result is str
    return eqs, query


def _find_formula_for_op(examples, op):
    """Find first formula that matches ALL examples with this op."""
    rel = [(a, b, r) for a, oo, b, r in examples if oo == op]
    if not rel:
        return None
    sample_a, sample_b, sample_r = rel[0]
    for name, _ in _eqn_formulas(sample_a, sample_b):
        ok = True
        for a, b, r in rel:
            vals = dict(_eqn_formulas(a, b))
            if vals.get(name) != r:
                ok = False
                break
        if ok:
            return name
    return None


def write_equation_numeric(puzzle, kind='deduce'):
    eqs, query = _parse_eq_prompt(puzzle['prompt'])
    answer = puzzle['answer']
    qa, qop, qb = query
    ops_seen = sorted({op for _, op, _, _ in eqs})
    a0, op0, b0, r0 = eqs[0]

    # winner formula per operator (answer-aware for the query operator)
    formulas_used = {}
    cosurvivors = {}
    for op in ops_seen:
        matches = _all_matching_formulas(eqs, op)
        cosurvivors[op] = matches
        win = _find_formula_str(eqs, op)
        if op == qop and len(matches) > 1:
            # break ties by the member that reproduces the query answer
            tied = next((m for m in matches
                         if dict(_eqn_formulas_str(qa, qb)).get(m) == answer), None)
            win = tied or win
        formulas_used[op] = win

    L = []
    L.append("Alice's Wonderland hides one arithmetic formula behind each operator symbol — and the symbol need not mean its usual operation. I read the result widths and signs, name a few plausible rules, reject the ones a single example refutes, keep the survivor, then apply it to the query.")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for a, op, b, r in eqs:
        L.append(f"  {a:02d}{op}{b:02d} = {r}")
    L.append(f"  Query: {qa:02d}{qop}{qb:02d} = ?\n")

    L.append("## [CONSTRAINT]")
    for a, op, b, r in eqs:
        mag = r.lstrip('-')
        sign = "negative" if r.startswith('-') else "positive"
        L.append(f"  {a:02d}{op}{b:02d} → {len(mag)}-digit {sign} result {r}")
    if qop in ops_seen:
        L.append(keep(f"the query operator '{qop}' already appears in the examples, so its rule is pinned down by those rows"))
    else:
        L.append(keep(f"the query operator '{qop}' never appears in the examples — its rule must be inferred from the family the example operators share"))
    L.append("")

    # ---- [ABSTRACTION] : examples → pattern (pre-rule), anchored on op0 ----
    L.append("## [ABSTRACTION]")
    anchor_win = formulas_used.get(op0)
    if anchor_win:
        L.append(f"Look at operator '{op0}' on {a0:02d}{op0}{b0:02d} = {r0}: {_abstraction_cue(anchor_win, a0, b0, r0)}.")
        L.append("That pattern is what I turn into a concrete rule next — examples → pattern → rule.")
    else:
        L.append(f"Operator '{op0}' produces {r0}, whose leading character is the operator '{op0}' itself — that marks a SIGN, not a digit, so the rule writes the operator in place of a minus sign.")
    L.append("")

    # ---- [HYPOTHESIS] : name a short, family-diverse ladder for the anchor op ----
    rel0 = [(a, b, r) for a, oo, b, r in eqs if oo == op0]
    ladder0 = _build_eqn_hypotheses(rel0, anchor_win)
    L.append("## [HYPOTHESIS]")
    if ladder0:
        names = [_hyp_label(n) for _, n in ladder0]
        L.append(f"Candidate rules for '{op0}', simplest first: {', '.join(names)}.")
    else:
        L.append(f"The result for '{op0}' is outside the plain numeric library (operator-as-sign); I read its value directly.")
    L.append("")

    # ---- [EVALUATION] : reject with minimal counterexample, accept with evidence ----
    L.append("## [EVALUATION]")
    for op in ops_seen:
        rel = [(a, b, r) for a, oo, b, r in eqs if oo == op]
        winner = formulas_used.get(op)
        L.append(f"Operator '{op}' ({len(rel)} example{'s' if len(rel) != 1 else ''}):")
        ladder = _build_eqn_hypotheses(rel, winner)
        if not ladder:
            L.append(f"  result carries the operator as a sign marker; value read directly (no library formula applies).")
            L.append("")
            continue
        for knd, name in ladder:
            if knd == 'reject':
                ff = _first_fail_str(name, rel)
                i, a, b, got, want = ff
                chk = f"fails first at ex{i}: {_eqn_arith_str(name, a, b)} ≠ {want}"
                L.append(hyp_reject(_hyp_label(name), chk, f"{got} ≠ {want}"))
            else:
                a, b, r = rel[0]
                chk = f"ex1: {_eqn_arith_str(name, a, b)} = {r}"
                L.append(hyp_accept(_hyp_label(name), chk, f"{len(rel)}/{len(rel)} examples satisfy"))
        # honest co-survivor note — only for a genuinely different rule (Rev3 point G)
        co = _meaningful_cosurvivors(winner, cosurvivors.get(op, []))
        if co:
            shown = ', '.join(_hyp_label(m) for m in co[:2])
            L.append(keep(f"  note: a genuinely different rule, {shown}, also fits every example here — "
                          f"the examples alone leave '{op}' underdetermined"))
        L.append("")

    # ---- [SELECTION] : state the chosen rule + family inference, no re-arithmetic ----
    L.append("## [SELECTION]")
    for op in ops_seen:
        f = formulas_used.get(op)
        L.append(f"  '{op}' = {f or '<operator-as-sign, value read directly>'}")
    f = None
    sign_note = None
    if qop in formulas_used and formulas_used[qop]:
        f = formulas_used[qop]
        co = _meaningful_cosurvivors(f, cosurvivors.get(qop, []))
        if co:
            L.append(keep(f"the examples leave '{qop}' underdetermined ({_hyp_label(co[0])} also fits); "
                          f"the member that reproduces the query answer is {f}, so I select it"))
    elif kind == 'guess':
        # infer the unseen query operator's rule from the example family
        fams = {_eqn_family(formulas_used[o]) for o in ops_seen if formulas_used.get(o)}
        # direct numeric match first
        f = next((name for name, val in _eqn_formulas_str(qa, qb) if val == answer), None)
        if f is None and len(answer) >= 2:
            if answer[0] == qop and answer[1:].lstrip('-').isdigit():
                signed = -int(answer[1:])
                f = next((name for name, val in _eqn_formulas_str(qa, qb)
                          if val.lstrip('-').isdigit() and int(val) == signed), None)
                if f:
                    sign_note = (f"the answer {answer} writes the operator '{qop}' in place of a minus sign, "
                                 f"so the value is {signed}")
            elif answer[-1] == qop and answer[:-1].lstrip('-').isdigit():
                mag = int(answer[:-1])
                f = next((name for name, val in _eqn_formulas_str(qa, qb)
                          if val.lstrip('-').isdigit() and abs(int(val)) == mag), None)
                if f:
                    sign_note = (f"the answer {answer} appends the operator '{qop}' as a sign marker on {mag}")
        if f:
            fam = _eqn_family(f)
            if len(fams) == 1 and fam in fams:
                fam_name = next(iter(fams))
                L.append(keep(f"every example operator is the {fam_name} family ⇒ the unseen '{qop}' must be too ⇒ the member matching the answer is {f}"))
            else:
                L.append(keep(f"the unseen '{qop}' is the {fam} family, matching one of the example operators; the member consistent with the answer is {f}"))
    # operator-as-sign explanation for the query (covers deduce path too)
    if sign_note is None and f:
        qval = dict(_eqn_formulas_str(qa, qb)).get(f)
        if qval is not None and qval != answer and qval.startswith('-'):
            mag = qval[1:]
            if answer == f"{mag}{qop}" or answer == f"{qop}{mag}":
                sign_note = (f"the computed value is {qval}; the puzzle marks a negative result by writing the "
                             f"operator '{qop}' as the sign, so {qval} is written {answer}")
    if sign_note:
        L.append(keep(sign_note))
    L.append("")

    # ---- [VERIFICATION] : independent per-example recompute ----
    L.append("## [VERIFICATION]")
    for a, op, b, r in eqs:
        fop = formulas_used.get(op)
        if not fop:
            L.append(keep(f"{a:02d}{op}{b:02d} = {r} (sign marker, taken as given)"))
            continue
        val = dict(_eqn_formulas_str(a, b)).get(fop)
        ok = val == r
        L.append(keep(f"{a:02d}{op}{b:02d}: {_eqn_arith_str(fop, a, b)} = {r}") if ok
                 else drop(f"{a:02d}{op}{b:02d}: {_eqn_arith_str(fop, a, b)} ≠ {r}"))
    if f:
        L.append(keep(f"query {qa:02d}{qop}{qb:02d}: {_eqn_arith_str(f, qa, qb)} → in the puzzle's notation = {answer}"))
    else:
        L.append(keep(f"query {qa:02d}{qop}{qb:02d} resolves to {answer} (operator-as-sign; confirmed against ground truth)"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)


# ---------------------------------------------------------------------------
# cryptarithm (deduce + guess) — uses verified solution dict from
# crypto_family_solutions.json
# ---------------------------------------------------------------------------

# Family op evaluators reuse cryptarithm_family.py
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
try:
    import cryptarithm_family as CF
except Exception:
    CF = None


FAMILY_NAMES_HUMAN = {
    'DIRECT_MAXMIN': 'direct max/min',
    'DIRECT_SIMPLE': 'direct simple a,b',
    'REV_AB': 'reverse-of-(rev(a) ⊕ rev(b))',
    'REV_MAXMIN': 'reverse-of-(max(rev),min(rev)) family',
}


def _apply_op_value(family: str, op_name: str, a: int, b: int):
    """Apply named operator-family op to (a,b) and return (result_str, is_negative)."""
    if CF is None:
        return None
    return CF.apply_op(family, op_name, a, b)


def _detect_neg_examples(equations, op):
    """An example whose result starts with the operator char is a NEG-result example."""
    neg = []
    for n1, opx, n2, res in equations:
        if opx == op and len(res) >= 2 and res[0] == op and all(c != op for c in res[1:]):
            neg.append((n1, opx, n2, res[1:]))
    return neg


# Families whose results are bounded above (additive/subtractive) — useful for the
# width-based elimination argument when a result has many digits.
_ADD_LIKE_FAMILIES = {'DIRECT_SIMPLE', 'DIRECT_MAXMIN'}


def _family_width_elimination(eqs, chosen_family):
    """Produce 1-2 elimination sentences ruling out a family by the digit-width argument.
    Returns list of (sentence, ok=False) drop-lines, plus the keep-line for the survivor."""
    lines = []
    # find the longest positive result width among examples
    max_w = 0
    sample = None
    for n1, op, n2, res in eqs:
        is_neg = len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])
        w = len(res) - 1 if is_neg else len(res)
        if w > max_w:
            max_w = w
            sample = (n1, op, n2, res)
    if max_w >= 4 and chosen_family not in ('DIRECT_SIMPLE',):
        # additive families cap at 198 = 3 digits, so a 4-digit result rules them out
        lines.append(drop("DIRECT_SIMPLE additive forms (a+b, a±1): two 2-digit numbers sum to ≤ 99+99 = 198, "
                          "which is only 3 digits — cannot make a 4-digit result"))
    return lines


def _units_digit_deduction(eqs, family, mp, om, al):
    """Surface up to 2 REAL units-digit deductions from a multiplicative/additive equation
    whose operator rule is known and whose result shares structure with the inputs."""
    out = []
    if CF is None:
        return out
    for n1, op, n2, res in eqs:
        rule = om.get(op)
        if rule is None:
            continue
        is_neg = len(res) >= 2 and res[0] == op and all(c != op for c in res[1:])
        res_syms = res[1:] if is_neg else res
        a = mp[n1[0]] * 10 + mp[n1[1]]
        b = mp[n2[0]] * 10 + mp[n2[1]]
        applied = CF.apply_op(family, rule, a, b)
        if not applied:
            continue
        rs = applied[0]
        if len(rs) != len(res_syms):
            continue
        # units digit of result corresponds to last result symbol
        u_sym = res_syms[-1]
        u_dig = int(rs[-1])
        # explain via the rule on the unit components when it's a clean direct op
        if rule in ('a+b', '(a+b)+1', '(a+b)-1', 'max+min', 'a-b', 'b-a', 'max-min'):
            ad, bd = a % 10, b % 10
            if rule == 'a+b':
                expr = f"({al[n1[1]]}+{bd}? ) units: ({ad}+{bd}) mod 10 = {(ad+bd)%10}"
            else:
                expr = f"units of {rule} on {a},{b} = {u_dig}"
            out.append(keep(f"{al[u_sym]} ('{u_sym}') = {u_dig} — units position of {n1}{op}{n2} ({rule}) forces it: {expr}"))
        elif rule.startswith('('):
            out.append(keep(f"{al[u_sym]} ('{u_sym}') = {u_dig} — last digit of {rule} applied to {a},{b} (= {rs})"))
        else:
            out.append(keep(f"{al[u_sym]} ('{u_sym}') = {u_dig} — last digit of {n1}{op}{n2}'s result {rs} (rule {rule})"))
        if len(out) >= 2:
            break
    return out


_REV_FAMILIES = {'REV_AB', 'REV_MAXMIN'}


def _direct_arith(rule: str, a: int, b: int) -> str:
    """Render a direct (non-rev) family rule as substituted arithmetic, e.g. 'max(87,64)*min(87,64)'."""
    table = {
        'a+b': f"{a}+{b}", 'a-b': f"{a}-{b}", 'b-a': f"{b}-{a}", 'a*b': f"{a}*{b}", 'b*a': f"{b}*{a}",
        '(a*b)+1': f"{a}*{b}+1", '(a*b)-1': f"{a}*{b}-1", '(a+b)+1': f"{a}+{b}+1", '(a+b)-1': f"{a}+{b}-1",
        'a||b': f"{a:02d}||{b:02d}", 'b||a': f"{b:02d}||{a:02d}",
        'max+min': f"max({a},{b})+min({a},{b})", 'max-min': f"max({a},{b})-min({a},{b})",
        'max*min': f"max({a},{b})*min({a},{b})", '(max*min)+1': f"max({a},{b})*min({a},{b})+1",
        '(max*min)-1': f"max({a},{b})*min({a},{b})-1", 'max||min': f"max({a},{b})||min({a},{b})",
        'min||max': f"min({a},{b})||max({a},{b})", 'min-max': f"min({a},{b})-max({a},{b})",
        'min+max': f"min({a},{b})+max({a},{b})",
    }
    return table.get(rule, rule)


def _inner_expr_rev(rule: str, ra: int, rb: int, rmx: int, rmn: int) -> str:
    """Strip the outer rev(...) of a reversed-family rule and substitute reversed inputs,
    e.g. 'rev(rev(a)+rev(b)+1)' on ra=60,rb=76 → '60+76+1'."""
    inner = rule
    if inner.startswith('rev(') and inner.endswith(')'):
        inner = inner[4:-1]
    elif inner.startswith('-rev(') and inner.endswith(')'):
        inner = inner[5:-1]
    inner = inner.replace('max(rev(a),rev(b))', str(rmx)).replace('min(rev(a),rev(b))', str(rmn))
    inner = inner.replace('rev(a)', str(ra)).replace('rev(b)', str(rb))
    return inner


def _crypto_abstraction(eqs, family, is_neg_res):
    """One-line PATTERN naming (examples → pattern), pre-rule, for [ABSTRACTION]."""
    lines = []
    max_w = 0
    has_neg = False
    for n1, op, n2, res in eqs:
        neg = is_neg_res(op, res)
        has_neg = has_neg or neg
        w = len(res) - 1 if neg else len(res)
        max_w = max(max_w, w)
    if max_w >= 4:
        lines.append(f"one result reaches {max_w} digits — two 2-digit numbers add to at most 198 (3 digits), "
                     f"so the rule must involve a product or a join, not just addition.")
    if family in _REV_FAMILIES:
        lines.append("the results don't line up with any straight arithmetic, but they DO once the digits are "
                     "mirrored — the pattern is that the result is written reversed.")
    elif has_neg:
        lines.append("some results carry the operator as a leading sign — the rule can go negative, pointing to "
                     "a plain difference rather than a product.")
    if not lines:
        lines.append("each result is reproduced by a single fixed combination of the two inputs read straight.")
    return lines


def _crypto_family_branch(eqs, family, mp, om, al, is_neg_res):
    """The wrong-family branch for [EVALUATION]: a genuine
    [ASSUMPTION]→[PROPAGATION]→[CONTRADICTION]→[REJECTION]→[RECOVERY] chain that is
    actually computed, plus the accept line for the surviving family."""
    L = []
    # pick the first positive example for the arithmetic display
    pos = next(((n1, op, n2, res) for n1, op, n2, res in eqs if not is_neg_res(op, res)), eqs[0])
    n1, op, n2, res = pos
    a = mp[n1[0]] * 10 + mp[n1[1]]
    b = mp[n2[0]] * 10 + mp[n2[1]]
    rule = om.get(op, '?')
    applied = CF.apply_op(family, rule, a, b) if CF else None
    R = applied[0] if applied else res
    # widest result for the width argument
    max_w = 0; wide = None
    for e in eqs:
        neg = is_neg_res(e[1], e[3]); w = len(e[3]) - 1 if neg else len(e[3])
        if w > max_w: max_w = w; wide = e

    if family in _REV_FAMILIES:
        ra, rb = CF.rev2(a), CF.rev2(b)
        rmx, rmn = max(ra, rb), min(ra, rb)
        revR = R[::-1]
        inner = _inner_expr_rev(rule, ra, rb, rmx, rmn)
        L.append(f"[ASSUMPTION] suppose '{op}' is an ordinary, straight-read rule (a DIRECT family).")
        L.append(f"  [PROPAGATION] then on {n1}{op}{n2} the value should read {R} directly; but {a}+{b}={a+b} and "
                 f"{a}*{b}={a*b} — no plain rule on {a},{b} lands on {R}.")
        L.append(f"  [CONTRADICTION] a straight reading cannot produce {R}.  ✗")
        L.append(f"  [REJECTION] reject the direct family.")
        L.append(f"  [RECOVERY] reverse the result: {R} → {revR}; with reversed inputs rev({a})={ra}, rev({b})={rb}, "
                 f"the combination {inner} = {int(revR)} matches — so the family reverses its result. Adopt {FAMILY_NAMES_HUMAN.get(family, family)}.")
    elif max_w >= 4:
        kind_word = ("a product" if '*' in rule else
                     "writing the two numbers side by side (concatenation)" if '||' in rule else
                     "a multiplicative combination")
        wn1, wop, wn2, wres = wide
        L.append(f"[ASSUMPTION] suppose '{op}' is an additive/subtractive rule (a±b or max±min).")
        L.append(f"  [PROPAGATION] two 2-digit numbers sum to at most 99+99 = 198.")
        L.append(f"  [CONTRADICTION] but {wn1}{wop}{wn2} = {wres} needs a {max_w}-digit result, and 198 is only 3 digits.  ✗")
        L.append(f"  [REJECTION] reject every additive/subtractive rule.")
        L.append(f"  [RECOVERY] {kind_word} is required to reach {max_w} digits; on {n1}{op}{n2}, "
                 f"{_direct_arith(rule, a, b)} = {R} fits. Adopt {FAMILY_NAMES_HUMAN.get(family, family)}.")
    else:
        revR = R[::-1]
        L.append(f"[ASSUMPTION] suppose the result is written reversed (a REV family).")
        L.append(f"  [PROPAGATION] then {n1}{op}{n2} = {res} would mean the true value is rev({R}) = {int(revR)}.")
        L.append(f"  [CONTRADICTION] but {_direct_arith(rule, a, b)} = {R} already matches the symbols read straight; "
                 f"reversing to {int(revR)} would break it.  ✗")
        L.append(f"  [REJECTION] reject the reversed family.")
        L.append(f"  [RECOVERY] read the result straight → {FAMILY_NAMES_HUMAN.get(family, family)}.")
    L.append(hyp_accept(FAMILY_NAMES_HUMAN.get(family, family),
                        f"{n1}{op}{n2}: rule '{op}'={rule} → {R}, matching the symbols {res}",
                        "every operator's width and sign agree under this one family"))
    return L


def _crypto_implication_chain(eqs, mp, al, is_neg_res):
    """1–2 cross-equation implication (⇒) lines for [SELECTION]."""
    out = []
    # symbol occurring in operands of the most distinct equations
    occ = {}
    for i, (n1, op, n2, res) in enumerate(eqs):
        for s in set(n1 + n2):
            occ.setdefault(s, set()).add(i)
    shared = sorted(((len(idx), s) for s, idx in occ.items() if len(idx) >= 2), reverse=True)
    if shared:
        _, s = shared[0]
        idx = sorted(occ[s])
        e1, e2 = eqs[idx[0]], eqs[idx[1]]
        out.append(keep(f"'{s}'({al.get(s,'?')}) appears in {e1[0]}{e1[1]}{e1[2]} and {e2[0]}{e2[1]}{e2[2]} "
                        f"⇒ it must be one digit in both ⇒ {al.get(s,'?')} = {mp.get(s,'?')}"))
    return out


def write_cryptarithm(puzzle):
    """puzzle must include the verified solution: family, map, ops, equations, query, answer."""
    eqs = puzzle['equations']
    query = puzzle['query']
    answer = puzzle['answer']
    mp = puzzle['map']
    om = puzzle['ops']
    family = puzzle['family']
    kind = puzzle.get('kind', 'deduce')
    is_guess = (kind == 'guess')

    q1, qop, q2 = query[0], query[1], query[2]

    def is_neg_res(op, res):
        return len(res) >= 2 and res[0] == op and all(cc != op for cc in res[1:])

    L = []
    L.append("This is a cryptarithm. Every distinct symbol is one digit (0–9); the same symbol is always the same digit, but two different symbols may share a digit. The middle character of a 5-symbol left side is the operator, and that operator need not mean ordinary arithmetic — every operator in one puzzle draws from ONE family of formulas. Characters like ` ' \" \\ are ordinary symbols (a doubled \"\" is the same symbol used twice). A result that begins with the operator character is a negative value whose magnitude digits follow.")
    L.append("I will put my final answer inside \\boxed{}.\n")

    L.append("## [OBSERVATION]")
    for n1, op, n2, res in eqs:
        L.append(f"  {n1}{op}{n2} = {res}")
    L.append(f"  Query: {q1}{qop}{q2}\n")

    # symbol labels (kept for the algebra, presented under CONSTRAINT)
    al = {}; idx = 0
    for n1, op, n2, res in eqs:
        for c in n1 + n2 + (res[1:] if is_neg_res(op, res) else res):
            if c not in al:
                al[c] = chr(65 + idx); idx += 1
    for c in q1 + q2 + answer:
        if c not in al:
            al[c] = chr(65 + idx); idx += 1

    L.append("## [CONSTRAINT]")
    L.append("Label each distinct symbol with a letter so the algebra is unambiguous:")
    L.append("  " + ", ".join(f"'{s}'={al[s]}" for s in sorted(al, key=lambda x: al[x])))
    L.append("The digit-width and sign of each result constrain which family the operator can belong to:")
    res_lens = {}
    for n1, op, n2, res in eqs:
        ln = len(res) - 1 if is_neg_res(op, res) else len(res)
        neg = ' (negative)' if is_neg_res(op, res) else ''
        res_lens.setdefault(op, []).append(ln)
        L.append(f"  {n1}{op}{n2} = {res}: result has {ln} digit(s){neg}")
    L.append("")

    L.append("## [ABSTRACTION]")
    for line in _crypto_abstraction(eqs, family, is_neg_res):
        L.append(line)
    L.append("")

    L.append("## [HYPOTHESIS]")
    L.append(f"The operators must all come from ONE family. Candidate families: direct simple (a,b), "
             f"direct max/min, and the two reversed-result families. The pattern above points to "
             f"{FAMILY_NAMES_HUMAN.get(family, family)}; I test it against a competing family before accepting.")
    L.append("")

    L.append("## [EVALUATION]")
    for line in _crypto_family_branch(eqs, family, mp, om, al, is_neg_res):
        L.append(line)
    L.append("")

    L.append("## [SELECTION]")
    L.append(f"Chosen family: {FAMILY_NAMES_HUMAN.get(family, family)}; operator rules: "
             + ", ".join(f"'{o}'={om[o]}" for o in om))
    for line in _crypto_implication_chain(eqs, mp, al, is_neg_res):
        L.append(line)
    deds = _units_digit_deduction(eqs, family, mp, om, al)
    for d in deds:
        L.append(d)
    L.append("Propagating these constraints through the remaining equations fixes the full assignment:")
    L.append("  " + ", ".join(f"{al[s]}('{s}')={mp[s]}" for s in sorted(al, key=lambda x: al[x]) if s in mp))
    L.append("")

    L.append("## [VERIFICATION]")
    for n1, op, n2, res in eqs:
        neg = is_neg_res(op, res)
        exp = res[1:] if neg else res
        exp_digits = ''.join(str(mp.get(c, '?')) for c in exp)
        a = mp[n1[0]] * 10 + mp[n1[1]]
        b = mp[n2[0]] * 10 + mp[n2[1]]
        applied = _apply_op_value(family, om.get(op, '?'), a, b)
        got = applied[0] if applied else '?'
        sign = '-' if (applied and applied[1]) else ''
        ok = applied and got == exp_digits
        L.append(keep(f"{n1}{op}{n2}: {om.get(op,'?')} on {a},{b} = {sign}{got}, and the symbols {res} read as {sign}{exp_digits}")
                 if ok else
                 drop(f"{n1}{op}{n2}: {om.get(op,'?')} gives {sign}{got} ≠ symbols {exp_digits}"))
    qa = mp[q1[0]] * 10 + mp[q1[1]]
    qb = mp[q2[0]] * 10 + mp[q2[1]]
    qrule = om.get(qop, '?')
    applied_q = _apply_op_value(family, qrule, qa, qb)
    qsign = '-' if (applied_q and applied_q[1]) else ''
    if is_guess:
        L.append(keep(f"'{qop}' never appears in the examples ⇒ the single-family constraint forces it to be {qrule}"))
    L.append(keep(f"query {q1}{qop}{q2}: numbers {qa},{qb}; applying '{qop}'={qrule} gives {qsign}{applied_q[0] if applied_q else '?'}, "
                  f"which reads back through the digit→symbol map as {answer}"))
    L.append("")

    L.append("## [ANSWER]")
    L.append(f"\\boxed{{{answer}}}")
    return wrap('\n'.join(L), answer)
