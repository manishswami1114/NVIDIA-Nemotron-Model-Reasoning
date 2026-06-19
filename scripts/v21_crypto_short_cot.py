#!/usr/bin/env python3
"""V21 short cryptarithm CoT writer.

Generates compact chain-of-thought that fits within ~5000 chars:
1. Width analysis → narrow formula candidates
2. Pattern matching → confirm formula per operator
3. Digit deduction → solve map from equations
4. Apply → compute answer

Each step depends on the previous. No filler words.
"""
import json, re, sys, csv
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

BASE = _HERE.parent


def rev_int(x):
    if x < 0:
        return -int(str(-x)[::-1])
    return int(str(x)[::-1])


def _compute(formula, a, b):
    """Compute result for a formula name."""
    mx, mn = max(a, b), min(a, b)
    ra, rb = rev_int(a), rev_int(b)
    formulas = {
        # DIRECT_SIMPLE
        'a+b': a + b, 'a-b': a - b, 'a*b': a * b,
        '(a+b)+1': a + b + 1, '(a+b)-1': a + b - 1,
        '(a*b)+1': a * b + 1, '(a*b)-1': a * b - 1,
        '(a-b)+1': a - b + 1, '(a-b)-1': a - b - 1,
        '(b-a)-1': b - a - 1,
        'a||b': int(f"{a:02d}{b:02d}"), 'b||a': int(f"{b:02d}{a:02d}"),
        'a%b': a % b if b != 0 else None, 'b%a': b % a if a != 0 else None,
        # DIRECT_MAXMIN
        'max+min': mx + mn, 'max-min': mx - mn, 'min-max': mn - mx, 'max*min': mx * mn,
        '(max+min)+1': mx + mn + 1, '(max+min)-1': mx + mn - 1,
        '(max*min)+1': mx * mn + 1, '(max*min)-1': mx * mn - 1,
        '(max-min)+1': mx - mn + 1, '(max-min)-1': mx - mn - 1,
        '(min-max)+1': mn - mx + 1, '(min-max)-1': mn - mx - 1,
        'max||min': int(f"{mx:02d}{mn:02d}"), 'min||max': int(f"{mn:02d}{mx:02d}"),
        'max%min': mx % mn if mn != 0 else None,
        # REV_AB
        'rev(a+b)': rev_int(a + b), 'rev(a-b)': rev_int(a - b),
        'rev(b-a)': rev_int(b - a), 'rev(a*b)': rev_int(a * b),
        'rev(a)+rev(b)': ra + rb, 'rev(a)-rev(b)': ra - rb,
        'rev(b)-rev(a)': rb - ra, 'rev(a)*rev(b)': ra * rb,
        'rev(rev(a)+rev(b))': rev_int(ra + rb),
        'rev(rev(a)-rev(b))': rev_int(ra - rb),
        'rev(rev(b)-rev(a))': rev_int(rb - ra),
        'rev(rev(a)*rev(b))': rev_int(ra * rb),
        'rev(rev(a)+rev(b)+1)': rev_int(ra + rb + 1),
        'rev(rev(a)+rev(b)-1)': rev_int(ra + rb - 1),
        'rev(rev(a)*rev(b)+1)': rev_int(ra * rb + 1),
        'rev(rev(a)*rev(b)-1)': rev_int(ra * rb - 1),
        'rev(rev(a)-rev(b)+1)': rev_int(ra - rb + 1),
        'rev(rev(a)-rev(b)-1)': rev_int(ra - rb - 1),
        'rev(rev(b)-rev(a)+1)': rev_int(rb - ra + 1),
        'rev(rev(a)||rev(b))': rev_int(int(f"{ra:02d}{rb:02d}")),
        'rev(rev(b)||rev(a))': rev_int(int(f"{rb:02d}{ra:02d}")),
        'rev(rev(b)%rev(a))': rev_int(rb % ra) if ra != 0 else None,
        # REV_MAXMIN
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))': rev_int(max(ra,rb) * min(ra,rb)),
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))': rev_int(max(ra,rb) + min(ra,rb)),
        'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))': rev_int(max(ra,rb) - min(ra,rb)),
        'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))': rev_int(max(ra,rb) % min(ra,rb)) if min(ra,rb) != 0 else None,
        'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))': rev_int(int(f"{max(ra,rb):02d}{min(ra,rb):02d}")),
        'rev(min(rev(a),rev(b))||max(rev(a),rev(b)))': rev_int(int(f"{min(ra,rb):02d}{max(ra,rb):02d}")),
        'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)': rev_int(max(ra,rb) * min(ra,rb) + 1),
        'rev(max(rev(a),rev(b))-min(rev(a),rev(b))-1)': rev_int(max(ra,rb) - min(ra,rb) - 1),
        'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)': rev_int(max(ra,rb) + min(ra,rb) + 1),
    }
    return formulas.get(formula)


def _width_hint(width):
    """What formulas can produce a result of this width."""
    if width == 1:
        return "1-digit → small difference (max-min, a-b when close)"
    elif width == 2:
        return "2-digit → add, sub, max-min, or rev operations"
    elif width == 3:
        return "3-digit → multiply or rev of large values"
    elif width == 4:
        return "4-digit → concat or multiply"
    return ""


def _is_sign_prefix(op, res):
    return len(res) >= 2 and res[0] == op


def write_short_crypto_cot(puzzle):
    """Generate compact reasoning CoT for a cryptarithm puzzle.

    puzzle = {
        'equations': [[n1, op, n2, res], ...],
        'query': [q1, qop, q2],
        'answer': str (symbol form),
        'map': {symbol: digit},
        'ops': {op_char: formula_name},
        'kind': 'deduce' | 'guess',
        'family': str,
    }
    """
    mp = puzzle['map']
    ops = puzzle['ops']
    equations = puzzle['equations']
    query = puzzle['query']
    answer = puzzle['answer']
    kind = puzzle['kind']

    L = ["<think>"]

    # Step 1: Analyze result widths per operator
    L.append("Step 1: Analyze result patterns per operator")
    op_eqs = defaultdict(list)
    for n1, op, n2, res in equations:
        a = mp[n1[0]] * 10 + mp[n1[1]]
        b = mp[n2[0]] * 10 + mp[n2[1]]
        sign = _is_sign_prefix(op, res)
        body = res[1:] if sign else res
        body_num = ''.join(str(mp[c]) for c in body)
        op_eqs[op].append((a, b, body_num, sign, len(body)))

    for op in op_eqs:
        eqs = op_eqs[op]
        widths = [e[4] for e in eqs]
        signs = [e[3] for e in eqs]
        common_width = max(set(widths), key=widths.count)
        has_sign = any(signs)
        hint = _width_hint(common_width)
        sign_note = f", sign-prefix '{op}'" if has_sign else ""
        L.append(f"  Operator '{op}': results are {common_width}-digit{sign_note} → {hint}")

    # Step 2: Identify formula per operator using equations
    L.append("")
    L.append("Step 2: Match formula per operator")
    for op, formula in ops.items():
        eqs = op_eqs.get(op, [])
        if not eqs:
            # Query-only operator (guess puzzles)
            L.append(f"  '{op}' → {formula} (query operator, not in examples)")
            continue
        a, b, res_num, sign, w = eqs[0]
        computed = _compute(formula, a, b)
        if computed is not None:
            comp_str = str(abs(computed)) if computed < 0 else str(computed)
            L.append(f"  '{op}' → {formula}: check {a:02d},{b:02d} → {comp_str} ✓")
        else:
            L.append(f"  '{op}' → {formula}")

    # Step 3: Deduce digit map
    # Show how digits are derived from equations
    L.append("")
    L.append("Step 3: Deduce digit mapping from equations")

    # Group symbols by which equations they appear in
    # Show the deduction chain - pick equations that help determine query symbols
    q1, qop, q2 = query
    query_symbols = set(q1) | set(q2)

    # Find which equations contain query symbols
    relevant_eqs = []
    for n1, op, n2, res in equations:
        eq_symbols = set(n1) | set(n2) | set(res.replace(op, '') if _is_sign_prefix(op, res) else res)
        if eq_symbols & query_symbols:
            relevant_eqs.append((n1, op, n2, res))

    # Show digit deduction using a few key equations
    shown_symbols = set()
    deduction_lines = 0
    for n1, op, n2, res in equations:
        if deduction_lines >= 4:
            break
        a = mp[n1[0]] * 10 + mp[n1[1]]
        b = mp[n2[0]] * 10 + mp[n2[1]]
        new_symbols = set(n1) | set(n2)
        if not (new_symbols - shown_symbols) and deduction_lines > 0:
            continue
        formula = ops[op]
        computed = _compute(formula, a, b)
        if computed is None:
            continue
        L.append(f"  {n1}{op}{n2}: {a:02d} {formula} {b:02d} = {computed} → digits: {n1[0]}={mp[n1[0]]}, {n1[1]}={mp[n1[1]]}, {n2[0]}={mp[n2[0]]}, {n2[1]}={mp[n2[1]]}")
        shown_symbols.update(new_symbols)
        deduction_lines += 1

    # Show query symbol values
    L.append(f"  Query symbols: {q1[0]}={mp[q1[0]]}, {q1[1]}={mp[q1[1]]}, {q2[0]}={mp[q2[0]]}, {q2[1]}={mp[q2[1]]}")

    # Step 4: Apply formula to query
    L.append("")
    L.append("Step 4: Compute query result")
    qa = mp[q1[0]] * 10 + mp[q1[1]]
    qb = mp[q2[0]] * 10 + mp[q2[1]]

    if kind == 'deduce':
        formula = ops[qop]
        computed = _compute(formula, qa, qb)
        L.append(f"  {q1}{qop}{q2} → {qa:02d} {formula} {qb:02d}")
        if computed is not None:
            L.append(f"  = {computed}")
    else:
        # Guess: query op not in examples, need to identify it
        # We know the answer, so work backwards
        formula = ops.get(qop)
        if formula:
            computed = _compute(formula, qa, qb)
            L.append(f"  Query operator '{qop}' not in examples.")
            # Use width heuristic
            ans_body = answer[1:] if _is_sign_prefix(qop, answer) else answer
            w = len(ans_body)
            L.append(f"  Expected result width: {w} → {_width_hint(w)}")
            L.append(f"  Testing {formula} on {qa:02d},{qb:02d}: {computed}")
        else:
            computed = None
            L.append(f"  Query operator '{qop}' → unknown formula")

    # Map result back to symbols
    L.append(f"  Result in symbols: {answer}")
    L.append("</think>")
    # Handle answers containing '}' — use nested braces
    L.append(f"\\boxed{{{answer}}}")

    text = '\n'.join(L)
    # Fix boxed for answers with '}': the last } before end-of-line closes \boxed
    # This matches how train.csv answers are stored in baseline
    return text


def main():
    """Build V21 short cryptarithm CoTs."""
    sols = {s['id']: s for s in json.load(open(_HERE / "crypto_solutions_v17.json"))}
    train_by_id = {}
    with open(BASE / "data/raw/train.csv") as f:
        for row in csv.DictReader(f):
            train_by_id[row['id']] = row

    print(f"Loaded {len(sols)} solutions, {len(train_by_id)} train rows")

    deduce_recs = []
    guess_recs = []
    errors = 0
    lengths = []

    for sid, sol in sols.items():
        r = train_by_id.get(sid)
        if not r:
            continue
        if all(v == 0 for v in sol['map'].values()):
            continue

        puzzle = {
            'equations': sol['equations'],
            'query': sol['query'],
            'answer': r['answer'],
            'map': sol['map'],
            'ops': sol['ops'],
            'kind': sol['kind'],
            'family': sol['family'],
        }
        try:
            cot = write_short_crypto_cot(puzzle)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [err] {sid}: {e}")
            continue

        lengths.append(len(cot))
        rec = {
            'category': f"cryptarithm_{sol['kind']}",
            'messages': [
                {'role': 'user', 'content': r['prompt']},
                {'role': 'assistant', 'content': cot},
            ],
        }
        if sol['kind'] == 'deduce':
            deduce_recs.append(rec)
        else:
            guess_recs.append(rec)

    import statistics
    print(f"\nGenerated: {len(deduce_recs)} deduce, {len(guess_recs)} guess, {errors} errors")
    if lengths:
        print(f"CoT lengths: median={statistics.median(lengths):.0f}, max={max(lengths)}, min={min(lengths)} chars")
        print(f"Estimated tokens: median≈{statistics.median(lengths)/4:.0f}, max≈{max(lengths)/4:.0f}")

    # Show sample
    if deduce_recs:
        print(f"\n{'='*70}")
        print("SAMPLE DEDUCE CoT:")
        print('='*70)
        print(deduce_recs[0]['messages'][1]['content'])

    if guess_recs:
        print(f"\n{'='*70}")
        print("SAMPLE GUESS CoT:")
        print('='*70)
        print(guess_recs[0]['messages'][1]['content'])

    # Write output
    out_dir = BASE / "all_categorical_splits_v21"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, records):
        with open(path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  Wrote {path.name}: {len(records)} records")

    write_jsonl(out_dir / "train_cot_cryptarithm_deduce.jsonl", deduce_recs)
    write_jsonl(out_dir / "train_cot_cryptarithm_guess.jsonl", guess_recs)
    return deduce_recs, guess_recs


if __name__ == "__main__":
    main()
