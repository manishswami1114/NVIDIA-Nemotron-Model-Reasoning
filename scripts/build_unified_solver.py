#!/usr/bin/env python3
"""
Build perfect_solver_complete.md for ALL equation_numeric_guess puzzles.

KEY RULE: Within a puzzle, same operator → same formula for ALL examples.
We find the formula that works for ALL examples of an operator GROUP,
not individually.

DOMAIN CONSISTENCY: Each puzzle uses either "direct" or "rev" domain.
When an operator is ambiguous (both domains match), we prefer the
dominant domain determined by other operators in the puzzle.
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
JSONL = BASE / "all_categorical_splits_v13" / "train_cot_equation_numeric_guess.jsonl"
SOLVER_OUT = BASE / "all_categorical_splits_v13" / "perfect_solver_complete.md"

# Domain tags
DIRECT = "direct"
REV = "rev"
NEUTRAL = "neutral"  # concat, a||b, etc. — no domain preference


def rev_s(s):
    neg = s.startswith('-')
    if neg: s = s[1:]
    r = s[::-1]
    return '-' + r if neg else r


def _rv(a, b):
    ra, rb = int(rev_s(a)), int(rev_s(b))
    return max(ra, rb), min(ra, rb)


# ---- UNIFIED FORMULA LIBRARY with domain tags ----
# Each entry: (fn, name, domain)
FORMULAS = []
def _f(fn, name, domain):
    FORMULAS.append((fn, name, domain))

# Direct domain formulas
_f(lambda a,b: str(max(int(a),int(b)) - min(int(a),int(b))),                    "max(a,b)-min(a,b)",                    DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) + min(int(a),int(b))),                    "max(a,b)+min(a,b)",                    DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) * min(int(a),int(b))),                    "max(a,b)*min(a,b)",                    DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) + min(int(a),int(b)) - 1),               "max(a,b)+min(a,b)-1",                  DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) * min(int(a),int(b)) + 1),               "max(a,b)*min(a,b)+1",                  DIRECT)
_f(lambda a,b: str(-(max(int(a),int(b)) - min(int(a),int(b)))),                 "-(max(a,b)-min(a,b))",                 DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) + min(int(a),int(b)) + 1),               "max(a,b)+min(a,b)+1",                  DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) * min(int(a),int(b)) - 1),               "max(a,b)*min(a,b)-1",                  DIRECT)
_f(lambda a,b: str(max(int(a),int(b)) % min(int(a),int(b))) if min(int(a),int(b))!=0 else None, "max(a,b)%min(a,b)",    DIRECT)

# Rev domain formulas
_f(lambda a,b: rev_s(str(_rv(a,b)[0] * _rv(a,b)[1])),                           "rev(max(rev(a),rev(b))*min(rev(a),rev(b)))",                REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] + _rv(a,b)[1])),                           "rev(max(rev(a),rev(b))+min(rev(a),rev(b)))",                REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] * _rv(a,b)[1] - 1)),                      "rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)",              REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] + _rv(a,b)[1] + 1)),                      "rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)",              REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] * _rv(a,b)[1] + 1)),                      "rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)",              REV)
_f(lambda a,b: '-' + rev_s(str(_rv(a,b)[0] - _rv(a,b)[1])),                    "-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))",               REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] - _rv(a,b)[1])),                          "rev(max(rev(a),rev(b))-min(rev(a),rev(b)))",                REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] + _rv(a,b)[1] - 1)),                      "rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)",              REV)
_f(lambda a,b: rev_s(str(_rv(a,b)[0] % _rv(a,b)[1])) if _rv(a,b)[1]!=0 else None, "rev(max(rev(a),rev(b))%min(rev(a),rev(b)))",             REV)

# Neutral (concat) formulas — no domain preference
_f(lambda a,b: a+b,                                                              "a||b",                                 NEUTRAL)
_f(lambda a,b: b+a,                                                              "b||a",                                 NEUTRAL)
_f(lambda a,b: str(max(int(a),int(b))) + str(min(int(a),int(b))).zfill(2),      "max(a,b)||min(a,b)",                   NEUTRAL)
_f(lambda a,b: str(min(int(a),int(b))).zfill(2) + str(max(int(a),int(b))),      "min(a,b)||max(a,b)",                   NEUTRAL)
_f(lambda a,b: rev_s(str(_rv(a,b)[0])) + rev_s(str(_rv(a,b)[1])),              "rev(max(rev(a),rev(b)))||rev(min(rev(a),rev(b)))", REV)

# Simple a op b (for cases with sign in answer) — direct-ish but treated neutral
_f(lambda a,b: str(int(a)-int(b)),                                               "(a-b)",                                NEUTRAL)
_f(lambda a,b: str(int(a)*int(b)),                                               "(a*b)",                                NEUTRAL)
_f(lambda a,b: str(int(a)+int(b)),                                               "(a+b)",                                NEUTRAL)
_f(lambda a,b: str(int(a)+int(b)+1),                                             "(a+b)+1",                              NEUTRAL)
_f(lambda a,b: str(int(a)+int(b)-1),                                             "(a+b)-1",                              NEUTRAL)


def get_formula_domain(name):
    """Get domain tag for a formula name."""
    for _, n, d in FORMULAS:
        if n == name:
            return d
    # Op-in-answer formula domain detection
    if 'rev(' in name:
        return REV
    if 'max(a,b)' in name or 'min(a,b)' in name:
        return DIRECT
    return NEUTRAL


# ---- OP-IN-ANSWER FORMULA LIBRARY ----
# These are formulas used when the operator character appears in the answer.
# Each: (test_fn, name, domain, position)  where position is "prefix" or "suffix"
OP_IN_ANSWER_FORMULAS = []
def _oia(test_fn, name, domain, position):
    OP_IN_ANSWER_FORMULAS.append((test_fn, name, domain, position))

# Prefix patterns (op appears at start of answer)
_oia(lambda a,b: str(max(int(a),int(b)) - min(int(a),int(b))),
     "-(max(a,b)-min(a,b))", DIRECT, "prefix")
_oia(lambda a,b: str(max(int(a),int(b)) % min(int(a),int(b))) if min(int(a),int(b))!=0 else None,
     "-(max(a,b)%min(a,b))", DIRECT, "prefix")
_oia(lambda a,b: rev_s(str(_rv(a,b)[0] - _rv(a,b)[1])),
     "-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))", REV, "prefix")

# Suffix patterns (op appears at end of answer)
_oia(lambda a,b: str(max(int(a),int(b)) - min(int(a),int(b))),
     "max(a,b)-min(a,b) [op suffix]", DIRECT, "suffix")
_oia(lambda a,b: str(max(int(a),int(b)) % min(int(a),int(b))) if min(int(a),int(b))!=0 else None,
     "max(a,b)%min(a,b) [op suffix]", DIRECT, "suffix")
_oia(lambda a,b: rev_s(str(_rv(a,b)[0] - _rv(a,b)[1])),
     "rev(max(rev(a),rev(b))-min(rev(a),rev(b))) [op suffix]", REV, "suffix")
_oia(lambda a,b: rev_s(str(_rv(a,b)[0] % _rv(a,b)[1])) if _rv(a,b)[1]!=0 else None,
     "rev(max(rev(a),rev(b))%min(rev(a),rev(b))) [op suffix]", REV, "suffix")


def find_all_group_formulas(examples):
    """Find ALL formulas matching ALL examples in a group.
    Returns list of (fn, name, domain)."""
    matches = []
    for fn, name, domain in FORMULAS:
        ok = True
        for a, b, result in examples:
            try:
                c = fn(a, b)
                if c is None or c != result:
                    ok = False; break
            except:
                ok = False; break
        if ok:
            matches.append((fn, name, domain))
    return matches


def find_all_group_formulas_with_op(examples, op_char):
    """Find ALL formulas matching ALL examples, including op-in-answer.
    Returns list of (fn_or_none, name, domain)."""
    # First try standard formulas
    matches = find_all_group_formulas(examples)

    # Check for op-in-answer patterns
    # Determine if all examples have op in answer
    # For +/- operators: only check if no standard formula matched AND
    # the result contains the op in a non-numeric position
    all_op_prefix = True
    all_op_suffix = True
    for a, b, result in examples:
        has_op = op_char in result
        if op_char in '+-':
            # For +/-, only treat as op-in-answer if standard formulas failed
            # and result starts/ends with the op (not just a sign)
            if not has_op or (matches and any(fn is not None for fn, _, _ in matches)):
                all_op_prefix = False
                all_op_suffix = False
                break
        else:
            if not has_op:
                all_op_prefix = False
                all_op_suffix = False
                break
        if not result.startswith(op_char):
            all_op_prefix = False
        if not result.endswith(op_char):
            all_op_suffix = False

    if all_op_prefix:
        for test_fn, name, domain, position in OP_IN_ANSWER_FORMULAS:
            if position != "prefix":
                continue
            ok = True
            for a, b, result in examples:
                numeric = result[1:]  # remove op prefix
                try:
                    c = test_fn(a, b)
                    if c is None or c != numeric:
                        ok = False; break
                except:
                    ok = False; break
            if ok:
                matches.append((None, name, domain))

    if all_op_suffix:
        for test_fn, name, domain, position in OP_IN_ANSWER_FORMULAS:
            if position != "suffix":
                continue
            ok = True
            for a, b, result in examples:
                numeric = result[:-1]  # remove op suffix
                try:
                    c = test_fn(a, b)
                    if c is None or c != numeric:
                        ok = False; break
                except:
                    ok = False; break
            if ok:
                matches.append((None, name, domain))

    return matches


def determine_puzzle_domain(op_all_matches):
    """Determine dominant domain for a puzzle based on all operator matches.

    Logic:
    - For each operator, check if it ONLY matches one domain
    - Count exclusive-direct vs exclusive-rev operators
    - The domain with more exclusive operators wins
    - If tied or all ambiguous, default to direct
    """
    rev_exclusive = 0
    direct_exclusive = 0
    rev_any = 0
    direct_any = 0

    for op, matches in op_all_matches.items():
        if not matches:
            continue
        domains = set()
        for _, _, d in matches:
            if d != NEUTRAL:
                domains.add(d)

        if domains == {REV}:
            rev_exclusive += 1
        elif domains == {DIRECT}:
            direct_exclusive += 1

        if REV in domains:
            rev_any += 1
        if DIRECT in domains:
            direct_any += 1

    # If any operator is exclusively rev-domain, puzzle is rev-domain
    if rev_exclusive > 0 and direct_exclusive == 0:
        return REV
    # If any operator is exclusively direct-domain, puzzle is direct-domain
    if direct_exclusive > 0 and rev_exclusive == 0:
        return DIRECT
    # If both exist, go with whichever has more exclusive operators
    if rev_exclusive > direct_exclusive:
        return REV
    if direct_exclusive > rev_exclusive:
        return DIRECT
    # Tiebreak: count total domain appearances
    if rev_any > direct_any:
        return REV
    return DIRECT


def select_formula_for_domain(matches, domain):
    """From a list of matching formulas, select the best one for the given domain.

    Priority:
    1. Exact domain match
    2. Neutral domain
    3. Any match (fallback)
    """
    # Prefer same-domain match
    for fn, name, d in matches:
        if d == domain:
            return fn, name
    # Then neutral
    for fn, name, d in matches:
        if d == NEUTRAL:
            return fn, name
    # Then any match
    if matches:
        return matches[0][0], matches[0][1]
    return None, None


def find_single_formula_all(a, b, result, op_char=None):
    """Find ALL formulas for a single equation. Returns list of (fn, name, domain)."""
    matches = []

    # Standard formulas
    for fn, name, domain in FORMULAS:
        try:
            c = fn(a, b)
            if c is not None and c == result:
                matches.append((fn, name, domain))
        except:
            continue

    # Op-in-answer patterns
    if op_char and op_char in result:
        # For +/-, only treat as op-in-answer if:
        # - result doesn't parse as normal number, OR
        # - no standard formula matched (meaning +/- is truly the operator symbol)
        if op_char in '+-':
            try:
                int(result)
                if matches:
                    return matches  # Normal signed number with formula matches, not op-in-answer
                # No standard matches — fall through to try op-in-answer
            except:
                pass

        numeric = result.replace(op_char, '')
        # Validate numeric part
        if not (numeric.lstrip('-0').isdigit() or numeric == '0' or
                (numeric and numeric.lstrip('0') == '' and len(numeric) > 0)):
            return matches

        if result.startswith(op_char):
            for test_fn, name, domain, position in OP_IN_ANSWER_FORMULAS:
                if position != "prefix":
                    continue
                try:
                    c = test_fn(a, b)
                    if c is not None and c == numeric:
                        matches.append((None, name, domain))
                except:
                    continue

        if result.endswith(op_char):
            for test_fn, name, domain, position in OP_IN_ANSWER_FORMULAS:
                if position != "suffix":
                    continue
                try:
                    c = test_fn(a, b)
                    if c is not None and c == numeric:
                        matches.append((None, name, domain))
                except:
                    continue

    return matches


def parse_prompt(prompt):
    lines = prompt.strip().split('\n')
    examples = []; qa = qop = qb = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if any(kw in line.lower() for kw in ['alice','secret','transformation','example','please','boxed','final']): continue
        if 'determine' in line.lower():
            m = re.search(r'(\d+)\s*(.)\s*(\d+)', line)
            if m: qa, qop, qb = m.group(1), m.group(2), m.group(3)
            continue
        m = re.match(r'\s*(\d+)\s*(.)\s*(\d+)\s*=\s*(.+?)(?:\s*$)', line)
        if m: examples.append((line.strip(), m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    return examples, qa, qop, qb


def main():
    # Ground truth
    gt = {}
    with open(BASE / "data" / "raw" / "train.csv") as f:
        csv = f.read()
    for m in re.finditer(r'determine the result for:\s*(\d+.{1}\d+).*?,\s*(.+?)$', csv, re.MULTILINE):
        q = m.group(1).strip().strip('"').replace(' ','')
        a = m.group(2).strip().strip('"').strip()
        if a: gt[q] = a

    with open(JSONL) as f:
        records = [json.loads(line) for line in f]

    lines = []
    total_eqs = 0; solved_eqs = 0; total_queries = 0; solved_queries = 0
    formula_usage = defaultdict(int)
    domain_stats = defaultdict(int)

    for i, rec in enumerate(records):
        user_msg = [m for m in rec['messages'] if m['role'] == 'user'][0]
        examples, qa, qop, qb = parse_prompt(user_msg['content'])
        if not examples or qa is None: continue

        # Group by operator
        op_groups = defaultdict(list)
        for eq_str, a, op, b, result in examples:
            op_groups[op].append((a, b, result))

        # Phase 1: Find ALL matching formulas for each operator group
        op_all_matches = {}  # op -> list of (fn, name, domain)
        for op, exs in op_groups.items():
            op_all_matches[op] = find_all_group_formulas_with_op(exs, op)

        # Phase 2: Determine the puzzle's dominant domain
        puzzle_domain = determine_puzzle_domain(op_all_matches)
        domain_stats[puzzle_domain] += 1

        # Phase 3: Select best formula per operator, respecting domain
        op_formulas = {}  # op -> (fn, name)
        for op, matches in op_all_matches.items():
            if matches:
                fn, name = select_formula_for_domain(matches, puzzle_domain)
                if fn is not None or name is not None:
                    op_formulas[op] = (fn, name)

        # Write each equation with its formula
        for eq_str, a, op, b, result in examples:
            total_eqs += 1
            if op in op_formulas:
                fn, name = op_formulas[op]
                lines.append(f"{eq_str} = {name}")
                formula_usage[name] += 1
                solved_eqs += 1
            else:
                # Try individual match with domain preference
                all_matches = find_single_formula_all(a, b, result, op)
                if all_matches:
                    fn, name = select_formula_for_domain(all_matches, puzzle_domain)
                    lines.append(f"{eq_str} = {name}")
                    formula_usage[name] += 1
                    solved_eqs += 1
                else:
                    lines.append(f"{eq_str} = UNSOLVED")

        # Query line
        total_queries += 1
        qkey = f"{qa}{qop}{qb}"
        gt_ans = gt.get(qkey, "")

        if gt_ans:
            all_matches = find_single_formula_all(qa, qb, gt_ans, qop)
            if all_matches:
                fn, name = select_formula_for_domain(all_matches, puzzle_domain)
                lines.append(f"Now, determine the result for: {qkey} = {name}")
                formula_usage[name] += 1
                solved_queries += 1
            else:
                lines.append(f"Now, determine the result for: {qkey} = UNSOLVED (gt={gt_ans})")
        else:
            # Use existing answer from record
            content = rec['messages'][-1]['content']
            boxed = re.findall(r'\\boxed\{([^}]*)\}', content)
            ans = boxed[-1] if boxed else ""
            if ans:
                all_matches = find_single_formula_all(qa, qb, ans, qop)
                if all_matches:
                    fn, name = select_formula_for_domain(all_matches, puzzle_domain)
                    lines.append(f"Now, determine the result for: {qkey} = {name}")
                    formula_usage[name] += 1
                    solved_queries += 1
                else:
                    lines.append(f"Now, determine the result for: {qkey} = UNSOLVED (ans={ans})")
            else:
                lines.append(f"Now, determine the result for: {qkey} = NO_ANSWER")

        lines.append("")

    with open(SOLVER_OUT, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Equations: {solved_eqs}/{total_eqs} solved")
    print(f"Queries: {solved_queries}/{total_queries} solved")
    print(f"\nDomain distribution: {dict(domain_stats)}")
    print(f"\nFormula usage (top 25):")
    for name, count in sorted(formula_usage.items(), key=lambda x: -x[1])[:25]:
        domain = get_formula_domain(name)
        print(f"  {count:3d}  [{domain:7s}] {name}")

    # Count unsolved
    unsolved_lines = [l for l in lines if 'UNSOLVED' in l]
    if unsolved_lines:
        print(f"\nUnsolved ({len(unsolved_lines)}):")
        for l in unsolved_lines[:20]:
            print(f"  {l}")


if __name__ == "__main__":
    main()
