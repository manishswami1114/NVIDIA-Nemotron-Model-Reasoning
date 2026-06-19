"""Bit-manipulation CoT reasoner with 3-operand composite fallback.

Based on the user-provided code. Minimal Problem types are inlined here so we
do not need to import the `reasoners` package.

When the 2-operand search fails for an output bit and no other candidate matches,
this version brute-forces a 3-operand composite (op1(IN[a], op2(IN[b], IN[c])))
before falling back to `default 1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple


# ── Minimal Problem types (so we don't need the `reasoners` package) ──
@dataclass
class _Example:
    input_value: str
    output_value: str


@dataclass
class Problem:
    examples: List[_Example]
    question: str


N_BITS = 8

SYM_FAMILIES = ("XOR", "OR", "AND")
ASYM_FAMILIES = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")
DEFAULT_FAMILY = "DEFAULT"
SECTION_ORDER = (
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
)

_SECTION_TO_FAMILIES = {
    "Identity": ("I",),
    "NOT": ("NOT",),
    "Constant": ("0", "1"),
}

_FAMILY_TO_SECTION: dict[str, str] = {}
for _section in SECTION_ORDER:
    for _fam in _SECTION_TO_FAMILIES.get(_section, (_section,)):
        _FAMILY_TO_SECTION[_fam] = _section


RuleFamily = Literal[
    "I", "NOT", "0", "1",
    "XOR", "OR", "AND",
    "AND-NOT", "XOR-NOT", "OR-NOT",
    "DEFAULT",
    "COMPOSITE",
]


@dataclass(frozen=True)
class RuleCandidate:
    family: RuleFamily
    primary: Optional[int]
    secondary: Optional[int]
    expr: str
    primary_stride: Optional[int] = None
    secondary_stride: Optional[int] = None
    primary_offset: Optional[int] = None
    secondary_offset: Optional[int] = None
    tertiary: Optional[int] = None
    op1: Optional[str] = None
    op2: Optional[str] = None

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


@dataclass(frozen=True)
class Record:
    label: str
    col: str
    hash_: str
    matches: Tuple[int, ...]


def _normalize_bits(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    if len(bits) != N_BITS:
        return ""
    return bits


def _column_bits(values: Sequence[str], bit: int) -> str:
    return "".join(v[bit] for v in values)


def _bit_not(bit: str) -> str:
    return "1" if bit == "0" else "0"


def _invert(bits: str) -> str:
    return "".join(_bit_not(b) for b in bits)


def _column_hash(bits: str, total_examples: int) -> str:
    ones = bits.count("1")
    if ones == 0 or ones == total_examples:
        return "a"
    return format(ones, "x")


def _evaluate_binary(a: str, b: str, family: str) -> str:
    if family in ("AND", "AND-NOT"):
        return "1" if a == "1" and b == "1" else "0"
    if family in ("OR", "OR-NOT"):
        return "1" if a == "1" or b == "1" else "0"
    if family in ("XOR", "XOR-NOT"):
        return "1" if a != b else "0"
    raise ValueError(f"Unsupported family {family}")


def _apply_family(a_bits: str, b_bits: str, family: str, invert_second: bool = False) -> str:
    b_eff = _invert(b_bits) if invert_second else b_bits
    return "".join(_evaluate_binary(x, y, family) for x, y in zip(a_bits, b_eff))


def _search_composite_for_col(out_col: str, input_columns: List[str]) -> Optional[RuleCandidate]:
    OP_PAIRS = [
        ("XOR", "AND"), ("XOR", "OR"), ("XOR", "XOR"),
        ("OR", "AND"), ("OR", "XOR"),
        ("AND", "OR"), ("AND", "XOR"),
        ("XOR", "AND-NOT"), ("OR", "AND-NOT"),
    ]
    n_rows = len(out_col)
    for op1, op2 in OP_PAIRS:
        for a in range(N_BITS):
            for b in range(N_BITS):
                if a == b: continue
                for c in range(N_BITS):
                    if c == a or c == b: continue
                    if op2 in ("AND", "OR", "XOR") and b > c:
                        continue
                    match = True
                    for r in range(n_rows):
                        val_a = input_columns[a][r]
                        val_b = input_columns[b][r]
                        val_c = input_columns[c][r]
                        if op2.endswith("-NOT"):
                            base_op2 = op2.split("-")[0]
                            val_bc = _evaluate_binary(val_b, _bit_not(val_c), base_op2)
                        else:
                            val_bc = _evaluate_binary(val_b, val_c, op2)
                        res = _evaluate_binary(val_a, val_bc, op1)
                        if res != out_col[r]:
                            match = False
                            break
                    if match:
                        expr = f"{op1}(IN{a}, {op2}(IN{b}, IN{c}))"
                        return RuleCandidate(
                            family="COMPOSITE",
                            primary=a, secondary=b, tertiary=c,
                            op1=op1, op2=op2, expr=expr,
                        )
    return None


def _find_match(candidates: List[RuleCandidate], fam: str,
                ep: Optional[int], es: Optional[int]) -> Optional[RuleCandidate]:
    for c in candidates:
        if c.family != fam: continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None


def _exists_anywhere(all_matches: List[List[RuleCandidate]], fam: str,
                     ep: Optional[int], es: Optional[int]) -> bool:
    return any(_find_match(bit_cands, fam, ep, es) is not None for bit_cands in all_matches)


def _fail_suffix(all_matches, fam, ep, es) -> str:
    return "y" if _exists_anywhere(all_matches, fam, ep, es) else "x"


def _find_all_left_runs(all_matches):
    if not all_matches or not all_matches[0]:
        return []
    runs = []
    for start_cand in all_matches[0]:
        fam = start_cand.family
        for p_step, s_step in [(1, 1)]:
            chain = [start_cand]
            cur_p, cur_s = start_cand.primary, start_cand.secondary
            failed_next = None
            for b in range(1, len(all_matches)):
                ep = (cur_p + p_step) % N_BITS if cur_p is not None else None
                es = (cur_s + s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, ep, es)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, ep, es)
                    if ep is not None and es is not None:
                        failed_next = f"{ep}{es}{suffix}"
                    elif ep is not None:
                        failed_next = f"{ep}{suffix}"
                    break
                chain.append(found)
                cur_p, cur_s = ep, es
            runs.append((chain, failed_next))
    return runs


def _find_all_right_runs(all_matches):
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []
    runs = []
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        for p_step, s_step in [(1, 1)]:
            chain = [end_cand]
            cur_p, cur_s = end_cand.primary, end_cand.secondary
            failed_next = None
            for k in range(1, n):
                b = n - 1 - k
                pp = (cur_p - p_step) % N_BITS if cur_p is not None else None
                ps = (cur_s - s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, pp, ps)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, pp, ps)
                    if pp is not None and ps is not None:
                        failed_next = f"{pp}{ps}{suffix}"
                    elif pp is not None:
                        failed_next = f"{pp}{suffix}"
                    break
                chain.insert(0, found)
                cur_p, cur_s = pp, ps
            runs.append((chain, failed_next))
    return runs


def _lr_from_matches(all_matches):
    all_left_runs = _find_all_left_runs(all_matches)
    all_right_runs = _find_all_right_runs(all_matches)
    left_run = max(all_left_runs, key=lambda t: len(t[0])) if all_left_runs else ([], None)
    right_run = max(all_right_runs, key=lambda t: len(t[0])) if all_right_runs else ([], None)
    left_lines = ([_format_list(chain, failed=failed) for chain, failed in all_left_runs]
                  if all_left_runs else ["none"])
    left_best = _format_list(left_run[0], with_count=True)
    right_lines = ([_format_list(list(reversed(chain)), failed=failed)
                    for chain, failed in all_right_runs] if all_right_runs else ["none"])
    right_best = _format_list(list(reversed(right_run[0])), with_count=True)
    return left_lines, left_best, right_lines, right_best


def _format_list(cands, with_count=False, failed=None):
    if not cands:
        return "none"
    if with_count:
        parts = []
        for i, c in enumerate(cands):
            if i == 0: parts.append(c.expr)
            else: parts.append(_compact_rule(c))
        return " ".join(parts) + f": {len(cands)}"
    parts = [_compact_rule(c) for c in cands]
    if failed:
        parts.append(failed)
    return " ".join(parts)


def _compact_rule(c):
    if c.primary is not None and c.secondary is not None:
        return f"{c.primary}{c.secondary}"
    if c.primary is not None:
        return str(c.primary)
    return c.family


def _evaluate_rule(bits: str, rule: RuleCandidate) -> str:
    if rule.family == "DEFAULT": return "1"
    if rule.family == "0": return "0"
    if rule.family == "1": return "1"
    if rule.family == "I":
        return bits[rule.primary]
    if rule.family == "NOT":
        return _bit_not(bits[rule.primary])
    if rule.family in PAIR_FAMILIES:
        a = bits[rule.primary]
        b = bits[rule.secondary]
        if "-NOT" in rule.family:
            b = _bit_not(b)
        return _evaluate_binary(a, b, rule.family)
    if rule.family == "COMPOSITE":
        a = bits[rule.primary]; b = bits[rule.secondary]; c = bits[rule.tertiary]
        if rule.op2.endswith("-NOT"):
            base_op2 = rule.op2.split("-")[0]
            val_bc = _evaluate_binary(b, _bit_not(c), base_op2)
        else:
            val_bc = _evaluate_binary(b, c, rule.op2)
        return _evaluate_binary(a, val_bc, rule.op1)
    raise ValueError(f"Unknown family {rule.family}")


def _emit_apply(lines, question_bits, vector):
    lines.append(f"Applying to {question_bits}")
    lines.append("Input")
    for i, bit in enumerate(question_bits):
        lines.append(f"{i} {bit}")
    lines.append("Output")
    answer_bits = []
    for i, rule in enumerate(vector):
        if rule.family == "DEFAULT":
            lines.append(f"{i} default 1 = 1")
            answer_bits.append("1"); continue
        if rule.family in CONSTANT_FAMILIES:
            lines.append(f"{i} {rule.expr} = {rule.family}")
            answer_bits.append(rule.family); continue
        if rule.family == "I":
            val = question_bits[rule.primary]
            lines.append(f"{i} {rule.expr} = {val}")
            answer_bits.append(val); continue
        if rule.family == "NOT":
            val = question_bits[rule.primary]; nval = _bit_not(val)
            lines.append(f"{i} {rule.expr} = NOT({val}) = {nval}")
            answer_bits.append(nval); continue
        if rule.family == "COMPOSITE":
            a = question_bits[rule.primary]; b = question_bits[rule.secondary]; c = question_bits[rule.tertiary]
            if rule.op2.endswith("-NOT"):
                base_op2 = rule.op2.split("-")[0]
                n_c = _bit_not(c)
                inner_val = _evaluate_binary(b, n_c, base_op2)
                inner_expr = f"{base_op2}({b},NOT({c}))={inner_val}"
            else:
                inner_val = _evaluate_binary(b, c, rule.op2)
                inner_expr = f"{rule.op2}({b},{c})={inner_val}"
            final_val = _evaluate_binary(a, inner_val, rule.op1)
            lines.append(f"{i} {rule.expr} -> {rule.op1}({a}, {inner_val}) [{inner_expr}] = {final_val}")
            answer_bits.append(final_val); continue
        a = question_bits[rule.primary]; b = question_bits[rule.secondary]
        if rule.family in SYM_FAMILIES:
            result = _evaluate_rule(question_bits, rule)
            lines.append(f"{i} {rule.expr} = {rule.family}({a},{b}) = {result}")
            answer_bits.append(result); continue
        base = rule.family.split("-")[0]
        result = _evaluate_rule(question_bits, rule)
        lines.append(f"{i} {rule.expr} = {base}({a},NOT({b})) = {result}")
        answer_bits.append(result)
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{''.join(answer_bits)}}}")


def reasoning_bit_manipulation(problem: Problem) -> Optional[str]:
    examples = problem.examples
    if not examples:
        return None
    outputs = [_normalize_bits(ex.output_value) for ex in examples]
    inputs = [_normalize_bits(ex.input_value) for ex in examples]
    question_bits = _normalize_bits(problem.question)
    if any(not bits for bits in outputs + inputs) or not question_bits:
        return None
    if len(outputs[0]) != N_BITS or len(inputs[0]) != N_BITS:
        return None
    if len(outputs) != len(inputs):
        return None
    n_examples = len(outputs)

    output_columns = [_column_bits(outputs, i) for i in range(N_BITS)]
    input_columns = [_column_bits(inputs, i) for i in range(N_BITS)]
    input_inverted = [_invert(col) for col in input_columns]

    all_records: Dict[str, List[Record]] = {name: [] for name in SECTION_ORDER}
    all_matches: Dict[str, List[List[RuleCandidate]]] = {
        name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER
    }

    for out_idx, out_col in enumerate(output_columns):
        for i_col, in_col in enumerate(input_columns):
            if in_col == out_col:
                all_matches["Identity"][out_idx].append(RuleCandidate("I", i_col, None, f"I{i_col}"))
            if input_inverted[i_col] == out_col:
                all_matches["NOT"][out_idx].append(RuleCandidate("NOT", i_col, None, f"NOT{i_col}"))
        if out_col.count("1") == 0:
            all_matches["Constant"][out_idx].append(RuleCandidate("0", None, None, "C0"))
        if out_col.count("1") == n_examples:
            all_matches["Constant"][out_idx].append(RuleCandidate("1", None, None, "C1"))

    for label, col in zip([str(i) for i in range(N_BITS)], input_columns):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Identity"].append(Record(label, col, _column_hash(col, n_examples), matches))
    for label, col in zip([str(i) for i in range(N_BITS)], input_inverted):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["NOT"].append(Record(label, col, _column_hash(col, n_examples), matches))
    for val in ("0", "1"):
        col = val * n_examples
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Constant"].append(Record(val, col, _column_hash(col, n_examples), matches))

    for fam in ("XOR", "OR", "AND"):
        for circ_diff in range(1, N_BITS // 2 + 1):
            n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + circ_diff) % N_BITS
                lo, hi = min(a, b), max(a, b)
                col = _apply_family(input_columns[lo], input_columns[hi], fam)
                matches = tuple(i for i, out_col in enumerate(output_columns) if col == out_col)
                all_records[fam].append(Record(f"{a}{b} {b}{a}", col, _column_hash(col, n_examples), matches))
                for out_idx in matches:
                    all_matches[fam][out_idx].append(RuleCandidate(fam, a, b, f"{fam}{a}{b}"))
                    all_matches[fam][out_idx].append(RuleCandidate(fam, b, a, f"{fam}{b}{a}"))

    for fam in ("AND-NOT", "XOR-NOT", "OR-NOT"):
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                col = _apply_family(input_columns[a], input_columns[b], fam, invert_second=True)
                matches = tuple(i for i, out_col in enumerate(output_columns) if col == out_col)
                all_records[fam].append(Record(f"{a}{b}", col, _column_hash(col, n_examples), matches))
                for out_idx in matches:
                    all_matches[fam][out_idx].append(RuleCandidate(fam, a, b, f"{fam}{a}{b}"))

    for name in ("Identity", "NOT", "Constant"):
        all_records[name].sort(key=lambda r: r.label)

    lines: List[str] = []
    lines.append("We need to deduce the transformation by matching the example outputs.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    for i, out in enumerate(outputs):
        lines.append(f"Output {i}: {out}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {out[bit]}")
        lines.append("")

    lines.append("Output bit columns (with bitsum as hash)")
    for bit in range(N_BITS):
        lines.append(f"{bit} {output_columns[bit]} {_column_hash(output_columns[bit], n_examples)}")
    lines.append("")
    for i, inp in enumerate(inputs):
        lines.append(f"Input {i}: {inp}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {inp[bit]}")
        lines.append("")

    lines.append("When matching output")
    lines.append("x: not in operator")
    lines.append("y: wrong position")
    lines.append("")

    section_lefts = []
    section_rights = []

    def _add_section(name):
        records = all_records[name]
        per_bit = all_matches[name]
        lines.append(name)
        prev_diff = None
        for rec in records:
            if len(rec.label) >= 2 and rec.label[0].isdigit() and rec.label[1].isdigit():
                diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                if prev_diff is not None and diff != prev_diff:
                    lines.append("")
                prev_diff = diff
            line = f"{rec.label} {rec.col} {rec.hash_}"
            if rec.matches:
                line += " match " + " ".join(str(i) for i in rec.matches)
            lines.append(line)
        lines.append("")
        lines.append("Matching output")
        for i in range(N_BITS):
            cands = per_bit[i]
            if cands:
                def _compact(c):
                    if c.primary is not None and c.secondary is not None:
                        return f"{c.primary}{c.secondary}"
                    if c.primary is not None:
                        return str(c.primary)
                    return c.expr
                lines.append(f"{i} " + " ".join(_compact(c) for c in cands))
            else:
                lines.append(f"{i} absent")
        lines.append("")
        left_lines, left_best, right_lines, right_best = _lr_from_matches(per_bit)
        section_lefts.append((name, left_best))
        section_rights.append((name, right_best))
        lines.append("Left")
        for ll in left_lines: lines.append(ll)
        lines.append(f"Best: {left_best}")
        lines.append("")
        lines.append("Right")
        for rl in right_lines: lines.append(rl)
        lines.append(f"Best: {right_best}")
        lines.append("")

    for name in all_records:
        _add_section(name)

    lines.append("Selecting"); lines.append("")

    def _parse_count(val):
        if val == "none": return 0
        try: return int(val.rsplit(": ", 1)[-1])
        except ValueError: return 0

    def _pick_winner(entries):
        best_name = None; best_text = "none"; best_count = 0
        for name, val in entries:
            count = _parse_count(val)
            if count > best_count:
                best_count = count; best_name = name; best_text = val
        return best_name, best_text, best_count

    left_winner_name, left_winner_text, left_winner_count = _pick_winner(section_lefts)
    right_winner_name, right_winner_text, right_winner_count = _pick_winner(section_rights)

    def _get_section_run(winner_name, direction):
        if winner_name is None: return []
        per_bit = all_matches[winner_name]
        runs = _find_all_left_runs(per_bit) if direction == "left" else _find_all_right_runs(per_bit)
        if not runs: return []
        best_chain, _ = max(runs, key=lambda t: len(t[0]))
        return best_chain

    left_run = _get_section_run(left_winner_name, "left")
    right_run = _get_section_run(right_winner_name, "right")

    lines.append("Lefts")
    for name, lb in section_lefts: lines.append(f"{name} {lb}")
    lines.append("")
    lines.append("Rights")
    for name, rb in section_rights: lines.append(f"{name} {rb}")
    lines.append("")
    lines.append(f"Left longest: {left_winner_count}")
    lines.append(f"Right longest: {right_winner_count}")
    lines.append("")

    def _matching_line(label, winner_name, entries):
        parts = [f"{name} {'yes' if name == winner_name else 'no'}" for name, _val in entries]
        return f"{label} winner: {', '.join(parts)}"

    if right_winner_count > left_winner_count:
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append("")
        lines.append(f"Best right: {right_winner_text}")
        lines.append(f"Best left: {left_winner_text}")
    else:
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append("")
        lines.append(f"Best left: {left_winner_text}")
        lines.append(f"Best right: {right_winner_text}")
    lines.append("")

    left_len_final = left_winner_count
    right_len_final = right_winner_count
    if left_len_final + right_len_final > N_BITS:
        if right_len_final > left_len_final:
            left_len_final = N_BITS - right_len_final
            left_run = left_run[:left_len_final]
        else:
            right_len_final = N_BITS - left_len_final
            right_run = right_run[-right_len_final:] if right_len_final else []
    left_was_truncated = left_len_final < left_winner_count
    right_was_truncated = right_len_final < right_winner_count
    trunc_left = f"Truncated left: {_format_list(left_run, with_count=True)}"
    if left_was_truncated: trunc_left += " truncated"
    trunc_right = f"Truncated right: {_format_list(list(reversed(right_run)), with_count=True)}"
    if right_was_truncated: trunc_right += " truncated"
    if right_winner_count > left_winner_count:
        lines.append(trunc_right); lines.append(trunc_left)
    else:
        lines.append(trunc_left); lines.append(trunc_right)
    lines.append("")

    right_start_final = N_BITS - right_len_final
    lines.append("Tentative from right")
    for i in range(N_BITS - 1, -1, -1):
        if i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")
    lines.append("Tentative")
    for i in range(N_BITS):
        if i < left_len_final:
            lines.append(f"{i} {left_run[i].expr}")
        elif i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")

    def _extrap_from(run, bit, run_start_bit, side="left"):
        if not run: return None
        r = run[0]; p = r.primary; s = r.secondary
        if p is not None:
            p_off = (p - run_start_bit) % N_BITS
            ep = (p_off + bit) % N_BITS
        else: ep = None
        if s is not None:
            s_off = (s - run_start_bit) % N_BITS
            es = (s_off + bit) % N_BITS
        else: es = None
        if ep is not None and es is not None: return f"?{ep}{es}"
        if ep is not None:
            return f"?{ep}?" if side == "left" else f"??{ep}"
        return None

    left_fam = left_run[0].family if left_run else None
    right_fam = right_run[0].family if right_run else None
    left_is_binary = left_fam in PAIR_FAMILIES if left_fam else False
    right_is_binary = right_fam in PAIR_FAMILIES if right_fam else False
    left_is_unary = left_fam in UNARY_FAMILIES if left_fam else False
    right_is_unary = right_fam in UNARY_FAMILIES if right_fam else False

    if right_winner_count > left_winner_count:
        preferred = []
        for i in range(N_BITS):
            if i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif i < left_len_final:
                preferred.append(left_run[i].expr)
            elif right_is_binary or right_is_unary:
                preferred.append(_extrap_from(right_run, i, right_start_final, "right") or "pending")
            else:
                preferred.append("pending")
        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1): lines.append(f"{i} {preferred[i]}")
        lines.append("")
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if left_is_binary or left_is_unary:
                    preferred[i] = _extrap_from(left_run, i, 0, "left") or "?"
                else: preferred[i] = "?"
            elif "?" in preferred[i][1:] and left_is_unary:
                el = _extrap_from(left_run, i, 0, "left")
                if el:
                    merged = list(preferred[i]); el_chars = list(el)
                    for j in range(1, min(len(merged), len(el_chars))):
                        if merged[j] == "?" and el_chars[j] != "?":
                            merged[j] = el_chars[j]
                    preferred[i] = "".join(merged)
        lines.append("Preferred from left")
        for i in range(N_BITS): lines.append(f"{i} {preferred[i]}")
        lines.append("")
    else:
        preferred = []
        for i in range(N_BITS):
            if i < left_len_final:
                preferred.append(left_run[i].expr)
            elif i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif left_is_binary or left_is_unary:
                preferred.append(_extrap_from(left_run, i, 0, "left") or "pending")
            else:
                preferred.append("pending")
        lines.append("Preferred from left")
        for i in range(N_BITS): lines.append(f"{i} {preferred[i]}")
        lines.append("")
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if right_is_binary or right_is_unary:
                    preferred[i] = _extrap_from(right_run, i, right_start_final, "right") or "?"
                else: preferred[i] = "?"
            elif "?" in preferred[i][1:] and right_is_unary:
                er = _extrap_from(right_run, i, right_start_final, "right")
                if er:
                    merged = list(preferred[i]); er_chars = list(er)
                    for j in range(1, min(len(merged), len(er_chars))):
                        if merged[j] == "?" and er_chars[j] != "?":
                            merged[j] = er_chars[j]
                    preferred[i] = "".join(merged)
        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1): lines.append(f"{i} {preferred[i]}")
        lines.append("")

    lines.append("Preferred")
    for i, pref in enumerate(preferred):
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            lines.append(f"{i} {pref} ?{pref[2]}{pref[1]}")
        else:
            lines.append(f"{i} {pref}")
    lines.append("")

    default_cand = RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
    best = [default_cand] * N_BITS

    for i, rc in enumerate(left_run): best[i] = rc
    for i, rc in enumerate(right_run): best[right_start_final + i] = rc

    lines.append("Matching")
    pending_indices = []
    per_bit_cat = {name: {} for name in SECTION_ORDER}

    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith("?") or pref == "?":
            lines.append(f"{i} {best[i].expr}")
            continue
        pending_indices.append(i)
        digits_str = pref[1:]
        pref_digits = [int(d) for d in digits_str if d != "?"]
        checks = []
        for section_name in SECTION_ORDER:
            cands = all_matches[section_name][i]
            if section_name in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    checks.append(section_name + " " + " ".join(c.expr for c in found))
                    per_bit_cat[section_name][i] = found
                else:
                    checks.append(f"{section_name} absent")
            elif section_name == "Constant":
                if cands:
                    checks.append("Constant " + " ".join(c.expr for c in cands))
                    per_bit_cat["Constant"][i] = list(cands)
                else:
                    checks.append("Constant absent")
            else:
                found_c = None
                orderings = []
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                orderings.append((want_p, want_s))
                if want_p is not None and want_s is not None and want_p != want_s:
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for c in cands:
                        if (wp is None or c.primary == wp) and (ws is None or c.secondary == ws):
                            found_c = c; break
                    if found_c is not None: break
                if found_c is not None:
                    checks.append(found_c.expr); per_bit_cat[section_name][i] = [found_c]
                else:
                    checks.append(f"{section_name} absent")
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            pref_display = f"{pref} ?{pref[2]}{pref[1]}"
        else:
            pref_display = pref
        lines.append(f"{i} {pref_display} - {', '.join(checks)}")
    lines.append("")

    lines.append("Perfect match")
    chosen_cat = None
    for cat in SECTION_ORDER:
        is_perfect = (chosen_cat is None and bool(pending_indices)
                      and all(i in per_bit_cat[cat] for i in pending_indices))
        lines.append(f"{cat} {'yes' if is_perfect else 'no'}")
        if is_perfect: chosen_cat = cat
    lines.append("")

    pending_set = set(pending_indices)
    lines.append("Matched")
    for i in range(N_BITS):
        if i in pending_set:
            if chosen_cat and i in per_bit_cat[chosen_cat]:
                best[i] = per_bit_cat[chosen_cat][i][0]
                lines.append(f"{i} {best[i].expr}")
            else:
                all_cands = []
                for name in SECTION_ORDER:
                    if i in per_bit_cat[name]:
                        all_cands.extend(per_bit_cat[name][i])
                if all_cands:
                    lines.append(f"{i} " + " ".join(c.expr for c in all_cands))
                    best[i] = all_cands[0]
                else:
                    # ── COMPOSITE 3-OPERAND FALLBACK ──
                    comp_cand = _search_composite_for_col(output_columns[i], input_columns)
                    if comp_cand:
                        lines.append(f"{i} COMPOSITE MATCH: {comp_cand.expr}")
                        best[i] = comp_cand
                    else:
                        lines.append(f"{i} none")
                        best[i] = default_cand
        else:
            lines.append(f"{i} {best[i].expr}")
    lines.append("")

    if all(r.is_default for r in best):
        return None

    lines.append("Selected")
    for i, rule in enumerate(best):
        lines.append(f"{i} {rule.expr}")
    lines.append("")
    _emit_apply(lines, question_bits, best)

    return "\n".join(lines)
