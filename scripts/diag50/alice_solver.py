"""AliceEquationSolver — ported from the public 97.2% gold-conditioned solver.

Used only for the 50-record diagnostic dataset. Python-only search (no Rust
helper). Gold-conditioned mode is enabled when answer_hint is supplied.

For the diagnostic we restrict ourselves to:
  - standard mode (no alice/little_endian)
  - simple ops (add, sub, rsub, absdiff, mul)
  - 10-symbol bijection (base 10)
  - unsigned answers (no operator-prefix on rhs)

This keeps the search fast and produces CoTs in the simplest format that
matches the baseline cryptarithm template.
"""
from __future__ import annotations
from itertools import permutations
from collections import defaultdict
from math import gcd


def _ss(a, b): return a - b if a >= b else None
def _rs(a, b): return b - a if b >= a else None


OPERATIONS = {
    "add":     (lambda a, b: a + b,             "+",          "addition"),
    "sub":     (_ss,                            "-",          "subtraction"),
    "rsub":    (_rs,                            "rev-sub",    "reverse subtraction"),
    "absdiff": (lambda a, b: abs(a - b),        "|a-b|",      "absolute difference"),
    "mul":     (lambda a, b: a * b,             "*",          "multiplication"),
    "gcd":     (gcd,                            "gcd",        "gcd"),
}


class AliceEquationSolver:
    def __init__(self, prompt: str, answer_hint: str | None = None):
        self.prompt = prompt
        self.answer_hint = answer_hint
        self.op_types = list(OPERATIONS.keys())
        self.examples, self.query = self._parse(prompt)
        self._analyze()

    @staticmethod
    def _parse(prompt):
        lines = prompt.strip().split("\n")
        examples, query = [], None
        for line in lines:
            line = line.strip()
            if not line:
                continue
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

    def _analyze(self):
        self.op_examples = defaultdict(list)
        self.example_op_chars = set()
        for lhs, rhs in self.examples:
            if len(lhs) >= 5:
                self.example_op_chars.add(lhs[2])
                self.op_examples[lhs[2]].append((lhs, rhs))
        self.op_chars = set(self.example_op_chars)
        if self.query and len(self.query) >= 5:
            self.op_chars.add(self.query[2])
        all_chars = set()
        for lhs, rhs in self.examples:
            all_chars.update(lhs); all_chars.update(rhs)
        if self.query:
            all_chars.update(self.query)
        if self.answer_hint:
            for ch in self.answer_hint:
                if ch.strip() and ch not in self.op_chars:
                    all_chars.add(ch)
        self.content_symbols = sorted(all_chars - self.op_chars)
        self.base = len(self.content_symbols)

    def solve(self):
        """Returns (answer_str, details) or (None, None).

        details = {
            'mode': 'standard',
            'mapping': {symbol: digit, ...},
            'ops': {op_char: op_name, ...},   # e.g. {'+': 'add'}
            'q_op': '+',
            'q_op_name': 'add',
            'qL': int, 'qR': int, 'q_result': int,
            'answer': str,
        }
        """
        if not self.query or len(self.query) < 5:
            return None, None
        if self.base > 10:
            return None, None

        # Reject puzzles with operator-prefix on rhs (signed)
        for lhs, rhs in self.examples:
            if len(rhs) > 1 and rhs[0] == lhs[2]:
                return None, None
        if self.answer_hint and len(self.answer_hint) > 1 and self.answer_hint[0] == self.query[2]:
            return None, None

        sym2i = {s: i for i, s in enumerate(self.content_symbols)}
        op_eqs = defaultdict(list)
        for lhs, rhs in self.examples:
            if len(lhs) < 5: continue
            chars = [lhs[0], lhs[1], lhs[3], lhs[4]] + list(rhs)
            if not all(c in sym2i for c in chars):
                return None, None
            op_eqs[lhs[2]].append((
                sym2i[lhs[0]], sym2i[lhs[1]],
                sym2i[lhs[3]], sym2i[lhs[4]],
                tuple(sym2i[c] for c in rhs),
                len(rhs),
            ))

        qo = self.query[2]
        ql = (sym2i[self.query[0]], sym2i[self.query[1]])
        qr = (sym2i[self.query[3]], sym2i[self.query[4]])

        # Gold-conditioning: add the gold answer as an extra equation for qo
        if self.answer_hint:
            gold = self.answer_hint.strip()
            if not all(c in sym2i for c in gold) and not all(c in sym2i for c in self.query):
                return None, None
            if all(c in sym2i for c in gold) and all(c in sym2i for c in self.query):
                op_eqs[qo].append((
                    sym2i[self.query[0]], sym2i[self.query[1]],
                    sym2i[self.query[3]], sym2i[self.query[4]],
                    tuple(sym2i[c] for c in gold),
                    len(gold),
                ))

        sorted_ops = sorted(op_eqs.keys(), key=lambda op: -len(op_eqs[op]))
        n = self.base
        for perm in permutations(range(10), n):
            ops_valid = {}
            ok = True
            for op in sorted_ops:
                valid = []
                for ot in self.op_types:
                    fn = OPERATIONS[ot][0]
                    all_match = True
                    for l0, l1, r0, r1, ridx, rl in op_eqs[op]:
                        L = perm[l0]*10 + perm[l1]
                        R = perm[r0]*10 + perm[r1]
                        v = fn(L, R)
                        if v is None or v < 0:
                            all_match = False; break
                        sv = str(v).zfill(rl)
                        if len(sv) != rl:
                            all_match = False; break
                        expected = [perm[i] for i in ridx]
                        if [int(c) for c in sv] != expected:
                            all_match = False; break
                    if all_match:
                        valid.append(ot)
                if not valid:
                    ok = False; break
                ops_valid[op] = valid
            if not ok:
                continue
            # Found a consistent perm + op-set. Encode answer.
            mapping = {s: perm[i] for i, s in enumerate(self.content_symbols)}
            inv = {v: k for k, v in mapping.items()}
            q_op_name = ops_valid[qo][0]
            fn = OPERATIONS[q_op_name][0]
            qL = mapping[self.query[0]]*10 + mapping[self.query[1]]
            qR = mapping[self.query[3]]*10 + mapping[self.query[4]]
            qv = fn(qL, qR)
            if qv is None or qv < 0:
                continue
            # Encode using the smallest representation length consistent with hint
            target_len = len(self.answer_hint) if self.answer_hint else len(str(qv))
            if target_len < len(str(qv)):
                continue
            sv = str(qv).zfill(target_len)
            try:
                ans = "".join(inv[int(c)] for c in sv)
            except KeyError:
                continue
            if self.answer_hint and ans != self.answer_hint:
                continue
            details = {
                "mode": "standard",
                "mapping": mapping,
                "ops": {op: vs[0] for op, vs in ops_valid.items()},
                "q_op": qo,
                "q_op_name": q_op_name,
                "qL": qL, "qR": qR, "q_result": qv,
                "answer": ans,
            }
            return ans, details
        return None, None
