"""Numpy-vectorized cryptarithm solver — gold-conditioned.

Evaluates all P(10, n) permutations at once per puzzle using numpy broadcasting.
Targets standard-mode cryptarithms with unsigned answers and simple ops.
"""
from __future__ import annotations
import numpy as np
from itertools import permutations as iperm
from math import gcd


# Pre-compute all permutations of 10 digits taken k=2..10 at a time on demand.
_PERM_CACHE: dict[int, np.ndarray] = {}


def _all_perms(k: int) -> np.ndarray:
    if k not in _PERM_CACHE:
        # Shape (perm_count, k), each row is a permutation of size-k from 0..9.
        _PERM_CACHE[k] = np.array(list(iperm(range(10), k)), dtype=np.int8)
    return _PERM_CACHE[k]


# ─────────────── Op semantics ─────────────────
def _op_add(L, R): return L + R
def _op_sub(L, R):
    out = L - R
    return np.where(out >= 0, out, -1)
def _op_rsub(L, R):
    out = R - L
    return np.where(out >= 0, out, -1)
def _op_absdiff(L, R): return np.abs(L - R)
def _op_mul(L, R): return L * R
def _op_gcd(L, R):
    # Vectorized gcd via reduce
    a, b = L.astype(np.int32), R.astype(np.int32)
    while np.any(b != 0):
        a, b = np.where(b != 0, b, a), np.where(b != 0, a % np.where(b != 0, b, 1), 0)
    return a

OP_FNS = {
    "add":     _op_add,
    "sub":     _op_sub,
    "rsub":    _op_rsub,
    "absdiff": _op_absdiff,
    "mul":     _op_mul,
    "gcd":     _op_gcd,
}
OP_CHARS = {  # display name per op_name
    "add": "addition", "sub": "subtraction", "rsub": "reverse subtraction",
    "absdiff": "absolute difference", "mul": "multiplication", "gcd": "gcd",
}


def parse_prompt(prompt: str):
    examples, query = [], None
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


def _solve_one_mode(content, sym2i, op_eqs_packed, query, gold, perms, mode):
    """Try to solve under a single mode ('standard' or 'reversed'). Return
    (mapping, q_op_name, q_result_int, answer_str) or None.
    """
    P = perms.shape[0]
    alive = np.ones(P, dtype=bool)
    for op_char, eqs in op_eqs_packed.items():
        any_op_ok = np.zeros(P, dtype=bool)
        for op_name, op_fn in OP_FNS.items():
            mask = np.ones(P, dtype=bool)
            for (l0, l1, r0, r1, ridx, rl) in eqs:
                if mode == "standard":
                    L = perms[:, l0].astype(np.int32) * 10 + perms[:, l1]
                    R = perms[:, r0].astype(np.int32) * 10 + perms[:, r1]
                else:  # reversed (alice / little_endian)
                    L = perms[:, l1].astype(np.int32) * 10 + perms[:, l0]
                    R = perms[:, r1].astype(np.int32) * 10 + perms[:, r0]
                v = op_fn(L, R)
                v = np.where(v >= 0, v, -1)
                exp = np.zeros(P, dtype=np.int32)
                if mode == "standard":
                    for sym_idx in ridx:
                        exp = exp * 10 + perms[:, sym_idx]
                else:
                    for sym_idx in ridx[::-1]:
                        exp = exp * 10 + perms[:, sym_idx]
                in_range = (v >= 0) & (v < 10**rl)
                mask &= in_range & (v == exp)
                if not mask.any():
                    break
            any_op_ok |= mask
        alive &= any_op_ok
        if not alive.any():
            return None

    survivor_idx = int(np.flatnonzero(alive)[0])
    perm = perms[survivor_idx]
    mapping = {c: int(perm[i]) for i, c in enumerate(content)}
    inv = {int(perm[i]): c for i, c in enumerate(content)}

    if mode == "standard":
        qL = mapping[query[0]] * 10 + mapping[query[1]]
        qR = mapping[query[3]] * 10 + mapping[query[4]]
    else:
        qL = mapping[query[1]] * 10 + mapping[query[0]]
        qR = mapping[query[4]] * 10 + mapping[query[3]]

    target_len = len(gold)
    if mode == "standard":
        gold_int = 0
        for c in gold:
            gold_int = gold_int * 10 + mapping[c]
    else:
        gold_int = 0
        for c in gold[::-1]:
            gold_int = gold_int * 10 + mapping[c]

    q_op_name = None
    q_result = None
    for op_name, op_fn in OP_FNS.items():
        v = op_fn(np.array([qL], dtype=np.int32), np.array([qR], dtype=np.int32))[0]
        if v >= 0 and v == gold_int and v < 10**target_len:
            q_op_name = op_name
            q_result = int(v)
            break
    if q_op_name is None:
        return None

    s = str(q_result).zfill(target_len)
    if mode == "standard":
        try:
            ans = "".join(inv[int(c)] for c in s)
        except KeyError:
            return None
    else:
        try:
            ans = "".join(inv[int(c)] for c in s[::-1])
        except KeyError:
            return None

    if ans != gold:
        return None
    return mapping, q_op_name, q_result, ans, qL, qR


def solve_fast(prompt: str, gold: str) -> dict | None:
    """Return solution dict or None.

    Tries standard mode then reversed (alice/little_endian) mode.
    """
    examples, query = parse_prompt(prompt)
    if not query or len(query) < 5:
        return None
    for lhs, rhs in examples:
        if len(rhs) > 1 and len(lhs) >= 5 and rhs[0] == lhs[2]:
            return None
    if len(gold) > 1 and gold[0] == query[2]:
        return None

    op_chars = {lhs[2] for lhs, _ in examples if len(lhs) >= 5} | {query[2]}
    all_chars = set()
    for lhs, rhs in examples:
        all_chars.update(lhs); all_chars.update(rhs)
    all_chars.update(query); all_chars.update(gold)
    content = sorted(all_chars - op_chars)
    n = len(content)
    if n > 10:
        return None
    sym2i = {c: i for i, c in enumerate(content)}

    op_eqs_packed: dict[str, list[tuple]] = {}
    for lhs, rhs in examples:
        if len(lhs) < 5: continue
        op_eqs_packed.setdefault(lhs[2], []).append((
            sym2i[lhs[0]], sym2i[lhs[1]],
            sym2i[lhs[3]], sym2i[lhs[4]],
            tuple(sym2i[c] for c in rhs),
            len(rhs),
        ))
    qo = query[2]
    op_eqs_packed.setdefault(qo, []).append((
        sym2i[query[0]], sym2i[query[1]],
        sym2i[query[3]], sym2i[query[4]],
        tuple(sym2i[c] for c in gold),
        len(gold),
    ))

    perms = _all_perms(n)
    for mode in ("standard", "reversed"):
        out = _solve_one_mode(content, sym2i, op_eqs_packed, query, gold, perms, mode)
        if out is None: continue
        mapping, q_op_name, q_result, ans, qL, qR = out
        return {
            "mode": mode,
            "mapping": mapping,
            "q_op": qo,
            "q_op_name": q_op_name,
            "qL": qL, "qR": qR, "q_result": q_result,
            "answer": ans,
        }
    return None


if __name__ == "__main__":
    # Smoke test on one puzzle from train.csv
    import csv, sys, time
    csv.field_size_limit(sys.maxsize)
    with open("data/raw/train.csv") as f:
        for r in csv.DictReader(f):
            p = r["prompt"]
            if "secret set of transformation rules is applied to equations" in p.lower():
                # cryptarithm or eq_num — distinguish
                exs, q = parse_prompt(p)
                lhs0 = exs[0][0] if exs else ""
                if exs and len(lhs0) == 5 and not (lhs0[0]+lhs0[1]+lhs0[3]+lhs0[4]).isdigit():
                    t0 = time.time()
                    sol = solve_fast(p, r["answer"])
                    dt = time.time() - t0
                    print(f"id={r['id']}  gold={r['answer']!r}  dt={dt:.3f}s  sol={'ok' if sol else 'none'}")
                    if sol:
                        print(f"  mapping: {sol['mapping']}")
                        print(f"  q_op_name: {sol['q_op_name']}")
                    break
