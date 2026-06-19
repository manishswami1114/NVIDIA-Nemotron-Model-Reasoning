"""
generate_cryptarithm_cot_v2_kaggle.py
Generates real trace-guided CoTs for cryptarithm puzzles.
Designed to run on Kaggle (saves progress iteratively, verbose output).

Usage: python generate_cryptarithm_cot_v2_kaggle.py
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path
from collections import defaultdict
from itertools import product as iprod

# If running on Kaggle, adjust the ROOT path to point to your dataset directory
# e.g., ROOT = Path('/kaggle/working') or ROOT = Path('/kaggle/input/your-dataset')
ROOT = Path(__file__).parent.parent
TONG = ROOT / "tong_reasoners"
sys.path.insert(0, str(TONG))
# os.chdir(TONG) # Removed so it doesn't break relative paths

from reasoners.store_types import Problem
from reasoners.cryptarithm import reasoning_cryptarithm

try:
    import z3
except ImportError:
    print("ERROR: pip install z3-solver"); sys.exit(1)

EVAL_SUFFIX = "\nPlease put your final answer inside `\\boxed{}`."
OUT_DIR = ROOT / "data" / "processed" / "baselined_plus_data" / "baseline_plus_sequence"

LINEAR_OPS = ["addition", "subtraction", "abs_diff", "add+1", "add-1"]
MUL_OPS = ["multiplication", "multiply+1", "multiply-1"]
ALL_OPS = LINEAR_OPS + MUL_OPS

LIKELY_PAIRS = [
    ("multiply+1", "abs_diff"), ("multiply+1", "subtraction"),
    ("multiply+1", "addition"), ("multiplication", "abs_diff"),
    ("multiplication", "addition"), ("multiplication", "subtraction"),
    ("addition", "abs_diff"), ("addition", "subtraction"),
    ("abs_diff", "multiply+1"), ("subtraction", "multiply+1"),
    ("addition", "multiply+1"), ("abs_diff", "multiplication"),
    ("add+1", "abs_diff"), ("add-1", "abs_diff"),
    ("multiply-1", "abs_diff"), ("multiply-1", "subtraction"),
]

def _to_z3(s, zvars):
    val = z3.IntVal(0)
    for i, c in enumerate(reversed(s)):
        if c in zvars: val = val + zvars[c] * (10 ** i)
    return val

def _add_op(solver, a, b, o, op):
    if op == "addition": solver.add(a+b == o)
    elif op == "subtraction": solver.add(a-b == o)
    elif op == "abs_diff": solver.add(z3.If(a>=b, a-b, b-a) == o)
    elif op == "multiplication": solver.add(a*b == o)
    elif op == "multiply+1": solver.add(a*b+1 == o)
    elif op == "multiply-1": solver.add(a*b-1 == o)
    elif op == "add+1": solver.add(a+b+1 == o)
    elif op == "add-1": solver.add(a+b-1 == o)


def _python_compute(a_n, b_n, op):
    """Mirror of _add_op in pure Python for end-to-end verification."""
    if op == "addition": return a_n + b_n
    if op == "subtraction": return a_n - b_n
    if op == "abs_diff": return abs(a_n - b_n)
    if op == "multiplication": return a_n * b_n
    if op == "multiply+1": return a_n * b_n + 1
    if op == "multiply-1": return a_n * b_n - 1
    if op == "add+1": return a_n + b_n + 1
    if op == "add-1": return a_n + b_n - 1
    return None


def _verify_solution(mapping, examples_raw, question, answer, si,
                     concat_ops, arith_map, rev_ops, rev_res):
    """End-to-end sanity check: re-execute the discovered (mapping, ops, modes)
    against every example AND the query. Returns True only when every encoded
    prediction matches the ground-truth string byte-for-byte.

    Catches three failure modes that Z3's `check()=sat` alone does NOT catch:
      (a) non-bijective mapping (multiple symbols → same digit)
      (b) mapping with negative or wrong-length intermediate results
      (c) edge cases where Z3 satisfies modular arithmetic but produces a
          result string with leading zeros / different length than target
    """
    # Strict bijection: each symbol gets its own digit
    if len(set(mapping.values())) != len(mapping):
        return False

    inv_map = {v: k for k, v in mapping.items()}

    def decode(s):
        rs = s[::-1] if rev_ops else s
        try:
            return int("".join(str(mapping[c]) for c in rs))
        except (KeyError, ValueError):
            return None

    def encode(n, length):
        if n < 0:
            return None
        s = str(n)
        if len(s) > length:
            return None        # number too large for target — genuine reject
        s = s.zfill(length)    # FIX: pad leading zeros so length matches Z3's expectation
        if rev_res:
            s = s[::-1]
        try:
            return "".join(inv_map[int(c)] for c in s)
        except KeyError:
            return None

    for inp, out in examples_raw:
        if len(inp) <= si:
            return False
        op_ch = inp[si]
        la, lb = inp[:si], inp[si+1:]
        if op_ch in concat_ops:
            # Concat ops: re-derive at the string level
            la_r = la[::-1] if rev_ops else la
            lb_r = lb[::-1] if rev_ops else lb
            pred = (la_r + lb_r) if concat_ops[op_ch] == "fwd_concat" else (lb_r + la_r)
            if pred != out:
                return False
            continue
        op_name = arith_map.get(op_ch)
        if op_name is None:
            return False
        L, R = decode(la), decode(lb)
        if L is None or R is None:
            return False
        res = _python_compute(L, R, op_name)
        if res is None:
            return False
        pred_out = encode(res, len(out))
        if pred_out != out:
            return False

    # Verify query produces the GT answer
    q_op = question[si]
    qa, qb = question[:si], question[si+1:]
    if q_op in concat_ops:
        qa_r = qa[::-1] if rev_ops else qa
        qb_r = qb[::-1] if rev_ops else qb
        pred = (qa_r + qb_r) if concat_ops[q_op] == "fwd_concat" else (qb_r + qa_r)
        return pred == answer
    op_name = arith_map.get(q_op)
    if op_name is None:
        return False
    L, R = decode(qa), decode(qb)
    if L is None or R is None:
        return False
    res = _python_compute(L, R, op_name)
    if res is None:
        return False
    return encode(res, len(answer)) == answer


def _try_z3(examples_raw, question, answer, si, symbols, ops_seen,
            concat_ops, arith_map, rev_ops, rev_res, distinct, timeout_ms=500):
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    zvars = {s: z3.Int(f"v{i}") for i, s in enumerate(symbols)}
    for v in zvars.values(): solver.add(v >= 0, v < 10)
    if distinct and len(symbols) <= 10:
        solver.add(z3.Distinct(*list(zvars.values())))

    for inp, out in examples_raw:
        if len(inp) <= si: return None
        op_ch = inp[si]
        if op_ch in concat_ops: continue
        if op_ch not in arith_map: return None
        la, lb, o_s = inp[:si], inp[si+1:], out
        if rev_ops: la, lb = la[::-1], lb[::-1]
        if rev_res: o_s = o_s[::-1]
        _add_op(solver, _to_z3(la, zvars), _to_z3(lb, zvars), _to_z3(o_s, zvars), arith_map[op_ch])

    q_op = question[si]
    qa_s, qb_s, qo_s = question[:si], question[si+1:], answer
    if q_op in concat_ops:
        if rev_ops: qa_r, qb_r = qa_s[::-1], qb_s[::-1]
        else: qa_r, qb_r = qa_s, qb_s
        pred = (qa_r + qb_r) if concat_ops[q_op] == "fwd_concat" else (qb_r + qa_r)
        if pred != answer: return None
    elif q_op in arith_map:
        if rev_ops: qa_s, qb_s = qa_s[::-1], qb_s[::-1]
        if rev_res: qo_s = qo_s[::-1]
        _add_op(solver, _to_z3(qa_s, zvars), _to_z3(qb_s, zvars), _to_z3(qo_s, zvars), arith_map[q_op])
    else:
        return None

    if solver.check() == z3.sat:
        m = solver.model()
        mapping = {s: m[zvars[s]].as_long() for s in symbols}
        # STRICT bijection + end-to-end verification (catches the bug where
        # multiple symbols mapped to the same digit, e.g. 6 symbols → 0).
        if _verify_solution(mapping, examples_raw, question, answer, si,
                            concat_ops, arith_map, rev_ops, rev_res):
            return mapping
    return None

def _try_mul_subprocess(examples_raw, question, answer, si, symbols, ops_seen,
                        concat_ops, sorted_arith, combos_to_test, timeout_secs=15):
    """Run Z3 with multiplication in a subprocess, testing ALL combos inside the child."""
    args_json = json.dumps({
        "examples_raw": examples_raw, "question": question, "answer": answer,
        "si": si, "symbols": symbols, "ops_seen": list(ops_seen),
        "concat_ops": concat_ops, "sorted_arith": sorted_arith,
        "combos_to_test": combos_to_test
    })
    
    script = f'''
import json, sys, z3

args = json.loads(sys.stdin.read())
e = args["examples_raw"]; q = args["question"]; a = args["answer"]
si = args["si"]; syms = args["symbols"]; co = args["concat_ops"]
sa = args["sorted_arith"]; combos = args["combos_to_test"]

def to_z3(s, zv):
    val = z3.IntVal(0)
    for i, c in enumerate(reversed(s)):
        if c in zv: val = val + zv[c] * (10**i)
    return val

def add_op(slv, a, b, o, op):
    if op=="addition": slv.add(a+b==o)
    elif op=="subtraction": slv.add(a-b==o)
    elif op=="abs_diff": slv.add(z3.If(a>=b,a-b,b-a)==o)
    elif op=="multiplication": slv.add(a*b==o)
    elif op=="multiply+1": slv.add(a*b+1==o)
    elif op=="multiply-1": slv.add(a*b-1==o)
    elif op=="add+1": slv.add(a+b+1==o)
    elif op=="add-1": slv.add(a+b-1==o)

qop = q[si]; qa, qb, qo = q[:si], q[si+1:], a

# Test all combos in one subprocess
for c_data in combos:
    combo, ro, rr, di = c_data
    am = dict(zip(sa, combo))
    
    slv = z3.Solver(); slv.set("timeout", 2000)
    zv = {{s: z3.Int(f"v{{i}}") for i, s in enumerate(syms)}}
    for v in zv.values(): slv.add(v>=0, v<10)
    if di and len(syms)<=10: slv.add(z3.Distinct(*list(zv.values())))

    ok = True
    for inp, out in e:
        if len(inp)<=si: ok=False; break
        oc = inp[si]
        if oc in co: continue
        if oc not in am: ok=False; break
        la, lb, os_ = inp[:si], inp[si+1:], out
        if ro: la, lb = la[::-1], lb[::-1]
        if rr: os_ = os_[::-1]
        add_op(slv, to_z3(la, zv), to_z3(lb, zv), to_z3(os_, zv), am[oc])
    if not ok: continue

    if qop in co:
        if ro: qar, qbr = qa[::-1], qb[::-1]
        else: qar, qbr = qa, qb
        pred = (qar+qbr) if co[qop]=="fwd_concat" else (qbr+qar)
        if pred!=a: continue
    elif qop in am:
        qa_s, qb_s, qo_s = qa, qb, qo
        if ro: qa_s, qb_s = qa_s[::-1], qb_s[::-1]
        if rr: qo_s = qo_s[::-1]
        add_op(slv, to_z3(qa_s, zv), to_z3(qb_s, zv), to_z3(qo_s, zv), am[qop])
    else: continue

    if slv.check()==z3.sat:
        m = slv.model()
        mp = {{s: m[zv[s]].as_long() for s in syms}}
        # Require STRICT bijection (catches degenerate "all → 0" mappings)
        if len(set(mp.values())) != len(mp): continue
        # End-to-end verify: re-execute the candidate and check it matches
        # every example AND the query answer byte-for-byte.
        def _pcomp(a_n, b_n, op):
            if op=="addition": return a_n+b_n
            if op=="subtraction": return a_n-b_n
            if op=="abs_diff": return abs(a_n-b_n)
            if op=="multiplication": return a_n*b_n
            if op=="multiply+1": return a_n*b_n+1
            if op=="multiply-1": return a_n*b_n-1
            if op=="add+1": return a_n+b_n+1
            if op=="add-1": return a_n+b_n-1
            return None
        inv = {{v: k for k, v in mp.items()}}
        def _dec(s, mo):
            rs = s[::-1] if mo else s
            try: return int("".join(str(mp[c]) for c in rs))
            except: return None
        def _enc(n, length, mr):
            if n < 0: return None
            ss = str(n)
            if len(ss) > length: return None   # too large — genuine reject
            ss = ss.zfill(length)              # FIX: zero-pad to match length
            if mr: ss = ss[::-1]
            try: return "".join(inv[int(c)] for c in ss)
            except: return None
        # Verify each example
        bad = False
        for inp, out in e:
            if len(inp) <= si: bad=True; break
            oc = inp[si]; la, lb = inp[:si], inp[si+1:]
            if oc in co:
                la_r = la[::-1] if ro else la
                lb_r = lb[::-1] if ro else lb
                pred = (la_r+lb_r) if co[oc]=="fwd_concat" else (lb_r+la_r)
                if pred != out: bad=True; break
                continue
            opn = am.get(oc)
            if opn is None: bad=True; break
            L = _dec(la, ro); R = _dec(lb, ro)
            if L is None or R is None: bad=True; break
            r_val = _pcomp(L, R, opn)
            if r_val is None: bad=True; break
            if _enc(r_val, len(out), rr) != out: bad=True; break
        if bad: continue
        # Verify query
        if qop in co:
            qa_r = qa[::-1] if ro else qa
            qb_r = qb[::-1] if ro else qb
            pred_q = (qa_r+qb_r) if co[qop]=="fwd_concat" else (qb_r+qa_r)
            if pred_q != a: continue
        else:
            opn = am.get(qop)
            if opn is None: continue
            qL = _dec(qa, ro); qR = _dec(qb, ro)
            if qL is None or qR is None: continue
            qres = _pcomp(qL, qR, opn)
            if qres is None: continue
            if _enc(qres, len(a), rr) != a: continue
        result = {{"mapping": mp, "combo": combo, "rev_ops": ro, "rev_res": rr}}
        print(json.dumps(result)); sys.exit(0)

print("null")
'''
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=args_json, capture_output=True, text=True, timeout=timeout_secs
        )
        if proc.returncode == 0 and proc.stdout.strip() != "null":
            return json.loads(proc.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None

def solve_problem(problem: Problem):
    q, a = str(problem.question), str(problem.answer)
    t0 = time.time()

    for si in [2, 1, 3]:
        if len(q) <= si: continue

        ops_seen = set()
        by_op = defaultdict(list)
        for ex in problem.examples:
            inp = str(ex.input_value)
            if len(inp) <= si: continue
            op_ch = inp[si]
            ops_seen.add(op_ch)
            by_op[op_ch].append((inp[:si], inp[si+1:], str(ex.output_value)))
        if not ops_seen: continue

        all_text = "".join(str(e.input_value)+str(e.output_value) for e in problem.examples) + q + a
        symbols = sorted(set(re.findall(r"[^\s0-9a-zA-Z]", all_text)) - ops_seen)
        if not symbols or len(symbols) > 12: continue

        concat_ops = {}
        arith_ops = []
        for op_ch, group in by_op.items():
            if all(la+lb == out for la, lb, out in group):
                concat_ops[op_ch] = "fwd_concat"
            elif all(lb+la == out for la, lb, out in group):
                concat_ops[op_ch] = "rev_concat"
            else:
                arith_ops.append(op_ch)

        if not arith_ops:
            q_op = q[si]
            if q_op in concat_ops:
                qa, qb = q[:si], q[si+1:]
                pred = (qa+qb) if concat_ops[q_op] == "fwd_concat" else (qb+qa)
                if pred == a:
                    return {"mapping": {}, "op_configs": concat_ops, "rev_ops": False,
                            "rev_res": False, "split_idx": si, "ops_seen": ops_seen, "is_concat": True}
            continue

        sorted_arith = sorted(arith_ops)
        examples_raw = [(str(ex.input_value), str(ex.output_value)) for ex in problem.examples]

        # Exhaustive search for Kaggle!
        if len(sorted_arith) <= 3:
            combos = list(iprod(ALL_OPS, repeat=len(sorted_arith)))
        else:
            combos = [(op,)*len(sorted_arith) for op in ALL_OPS]

        # Phase 1: Linear ops
        for rev_ops in [False, True]:
            for rev_res in [False, True]:
                for distinct in [True]:   # FIX: distinct=False creates degenerate bijections
                    for combo in combos:
                        arith_map = dict(zip(sorted_arith, combo))
                        if any(op in MUL_OPS for op in arith_map.values()):
                            continue 
                        q_op = q[si]
                        if q_op not in concat_ops and q_op not in arith_map:
                            continue
                        mapping = _try_z3(examples_raw, q, a, si, symbols, ops_seen,
                                          concat_ops, arith_map, rev_ops, rev_res, distinct, 2000)
                        if mapping:
                            full = dict(concat_ops); full.update(arith_map)
                            return {"mapping": mapping, "op_configs": full, "rev_ops": rev_ops,
                                    "rev_res": rev_res, "split_idx": si, "ops_seen": ops_seen, "is_concat": False}

        # Phase 2: Multiplication ops (batched subprocess)
        mul_combos_to_test = []
        for rev_ops in [False, True]:
            for rev_res in [False, True]:
                for distinct in [True]:   # FIX: distinct=False creates degenerate bijections
                    for combo in combos:
                        arith_map = dict(zip(sorted_arith, combo))
                        if not any(op in MUL_OPS for op in arith_map.values()):
                            continue 
                        q_op = q[si]
                        if q_op not in concat_ops and q_op not in arith_map:
                            continue
                        mul_combos_to_test.append((combo, rev_ops, rev_res, distinct))
                        
        if mul_combos_to_test:
            result = _try_mul_subprocess(examples_raw, q, a, si, symbols, ops_seen,
                                           concat_ops, sorted_arith, mul_combos_to_test, timeout_secs=120)
            if result:
                mapping = result["mapping"]
                combo = result["combo"]
                ro = result["rev_ops"]
                rr = result["rev_res"]
                arith_map = dict(zip(sorted_arith, combo))
                full = dict(concat_ops); full.update(arith_map)
                return {"mapping": mapping, "op_configs": full, "rev_ops": ro,
                        "rev_res": rr, "split_idx": si, "ops_seen": ops_seen, "is_concat": False}
    return None

def _compute(a_n, b_n, op_n):
    if "multiply+1" in op_n: return a_n * b_n + 1
    elif "multiply-1" in op_n: return a_n * b_n - 1
    elif "multiplication" in op_n: return a_n * b_n
    elif "add+1" in op_n: return a_n + b_n + 1
    elif "add-1" in op_n: return a_n + b_n - 1
    elif "addition" in op_n: return a_n + b_n
    elif "abs_diff" in op_n: return abs(a_n - b_n)
    elif "subtraction" in op_n: return a_n - b_n
    return "?"

def generate_cot(problem, sol):
    mapping = sol["mapping"]; si = sol["split_idx"]
    q, a = str(problem.question), str(problem.answer)
    is_concat = sol.get("is_concat", False)
    L = [
        "We need to crack a cipher-digit puzzle. Each character is an encrypted "
        "digit, and the operator symbols are also encrypted.",
        "I will put my final answer inside \\boxed{}.", ""
    ]
    L.append("Examples (encrypted):")
    for i, ex in enumerate(problem.examples):
        L.append(f"  EX{i+1}: 【{ex.input_value}】 = 【{ex.output_value}】")
    L.append(f"Query: 【{q}】 = ?"); L.append("")
    L.append(f"Each example has form 【L op R】 = 【C】 with operator at position {si+1}.")
    L.append(f"Operator symbols: {', '.join(repr(o) for o in sorted(sol['ops_seen']))}.")
    if sol["rev_ops"]: L.append("Reading mode: reversed operands.")
    if sol["rev_res"]: L.append("Reading mode: reversed result.")
    L.append("")
    if not is_concat and mapping:
        sm = sorted(mapping.items(), key=lambda x: x[1])
        L.append("After testing assignments, the symbol→digit mapping is:")
        L.append("  " + "    ".join(f"{k}={v}" for k, v in sm)); L.append("")
    L.append("Operator-symbol → operation:")
    for oc, on in sorted(sol["op_configs"].items()):
        L.append(f"  '{oc}' → {on}")
    L.append("")
    def _num(s): return int("".join(str(mapping.get(c, 0)) for c in s))
    L.append("Verification:")
    for i, ex in enumerate(problem.examples):
        inp, out = str(ex.input_value), str(ex.output_value)
        a_s, b_s = inp[:si], inp[si+1:]
        op_n = sol["op_configs"].get(inp[si], "?")
        if "concat" in op_n:
            L.append(f"  EX{i+1}: {a_s} {op_n} {b_s} → {out} ✓")
        else:
            ad = a_s[::-1] if sol["rev_ops"] else a_s
            bd = b_s[::-1] if sol["rev_ops"] else b_s
            res = _compute(_num(ad), _num(bd), op_n)
            L.append(f"  EX{i+1}: L={_num(ad)}, R={_num(bd)}, op={op_n}")
            L.append(f"        {op_n}({_num(ad)}, {_num(bd)}) = {res} → encode → {out} ✓")
    L.append("")
    qa, qb = q[:si], q[si+1:]
    qon = sol["op_configs"].get(q[si], "?")
    L.append("Apply to query:")
    L.append(f"  Query 【{q}】")
    if "concat" in qon:
        L.append(f"  Operation = {qon}")
    elif mapping:
        qad = qa[::-1] if sol["rev_ops"] else qa
        qbd = qb[::-1] if sol["rev_ops"] else qb
        res = _compute(_num(qad), _num(qbd), qon)
        L.append(f"  L = {_num(qad)}, R = {_num(qbd)}")
        L.append(f"  Operation = {qon}, Numeric result = {res}")
    L.append(f"  Encode result → {a}"); L.append("")
    L.append(f"\\boxed{{{a}}}")
    return "\n".join(L)

def try_tong(problem):
    try:
        r = reasoning_cryptarithm(problem)
        if r:
            m = re.findall(r"\\boxed\{([^}]*)\}", r)
            if m and m[-1].strip() == str(problem.answer).strip(): return r
    except: pass
    return None

def main():
    t0 = time.time()
    os.chdir(TONG)
    meta = {}
    problems_file = TONG / "problems.jsonl"
    print(f"Loading problems from {problems_file}")
    with problems_file.open() as f:
        for line in f: r = json.loads(line); meta[r["id"]] = r

    for target in ["cryptarithm_deduce", "cryptarithm_guess"]:
        s_z3, s_concat, s_fail = 0, 0, 0
        pids = [p for p, m in meta.items() if m["category"] == target]
        print(f"\n{'='*60}\n{target}: {len(pids)} problems\n{'='*60}", flush=True)

        out = OUT_DIR / f"train_cot_{target}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        
        # Append-mode writing so we don't lose progress if it crashes
        with out.open("w") as f: # Overwrite initially
            pass

        for idx, pid in enumerate(pids):
            t_prob_start = time.time()
            problem = Problem.load_from_json(pid)
            sol = solve_problem(problem)
            if sol:
                cot = generate_cot(problem, sol)
                s_z3 += 1
            else:
                cot = try_tong(problem)
                if cot: s_concat += 1
                else: s_fail += 1
            
            elapsed = time.time() - t_prob_start

            if cot:
                user = problem.prompt + EVAL_SUFFIX
                asst = f"<think>\n{cot.strip()}\n</think>\n\\boxed{{{problem.answer}}}"
                if len(asst)/4 <= 7200:
                    record = {"id": pid, "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": asst},
                    ]}
                    # Append immediately
                    with out.open("a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            # Print progress for every single problem so you can monitor on Kaggle
            status_str = "Z3" if sol else ("Concat" if cot else "FAILED")
            print(f"[{idx+1}/{len(pids)}] {pid} | {status_str} | {elapsed:.1f}s | Stats: z3={s_z3} concat={s_concat} fail={s_fail}", flush=True)

        print(f"\nDONE {target}: Z3={s_z3} Concat={s_concat} Failed={s_fail}", flush=True)

    eq_sym = OUT_DIR / "train_cot_equation_symbolic.jsonl"
    if eq_sym.exists(): eq_sym.unlink(); print(f"\nDeleted: {eq_sym}")
    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
