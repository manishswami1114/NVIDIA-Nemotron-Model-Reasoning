"""
manual_explore.py — interactive viewer for failed cipher-equation puzzles.

Browse the puzzles your 0.86 model failed on, one at a time, and try
to crack them manually. Shows:
  - The puzzle (examples + query)
  - The ground-truth answer
  - Whether our solver covered it
  - A scratchpad area for your guesses
  - Auto-test of common operations against the examples

Commands:
  n / next       — next puzzle
  p / prev       — previous puzzle
  s / show       — show full prompt
  t / test <expr>— test a Python expression with L, R variables
                    e.g.:  t L*R  /  t abs(L-R)+1  /  t int(str(L)[::-1])*R
  o / ops        — auto-check common operations against all examples
  f / filter <type>  — filter by 'plain' / 'enc' / 'unsolved' / 'all'
  g / goto <n>   — jump to puzzle index
  q / quit       — exit

Usage:
    python manual_explore.py \\
        --eval-csv ../../evaluation_results_086.csv \\
        --train-csv ../../data/raw/train.csv \\
        --solver-jsonl sequence_solved.jsonl \\
        --plain-jsonl plain_solved.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys

csv.field_size_limit(sys.maxsize)


# Operations to auto-test
def _safe(fn):
    def wrapped(L, R):
        try:
            r = fn(L, R)
            return r if r is not None else None
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
    return wrapped

def _rev(n): return int(str(n)[::-1].lstrip('0') or '0')

COMMON_OPS = {
    "L + R":              _safe(lambda L, R: L + R),
    "L - R":              _safe(lambda L, R: L - R),
    "R - L":              _safe(lambda L, R: R - L),
    "|L - R|":            _safe(lambda L, R: abs(L - R)),
    "L * R":              _safe(lambda L, R: L * R),
    "L * R + 1":          _safe(lambda L, R: L * R + 1),
    "L * R - 1":          _safe(lambda L, R: L * R - 1),
    "L + R + 1":          _safe(lambda L, R: L + R + 1),
    "L + R - 1":          _safe(lambda L, R: L + R - 1),
    "L mod R":            _safe(lambda L, R: L % R),
    "R mod L":            _safe(lambda L, R: R % L),
    "gcd(L, R)":          _safe(lambda L, R: math.gcd(L, R)),
    "lcm(L, R)":          _safe(lambda L, R: math.lcm(L, R)),
    "concat(L, R)":       _safe(lambda L, R: int(f"{L:02d}{R:02d}")),
    "concat(R, L)":       _safe(lambda L, R: int(f"{R:02d}{L:02d}")),
    "rev(L) + R":         _safe(lambda L, R: _rev(L) + R),
    "L + rev(R)":         _safe(lambda L, R: L + _rev(R)),
    "rev(L) - R":         _safe(lambda L, R: _rev(L) - R),
    "L - rev(R)":         _safe(lambda L, R: L - _rev(R)),
    "rev(L) * R":         _safe(lambda L, R: _rev(L) * R),
    "L * rev(R)":         _safe(lambda L, R: L * _rev(R)),
    "rev(L) + rev(R)":    _safe(lambda L, R: _rev(L) + _rev(R)),
    "rev(L + R)":         _safe(lambda L, R: _rev(L + R)),
    "rev(L * R)":         _safe(lambda L, R: _rev(L * R)),
    "rev(|L - R|)":       _safe(lambda L, R: _rev(abs(L - R))),
    "L² + R":             _safe(lambda L, R: L * L + R),
    "L² - R":             _safe(lambda L, R: L * L - R),
    "L² + R²":            _safe(lambda L, R: L * L + R * R),
    "L² - R²":            _safe(lambda L, R: L * L - R * R),
    "sum digits(L, R)":   _safe(lambda L, R: sum(int(c) for c in f"{L:02d}{R:02d}")),
    "prod digits(L, R)":  _safe(lambda L, R: int(str(L)[0])*int(str(L)[1])*int(str(R)[0])*int(str(R)[1])),
}


def parse_examples(prompt):
    """Pull (L, op, R, C) from each example line."""
    examples = []
    query = None
    LINE = re.compile(r"^(\S+)\s*=\s*(\S+)\s*$")
    QUERY = re.compile(r"determine the result for:\s*(\S+)\s*$", re.IGNORECASE)
    for line in prompt.split("\n"):
        line = line.rstrip()
        m = LINE.match(line)
        if m and len(m.group(1)) == 5 and "determine" not in line.lower():
            lhs = m.group(1)
            examples.append({"lhs": lhs, "rhs": m.group(2)})
            continue
        m = QUERY.search(line)
        if m and len(m.group(1)) == 5:
            query = m.group(1)
    return examples, query


def fmt_result_candidates(val, target_C):
    """List of formats that match target_C, useful as hints."""
    if val is None: return []
    candidates = []
    fmts = {
        "raw":       str(val),
        "rev":       (str(val)[::-1] if val >= 0 else "-"+str(-val)[::-1]),
        "abs":       str(abs(val)),
        "abs_rev":   str(abs(val))[::-1],
        "pad2":      f"{val:02d}" if val >= 0 else None,
        "pad3":      f"{val:03d}" if val >= 0 else None,
        "pad4":      f"{val:04d}" if val >= 0 else None,
    }
    for name, s in fmts.items():
        if s is not None and s == target_C:
            candidates.append(name)
    return candidates


def auto_check_ops(examples_parsed):
    """For each known operation, check if any consistent format makes ALL
    examples match. Returns list of (op_name, fmt_name) candidates."""
    matches = []
    for op_name, op_fn in COMMON_OPS.items():
        # Collect format hits for each example
        ex_fmt_sets = []
        all_have_match = True
        for L, R, C in examples_parsed:
            val = op_fn(L, R)
            fmts = fmt_result_candidates(val, C)
            if not fmts:
                all_have_match = False
                break
            ex_fmt_sets.append(set(fmts))
        if all_have_match and ex_fmt_sets:
            common = set.intersection(*ex_fmt_sets)
            for fmt in common:
                matches.append((op_name, fmt))
    return matches


def show_puzzle(puzzles, idx):
    p = puzzles[idx]
    print(f"\n{'='*70}")
    print(f"[{idx+1}/{len(puzzles)}] id={p['id']}  type={p.get('type','?')}  "
          f"in_solver={p.get('in_solver', False)}")
    print(f"{'='*70}")
    print(f"GROUND TRUTH: {p['answer']!r}")
    print(f"MODEL PRED  : {p.get('prediction', '?')!r}\n")
    print("Examples:")
    examples, query = parse_examples(p["prompt"])
    for i, e in enumerate(examples, 1):
        # Try to identify operator
        lhs = e["lhs"]
        if len(lhs) == 5:
            print(f"  EX{i}: {lhs[0:2]} [{lhs[2]}] {lhs[3:5]}  =  {e['rhs']}")
        else:
            print(f"  EX{i}: {lhs}  =  {e['rhs']}")
    if query:
        if len(query) == 5:
            print(f"  Q:   {query[0:2]} [{query[2]}] {query[3:5]}  =  ?")
        else:
            print(f"  Q:   {query}  =  ?")


def cmd_test(puzzles, idx, expr):
    p = puzzles[idx]
    examples, query = parse_examples(p["prompt"])
    if not examples:
        print("No parseable examples."); return
    print(f"\nTesting:  {expr}")
    print(f"  (L, R variables refer to operands of each example/query)\n")
    for i, e in enumerate(examples, 1):
        if len(e["lhs"]) != 5: continue
        try:
            L = int(e["lhs"][:2])
            R = int(e["lhs"][3:5])
        except ValueError:
            print(f"  EX{i}: operands not pure digits, skipping"); continue
        try:
            val = eval(expr, {"L": L, "R": R, "abs": abs, "rev": _rev,
                              "str": str, "int": int, "math": math})
            fmts = fmt_result_candidates(val, e["rhs"])
            tag = f"  MATCH via [{', '.join(fmts)}]" if fmts else ""
            print(f"  EX{i}: L={L}, R={R} → {val}  (target {e['rhs']!r}){tag}")
        except Exception as ex:
            print(f"  EX{i}: error: {ex}")
    # Apply to query
    if query and len(query) == 5:
        try:
            L = int(query[:2]); R = int(query[3:5])
            val = eval(expr, {"L": L, "R": R, "abs": abs, "rev": _rev,
                              "str": str, "int": int, "math": math})
            fmts = fmt_result_candidates(val, p["answer"])
            tag = f"  MATCH via [{', '.join(fmts)}]" if fmts else ""
            print(f"  Q:   L={L}, R={R} → {val}  (GT {p['answer']!r}){tag}")
        except Exception as ex:
            print(f"  Q:   error: {ex}")


def cmd_ops(puzzles, idx):
    p = puzzles[idx]
    examples, query = parse_examples(p["prompt"])
    # Build parsed example list
    parsed = []
    for e in examples:
        if len(e["lhs"]) == 5:
            try:
                L = int(e["lhs"][:2]); R = int(e["lhs"][3:5])
                parsed.append((L, R, e["rhs"]))
            except ValueError:
                pass
    if not parsed:
        print("Examples can't be parsed as plain digits."); return
    matches = auto_check_ops(parsed)
    if not matches:
        print("No common operation matches ALL examples. This puzzle uses something more exotic.")
        print("Try:  t <python expression>")
        return
    print(f"\n{len(matches)} candidate operation(s) match ALL examples:\n")
    for op_name, fmt_name in matches:
        # Apply to query too
        if query and len(query) == 5:
            try:
                qL = int(query[:2]); qR = int(query[3:5])
                qval = COMMON_OPS[op_name](qL, qR)
                qfmts = fmt_result_candidates(qval, p["answer"])
                ans_match = fmt_name in qfmts
                marker = " ← also matches GT answer ✓" if ans_match else f"  (gives {qval} via {fmt_name})"
                print(f"  {op_name:<28}  fmt={fmt_name:<10}{marker}")
            except Exception:
                print(f"  {op_name:<28}  fmt={fmt_name:<10}  (query eval failed)")
        else:
            print(f"  {op_name:<28}  fmt={fmt_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv",
                    default="/Users/manishswami/developer/NVIDIA-Nemotron Model/evaluation_results_086.csv")
    ap.add_argument("--train-csv",
                    default="/Users/manishswami/developer/NVIDIA-Nemotron Model/data/raw/train.csv")
    ap.add_argument("--solver-jsonl",
                    default="sequence_solved.jsonl")
    ap.add_argument("--plain-jsonl",
                    default="plain_solved.jsonl")
    args = ap.parse_args()

    # Load eval results
    with open(args.eval_csv) as f:
        eval_rows = list(csv.DictReader(f))
    # Load train prompts
    with open(args.train_csv) as f:
        train_prompts = {r["id"]: r["prompt"] for r in csv.DictReader(f)}
    # Load solver coverage
    enc_solved = set()
    try:
        with open(args.solver_jsonl) as f:
            for line in f: enc_solved.add(json.loads(line)["id"])
    except FileNotFoundError: pass
    plain_solved = set()
    try:
        with open(args.plain_jsonl) as f:
            for line in f: plain_solved.add(json.loads(line)["id"])
    except FileNotFoundError: pass

    # Build puzzle list: failed cipher-equation only
    puzzles = []
    for r in eval_rows:
        if r["is_correct"].lower() in ("true","1","yes"): continue
        prompt = train_prompts.get(r["id"], "")
        if "secret set of transformation rules" not in prompt: continue
        # Classify: plain-digit vs fully-encrypted
        is_plain = False
        for line in prompt.split("\n"):
            line = line.strip()
            if "=" in line and "determine" not in line.lower() and not line.startswith("In Alice"):
                lhs = line.split("=")[0].strip()
                if len(lhs) == 5 and re.search(r"\d", lhs):
                    is_plain = True
                break
        in_solver = r["id"] in (plain_solved if is_plain else enc_solved)
        puzzles.append({
            "id":          r["id"],
            "prompt":      prompt,
            "answer":      r["ground_truth"],
            "prediction":  r["prediction"],
            "type":        "plain" if is_plain else "enc",
            "in_solver":   in_solver,
        })

    print(f"Loaded {len(puzzles)} failed cipher-equation puzzles.")
    print(f"  Plain-digit:    {sum(1 for p in puzzles if p['type']=='plain'):>4}  "
          f"(solver covers {sum(1 for p in puzzles if p['type']=='plain' and p['in_solver']):>4})")
    print(f"  Fully-encrypt:  {sum(1 for p in puzzles if p['type']=='enc'):>4}  "
          f"(solver covers {sum(1 for p in puzzles if p['type']=='enc' and p['in_solver']):>4})")

    filtered = puzzles
    idx = 0
    show_puzzle(filtered, idx)

    print("\nCommands: n/p/s/o/t <expr>/f <filter>/g <n>/q  (h for help)")
    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not cmd: continue
        parts = cmd.split(maxsplit=1)
        c = parts[0].lower()

        if c in ("q", "quit", "exit"):
            break
        elif c in ("n", "next"):
            if idx < len(filtered) - 1: idx += 1
            show_puzzle(filtered, idx)
        elif c in ("p", "prev"):
            if idx > 0: idx -= 1
            show_puzzle(filtered, idx)
        elif c in ("s", "show"):
            print("\n" + filtered[idx]["prompt"])
        elif c in ("o", "ops"):
            cmd_ops(filtered, idx)
        elif c in ("t", "test"):
            if len(parts) < 2: print("Usage: t <expression>"); continue
            cmd_test(filtered, idx, parts[1])
        elif c in ("f", "filter"):
            arg = parts[1].lower() if len(parts) > 1 else "all"
            if arg == "plain":
                filtered = [p for p in puzzles if p["type"] == "plain"]
            elif arg == "enc":
                filtered = [p for p in puzzles if p["type"] == "enc"]
            elif arg == "unsolved":
                filtered = [p for p in puzzles if not p["in_solver"]]
            elif arg == "solved":
                filtered = [p for p in puzzles if p["in_solver"]]
            else:
                filtered = puzzles
            idx = 0
            print(f"Filtered to {len(filtered)} puzzles.")
            if filtered: show_puzzle(filtered, idx)
        elif c in ("g", "goto"):
            if len(parts) < 2: print("Usage: g <index>"); continue
            try:
                idx = max(0, min(int(parts[1]) - 1, len(filtered) - 1))
                show_puzzle(filtered, idx)
            except ValueError:
                print("Index must be an integer")
        elif c in ("h", "help", "?"):
            print(__doc__)
        else:
            print("Unknown command. Try: n/p/s/o/t <expr>/f <filter>/g <n>/q/h")


if __name__ == "__main__":
    main()
