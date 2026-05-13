"""gen_numeral.py — Donald-format Roman numeral generator (bidirectional).

Trains both directions 50/50: int→Roman AND Roman→int.
Forward:  DECOMPOSE → CAT (incremental) → VER (re-parse) → ANS
Reverse:  PARSE (running total) → VER (rebuild) → ANS

Usage:  python gen_numeral.py --n 1500 --out numeral_donald.jsonl
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"), (1, "I"),
]

PREAMBLE_FWD = (
    "\nI am a reasoning model. I am in a Kaggle competition. I follow the "
    "template strictly.\n\n"
    "RULE 1: This is a Roman numeral template. I am converting an integer to "
    "a Roman numeral. NOT binary, symbol, digit, or any other type.\n\n"
    "RULE 2: Flavor text wrapper does not matter. I solve the problem.\n\n"
    "RULE 3: Final answer in \\boxed{} at the end.\n\n"
)
PREAMBLE_REV = (
    "\nI am a reasoning model. I am in a Kaggle competition. I follow the "
    "template strictly.\n\n"
    "RULE 1: This is a Roman numeral template. I am converting a Roman "
    "numeral to an integer. NOT binary, symbol, digit, or any other type.\n\n"
    "RULE 2: Flavor text wrapper does not matter.\n\n"
    "RULE 3: Final answer in \\boxed{} at the end.\n\n"
)


def int_to_roman(n: int) -> str:
    out = []
    for v, sym in ROMAN_VALUES:
        while n >= v:
            out.append(sym); n -= v
    return "".join(out)


def roman_to_int(s: str) -> int:
    val = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total, prev = 0, 0
    for ch in reversed(s):
        v = val[ch]
        if v < prev: total -= v
        else:        total += v
        prev = v
    return total


def decompose_places(n: int):
    """Return (TH, HU, TE, ON) tuples of (digit, roman_segment)."""
    th = (n // 1000, "M" * (n // 1000) if n // 1000 else "SKIP")
    hu_d = (n // 100) % 10
    hu_r = int_to_roman(hu_d * 100) if hu_d else "SKIP"
    te_d = (n // 10) % 10
    te_r = int_to_roman(te_d * 10)  if te_d else "SKIP"
    on_d = n % 10
    on_r = int_to_roman(on_d)       if on_d else "SKIP"
    return th, (hu_d, hu_r), (te_d, te_r), (on_d, on_r)


def build_cot_forward(target_int: int) -> str:
    th, hu, te, on = decompose_places(target_int)
    th_d, th_r = th
    hu_d, hu_r = hu
    te_d, te_r = te
    on_d, on_r = on

    # CAT — incremental concat, segment by segment
    parts = [s for s in (th_r, hu_r, te_r, on_r) if s != "SKIP"]
    if not parts:
        parts = ["I"]  # edge case: 0 (shouldn't happen with our seed range)
    cat_lines = []
    acc = parts[0]
    for nxt in parts[1:]:
        cat_lines.append(f"{acc} + {nxt} = {acc + nxt}")
        acc = acc + nxt
    result = acc

    # VER — re-parse the result
    parsed = roman_to_int(result)

    decompose_str = (
        f"TH:{th_d}->{th_r} ({th_d * 1000})\n"
        f"HU:{hu_d}->{hu_r} ({hu_d * 100})\n"
        f"TE:{te_d}->{te_r} ({te_d * 10})\n"
        f"ON:{on_d}->{on_r} ({on_d})"
    )

    cat_block = "\n".join(cat_lines) if cat_lines else f"({result})"

    return PREAMBLE_FWD + (
        f"S1: I am converting integer {target_int} to a Roman numeral.\n\n"
        f"S2: DECOMPOSE {target_int}\n{decompose_str}\n\n"
        f"S3: CAT\n{cat_block}\n"
        f"RESULT: {result}\n\n"
        f"S4: VER - Re-parse my RESULT to verify.\n"
        f"REPARSE: {result} = {parsed}\n"
        f"CHK: Does REPARSE({parsed}) = TARGET({target_int})? "
        f"{'YES' if parsed == target_int else 'NO'}\n\n"
        f"S5: ANS={result}\n\n"
        f"\\boxed{{{result}}}"
    )


def build_cot_reverse(target_roman: str) -> str:
    val = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total, prev = 0, 0
    parse_lines = []
    g = 1
    # walk subtractive pairs as atomic units
    pairs = {"CM":900,"CD":400,"XC":90,"XL":40,"IX":9,"IV":4}
    i = 0; running = 0; gi = 1
    while i < len(target_roman):
        if i+1 < len(target_roman) and target_roman[i:i+2] in pairs:
            seg = target_roman[i:i+2]
            v = pairs[seg]
            running += v
            parse_lines.append(f"G{gi}: {seg}={v}, RT={running}")
            i += 2
        else:
            seg = target_roman[i]
            v = val[seg]
            running += v
            parse_lines.append(f"G{gi}: {seg}={v}, RT={running}")
            i += 1
        gi += 1

    rebuilt = int_to_roman(running)
    return PREAMBLE_REV + (
        f"S1: I am converting Roman numeral {target_roman} to an integer.\n\n"
        f"S2: PARSE {target_roman}\n" + "\n".join(parse_lines) + "\n\n"
        f"S3: VER - Rebuild from my answer to verify.\n"
        f"REBUILD: {running} -> {rebuilt}\n"
        f"CHK: Does REBUILD({rebuilt}) = INPUT({target_roman})? "
        f"{'YES' if rebuilt == target_roman else 'NO'}\n\n"
        f"S4: ANS={running}\n\n"
        f"\\boxed{{{running}}}"
    )


def gen_one(rng):
    direction = rng.random() < 0.5
    n_ex = rng.randint(2, 4)

    if direction:  # int → Roman
        target = rng.randint(1, 3999)
        ex_ints = [rng.randint(1, 3999) for _ in range(n_ex)]
        prompt_lines = [f"{i} -> {int_to_roman(i)}" for i in ex_ints]
        prompt = "\n".join(prompt_lines) + f"\nConvert {target} to Roman numerals."
        prompt += "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        cot = build_cot_forward(target)
        answer = int_to_roman(target)
    else:           # Roman → int
        target = rng.randint(1, 3999)
        target_roman = int_to_roman(target)
        ex_ints = [rng.randint(1, 3999) for _ in range(n_ex)]
        prompt_lines = [f"{int_to_roman(i)} -> {i}" for i in ex_ints]
        prompt = "\n".join(prompt_lines) + f"\nConvert {target_roman} to an integer."
        prompt += "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"
        cot = build_cot_reverse(target_roman)
        answer = str(target)

    return {
        "category": "numeral",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cot},
        ],
        "answer": answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out", type=str, default="numeral_donald.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    n_ok = 0
    with open(args.out, "w") as f:
        for _ in range(args.n):
            r = gen_one(rng)
            if f"\\boxed{{{r['answer']}}}" in r["messages"][-1]["content"]:
                f.write(json.dumps({"category": r["category"], "messages": r["messages"]},
                                   ensure_ascii=False) + "\n")
                n_ok += 1
    print(f"Wrote {n_ok} records to {args.out}")
    import statistics
    lens = [len(json.loads(l)["messages"][-1]["content"]) for l in open(args.out)]
    print(f"CoT chars: median={statistics.median(lens):.0f} max={max(lens)}, est tokens≈{statistics.median(lens)/3.5:.0f}")


if __name__ == "__main__":
    main()
