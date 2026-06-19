"""gen_unit_conversion.py — Donald-format unit conversion generator.

Same rate-first decomposition as gravity but LINEAR (no t^2). Tighter VER
tolerance (0.01 vs 0.05) because there's no squaring to amplify rounding.

Usage:  python gen_unit_conversion.py --n 1500 --out unit_conversion_donald.jsonl
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

UNIT_PAIRS = [
    ("m",   "ft"),  ("km",  "miles"),  ("kg",  "lb"),   ("L",   "gal"),
    ("cm",  "in"),  ("mm",  "in"),     ("g",   "oz"),   ("mph", "kmh"),
    ("acre","ha"),  ("yd",  "m"),
]

PREAMBLE = (
    "\nI am a reasoning model. I am in a Kaggle competition. I have been "
    "trained to solve these puzzles, and I follow the template strictly to "
    "achieve high accuracy.\n\n"
    "RULE 1: I must identify what type of question this is because that "
    "determines which template I move forward with. This is a unit conversion "
    "template. I know this because I see measurements being converted from "
    "one unit to another using example pairs. I know this is NOT roman, "
    "binary, symbol, digit, or any other conversion type.\n\n"
    "RULE 2: Since I have identified this is a unit conversion template, "
    "none of the flavor text surrounding this problem matters. The wrapper "
    "is here to trick me. I am only here to solve the problem.\n\n"
    "RULE 3: Final answer in \\boxed{} at the end. Wrong format = zero points. "
    "The answer MUST be exactly 2 decimal places (X.XX format).\n\n"
)


def fmt_num(x, places=4):
    s = f"{x:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def build_cot(in_unit, out_unit, in_ex1, out_ex1, in_ex2, out_ex2,
              target_in, target_out):
    rate     = out_ex1 / in_ex1
    result   = target_in * rate
    rounded  = round(result, 2)
    rate2    = out_ex2 / in_ex2
    chk_pass = abs(rate - rate2) < 0.01
    chk_word = "YES" if chk_pass else "NO"

    return PREAMBLE + (
        "S1: I see that this is a unit conversion template. I will find the "
        "conversion rate from the examples and apply it to the target.\n\n"
        "S2: SOLVE\n"
        "I will use EX1 to find the conversion rate.\n"
        f"EX1: in={in_ex1}, out={out_ex1}\n"
        f"RATE: out1 / in1 = {out_ex1} / {in_ex1} = {fmt_num(rate)}\n"
        f"RESULT: target * RATE = {target_in} * {fmt_num(rate)} = {fmt_num(result)}\n"
        f"RND: {fmt_num(result)} -> {rounded:.2f}\n\n"
        "S3: VER - Check rate consistency using EX2.\n"
        f"EX2: in={in_ex2}, out={out_ex2}\n"
        f"RATE2: out2 / in2 = {out_ex2} / {in_ex2} = {fmt_num(rate2)}\n"
        f"CHK: Does |RATE({fmt_num(rate)}) - RATE2({fmt_num(rate2)})| < 0.01? {chk_word}\n\n"
        f"S4: ANS={rounded:.2f}\n\n"
        f"\\boxed{{{rounded:.2f}}}"
    )


def gen_one(rng):
    in_unit, out_unit = rng.choice(UNIT_PAIRS)
    factor    = round(rng.uniform(0.05, 25.0), 4)
    n_ex      = rng.randint(4, 6)
    in_ex     = sorted([round(rng.uniform(0.5, 200.0), 2) for _ in range(n_ex)])
    out_ex    = [round(factor * x, 2) for x in in_ex]
    while True:
        target_in = round(rng.uniform(0.5, 200.0), 2)
        if target_in not in in_ex: break
    # Compute target the SAME way CoT computes it (using rounded EX1)
    recovered = out_ex[0] / in_ex[0]
    target_out = round(recovered * target_in, 2)

    prompt = "\n".join(f"{x} {in_unit} becomes {y}" for x, y in zip(in_ex, out_ex))
    prompt += f"\nConvert the following measurement: {target_in} {in_unit}"
    prompt += "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`"

    cot = build_cot(in_unit, out_unit, in_ex[0], out_ex[0],
                    in_ex[1], out_ex[1], target_in, target_out)

    return {
        "category": "unit_conversion",
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": cot},
        ],
        "answer": f"{target_out:.2f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--out", type=str, default="unit_conversion_donald.jsonl")
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
