"""
gen_gravity.py — Donald-format gravity (kinematics) generator
==============================================================

Produces training records for the `gravity` category in the EXACT format
Donald's playbook describes:
  - Strict template with meta-preamble + RULE 1/2/3 + S1/S2/S3/S4 + \\boxed{}
  - Rate-first decomposition (RATE = d/t^2, two ops instead of five)
  - Rate consistency VER (|RATE - RATE2| < 0.05) instead of full recompute
  - Answer EXACTLY 2 decimal places

Output: ~250 tokens per CoT (vs your 75K dataset's 1638 tokens — 6.5x shorter)

Usage:
    python gen_gravity.py --n 1500 --out gravity_donald.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ============================================================
# Donald-format CoT template for gravity
# ============================================================
PREAMBLE = (
    "\nI am a reasoning model. I am in a Kaggle competition. I have been "
    "trained to solve these puzzles, and I follow the template strictly to "
    "achieve high accuracy.\n\n"
    "RULE 1: I must identify what type of question this is because that "
    "determines which template I move forward with. This is a gravity "
    "kinematics template. I know this because I see time and distance values "
    "that follow d = 0.5gt^2, and the problem asks me to find distance for a "
    "new time value. I know this is NOT roman, binary, symbol, digit, unit "
    "conversion, or any other conversion type.\n\n"
    "RULE 2: Since I have identified this is a gravity kinematics template, "
    "none of the flavor text surrounding this problem matters. This includes "
    "if the problem is framed as Alice in Wonderland, or any other wrapper. "
    "The wrapper is here to trick me. I know this, so I am going to ignore "
    "any flavor text. I am only here to solve the problem.\n\n"
    "RULE 3: Final answer in \\boxed{} at the end. Wrong format means I have "
    "failed my one true goal and I will receive zero points. This invalidates "
    "all work if I do not box my answer. The answer MUST be exactly 2 decimal "
    "places (X.XX format).\n\n"
)


def fmt_num(x: float, places: int = 4) -> str:
    """Format with up to `places` decimals, strip trailing zeros."""
    s = f"{x:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def build_cot(t_ex1: float, d_ex1: float,
              t_ex2: float, d_ex2: float,
              target_t: float, target_d: float) -> str:
    """Build the Donald-style CoT trace for one gravity problem."""
    # SOLVE block math
    t_sq      = t_ex1 ** 2
    rate      = d_ex1 / t_sq
    tgt_sq    = target_t ** 2
    result    = rate * tgt_sq
    rounded   = round(result, 2)

    # VER block math
    t_sq2     = t_ex2 ** 2
    rate2     = d_ex2 / t_sq2
    chk_pass  = abs(rate - rate2) < 0.05
    chk_word  = "YES" if chk_pass else "NO"

    cot = PREAMBLE + (
        "S1: I see that this is a gravity kinematics template. I will find "
        "the rate constant (0.5g) from the examples and apply it to the "
        "target time. I am now going to fill out the template.\n\n"
        "S2: SOLVE\n"
        "I will use EX1 to find the rate constant (0.5g).\n"
        f"EX1: t={t_ex1}, d={d_ex1}\n"
        f"T_SQ: t^2 = {t_ex1}^2 = {fmt_num(t_sq)}\n"
        f"RATE: d / t^2 = {d_ex1} / {fmt_num(t_sq)} = {fmt_num(rate)}\n"
        f"TGT_SQ: target_t^2 = {target_t}^2 = {fmt_num(tgt_sq)}\n"
        f"RESULT: RATE * TGT_SQ = {fmt_num(rate)} * {fmt_num(tgt_sq)} = {fmt_num(result)}\n"
        f"RND: {fmt_num(result)} -> {rounded:.2f}\n\n"
        "S3: VER - Check rate consistency using EX2.\n"
        f"EX2: t={t_ex2}, d={d_ex2}\n"
        f"T_SQ2: t^2 = {t_ex2}^2 = {fmt_num(t_sq2)}\n"
        f"RATE2: d / t^2 = {d_ex2} / {fmt_num(t_sq2)} = {fmt_num(rate2)}\n"
        f"CHK: Does |RATE({fmt_num(rate)}) - RATE2({fmt_num(rate2)})| < 0.05? {chk_word}\n\n"
        f"S4: ANS={rounded:.2f}\n\n"
        f"\\boxed{{{rounded:.2f}}}"
    )
    return cot


def build_prompt(t_examples: list[float], d_examples: list[float],
                 target_t: float) -> str:
    """The user-facing prompt (same as the test format)."""
    lines = [f"For t = {t}s, distance = {d} m" for t, d in zip(t_examples, d_examples)]
    return (
        "\n".join(lines)
        + f"\nNow, determine the falling distance for t = {target_t}s "
          "given d = 0.5gt^2."
        "\nPlease put your final answer inside `\\boxed{}`. For example: "
        "`\\boxed{your answer}`"
    )


def gen_one(rng: random.Random) -> dict:
    """Generate a single (prompt, CoT, answer) record."""
    # Random gravity-like rate constant
    g           = round(rng.uniform(2.0, 30.0), 4)
    rate_const  = 0.5 * g  # this is what RATE recovers

    # 4-6 example pairs
    n_examples = rng.randint(4, 6)
    t_examples = sorted([round(rng.uniform(0.5, 9.0), 2) for _ in range(n_examples)])
    d_examples = [round(rate_const * t * t, 2) for t in t_examples]

    # Target time (different from examples)
    while True:
        target_t = round(rng.uniform(0.5, 9.0), 2)
        if target_t not in t_examples:
            break

    # CRITICAL: target_d must be computed the same way the CoT computes it,
    # i.e. using the rate RECOVERED from the rounded example (d_ex1, t_ex1).
    # The example is rounded to 2dp, so the recovered rate slightly differs
    # from the true rate_const. Using rate_const here would create
    # unverifiable records (CoT answer ≠ ground truth).
    recovered_rate = d_examples[0] / (t_examples[0] ** 2)
    target_d       = round(recovered_rate * target_t * target_t, 2)

    prompt    = build_prompt(t_examples, d_examples, target_t)
    cot       = build_cot(
        t_examples[0], d_examples[0],
        t_examples[1], d_examples[1],
        target_t, target_d,
    )

    return {
        "category": "gravity",
        "messages": [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": cot},
        ],
        "answer": f"{target_d:.2f}",  # for verification
    }


def verify_record(rec: dict) -> bool:
    """Check that the CoT's final \\boxed{} matches the ground-truth answer."""
    cot = rec["messages"][-1]["content"]
    expected = rec["answer"]
    return f"\\boxed{{{expected}}}" in cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  type=str, default="gravity_donald.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)

    n_ok, n_bad = 0, 0
    with out_path.open("w") as f:
        for _ in range(args.n):
            rec = gen_one(rng)
            if verify_record(rec):
                # Drop the "answer" helper field before writing — training
                # data only needs messages + category
                rec_out = {"category": rec["category"], "messages": rec["messages"]}
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                n_bad += 1

    print(f"Wrote {n_ok} records to {out_path} ({n_bad} dropped as malformed)")

    # Diagnostics: avg CoT length
    import statistics
    lens = []
    with out_path.open() as f:
        for line in f:
            r = json.loads(line)
            lens.append(len(r["messages"][-1]["content"]))
    if lens:
        print(f"CoT length (chars): median={statistics.median(lens):.0f} "
              f"mean={statistics.mean(lens):.0f} "
              f"max={max(lens)}")
        # Rough token estimate: chars / 3.5 (English text ≈ 3.5 chars/token)
        print(f"Estimated tokens : median≈{statistics.median(lens)/3.5:.0f} "
              f"mean≈{statistics.mean(lens)/3.5:.0f}")


if __name__ == "__main__":
    main()
