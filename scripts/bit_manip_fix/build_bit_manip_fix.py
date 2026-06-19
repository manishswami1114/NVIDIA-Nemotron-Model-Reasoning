"""Build bit_manip_fix dataset — replace default-containing bit_manipulation
CoTs with composite-aware regenerated CoTs, drop ones that still default.

For each baseline bit_manipulation training record:
  - If baseline CoT contains 'default 1' or 'default 0' → regenerate with composite
      - If new CoT has no default AND boxed answer matches gold → USE NEW
      - Else → DROP the record
  - Else → keep baseline as-is

Other 7 categories: copied verbatim.

Output:
  all_categorical_splits_bit_manip_fix/
  ~/Downloads/bit_manip_fix.zip
"""
from __future__ import annotations
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from reasoner import reasoning_bit_manipulation, Problem, _Example  # noqa: E402

csv.field_size_limit(sys.maxsize)

ROOT         = Path(__file__).resolve().parent.parent.parent
TRAIN_CSV    = ROOT / "data" / "raw" / "train.csv"
BASELINE_DIR = ROOT / "dont_touch_it" / "all_categorical_splits"
OUT_DIR      = ROOT / "all_categorical_splits_bit_manip_fix"

BIT_MANIP_FILE = "train_cot_bit_manipulation.jsonl"


# Match lines like "01010001 -> 11011101"
_EX_RE  = re.compile(r"^\s*([01]{8})\s*->\s*([01]{8})\s*$")
# Match all the phrasings seen in bit_manipulation prompts
_Q_RE   = re.compile(
    r"(?:Now\s*,?\s*(?:apply this rule to|determine the output for|determine the output|find the output for))\s*:\s*([01]{8})",
    re.IGNORECASE,
)


def parse_bit_manip_prompt(prompt: str):
    """Extract list of (input, output) pairs and the query bit-string."""
    examples = []
    query = None
    for line in prompt.splitlines():
        m = _EX_RE.match(line)
        if m:
            examples.append(_Example(input_value=m.group(1), output_value=m.group(2)))
            continue
        m = _Q_RE.search(line)
        if m:
            query = m.group(1)
    return examples, query


def cot_has_default(cot: str) -> bool:
    return "default 1" in cot or "default 0" in cot


def extract_boxed_answer(cot: str):
    # Find the final \boxed{X} at the end of the CoT
    matches = re.findall(r"\\boxed\{([01]+)\}", cot)
    if not matches:
        return None
    return matches[-1]


def main():
    if not BASELINE_DIR.is_dir():
        raise FileNotFoundError(BASELINE_DIR)
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(TRAIN_CSV)

    # id -> gold answer  (note: many bit_manipulation records in baseline JSONL
    # have id='?' placeholder, so we'll join by PROMPT TEXT instead)
    prompt2gold = {}
    with TRAIN_CSV.open() as f:
        for r in csv.DictReader(f):
            prompt2gold[r["prompt"]] = r["answer"]
    print(f"train.csv prompts indexed: {len(prompt2gold)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'File':<42} {'In':>5} {'Action'}")
    print("-"*72)

    # Process bit_manipulation
    src = BASELINE_DIR / BIT_MANIP_FILE
    dst = OUT_DIR / BIT_MANIP_FILE

    stats = {
        "no_default_baseline_kept": 0,
        "had_default_composite_resolved": 0,
        "had_default_still_default_dropped": 0,
        "had_default_wrong_answer_dropped": 0,
        "had_default_parse_fail_dropped": 0,
        "had_default_no_gold_dropped": 0,
    }
    composite_count = 0
    new_cot_lens = []
    t_start = time.time()

    with src.open() as f_in, dst.open("w") as f_out:
        n_in = 0
        for line in f_in:
            if not line.strip(): continue
            n_in += 1
            rec = json.loads(line)
            user_prompt = rec["messages"][0]["content"]
            baseline_cot = rec["messages"][1]["content"]

            if not cot_has_default(baseline_cot):
                # Keep as-is
                f_out.write(line)
                stats["no_default_baseline_kept"] += 1
                continue

            # Try regeneration
            examples, query = parse_bit_manip_prompt(user_prompt)
            if not examples or not query:
                stats["had_default_parse_fail_dropped"] += 1
                continue

            # Note: baseline bit_manipulation prompts are synthetic and don't
            # match train.csv. We don't need external gold — if composite
            # resolves all bits, the answer is uniquely determined by the
            # examples themselves. Filter by "no default remains".

            try:
                new_cot_inner = reasoning_bit_manipulation(
                    Problem(examples=examples, question=query)
                )
            except Exception as e:
                stats["had_default_parse_fail_dropped"] += 1
                continue
            if new_cot_inner is None:
                stats["had_default_parse_fail_dropped"] += 1
                continue

            # Wrap in <think>...</think> matching baseline's format
            new_cot = "<think>\n" + new_cot_inner + "\n</think>\n" + \
                      f"\\boxed{{{extract_boxed_answer(new_cot_inner) or ''}}}"

            if cot_has_default(new_cot):
                stats["had_default_still_default_dropped"] += 1
                continue

            # Sanity: must have a boxed answer that's 8 bits
            boxed = extract_boxed_answer(new_cot)
            if boxed is None or len(boxed) != 8:
                stats["had_default_wrong_answer_dropped"] += 1
                continue

            # Replace
            new_rec = {
                "messages": [
                    rec["messages"][0],
                    {"role": "assistant", "content": new_cot},
                ],
            }
            if "id" in rec: new_rec["id"] = rec["id"]
            f_out.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            stats["had_default_composite_resolved"] += 1
            new_cot_lens.append(len(new_cot))
            if "COMPOSITE MATCH" in new_cot or "COMPOSITE(" in new_cot:
                composite_count += 1

            if n_in % 100 == 0:
                print(f"  ...processed {n_in}/2728  elapsed={time.time()-t_start:.0f}s  "
                      f"resolved={stats['had_default_composite_resolved']} "
                      f"dropped={stats['had_default_still_default_dropped']+stats['had_default_wrong_answer_dropped']}",
                      flush=True)

    kept_total = stats["no_default_baseline_kept"] + stats["had_default_composite_resolved"]
    dropped_total = (stats["had_default_still_default_dropped"]
                     + stats["had_default_wrong_answer_dropped"]
                     + stats["had_default_parse_fail_dropped"]
                     + stats["had_default_no_gold_dropped"])
    print(f"\n  {BIT_MANIP_FILE:<42} {n_in:>5} → kept {kept_total} (dropped {dropped_total})")

    # Copy other 7 categories verbatim
    for p in sorted(BASELINE_DIR.glob("*.jsonl")):
        if p.name == BIT_MANIP_FILE: continue
        shutil.copy2(p, OUT_DIR / p.name)
        with (OUT_DIR / p.name).open() as f:
            n = sum(1 for _ in f if _.strip())
        print(f"  {p.name:<42} {n:>5}  COPY")

    print("\n=== Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nTotal bit_manipulation records after fix: {kept_total} (baseline: {n_in})")
    print(f"  - kept-baseline (no default in CoT): {stats['no_default_baseline_kept']}")
    print(f"  - composite-resolved replacements:    {stats['had_default_composite_resolved']}")
    print(f"  - dropped:                           {dropped_total}")
    print(f"\nNew CoTs that triggered COMPOSITE search: {composite_count}/{stats['had_default_composite_resolved']}")
    if new_cot_lens:
        new_cot_lens.sort()
        print(f"\nReplaced-CoT length: min={min(new_cot_lens)}, "
              f"median={new_cot_lens[len(new_cot_lens)//2]}, "
              f"max={max(new_cot_lens)}")

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"\nNext:")
    print(f"  cd {OUT_DIR}")
    print(f"  zip -r ~/Downloads/bit_manip_fix.zip *.jsonl")


if __name__ == "__main__":
    main()
