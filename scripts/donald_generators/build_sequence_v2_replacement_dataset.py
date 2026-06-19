"""
Build a training dataset with solved sequence-v2 cryptarithm rows replaced.

This keeps the proven baseline files intact and writes a new directory:
  all_categorical_splits_sequence_v2_replaced/

Only rows whose IDs are present in sequence_solved_v2.jsonl are replaced.
The replacement records keep the original user prompt and ID, but swap in a
solver-derived Tong-style CoT assistant response.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gen_sequence_cot_tong import build_cot  # noqa: E402


TARGET_FILES = {
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
}


def load_solved(path: Path) -> dict[str, dict]:
    solved: dict[str, dict] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec_id = rec["id"]
            if rec_id in solved:
                raise ValueError(f"duplicate solved id {rec_id!r} at {path}:{line_no}")
            solved[rec_id] = rec
    return solved


def replace_file(src: Path, dst: Path, solved: dict[str, dict]) -> tuple[int, int]:
    total = 0
    replaced = 0
    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            rec_id = rec.get("id")
            if rec_id in solved:
                solved_rec = dict(solved[rec_id])
                solved_rec["prompt"] = rec["messages"][0]["content"]
                rec = {
                    "id": rec_id,
                    "messages": [
                        rec["messages"][0],
                        {"role": "assistant", "content": build_cot(solved_rec)},
                    ],
                }
                replaced += 1
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return total, replaced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-dir",
        default="dont_touch_it/all_categorical_splits",
        help="Source directory with the original category JSONL files.",
    )
    ap.add_argument(
        "--solved",
        default="sequence_solved_v2.jsonl",
        help="Solver output JSONL from solve_sequence_v2.py.",
    )
    ap.add_argument(
        "--out-dir",
        default="all_categorical_splits_sequence_v2_replaced",
        help="Destination directory for the rebuilt training dataset.",
    )
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    solved_path = Path(args.solved)
    out_dir = Path(args.out_dir)

    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"baseline directory not found: {baseline_dir}")
    if not solved_path.exists():
        raise FileNotFoundError(f"solved JSONL not found: {solved_path}")
    if "dont_touch_it" in out_dir.parts:
        raise ValueError("refusing to write output under dont_touch_it")

    solved = load_solved(solved_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_replacements: set[str] = set()
    print(f"Source baseline : {baseline_dir}")
    print(f"Solved records  : {solved_path} ({len(solved)} rows)")
    print(f"Output dataset  : {out_dir}\n")

    for src in sorted(baseline_dir.glob("*.jsonl")):
        dst = out_dir / src.name
        if src.name in TARGET_FILES:
            total, replaced = replace_file(src, dst, solved)
            if replaced:
                with src.open() as f:
                    for line in f:
                        if line.strip():
                            rec_id = json.loads(line).get("id")
                            if rec_id in solved:
                                seen_replacements.add(rec_id)
            print(f"replaced {replaced:>4} / {total:<4} in {src.name}")
        else:
            shutil.copy2(src, dst)
            total = sum(1 for _ in dst.open())
            print(f"copied   {total:>4} rows     in {src.name}")

    missing = sorted(set(solved) - seen_replacements)
    if missing:
        raise RuntimeError(
            f"{len(missing)} solved IDs were not found in target files; "
            f"first few: {missing[:10]}"
        )

    print(f"\nDone. Replaced {len(seen_replacements)} solved sequence-v2 rows.")


if __name__ == "__main__":
    main()
