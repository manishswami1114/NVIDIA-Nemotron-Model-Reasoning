#!/bin/bash
# v17 finalisation pipeline — run AFTER scripts/v17_strict_resolve.py finishes.
# Steps:
#   1. Merge strict + originally-valid solutions → crypto_solutions_v17.json
#   2. Regenerate cryptarithm_deduce/guess JSONL with new solutions
#   3. Verify boxed answers match train.csv
#   4. Final audit — confirm no zero-pad fakery + no degenerate maps
set -e

cd "$(dirname "$0")/.."
echo "=========================================="
echo "STEP 1: merge solutions"
echo "=========================================="
python3 scripts/v17_merge_solutions.py

echo
echo "=========================================="
echo "STEP 2: regenerate v17 cryptarithm files"
echo "=========================================="
python3 scripts/v17_build_cots.py

echo
echo "=========================================="
echo "STEP 3: verify boxed answers"
echo "=========================================="
python3 scripts/v17_verify.py

echo
echo "=========================================="
echo "STEP 4: final audit (zero-pad + degenerate-map checks)"
echo "=========================================="
python3 scripts/v17_final_audit.py
