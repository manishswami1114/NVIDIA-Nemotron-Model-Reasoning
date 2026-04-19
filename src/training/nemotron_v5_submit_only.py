# -*- coding: utf-8 -*-
"""
Nemotron v5 — SUBMISSION-ONLY notebook
=======================================
This notebook does NOT train. It just packages a pre-trained adapter
into the competition submission zip.

WORKFLOW:
  1. Run the TRAINING notebook first (nemotron_v5_train_only).
     → Commit it. Its /kaggle/working/adapter/ becomes the notebook output.
  2. In THIS submission notebook, Add Data → Notebook Output → pick the training notebook.
     → The adapter files show up at /kaggle/input/<training-notebook-slug>/adapter/
  3. Run this notebook. Takes ~10 seconds. Produces /kaggle/working/submission.zip
  4. Submit submission.zip to the competition.

NO GPU NEEDED. NO MODEL LOADING. Just file operations.
"""

# ============================================================
# Cell 1: Imports
# ============================================================
import os, json, shutil, zipfile, glob
from pathlib import Path

print("Nemotron v5 Submission Packager")
print("=" * 50)


# ============================================================
# Cell 2: Find the trained adapter
# ============================================================
# Look for adapter_model.safetensors anywhere under /kaggle/input/
# The training notebook saves it to /kaggle/working/adapter/ which becomes
# /kaggle/input/<notebook-slug>/adapter/ when used as input to this notebook.

search_globs = [
    "/kaggle/input/*/adapter/adapter_model.safetensors",
    "/kaggle/input/*/adapter_model.safetensors",
    "/kaggle/input/*/*/adapter_model.safetensors",
    "/kaggle/input/*/*/*/adapter_model.safetensors",
    # If you uploaded the adapter as a Kaggle Dataset instead:
    "/kaggle/input/nemotron-adapter/adapter_model.safetensors",
    "/kaggle/input/nemotron-adapter/*/adapter_model.safetensors",
]

weights_path = None
for pattern in search_globs:
    matches = glob.glob(pattern)
    if matches:
        weights_path = matches[0]
        print(f"✓ Found adapter weights: {weights_path}")
        break

if weights_path is None:
    print("✗ Could not find adapter_model.safetensors. Searched:")
    for p in search_globs:
        print(f"    {p}")
    print("\nTo fix: add the training notebook's output as an input to this notebook.")
    print("  Right sidebar → Add Data → Notebook Output → pick your training notebook")
    raise FileNotFoundError("Adapter not found")

adapter_dir = os.path.dirname(weights_path)
config_path = os.path.join(adapter_dir, "adapter_config.json")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"adapter_config.json missing next to weights at {adapter_dir}")

print(f"✓ Found adapter config:  {config_path}")


# ============================================================
# Cell 3: Copy files to /kaggle/working/adapter/
# (We need write access; /kaggle/input/ is read-only)
# ============================================================
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

shutil.copy2(weights_path, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
shutil.copy2(config_path,  os.path.join(OUTPUT_DIR, "adapter_config.json"))

print(f"\nCopied to {OUTPUT_DIR}:")
for f in os.listdir(OUTPUT_DIR):
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
    print(f"  {f}  ({size_mb:.2f} MB)")


# ============================================================
# Cell 4: Patch adapter_config.json — base_model must be canonical
# ============================================================
config_file = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(config_file) as f:
    cfg = json.load(f)

original_base = cfg.get("base_model_name_or_path", "<missing>")
cfg["base_model_name_or_path"] = "metric/nemotron-3-nano-30b-a3b-bf16"

with open(config_file, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"\nPatched base_model_name_or_path:")
print(f"  was: {original_base}")
print(f"  now: {cfg['base_model_name_or_path']}")

# Sanity-check a few fields
print(f"\nAdapter config summary:")
print(f"  r (rank):         {cfg.get('r')}")
print(f"  lora_alpha:       {cfg.get('lora_alpha')}")
print(f"  target_modules:   {cfg.get('target_modules')}")
print(f"  task_type:        {cfg.get('task_type')}")


# ============================================================
# Cell 5: Verify adapter weights are non-zero (sanity check)
# ============================================================
try:
    from safetensors import safe_open
    with safe_open(os.path.join(OUTPUT_DIR, "adapter_model.safetensors"),
                   framework="pt") as f:
        keys = list(f.keys())
        print(f"\nAdapter tensors: {len(keys)} parameters")
        norms = []
        for k in keys[:5]:
            t = f.get_tensor(k)
            norms.append(t.norm().item())
        print(f"First 5 weight norms: {[f'{n:.4f}' for n in norms]}")
        if all(n < 1e-4 for n in norms):
            print("⚠️  WARNING: norms are near zero — adapter may be untrained!")
            print("    Do NOT submit. Re-run the training notebook.")
        else:
            print("✓ Adapter looks healthy (non-zero weights)")
except Exception as e:
    print(f"Could not verify: {e}")


# ============================================================
# Cell 6: Zip — exactly 2 files, nothing else
# ============================================================
ZIP_PATH = "/kaggle/working/submission.zip"
REQUIRED = {"adapter_config.json", "adapter_model.safetensors"}

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname in REQUIRED:
            zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)

with zipfile.ZipFile(ZIP_PATH) as zf:
    contents = zf.namelist()

zip_mb = os.path.getsize(ZIP_PATH) / 1024 / 1024
print(f"\n{'='*50}")
print(f"submission.zip ready ({zip_mb:.1f} MB)")
print(f"{'='*50}")
print(f"Contents: {contents}")
assert set(contents) == REQUIRED, f"Wrong files in zip: {contents}"
print("\n✓ Clean 2-file zip — ready to submit!")
print(f"  Path: {ZIP_PATH}")
