# -*- coding: utf-8 -*-
"""
Nemotron v5 — TRAINING-ONLY notebook
======================================
This notebook ONLY trains the LoRA adapter. It does NOT package a submission.
Pair it with `nemotron_v5_submit_only.py` which picks up this notebook's
output and zips it for submission.

WHY SPLIT?
  - Training takes 2–4 hours on a GPU. You want to iterate on the submission
    zip (category patches, base-model path tweaks, etc.) without re-training.
  - Submission notebook is CPU-only and runs in seconds — perfect for quick
    fixes + resubmits.

WORKFLOW:
  1. Upload train_cot_v4_real.jsonl as a Kaggle dataset named `nemotron-cot-v4`.
  2. Attach that dataset + the base model + offline packages to THIS notebook.
  3. Set accelerator to GPU (T4/P100/A100/L4 all OK), run all cells.
     → Adapter saved to /kaggle/working/adapter/
  4. Click "Save Version" → "Save & Run All (Commit)". Let it finish.
     → Kaggle now has this notebook's output (adapter files) saved.
  5. Open `nemotron_v5_submit_only` notebook → Add Data → Notebook Output →
     pick this training notebook → run all → submit resulting submission.zip.

v5 scored 0.66 (regression from 0.69) — removed too much data (5121 examples, 0 transform).

v6 FIXES:
  - train_cot_v5_merged.jsonl = 9406 VERIFIED examples (all answers correct)
  - v4_real (5121) as primary: solver traces for bit/cipher/unit/gravity/roman
  - v3 correct gap-fill (4285): verified-correct answers for transform + remaining categories
  - Bit manipulation: 722 real solver CoT + 880 honest-template CoT = 1602 total
  - Transformation rules: 1461 verified-correct examples (was 0 in v5!)
  - ALL 6 categories now covered with correct answers
  - 3 epochs on larger, verified dataset
"""

# ============================================================
# 1. OFFLINE DEPENDENCY INSTALLATION
# ============================================================
import subprocess, sys, os
from pathlib import Path

def resolve_python_path(target_dir):
    for pth_file in Path(target_dir).glob("*.pth"):
        with pth_file.open() as fp:
            relpath = fp.read().strip()
            rel_pack_path = pth_file.parent / relpath
            if rel_pack_path.exists():
                sys.path.append(str(rel_pack_path))

offline_dir = "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages"
target_dir  = "/kaggle/working/packages"
os.makedirs(target_dir, exist_ok=True)

resolve_python_path("/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/")

if os.path.exists(offline_dir):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-index", "--find-links", offline_dir,
        "--target", target_dir,
        "datasets", "trl", "peft"
    ])
    print("Offline packages installed.")

sys.path.append(target_dir)
resolve_python_path(target_dir)


# ============================================================
# 2. IMPORTS & ENVIRONMENT
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import stat, shutil, time, json, re
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm

print(f"PyTorch : {torch.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# ============================================================
# 3. TRITON / RMSNORM FIXES (critical for NemotronH)
# ============================================================
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for name, mod in list(sys.modules.items()):
    if hasattr(mod, "rmsnorm_fn"):
        mod.rmsnorm_fn = _pure_rmsnorm_fn

src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
dst = "/tmp/ptxas-blackwell"
if os.path.exists(src):
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    import triton.backends.nvidia as nv_backend
    src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
    dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
    for f in os.listdir(dst_bin):
        fp = os.path.join(dst_bin, f)
        if os.path.isfile(fp):
            os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"] = dst
    print("Triton ptxas fix applied.")


# ============================================================
# 4. HYPERPARAMETERS — v5 with real CoT data
# ============================================================
LORA_RANK    = 32
LORA_ALPHA   = 128      # ← NVIDIA uses 4:1 ratio (was 1:1 = 32)
MAX_SEQ_LEN  = 2048
NUM_EPOCHS   = 2        # ← 2 epochs (NVIDIA SFT default), less overfitting risk
BATCH_SIZE   = 4        # ← bump to 8 on A100/L4
GRAD_ACCUM   = 4        # effective batch = 16
LR           = 1e-5     # ← NVIDIA SFT official LR (was 2e-4 = 20x too high!)

MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# v5 merged = v4_real (5121 solver-verified) + v3 correct gap-fill (4285) = 9406 examples
# All answers verified against ground truth. All 6 categories covered.
OUR_DATA_PATHS = [
    "/kaggle/input/nemotron-cot-v5/train_cot_v5_merged.jsonl",        # primary: 9406 verified
    "/kaggle/input/nemotron-dataset/train_cot_v5_merged.jsonl",
    "/kaggle/input/nemotron-cot-v4/train_cot_v4_real.jsonl",          # fallback: 5121 (missing transform)
    "/kaggle/input/nemotron-dataset/train_cot_v4_real.jsonl",
    # last resort
    "/kaggle/input/nemotron-dataset/train_cot_v3_metric_aligned.jsonl",
    "/kaggle/input/nemotron-cot-v3/train_cot_v3_metric_aligned.jsonl",
]
EXTERNAL_CSV_PATHS = [
    "/kaggle/input/nemotron-30b-competition-trainingdata-cot-labels/final_Nemotron_training_data.csv",
    "/kaggle/input/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels/final_Nemotron_training_data.csv",
]

print(f"Config: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM} eff, rank {LORA_RANK}, alpha {LORA_ALPHA}, lr {LR}")


# ============================================================
# 5. PROGRESS BAR CALLBACK
# ============================================================
class LiveProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar       = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(
            total=state.max_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.pbar is None:
            return
        elapsed  = time.time() - self.start_time
        step     = state.global_step
        eta      = (elapsed / step) * (state.max_steps - step) if step > 0 else 0
        loss_str = (
            f"loss={state.log_history[-1]['loss']:.4f}"
            if state.log_history and "loss" in state.log_history[-1]
            else "loss=..."
        )
        self.pbar.set_postfix_str(
            f"{loss_str}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
        )
        self.pbar.update(1)
        sys.stdout.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()


# ============================================================
# 6. LOAD DATASET — combine BOTH sources for max coverage
# ============================================================
print("Loading datasets...\n")

our_data = []
ext_csv_path = None

# 1) Try our metric-aligned JSONL
for path in OUR_DATA_PATHS:
    if os.path.exists(path):
        print(f"✓ Found our JSONL: {path}")
        with open(path, 'r') as f:
            for line in f:
                our_data.append(json.loads(line))
        print(f"  → {len(our_data)} metric-aligned examples")
        break

if not our_data:
    print("✗ Our JSONL not found (searched all candidate paths)")

# 2) Try external CSV
for path in EXTERNAL_CSV_PATHS:
    if os.path.exists(path):
        ext_csv_path = path
        print(f"✓ Found external CSV: {path}")
        break

if not ext_csv_path:
    print("✗ External CSV not found")

if not our_data and not ext_csv_path:
    raise FileNotFoundError(
        "No training data found!\n"
        "Upload train_cot_v4_real.jsonl as Kaggle dataset named 'nemotron-cot-v4'\n"
        "OR add kienngx/nemotron-30b-competition-trainingdata-cot-labels as input"
    )

print(f"\nData sources: our_jsonl={len(our_data)}, ext_csv={'YES' if ext_csv_path else 'NO'}")


# ============================================================
# 7. TOKENIZER & FORMAT — unified pipeline for both sources
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

EVAL_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

all_texts = []

# ─── Source A: Our metric-aligned JSONL ───────────────────
if our_data:
    print(f"Formatting {len(our_data)} JSONL examples...")
    for example in our_data:
        messages = [m for m in example['messages'] if m['role'] != 'system']

        # Ensure <think> tags
        assistant_msg = messages[-1]
        if '<think>' not in assistant_msg['content']:
            content = assistant_msg['content']
            boxed_match = re.search(r'(\\boxed\{.*?\})\s*$', content)
            if boxed_match:
                reasoning = content[:boxed_match.start()].strip()
                boxed_answer = boxed_match.group(1)
                messages[-1] = {
                    'role': 'assistant',
                    'content': f"<think>\n{reasoning}\n</think>\n{boxed_answer}"
                }

        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = (
                f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n{messages[-1]['content']}<|im_end|>"
            )
        all_texts.append(text)
    print(f"  → {len(all_texts)} examples from JSONL")

# ─── Source B: External CSV (deduplicated) ────────────────
if ext_csv_path:
    import pandas as pd
    ext_df = pd.read_csv(ext_csv_path)
    print(f"\nExternal CSV: {len(ext_df)} rows, columns: {list(ext_df.columns)}")

    # Collect prompts we already have (to avoid duplicates)
    existing_prompts = set()
    if our_data:
        for example in our_data:
            user_content = example['messages'][0]['content'] if example['messages'][0]['role'] == 'user' else ''
            existing_prompts.add(user_content[:200])

    ext_added = 0
    for _, row in ext_df.iterrows():
        prompt = str(row.get('prompt', ''))
        if prompt[:200] in existing_prompts:
            continue

        answer = str(row.get('answer', ''))
        cot = str(row.get('generated_cot', ''))

        if not prompt.strip() or not answer.strip():
            continue

        user_msg = prompt + EVAL_SUFFIX
        assistant_msg = f"<think>\n{cot}\n</think>\n\\boxed{{{answer}}}"

        try:
            messages = [
                {'role': 'user',      'content': user_msg},
                {'role': 'assistant', 'content': assistant_msg},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            )
        all_texts.append(text)
        ext_added += 1

    print(f"  → {ext_added} NEW examples from external CSV (after dedup)")

# ─── Build final dataset ──────────────────────────────────
hf_dataset = Dataset.from_dict({'text': all_texts})
print(f"\n{'='*50}")
print(f"TOTAL DATASET: {len(hf_dataset)} examples")
print(f"{'='*50}")
print(f"\nSample (first 400 chars):")
print(hf_dataset[0]['text'][:400])


# ============================================================
# 8. DROP OVERSIZED SAMPLES
# ============================================================
print(f"Filtering samples > {MAX_SEQ_LEN} tokens...")
before = len(hf_dataset)

def get_token_length(example):
    ids = tokenizer(example['text'], truncation=False,
                    return_attention_mask=False)['input_ids']
    return {'token_len': len(ids)}

hf_dataset = hf_dataset.map(get_token_length, desc="Counting tokens")
hf_dataset = hf_dataset.filter(
    lambda x: x['token_len'] <= MAX_SEQ_LEN,
    desc="Dropping oversized",
)
hf_dataset = hf_dataset.remove_columns(['token_len'])
print(f"  Kept {len(hf_dataset)} / {before}  ({before - len(hf_dataset)} dropped)")

steps_estimate = len(hf_dataset) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS
print(f"\nSteps    : {steps_estimate}")
print(f"Est. time: ~{steps_estimate * 8 / 3600:.1f} hrs")


# ============================================================
# 9. LOAD MODEL — bf16, NO quantization
# ============================================================
# NemotronH is a HYBRID model (Transformer + Mamba-2 + MoE)
# It does NOT support flash_attention_2 — must use "eager" or default
# Flash attn wheel is still needed by some internal Triton kernels

flash_whl = "/kaggle/input/nvidia-nemotron-offline-packages/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
if os.path.exists(flash_whl):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", flash_whl])
        print("Installed flash_attn wheel (used by internal kernels)")
    except Exception as e:
        print(f"flash_attn install skipped: {e}")

print("Loading base model (bf16, eager attention)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},          # Single GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,   # Full bf16, NOT quantized
    # NO attn_implementation — NemotronH doesn't support flash_attention_2
)
model.gradient_checkpointing_enable()

# Critical fix: disable fast path for NemotronH
for name, mod in list(sys.modules.items()):
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False

print(f"Model loaded on GPU")


# ============================================================
# 10. FRESH LORA
# ============================================================
# NVIDIA explicitly EXCLUDES out_proj from LoRA targets:
# "When NemotronHMamba2Mixer uses cuda_kernels_forward, out_proj LoRA has no gradient."
# Using "all-linear" wastes rank budget on zero-gradient modules!
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
    "in_proj",                                  # Mamba-2 input projection (KEEP)
    # "out_proj",                               # EXCLUDED: no gradient in Mamba-2!
    "up_proj", "down_proj", "gate_proj",        # MLP / MoE layers
]

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,          # 128:32 = 4:1 ratio (NVIDIA standard)
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.0,               # NVIDIA uses 0 dropout for LoRA
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Triton compiler fix
try:
    import triton.backends.nvidia.compiler as nv_compiler
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
    nv_compiler.get_ptxas_version = lambda arch: "12.0"
except Exception as e:
    print(f"Triton compiler fix skipped: {e}")


# ============================================================
# 11. TRAINING
# ============================================================
# Enable tf32 for ~2x matmul speedup on Ampere/Blackwell GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    bf16=True,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",
    lr_scheduler_type="cosine",
    warmup_steps=100,             # ← more warmup for lower LR
    save_strategy="no",
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[LiveProgressCallback()],
)

total_steps = trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else len(hf_dataset) // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nStarting training ({len(hf_dataset)} samples, ~{total_steps} steps)...")
print(f"  batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, eff_batch={BATCH_SIZE*GRAD_ACCUM}")
print(f"  packing=True, fused_adamw=True, tf32=True, workers=4\n")

t0 = time.time()
trainer.train()
print(f"\nDone. Time: {(time.time() - t0) / 3600:.2f} hrs")


# ============================================================
# 12. SAVE — adapter only, NO tokenizer (breaks vLLM eval)
# ============================================================
# This is the ONLY output this notebook produces. The submission notebook
# picks it up from /kaggle/input/<this-notebook-slug>/adapter/
trainer.model.save_pretrained(OUTPUT_DIR)

# Fix base_model_name_or_path to canonical name (so the submit notebook
# doesn't have to — but it patches it again anyway for safety)
config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(config_path) as f:
    adapter_config = json.load(f)

adapter_config["base_model_name_or_path"] = "metric/nemotron-3-nano-30b-a3b-bf16"

with open(config_path, "w") as f:
    json.dump(adapter_config, f, indent=2)

print(f"base_model_name_or_path -> {adapter_config['base_model_name_or_path']}")

# Verify weights look trained
try:
    from safetensors import safe_open
    with safe_open(os.path.join(OUTPUT_DIR, "adapter_model.safetensors"),
                   framework="pt") as f:
        keys  = list(f.keys())
        norms = [f.get_tensor(k).norm().item() for k in keys[:5]]
    print(f"Weight norms (first 5): {[f'{n:.4f}' for n in norms]}")
    if all(n < 0.001 for n in norms):
        print("WARNING: Norms near 0 — do NOT use.")
    else:
        print("✓ Adapter looks healthy.")
except Exception as e:
    print(f"Could not verify: {e}")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Adapter saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Click 'Save Version' → 'Save & Run All (Commit)'")
print("  2. Wait for the commit to finish")
print("  3. In nemotron_v5_submit_only notebook:")
print("     Add Data → Notebook Output → pick THIS notebook")
print("  4. Run the submission notebook → submit submission.zip")
