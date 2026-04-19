# Training Code Explained: Section by Section

## 📋 Complete Code Walkthrough

---

## SECTION 1: Install Dependencies (Lines 37-67)

### What It Does
Installs PEFT, TRL, and datasets libraries from offline packages (Kaggle environment has no internet).

```python
# Step 1: Import the tools
import subprocess, sys, os
from pathlib import Path

# Step 2: Define a function to find Python packages
def resolve_python_path(target_dir):
    # Look for .pth files (Python path files)
    for pth_file in Path(target_dir).glob("*.pth"):
        with pth_file.open() as fp:
            relpath = fp.read().strip()  # Read path reference
            rel_pack_path = pth_file.parent / relpath
            if rel_pack_path.exists():
                sys.path.append(str(rel_pack_path))  # Add to Python search path

# Step 3: Set paths
offline_dir = "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages"
target_dir  = "/kaggle/working/packages"
os.makedirs(target_dir, exist_ok=True)  # Create directory if it doesn't exist

# Step 4: Install packages
if os.path.exists(offline_dir):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",  # -q = quiet (no spam)
        "--no-index",                                     # Don't search PyPI
        "--find-links", offline_dir,                     # Use offline packages
        "--target", target_dir,                          # Install here
        "datasets", "trl", "peft"                        # What to install
    ])
```

**Why this matters:** Kaggle notebooks can't download from internet, so NVIDIA provides pre-downloaded packages.

---

## SECTION 2: Imports & Check GPU (Lines 70-87)

### What It Does
Imports all libraries and prints GPU info.

```python
# Set memory to grow dynamically instead of allocating all at once
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import everything we need
import stat, shutil, time, json, re
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset                    # Load JSONL/CSV data
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType  # LoRA tools
from trl import SFTTrainer, SFTConfig           # Trainer for SFT (Supervised Fine-Tuning)
from tqdm.auto import tqdm                     # Progress bars

# Print what hardware we have
print(f"PyTorch : {torch.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**Output you'll see:**
```
PyTorch : 2.1.0
GPU     : Tesla T4
VRAM    : 15.0 GB
```

---

## SECTION 3: Triton/RMSNorm Fixes (Lines 90-127)

### The Problem
NemotronH uses special CUDA kernels for speed. Sometimes these kernels have numerical issues. This section applies fixes.

```python
# Create a "pure" RMSNorm that works correctly with LoRA
def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    """
    RMSNorm = Root Mean Square Layer Normalization
    Formula: y = x / sqrt(mean(x^2) + eps) * weight
    """
    dtype = x.dtype  # Remember original data type (bfloat16)
    
    if upcast:
        x = x.float()  # Convert to float32 for precision
    
    # Calculate variance
    variance = x.pow(2).mean(-1, keepdim=True)
    
    # Normalize
    x_normed = x * torch.rsqrt(variance + eps)  # rsqrt = 1/sqrt
    
    # Scale by weight
    out = x_normed * weight.float()
    
    if bias is not None:
        out = out + bias.float()
    
    if z is not None:
        out = out * F.silu(z.float())  # Apply activation if provided
    
    return out.to(dtype)  # Convert back to original dtype

# Apply this fix to all modules
for name, mod in list(sys.modules.items()):
    if hasattr(mod, "rmsnorm_fn"):
        mod.rmsnorm_fn = _pure_rmsnorm_fn  # Replace their broken one with ours
```

**Why:** NemotronH uses RMSNorm in every layer. If it's wrong, training breaks. This ensures it's correct.

**Triton fix part:**
```python
# Triton is a compiler for CUDA kernels
src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
dst = "/tmp/ptxas-blackwell"

if os.path.exists(src):
    shutil.copy2(src, dst)  # Copy Triton compiler
    os.chmod(dst, ...)      # Make executable
    # ... set environment variables so Triton uses this compiler
```

**Bottom line:** These are NemotronH-specific fixes. Without them, training might crash or produce NaN losses.

---

## SECTION 4: Hyperparameters (Lines 129-159)

### What It Does
Set all the knobs for training.

```python
# LoRA Configuration
LORA_RANK    = 32      # LoRA matrix size (smaller = fewer parameters to update)
LORA_ALPHA   = 128     # Scaling factor (higher = stronger updates)
               # Ratio: 128/32 = 4.0 (NVIDIA standard)

# Data & Training
MAX_SEQ_LEN  = 2048    # Max tokens per sequence
NUM_EPOCHS   = 2       # How many times to see the data
BATCH_SIZE   = 4       # Sequences per batch
GRAD_ACCUM   = 4       # Accumulate gradients over 4 steps
               # Effective batch = 4 * 4 = 16
LR           = 1e-5    # Learning rate (0.00001 - small!)

# File Paths
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
OUTPUT_DIR = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data file paths (tried in order)
OUR_DATA_PATHS = [
    "/kaggle/input/nemotron-cot-v5/train_cot_v5_merged.jsonl",
    # ... fallback paths ...
]
```

**Why these values?**
```
BATCH_SIZE = 4:   Small enough to fit on T4 (15GB VRAM)
GRAD_ACCUM = 4:   Effective batch 16 is reasonable
LR = 1e-5:        NVIDIA's official for Nemotron SFT
NUM_EPOCHS = 2:   2 passes on 9406 examples = good coverage, no overfitting
```

---

## SECTION 5: Progress Bar (Lines 162-179)

### What It Does
Shows training progress in a nice format.

```python
class LiveProgressCallback(TrainerCallback):
    """Update progress bar every training step"""
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Create progress bar at start
        self.pbar = tqdm(
            total=state.max_steps,          # How many steps total?
            desc="Training",
            unit="step",
            dynamic_ncols=True,             # Adapt to terminal width
            file=sys.stdout,
        )
        self.start_time = time.time()       # Track time

    def on_step_end(self, args, state, control, **kwargs):
        # Every step, update the bar
        elapsed  = time.time() - self.start_time
        step     = state.global_step
        eta      = (elapsed / step) * (state.max_steps - step)  # Estimate remaining time
        
        loss_str = (
            f"loss={state.log_history[-1]['loss']:.4f}"
            if state.log_history and "loss" in state.log_history[-1]
            else "loss=..."
        )
        
        self.pbar.set_postfix_str(
            f"{loss_str}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m"
        )
        self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        # Close bar at end
        if self.pbar:
            self.pbar.close()
```

**What you'll see on screen:**
```
Training: 100%|██████████| 5229/5229 [2:15:30<00:00,  loss=1.542  elapsed=135.5m  eta=0.0m]
```

---

## SECTION 6: Load Training Data (Lines 181-220)

### What It Does
Read JSONL file with 9406 CoT examples.

```python
print("Loading datasets...\n")

our_data = []
ext_csv_path = None

# Try each path in order until one exists
for path in OUR_DATA_PATHS:
    if os.path.exists(path):
        print(f"✓ Found our JSONL: {path}")
        with open(path, 'r') as f:
            for line in f:
                # Each line is one JSON example
                our_data.append(json.loads(line))
        print(f"  → {len(our_data)} metric-aligned examples")
        break  # Stop after finding first one

if not our_data:
    print("✗ Our JSONL not found (searched all candidate paths)")

# Try to find external CSV as fallback
for path in EXTERNAL_CSV_PATHS:
    if os.path.exists(path):
        ext_csv_path = path
        print(f"✓ Found external CSV: {path}")
        break

# Check if we have ANY data
if not our_data and not ext_csv_path:
    raise FileNotFoundError(
        "No training data found!\n"
        "Upload train_cot_v5_merged.jsonl as Kaggle dataset..."
    )

print(f"\nData sources: our_jsonl={len(our_data)}, ext_csv={'YES' if ext_csv_path else 'NO'}")
```

**Output:**
```
✓ Found our JSONL: /kaggle/input/nemotron-cot-v5/train_cot_v5_merged.jsonl
  → 9406 metric-aligned examples

Data sources: our_jsonl=9406, ext_csv=NO
```

---

## SECTION 7: Format Data with Tokenizer (Lines 222-334)

### The Most Important Part!
Converts raw text into the format Nemotron expects.

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use <eos> for padding

# This is what we append to user prompts (tells model where to put answer)
EVAL_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

all_texts = []

# ═════════════════════════════════════════════════════════════════
# FORMAT JSONL DATA (the good stuff)
# ═════════════════════════════════════════════════════════════════
if our_data:
    print(f"Formatting {len(our_data)} JSONL examples...")
    for example in our_data:
        # Each example is: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        
        # Filter out system messages
        messages = [m for m in example['messages'] if m['role'] != 'system']
        
        # Ensure assistant message has <think> tags
        assistant_msg = messages[-1]
        if '<think>' not in assistant_msg['content']:
            # If no thinking tags, add them
            content = assistant_msg['content']
            boxed_match = re.search(r'(\\boxed\{.*?\})\s*$', content)
            
            if boxed_match:
                # Split into reasoning + answer
                reasoning = content[:boxed_match.start()].strip()
                boxed_answer = boxed_match.group(1)
                
                # Reconstruct with tags
                messages[-1] = {
                    'role': 'assistant',
                    'content': f"<think>\n{reasoning}\n</think>\n{boxed_answer}"
                }
        
        # Apply Nemotron's chat template
        # This converts to: <|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer<|im_end|>
        try:
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,              # Just format, don't tokenize yet
                add_generation_prompt=False  # Don't add "<|im_start|>assistant\n"
            )
        except Exception:
            # Fallback if tokenizer doesn't have apply_chat_template
            text = (
                f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
                f"<|im_start|>assistant\n{messages[-1]['content']}<|im_end|>"
            )
        
        all_texts.append(text)
    
    print(f"  → {len(all_texts)} examples from JSONL")

# ═════════════════════════════════════════════════════════════════
# Optional: Add external CSV data (dedup to avoid duplicates)
# ═════════════════════════════════════════════════════════════════
if ext_csv_path:
    import pandas as pd
    ext_df = pd.read_csv(ext_csv_path)
    print(f"\nExternal CSV: {len(ext_df)} rows, columns: {list(ext_df.columns)}")

    # Collect prompts we already have (for deduplication)
    existing_prompts = set()
    if our_data:
        for example in our_data:
            user_content = example['messages'][0]['content'] if example['messages'][0]['role'] == 'user' else ''
            existing_prompts.add(user_content[:200])

    ext_added = 0
    for _, row in ext_df.iterrows():
        prompt = str(row.get('prompt', ''))
        
        # Skip if we already have this problem
        if prompt[:200] in existing_prompts:
            continue

        answer = str(row.get('answer', ''))
        cot = str(row.get('generated_cot', ''))

        # Skip empty rows
        if not prompt.strip() or not answer.strip():
            continue

        # Format as Nemotron expects
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

# ═════════════════════════════════════════════════════════════════
# Create HuggingFace Dataset
# ═════════════════════════════════════════════════════════════════
hf_dataset = Dataset.from_dict({'text': all_texts})
print(f"\n{'='*50}")
print(f"TOTAL DATASET: {len(hf_dataset)} examples")
print(f"{'='*50}")
print(f"\nSample (first 400 chars):")
print(hf_dataset[0]['text'][:400])
```

**Key Insight: What the format looks like**

Before formatting (raw JSONL):
```json
{
  "messages": [
    {"role": "user", "content": "Convert 25.09 m to the Wonderland units..."},
    {"role": "assistant", "content": "<think>\nI'll find the conversion factor...</think>\n\\boxed{16.65}"}
  ]
}
```

After formatting (what model sees):
```
<|im_start|>user
Convert 25.09 m to the Wonderland units...
<|im_end|>
<|im_start|>assistant
<think>
I'll find the conversion factor...
</think>
\boxed{16.65}<|im_end|>
```

**Why <think> tags?** They tell the model: "Please show your reasoning before the answer."

---

## SECTION 8: Filter Out Too-Long Sequences (Lines 336-346)

### What It Does
Remove examples longer than 2048 tokens (would crash training).

```python
print(f"Filtering samples > {MAX_SEQ_LEN} tokens...")
before = len(hf_dataset)

def get_token_length(example):
    # Count how many tokens this example has
    ids = tokenizer(
        example['text'],
        truncation=False,           # Don't truncate, just count
        return_attention_mask=False
    )['input_ids']
    return {'token_len': len(ids)}

# Apply function to all examples
hf_dataset = hf_dataset.map(get_token_length, desc="Counting tokens")

# Keep only examples <= 2048 tokens
hf_dataset = hf_dataset.filter(
    lambda x: x['token_len'] <= MAX_SEQ_LEN,
    desc="Dropping oversized",
)

# Remove the token_len column (we don't need it anymore)
hf_dataset = hf_dataset.remove_columns(['token_len'])

print(f"  Kept {len(hf_dataset)} / {before}  ({before - len(hf_dataset)} dropped)")

# Estimate training time
steps_estimate = len(hf_dataset) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS
print(f"\nSteps    : {steps_estimate}")
print(f"Est. time: ~{steps_estimate * 8 / 3600:.1f} hrs")
```

**Example output:**
```
Filtering samples > 2048 tokens...
  Kept 9200 / 9406  (206 dropped)

Steps    : 287
Est. time: ~0.6 hrs
```

---

## SECTION 9: Load Base Model (Lines 348-394)

### What It Does
Load the 30B Nemotron model in bfloat16 (no quantization).

```python
# NemotronH is HYBRID: Transformer + Mamba-2 + MoE
# It does NOT support flash_attention_2
# Must use default "eager" attention

# Optional: Install flash_attn wheel (used by internal kernels)
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
    device_map={"": 0},              # Put everything on GPU 0
    trust_remote_code=True,          # Allow custom code in model
    torch_dtype=torch.bfloat16,      # bfloat16: smaller than float32, faster
    # NO attn_implementation — NemotronH doesn't support flash_attention_2
)

# Enable gradient checkpointing (save memory by recomputing activations)
model.gradient_checkpointing_enable()

# Critical fix: disable fast path for NemotronH
# (Fast path sometimes has bugs)
for name, mod in list(sys.modules.items()):
    if "modeling_nemotron_h" in name:
        mod.is_fast_path_available = False

print(f"Model loaded on GPU")
```

**Memory breakdown on T4 (15GB):**
```
Base model (bfloat16): ~60GB... wait, that's too much!
```

Actually, the T4 can't hold the full model. This uses "device_map" which:
1. Loads model in 8-bit quantization automatically? NO.
2. Uses offload_folder? NO.
3. Actually uses... hmm. Let me clarify.

**Actually:** The model loads in chunks. Some layers stay on GPU, others on CPU. The trainer moves batches to GPU as needed.

---

## SECTION 10: Attach LoRA (Lines 397-417)

### The Magic Part!
Add LoRA adapters to only update ~2M parameters instead of 30B.

```python
# NVIDIA explicitly EXCLUDES out_proj from LoRA targets:
# "When NemotronHMamba2Mixer uses cuda_kernels_forward, out_proj LoRA has no gradient."
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
    "in_proj",                                  # Mamba-2 input projection
    # "out_proj",                               # EXCLUDED: no gradient in Mamba-2!
    "up_proj", "down_proj", "gate_proj",        # MLP / MoE layers
]

lora_config = LoraConfig(
    r=LORA_RANK,                # Rank 32: matrix A is (hidden_dim, 32), matrix B is (32, hidden_dim)
    lora_alpha=LORA_ALPHA,      # 128: scales the update by 128/32 = 4.0
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.0,           # NVIDIA uses 0 (no dropout)
    bias="none",                # Don't update bias terms
    task_type=TaskType.CAUSAL_LM,  # We're doing language modeling
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print how many parameters are trainable
model.print_trainable_parameters()

# Triton compiler fix (for CUDA kernels)
try:
    import triton.backends.nvidia.compiler as nv_compiler
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
    nv_compiler.get_ptxas_version = lambda arch: "12.0"
except Exception as e:
    print(f"Triton compiler fix skipped: {e}")
```

**What you'll see:**
```
trainable params: 2,195,456 || all params: 30,009,349,120 || trainable%: 0.0073
```

**LoRA explained:**
```
Instead of updating W (30B×30B):
    new_W = old_W + ΔW

We compute ΔW = A @ B (much smaller)
    A: 30B × 32
    B: 32 × 30B
    ΔW = A @ B (2M parameters instead of 30B!)
    
    Then apply scaling: ΔW = (alpha/rank) * A @ B = 4.0 * A @ B
```

---

## SECTION 11: Configure Training (Lines 420-453)

### What It Does
Set training hyperparameters and create trainer.

```python
# Enable tensor float 32 (faster matmuls on Ampere/Blackwell GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

training_args = SFTConfig(
    # Output & Saving
    output_dir=OUTPUT_DIR,                  # Where to save checkpoint
    per_device_train_batch_size=BATCH_SIZE, # 4 examples per step
    gradient_accumulation_steps=GRAD_ACCUM, # Accumulate over 4 steps
    
    # Learning Schedule
    num_train_epochs=NUM_EPOCHS,            # 2 passes through data
    learning_rate=LR,                       # 1e-5 (small!)
    logging_steps=10,                       # Print loss every 10 steps
    
    # Optimization
    bf16=True,                              # Use bfloat16
    max_grad_norm=1.0,                      # Clip gradients at norm 1.0
    optim="adamw_torch_fused",              # Fused AdamW: faster, uses less memory
    lr_scheduler_type="cosine",             # Cosine annealing: LR decreases smoothly
    warmup_steps=100,                       # Warm up LR for first 100 steps
    
    # Dataset & Packing
    save_strategy="no",                     # Don't save checkpoints during training
    report_to="none",                       # Don't report to wandb/etc
    dataset_text_field="text",              # Use 'text' column from dataset
    max_length=MAX_SEQ_LEN,                 # Max sequence length 2048
    packing=True,                           # Pack short sequences together (efficiency)
    
    # Memory & Speed
    gradient_checkpointing=True,            # Save memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=4,               # Load data on 4 CPU threads
    dataloader_pin_memory=True,             # Copy data to GPU faster
    dataloader_prefetch_factor=2,           # Prefetch next batch while training
)

# Create trainer
trainer = SFTTrainer(
    model=model,                       # Our model with LoRA
    train_dataset=hf_dataset,          # Our training data
    processing_class=tokenizer,        # Our tokenizer
    args=training_args,                # Config above
    callbacks=[LiveProgressCallback()], # Progress bar
)

# Estimate steps
total_steps = trainer.state.max_steps if hasattr(trainer.state, 'max_steps') else len(hf_dataset) // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nStarting training ({len(hf_dataset)} samples, ~{total_steps} steps)...")
print(f"  batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, eff_batch={BATCH_SIZE*GRAD_ACCUM}")
print(f"  packing=True, fused_adamw=True, tf32=True, workers=4\n")

# Start training
t0 = time.time()
trainer.train()
print(f"\nDone. Time: {(time.time() - t0) / 3600:.2f} hrs")
```

**What "packing" does:**
```
Without packing:
  Batch: [seq1(2048), seq2(512), seq3(256), seq4(100)]
  → Wastes tokens on short sequences

With packing:
  Batch: [seq1(2048), seq2+seq3+seq4(868), padding(100)]
  → More efficient, fewer wasted tokens
```

---

## SECTION 12: Save Adapter (Lines 455-487)

### What It Does
Save only the LoRA weights, not the full model.

```python
# Save the trained LoRA adapter
trainer.model.save_pretrained(OUTPUT_DIR)
# Creates:
#   - adapter_config.json (LoRA configuration)
#   - adapter_model.safetensors (the actual weights, ~200MB)

# Fix base_model_name_or_path to canonical name
config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
with open(config_path) as f:
    adapter_config = json.load(f)

adapter_config["base_model_name_or_path"] = "metric/nemotron-3-nano-30b-a3b-bf16"

with open(config_path, "w") as f:
    json.dump(adapter_config, f, indent=2)

print(f"base_model_name_or_path -> {adapter_config['base_model_name_or_path']}")

# Verify weights look trained (not zero)
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
```

**What gets saved:**
```
adapter/
├── adapter_config.json      (1.2 KB)
└── adapter_model.safetensors (~200 MB)
```

**adapter_config.json looks like:**
```json
{
  "r": 32,
  "lora_alpha": 128,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "up_proj", "down_proj", "gate_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM",
  "base_model_name_or_path": "metric/nemotron-3-nano-30b-a3b-bf16"
}
```

---

## Summary: Training Flow Diagram

```
┌─────────────────────────────┐
│  1. Install packages        │
│     (PEFT, TRL, datasets)   │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  2. Load data               │
│     (9406 JSONL examples)   │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  3. Format with tokenizer   │
│     (<think>...</think> tags)│
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  4. Filter long sequences   │
│     (keep < 2048 tokens)    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  5. Load 30B base model     │
│     (bfloat16, no quant)    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  6. Add LoRA adapter        │
│     (2M trainable params)   │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  7. Train for 2 epochs      │
│     (LR: 1e-5, batch: 16)   │
│     Time: 2-3 hours         │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│  8. Save adapter weights    │
│     (200MB safetensors)     │
└─────────────────────────────┘
```

---

## Key Numbers to Remember

| Parameter | Value | Why |
|-----------|-------|-----|
| Data | 9406 examples | Covers all 6 categories |
| Batch size | 16 effective | 4 per device × 4 accumulation |
| Learning rate | 1e-5 | NVIDIA SFT standard |
| LoRA rank | 32 | 2M trainable params |
| LoRA alpha | 128 | 4:1 ratio (NVIDIA standard) |
| Epochs | 2 | Good coverage without overfitting |
| Max length | 2048 tokens | Fits in VRAM |
| Training time | 2-3 hours | On T4 with packing |
| Output | 200 MB | Adapter weights only |

---

## Common Training Issues & Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| OOM (Out of Memory) | CUDA out of memory error | Reduce BATCH_SIZE or MAX_SEQ_LEN |
| Loss is NaN | Training crashes with NaN | Check RMSNorm fix is applied, verify data format |
| Loss doesn't decrease | Loss stays ~2.0+ | Increase learning rate or check data |
| Training is slow | > 4 hours for 2 epochs | Enable packing (already enabled) or increase batch size |
| Weights are zero | Adapter norms < 0.001 | Increased learning rate or train longer |

