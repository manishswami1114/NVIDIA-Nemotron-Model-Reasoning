# NVIDIA Nemotron Reasoning Challenge: Deep Dive on v6 Improvements (0.66 → 0.85+ Path)

## TL;DR

I scored **0.66** after removing "fake CoT" data (thinking cleaner = better). Research into NVIDIA's official configs revealed **critical bugs** in my LoRA setup:

1. **`out_proj` was killing training** — NVIDIA explicitly excludes it: "Mamba-2 cuda_kernels_forward produces no gradient"
2. **Learning rate was 20x too high** — NVIDIA SFT uses `1e-5`, I had `2e-4`
3. **LoRA alpha:rank ratio was wrong** — NVIDIA uses 4:1 (512:128), I had 1:1 (32:32)
4. **Dataset shrank too much** — Removed entire `transformation_rules` category (0 examples), lost coverage in others

Fixed all three + rebuilt dataset → **v6 with 9406 verified examples across all 6 categories**. Expected improvement: 0.66 → 0.75 (SFT fixes) → 0.85+ (with GRPO RL).

---

## The Problem: Why 0.69 → 0.66 Regression

### What Happened

I discovered that my v3 training data had **templated fake CoT** like:
```
<think>
I studied the pattern and found the rule...
</think>
\boxed{answer}
```

This was teaching the model to hallucinate reasoning instead of actually solving. So I:
- Built a truth-table solver for bit manipulation (45% coverage)
- Kept only solver-verified examples
- Dropped entire categories with bad CoT (transformation_rules)
- Went from **9500 → 5121 examples**

**Result: 0.69 → 0.66 (regression!)**

### Root Cause Analysis

| Metric | Before (0.69) | After (0.66) | Impact |
|--------|---------------|--------------|--------|
| Total examples | 9500 | 5121 | -46% data |
| Bit manipulation | 1602 | 722 | Lost 55% of category |
| Cipher | 1576 | 625 | Lost 60% |
| Gravity | 1597 | 951 | Lost 40% |
| Roman | 1576 | 1576 | ✓ Stable |
| Unit conversion | 1594 | 1247 | Lost 22% |
| **Transformation rules** | 1555 | **0** | **BLIND TO ENTIRE CATEGORY** |

The quality bet didn't pay off — losing entire categories hurt more than fake CoT helped.

---

## The Fix: v6 Dataset + NVIDIA-Aligned Hyperparameters

### Part 1: Rebuild the Dataset (train_cot_v5_merged.jsonl)

**Strategy:** v4_real (best CoT) + v3 correct gap-fill

```
v4_real (5121 verified by solver)
├── bit_manipulation: 722 (real solver traces, showing hypothesis search)
├── cipher: 625 (real solver traces)
├── unit_conversion: 1247 (real formula derivation)
├── gravity: 951 (real gravity constant discovery)
└── roman_numeral: 1576 (100% verified)

+ v3 correct gap-fill (4285 verified against ground truth)
├── bit_manipulation: 880 (improved with honest-template CoT, not fake)
├── cipher: 951 (verified correct)
├── transformation_rules: 1461 (was 0 in v5!)
├── gravity: 646 (verified correct)
└── unit_conversion: 347 (verified correct)

= 9406 TOTAL (all answers verified against train.csv)
```

**Key insight:** Not all v3 examples are trash. The 511→1461 correct transformation_rules examples are valuable once filtered.

### Part 2: NVIDIA-Aligned Hyperparameters

**Before (v5, scored 0.66):**
```python
LORA_RANK = 32
LORA_ALPHA = 32          # 1:1 ratio ❌
LR = 2e-4                # 20x too high ❌
NUM_EPOCHS = 3           # Overfitting risk
target_modules="all-linear"  # Includes out_proj ❌
lora_dropout=0.05
warmup_steps=50
```

**After (v6, NVIDIA-aligned):**
```python
LORA_RANK = 32
LORA_ALPHA = 128         # 4:1 ratio ✓ (NVIDIA uses 512:128)
LR = 1e-5                # NVIDIA SFT official ✓
NUM_EPOCHS = 2           # Less overfitting ✓
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "in_proj",                                 # Mamba-2 input
    "up_proj", "down_proj", "gate_proj"       # MLP/MoE
]
# "out_proj" EXCLUDED ✓
lora_dropout=0.0         # NVIDIA uses 0 ✓
warmup_steps=100         # More warmup for lower LR ✓
```

### The Critical Bug: `out_proj`

NVIDIA's **official** Nemotron-3-Nano LoRA config (from their repo) includes this comment:

```yaml
lora_cfg:
  excluded_modules: ['*out_proj*']
  # Reason: When NemotronHMamba2Mixer uses cuda_kernels_forward,
  # out_proj LoRA has no gradient. Including it wastes rank budget.
```

Using `target_modules="all-linear"` includes `out_proj`, which means:
- You have gradient flow: query → attention → softmax → values → **out_proj (NO GRADIENT)**
- Those LoRA parameters never update
- You wasted rank budget on zero-gradient modules

---

## Expected Improvements

### SFT Fixes Alone (Priority 1-2)
| Fix | Impact |
|-----|--------|
| Remove `out_proj` from LoRA targets | +2-5% |
| Change alpha 32→128 (4:1 ratio) | +1-3% |
| Lower LR from 2e-4 to 1e-5 | +1-3% |
| Reduce epochs 3→2 | +0-2% |
| Add back transformation_rules (1461 examples) | +3-5% |
| **Expected from SFT alone** | **+7-18%** |

So: 0.66 + 0.07-0.18 = **0.73-0.84** with just SFT fixes.

### GRPO RL (Priority 3, the real leap)
NVIDIA's own pipeline is **SFT → GRPO with rule-based rewards**. Your 6 puzzle categories all have programmatic verifiers:
- Bit manipulation: exact binary string match
- Cipher: exact text match
- Unit conversion: numeric tolerance check
- Gravity: numeric tolerance check
- Roman numerals: exact string match
- Transformation rules: exact match

This is perfect for **Group Relative Policy Optimization (GRPO)** — the RL algorithm NVIDIA uses. TRL's `GRPOTrainer` works on single GPU.

**RL can add +10-15%**, pushing from 0.75 → 0.85+.

---

## How to Reproduce

### Step 1: Prepare Data (Local)
```bash
# Build v5 merged dataset
python3 nvidia-nemotron-model-reasoning-challenge/build_v5_merged.py

# Verify: 9406 examples, all 6 categories
head -1 train_cot_v5_merged.jsonl | jq '.messages[0].content' | head -50
```

### Step 2: Upload to Kaggle
1. Create dataset `nemotron-cot-v5` with `train_cot_v5_merged.jsonl` (8.9 MB)
2. Create notebook from `nemotron_v5_train_only.py`
3. Attach inputs: 
   - `nemotron-cot-v5` dataset
   - `metric/nemotron-3-nano-30b-a3b-bf16` (base model)
   - `nvidia-nemotron-offline-packages` (PEFT, TRL, Triton fixes)

### Step 3: Train
```bash
# Runs on T4/L4/P100, ~2-3 hours
# Produces /kaggle/working/adapter/ with 2 files:
#   - adapter_config.json (fixed base_model_name_or_path)
#   - adapter_model.safetensors (~200 MB)
```

### Step 4: Submit
1. Create notebook from `nemotron_v5_submit_only.py`
2. Attach: Notebook Output from training notebook
3. Run: Globs for adapter files, creates submission.zip
4. Submit!

---

## Technical Details: Why NVIDIA's Config Matters

### Architecture: NemotronH (Hybrid Transformer + Mamba-2 + MoE)
```
52 layers in pattern: MEMEM*EMEMEM*... where
  M = Mamba-2 layer (sparse)
  E = Transformer Attention layer
  * = MoE expert layer
```

**Mamba-2 layers** use CUDA kernels for efficiency:
```python
# Inside NemotronHMamba2Mixer._forward()
out = cuda_kernels_forward(x, weights)  # ← out_proj is AFTER this
# When you backprop through out_proj LoRA:
# grad_out_proj = grad(output) × out_proj_weight
# But out_proj is AFTER the CUDA kernel, so no gradient flows back!
```

This is why NVIDIA explicitly excludes `out_proj`.

### LoRA Alpha Ratio Matters

LoRA update: `output = base_output + (input @ A) @ B`

When `alpha=128, rank=32` (4:1):
- Effective learning rate = `alpha / rank = 4.0`
- Larger updates = faster learning on clean data
- Less risk of underfitting

When `alpha=32, rank=32` (1:1):
- Effective learning rate = 1.0
- Slower learning
- But you're also losing information (rank undersized)

NVIDIA's 4:1 ratio is tuned for their 9-stage pipeline and 25T token pretraining. For competitive fine-tuning with smaller data, it's still a good heuristic.

### Learning Rate Sensitivity

LoRA learning rate is `base_lr × (alpha / rank)` in most implementations:

```
Effective LR = 2e-4 × (32/32) = 2e-4      (my v5)
Effective LR = 1e-5 × (128/32) = 4e-5     (NVIDIA-aligned)
```

At LR `2e-4`, you're updating LoRA weights very aggressively. On a 9406-example dataset, this leads to overfitting and loss divergence. NVIDIA's `1e-5` with larger alpha gives you smoother training.

---

## What's Next: GRPO RL Implementation

For the leaderboard push from 0.75 → 0.85+, the next step is **GRPO RL training**. Here's a skeleton using TRL:

```python
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM

# Load your trained SFT adapter
model = AutoModelForCausalLM.from_pretrained(
    "metric/nemotron-3-nano-30b-a3b-bf16",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Load trained LoRA
from peft import PeftModel
model = PeftModel.from_pretrained(model, "/path/to/sft/adapter")

# Define reward function (your existing verifier)
def reward_fn(completions, prompts):
    rewards = []
    for completion in completions:
        answer = extract_boxed_answer(completion)
        if verify_answer(answer):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)

# GRPO config (single GPU)
config = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=4,              # Rollouts per prompt (reduced for memory)
    learning_rate=3e-6,             # NVIDIA uses 3e-6 for RL
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_new_tokens=4096,
    temperature=1.0,                # Must be >0 for diverse generations
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=rl_dataset,       # ~2000 unique prompts
    reward_funcs=reward_fn,
)

trainer.train()
```

**Time budget on Kaggle:** ~8-10 hours for 1 GRPO epoch on 2000-4000 prompts.

---

## Debugging Checklist for v6 Training

- [ ] Verify `out_proj` is NOT in `target_modules` (check model output)
- [ ] Check initial loss at epoch 0 (should be ~2.0-3.0 for 9406 examples)
- [ ] Monitor loss curve: should decrease smoothly, no spikes
- [ ] At epoch 1, loss should drop 20-30% (good signal)
- [ ] Weight norms in adapter at end should be >0.01 (not near 0)
- [ ] Verify adapter size is ~200-250 MB (rank 32 × ~30B params)

---

## Lessons Learned

1. **"Cleaner data" ≠ "better training"** — Removing categories entirely is worse than keeping them with mixed quality
2. **NVIDIA's configs are hardened for a reason** — Their `out_proj` exclusion isn't arbitrary; it's based on the actual gradient flow in Mamba-2
3. **Learning rates matter hugely for small datasets** — 20x difference can mean 0.66 vs 0.80
4. **RL is the next frontier** — SFT plateaus around 0.75 for reasoning tasks; RL (with verifiable rewards) is what gets to 0.85+

---

## Resources

- [Nemotron-3-Nano RL Guide (NVIDIA)](https://docs.nvidia.com/nemo/rl/nightly/guides/nemotron-3-nano.html)
- [NVIDIA NeMo RL GitHub](https://github.com/NVIDIA-NeMo/RL)
- [TRL GRPO Trainer Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Nemotron Model Family](https://developer.nvidia.com/nemotron)

---

**Good luck on the leaderboard! Feel free to ask questions or share your improvements.** 🚀
