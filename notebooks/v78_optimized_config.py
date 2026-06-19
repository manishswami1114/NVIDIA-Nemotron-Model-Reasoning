# ============================================================
# 4. HYPERPARAMETERS — v7.8 (SPEED-OPTIMIZED, ALL 9 CATEGORIES KEPT)
# ============================================================
# Changes from v7.7 for speed (NO methodology/data changes):
#   - BATCH_SIZE 2 -> 4      (uses the 35GB VRAM headroom for throughput)
#   - GRAD_ACCUM 1 -> 2      (eff batch=8, smoother gradients)
#   - NUM_EPOCHS 1 -> 3      (user requirement)
#   - LR 1e-4 -> 5e-5        (lower for 3-epoch stability)
#   - WARMUP_STEPS 50 -> 30  (proportional to fewer steps/epoch)
#   - save_only_model=True   (FIX: avoids Kaggle disk-space crash on optimizer state)
#   - SAVE_EVERY_N_STEPS 200 -> 500 (fewer disk writes)
#
# UNCHANGED: MAX_SEQ_LEN=6144, all 9 categories, MoE tied, CCE, fast path
#
# Memory budget (bf16 + fast path + CCE, batch=4, seq=6144, GC=on):
#   60 GB base + ~12 GB acts (2x batch) + ~3 GB Mamba (fast path) +
#   ~1.5 GB optim = ~76.5 GB peak → ~18 GB headroom on 95 GB ✓
#
# WHY THE ORIGINAL CRASHED (disk space, NOT code):
#   Optimizer state for 884M trainable params = ~3.3 GB per checkpoint.
#   save_steps=200 × save_total_limit=3 = ~10 GB of checkpoints.
#   Kaggle /kaggle/working has limited disk. Adding adapter zips +
#   wandb logs fills it up → "iostream error" when writing optimizer.pt.
#   FIX: save_only_model=True (skip optimizer state, saves ~3.3 GB/ckpt)

LORA_RANK           = 32          # eval-server cap (matches 0.85)
LORA_ALPHA          = 64          # 2:1 vs their 1:1 → sharper adaptation
LORA_DROPOUT        = 0.0         # eval-server contract (vLLM math)

MAX_SEQ_LEN         = 6144        # UNCHANGED — keeps all bit_manipulation data
NUM_EPOCHS          = 3           # 3 epochs for proper convergence
BATCH_SIZE          = 4           # 2x throughput (was 2), fits with 18GB headroom
GRAD_ACCUM          = 2           # eff batch=8, smooth gradients for multi-epoch
LR                  = 5e-5        # lower for 3 epochs (alpha/r=2 already aggressive)
WARMUP_STEPS        = 30
SAVE_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_STEPS  = 500         # fewer checkpoints = less disk pressure

USE_PACKING             = True    # packs short samples into 6144-token sequences → 3-5x speedup
USE_STRATIFIED_BATCHING = False   # incompatible with packing; packing wins for speed
USE_CCE                 = True    # Cut Cross-Entropy (saves ~17 GB)
USE_MAMBA_FAST_PATH     = True    # Fused CUDA scan (saves ~30 GB)

# MoE LoRA strategy (UNCHANGED)
MOE_LORA_MODE = "tied"            # MoE weight tying (best memory/quality tradeoff)

# Mode selector — bf16 native is fastest on Blackwell tensor cores
FORCE_MODE = "bf16"

if FORCE_MODE:
    MODE = FORCE_MODE
else:
    MODE = "bf16"

if MODE == "nf4":
    LR = 2e-4
    print("NF4 mode: LR=2e-4")
else:
    print(f"bf16 mode (FAST native tensor cores on Blackwell): LR={LR:.1e}")

USE_QLORA = MODE in ("nf4", "int8")

# Disable fast path if wheels not present (Cell 1 set FAST_PATH_AVAILABLE)
if USE_MAMBA_FAST_PATH and not FAST_PATH_AVAILABLE:
    print("[warn] USE_MAMBA_FAST_PATH=True but wheels missing — forcing OFF")
    USE_MAMBA_FAST_PATH = False
    print("       This will cost ~30 GB. Reducing MAX_SEQ_LEN -> 4096, BATCH_SIZE -> 1")
    MAX_SEQ_LEN = 4096
    BATCH_SIZE  = 1

# Disable CCE if not installed
if USE_CCE and not CCE_AVAILABLE:
    print("[warn] USE_CCE=True but cut_cross_entropy missing — forcing OFF")
    USE_CCE = False

MODEL_PATH  = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
OUTPUT_DIR  = "/kaggle/working/adapter"
CKPT_DIR    = "/kaggle/working/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ALL 9 CATEGORIES KEPT — no data removed
CATEGORY_FILES = [
    "train_cot_bit_manipulation.jsonl",
    "train_cot_cipher.jsonl",
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_numeral.jsonl",
    "train_cot_unit_conversion.jsonl",
]

DATA_DIR_CANDIDATES = [
    "/kaggle/input/datasets/asharamkanderiwal/nvidia-dataset/all_categorical_splits",
    str(Path.cwd().parent / "data" / "processed" / "all_categorical_splits"),
    str(Path.cwd() / "data" / "processed" / "all_categorical_splits"),
]

assert LORA_RANK   <= 32,   f"LORA_RANK={LORA_RANK} exceeds eval cap 32"
assert MAX_SEQ_LEN <= 8192, f"MAX_SEQ_LEN={MAX_SEQ_LEN} exceeds eval max_model_len 8192"

print("=" * 60)
print("  v7.8 CONFIG — speed-optimized, all 9 categories")
print("=" * 60)
print(f"  Mode             : {MODE}")
print(f"  Epochs           : {NUM_EPOCHS}  (epoch ckpt every {SAVE_EVERY_N_EPOCHS}, step ckpt every {SAVE_EVERY_N_STEPS})")
print(f"  LR               : {LR:.1e}  (warmup {WARMUP_STEPS}, cosine)")
print(f"  Batch            : {BATCH_SIZE}×{GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} eff")
print(f"  LoRA             : r={LORA_RANK}, α={LORA_ALPHA}  (2:1 ratio)")
print(f"  Max seqlen       : {MAX_SEQ_LEN}")
print(f"  Packing          : {USE_PACKING}")
print(f"  Mamba fast path  : {USE_MAMBA_FAST_PATH}  (~30 GB saved)")
print(f"  Cut Cross-Entropy: {USE_CCE}              (~17 GB saved)")
print(f"  MoE LoRA mode    : {MOE_LORA_MODE}")
print(f"  Optimizer        : torch.optim.AdamW (Blackwell-stable)")
print(f"  Ckpt dir         : {CKPT_DIR}")
print(f"  Categories       : {len(CATEGORY_FILES)} (ALL kept)")
