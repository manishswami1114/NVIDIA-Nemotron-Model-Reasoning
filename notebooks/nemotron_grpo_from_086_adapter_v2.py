# %% [markdown]
# # Nemotron GRPO v2 — KL-Constrained RL From The 0.86 Adapter
#
# Goal: start from the LB 0.86 LoRA adapter and make small GRPO-style
# reward updates without destroying the SFT memorization that already works.
#
# This is intentionally conservative:
#
# - Policy adapter: trainable copy of the 0.86 adapter.
# - Reference adapter: frozen copy of the same 0.86 adapter.
# - KL anchor: sampled per-token log-prob distance to the frozen adapter.
# - LoRA-only updates: base model never trains.
# - Checkpoint every few accepted updates.
#
# Expected use: Kaggle notebook, Internet off, NVIDIA metric utility enabled.

# %%
# ============================================================
# 1. INSTALL DEPENDENCIES — same offline method as the working SFT notebook
# ============================================================
# IMPORTANT:
# - Kaggle internet is OFF.
# - Do NOT uninstall/reinstall torch or transformers.
# - Install only missing helper packages into /kaggle/working/packages.
# - Use --no-deps so pip does not pull incompatible torch/transformers wheels.
# - Append target dir first, then after imports purge utility-script mamba paths.

import os
import sys
import glob
import json
import math
import time
import stat
import random
import shutil
import zipfile
import hashlib
import subprocess
from pathlib import Path
from collections import Counter, defaultdict, deque

TARGET_DIR = "/kaggle/working/packages"
OFFLINE_DIR = "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages"
os.makedirs(TARGET_DIR, exist_ok=True)
if TARGET_DIR not in sys.path:
    sys.path.append(TARGET_DIR)  # append first: Kaggle torch/transformers remain authoritative


def _pip_install(pkgs, *, no_deps=True, no_index=False, find_links=None,
                 path_arg=None, label=None):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--target", TARGET_DIR]
    if no_deps:
        cmd.append("--no-deps")
    if no_index:
        cmd.append("--no-index")
    if find_links:
        cmd += ["--find-links", find_links]
    if path_arg:
        cmd.append(path_arg)
    else:
        cmd += pkgs
    try:
        subprocess.check_call(cmd)
        if label:
            print(f"[ok] {label}")
        return True
    except Exception as e:
        if label:
            print(f"[warn] {label} failed: {e}")
        return False


def _find_wheel(pattern, search_paths):
    for base in search_paths:
        if os.path.isdir(base):
            for f in glob.glob(f"{base}/**/{pattern}", recursive=True):
                return f
    return None


# nvidia-cutlass must be installed before CUDA-heavy imports if present.
CUTLASS_PATHS = ["/kaggle/input/datasets/rubyducklove/nvidia-cutlass"]
cutlass_wheel = (
    _find_wheel("nvidia_cutlass-*.whl", CUTLASS_PATHS)
    or _find_wheel("cutlass-*.whl", CUTLASS_PATHS)
)
CUTLASS_AVAILABLE = bool(cutlass_wheel) and _pip_install(
    [], path_arg=cutlass_wheel, label=f"nvidia-cutlass <- {cutlass_wheel}"
)

# GRPO only needs peft/accelerate at runtime, but installing the same small
# helper set as the working SFT notebook avoids missing-package surprises.
PKG_LIST = ["trl", "peft", "datasets", "bitsandbytes", "wandb", "cut-cross-entropy"]
if os.path.isdir(OFFLINE_DIR):
    _pip_install(PKG_LIST, no_index=True, find_links=OFFLINE_DIR,
                 label=f"core deps (offline): {PKG_LIST}")
else:
    _pip_install(PKG_LIST, label=f"core deps (online fallback): {PKG_LIST}")

# Blackwell Mamba CUDA wheels.
WHEEL_PATHS = ["/kaggle/input/datasets/mayukh18/nemotron-packages"]
ccv_wheel = _find_wheel("causal_conv1d-*.whl", WHEEL_PATHS)
mssm_wheel = _find_wheel("mamba_ssm-*.whl", WHEEL_PATHS)
CAUSAL_CONV1D_AVAILABLE = bool(ccv_wheel) and _pip_install(
    [], path_arg=ccv_wheel, label=f"causal_conv1d <- {ccv_wheel}"
)
MAMBA_AVAILABLE = bool(mssm_wheel) and _pip_install(
    [], path_arg=mssm_wheel, label=f"mamba_ssm <- {mssm_wheel}"
)
FAST_PATH_AVAILABLE = MAMBA_AVAILABLE and CAUSAL_CONV1D_AVAILABLE


def _resolve_pth(d):
    for pth in Path(d).glob("*.pth"):
        with pth.open() as fp:
            rel = fp.read().strip()
            p = pth.parent / rel
            if p.exists():
                sys.path.append(str(p))


_resolve_pth(TARGET_DIR)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"

import transformers
TRANSFORMERS_VERSION = tuple(int(x) for x in transformers.__version__.split(".")[:2])
TRANSFORMERS_PATH = transformers.__file__
NEW_ENOUGH = TRANSFORMERS_VERSION >= (4, 45)

if not NEW_ENOUGH:
    raise RuntimeError(
        f"Kaggle transformers {transformers.__version__} is too old for Nemotron-H. "
        "Use the same Kaggle image/input setup as the working SFT notebook."
    )

# Purge Kaggle utility-script mamba_ssm so our installed Blackwell wheel wins.
_BAD_PATH_FRAGS = ("nvidia_utility_script", "nvidia-utility-script")
sys.path[:] = [p for p in sys.path if not any(b in p for b in _BAD_PATH_FRAGS)]
for _m in list(sys.modules):
    _mfile = getattr(sys.modules[_m], "__file__", "") or ""
    if any(b in _mfile for b in _BAD_PATH_FRAGS):
        del sys.modules[_m]
if TARGET_DIR in sys.path:
    sys.path.remove(TARGET_DIR)
sys.path.insert(0, TARGET_DIR)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Defensive Triton ptxas shim from the working SFT notebook.
candidates = (
    glob.glob("/kaggle/usr/lib/notebooks/**/ptxas-blackwell", recursive=True)
    + glob.glob("/kaggle/usr/lib/notebooks/**/ptxas", recursive=True)
    + glob.glob("/usr/local/cuda*/bin/ptxas", recursive=True)
    + glob.glob("/usr/local/lib/python*/dist-packages/nvidia/cuda_nvcc/bin/ptxas",
                recursive=True)
)
src = next((c for c in candidates if "blackwell" in c), None) or (candidates[0] if candidates else None)
if src and os.path.exists(src):
    dst = "/tmp/ptxas-blackwell"
    shutil.copy2(src, dst)
    os.chmod(dst, os.stat(dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    for v in ("TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH",
              "TRITON_PTXAS_BIN", "TRITON_PTXAS"):
        os.environ[v] = dst
    try:
        import triton.backends.nvidia.compiler as nv_compiler
        try:
            nv_compiler.get_ptxas_version.cache_clear()
        except AttributeError:
            pass
        nv_compiler.get_ptxas_version = lambda arch: "release 12.8"
        from triton import knobs as triton_knobs
        for attr in ("ptxas", "ptxas_blackwell"):
            triton_knobs.nvidia.__dict__.pop(attr, None)
    except Exception as e:
        print(f"[warn] Triton cache clear: {e}")
    print(f"[ok] ptxas binary -> {dst} (copied from {src})")
else:
    print("[warn] no ptxas binary found — Mamba Triton kernel may crash")

print("=" * 60)
print("Dependency status")
print("=" * 60)
print(f"transformers     : {transformers.__version__} ({TRANSFORMERS_PATH})")
print(f"torch            : {torch.__version__} ({torch.__file__})")
print(f"nvidia-cutlass   : {'YES' if CUTLASS_AVAILABLE else 'NO'}")
print(f"causal_conv1d    : {'YES' if CAUSAL_CONV1D_AVAILABLE else 'NO'}")
print(f"mamba_ssm        : {'YES' if MAMBA_AVAILABLE else 'NO'}")
print(f"Mamba fast path  : {'YES' if FAST_PATH_AVAILABLE else 'NO'}")
print(f"cuda             : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu              : {torch.cuda.get_device_name(0)}")
    print(f"vram             : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# %%
# ============================================================
# 2. CONFIG — edit adapter/data paths here
# ============================================================
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"

# Preferred: set this to the Kaggle input directory containing the 0.86 adapter.
# It must contain adapter_config.json and adapter_model.safetensors.
ADAPTER_PATH = "/kaggle/input/models/manish756/nvidia-adapter/transformers/default/7"

# If ADAPTER_PATH is wrong, the notebook will auto-search /kaggle/input.
AUTO_FIND_ADAPTER = True

# Dataset candidates. The validated-only dataset is preferred if uploaded.
DATA_DIR_CANDIDATES = [
    "/kaggle/input/all-categorical-splits-sequence-v2-validated-only/all_categorical_splits_sequence_v2_validated_only",
    "/kaggle/input/all-categorical-splits-sequence-v2-replaced/all_categorical_splits_sequence_v2_replaced",
    "/kaggle/input/datasets/manish756/nemotron-dataset/all_categorical_splits",
    "/kaggle/input/datasets/asharamkanderiwal/nvidia-dataset/all_categorical_splits",
]

# Train on categories with enough answer-reward density. Numeral is usually
# saturated and can waste steps, so it is off by default.
TRAIN_CATEGORIES = [
    "train_cot_cipher.jsonl",
    "train_cot_equation_numeric_deduce.jsonl",
    "train_cot_equation_numeric_guess.jsonl",
    "train_cot_gravity.jsonl",
    "train_cot_unit_conversion.jsonl",
    "train_cot_cryptarithm_deduce.jsonl",
    "train_cot_cryptarithm_guess.jsonl",
]

# Oversample the categories where the 0.86 model still has realistic upside.
CATEGORY_WEIGHTS = {
    "cipher": 2.5,
    "equation_numeric_deduce": 2.0,
    "equation_numeric_guess": 1.5,
    "gravity": 1.0,
    "unit_conversion": 1.0,
    "cryptarithm_deduce": 0.75,
    "cryptarithm_guess": 0.75,
}

# Rollout settings. Keep max_new_tokens modest; RL log-prob passes are expensive.
K_ROLLOUTS = 4
ROLLOUT_TEMP = 0.8
ROLLOUT_TOP_P = 0.95
MAX_NEW_TOKENS = 1024
MAX_PROMPT_TOKENS = 4096

# Conservative GRPO/RLOO settings.
LR = 7.5e-7
KL_COEF = 0.03
KL_TARGET = 0.08
MAX_KL_PER_STEP = 0.20
ADV_NORM = True
MAX_GRAD_NORM = 0.35
NUM_ACCEPTED_UPDATES = 160
MAX_ATTEMPTED_PROMPTS = 1200

# Checkpointing.
SAVE_EVERY_ACCEPTED = 20
OUTPUT_DIR = "/kaggle/working/grpo_adapter"
CKPT_DIR = "/kaggle/working/grpo_checkpoints"
BEST_DIR = "/kaggle/working/grpo_best_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

SEED = 20260517
random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 72)
print("GRPO v2 config")
print("=" * 72)
for k in [
    "K_ROLLOUTS", "ROLLOUT_TEMP", "ROLLOUT_TOP_P", "MAX_NEW_TOKENS",
    "MAX_PROMPT_TOKENS", "LR", "KL_COEF", "KL_TARGET", "MAX_KL_PER_STEP",
    "NUM_ACCEPTED_UPDATES", "MAX_ATTEMPTED_PROMPTS",
]:
    print(f"{k:24s}: {globals()[k]}")


# %%
# ============================================================
# 3. PATH DISCOVERY — adapter + data
# ============================================================
def has_adapter_files(path: str) -> bool:
    return (
        os.path.exists(os.path.join(path, "adapter_config.json"))
        and os.path.exists(os.path.join(path, "adapter_model.safetensors"))
    )


if AUTO_FIND_ADAPTER and not has_adapter_files(ADAPTER_PATH):
    candidates = []
    for cfg in glob.glob("/kaggle/input/**/adapter_config.json", recursive=True):
        d = os.path.dirname(cfg)
        if has_adapter_files(d):
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError("No LoRA adapter found under /kaggle/input.")
    print("Adapter candidates:")
    for c in candidates[:20]:
        print(" ", c)
    ADAPTER_PATH = candidates[0]

if not has_adapter_files(ADAPTER_PATH):
    raise FileNotFoundError(f"Adapter files not found at ADAPTER_PATH={ADAPTER_PATH}")

with open(os.path.join(ADAPTER_PATH, "adapter_config.json")) as f:
    adapter_cfg = json.load(f)
print(f"[ok] adapter: {ADAPTER_PATH}")
print(f"     r={adapter_cfg.get('r')} alpha={adapter_cfg.get('lora_alpha')} "
      f"targets={adapter_cfg.get('target_modules')}")


def discover_data_dir() -> str:
    for cand in DATA_DIR_CANDIDATES:
        if cand and os.path.isdir(cand):
            if any(os.path.exists(os.path.join(cand, f)) for f in TRAIN_CATEGORIES):
                return cand
    # Fallback: search recursively for a directory containing at least one train_cot file.
    parents = Counter()
    for fp in glob.glob("/kaggle/input/**/train_cot_*.jsonl", recursive=True):
        parents[os.path.dirname(fp)] += 1
    if parents:
        print("Data-dir candidates:")
        for d, n in parents.most_common(20):
            print(f"  {n:2d} files  {d}")
        return parents.most_common(1)[0][0]
    raise FileNotFoundError("No train_cot_*.jsonl files found under /kaggle/input.")


DATA_DIR = discover_data_dir()
print(f"[ok] data dir: {DATA_DIR}")


# %%
# ============================================================
# 4. ANSWER EXTRACTION + REWARD
# ============================================================
def extract_boxed(text: str | None) -> str | None:
    """Robust last-box extractor that tolerates answers containing '}'.

    If the final line is exactly \\boxed{...}, take everything between the
    prefix and the final closing brace. This handles answers like '}'.
    """
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "\\boxed{" in ln:
            start = ln.rfind("\\boxed{") + len("\\boxed{")
            tail = ln[start:]
            if tail.endswith("}"):
                return tail[:-1].strip()
            return tail.strip()
    return None


def normalize_answer(s):
    if s is None:
        return ""
    # Do not strip quote characters: cryptarithm answers can literally be
    # "'" or '"'. Only normalize whitespace/case.
    return str(s).strip().lower().replace(" ", "")


def metric_match(gt, pred) -> bool:
    gt_s = str(gt).strip()
    pr_s = str(pred).strip()
    if gt_s == "":
        return pr_s == ""
    if set(gt_s) <= {"0", "1"} and gt_s:
        return pr_s.lower() == gt_s.lower()
    try:
        gt_f = float(gt_s)
        pr_f = float(pr_s)
        return math.isclose(gt_f, pr_f, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return normalize_answer(gt_s) == normalize_answer(pr_s)


def reward_fn(generated_text: str, gt_answer: str) -> float:
    """Mostly exact reward, with tiny format shaping.

    Exact answer dominates. Format reward is intentionally small so it cannot
    teach the model to output boxed wrong answers.
    """
    pred = extract_boxed(generated_text)
    reward = 0.0
    if pred is not None:
        reward += 0.03
    if "</think>" in (generated_text or ""):
        reward += 0.01
    if pred is not None and metric_match(gt_answer, pred):
        reward += 1.0
    return reward


for txt, gt in [
    ("reasoning\\n\\boxed{42}", "42"),
    ("reasoning\\n\\boxed{}}", "}"),
    ("no box", "42"),
]:
    print(repr(txt), "gt=", repr(gt), "pred=", repr(extract_boxed(txt)), "reward=", reward_fn(txt, gt))


# %%
# ============================================================
# 5. LOAD PROMPTS
# ============================================================
def category_from_fname(fname: str) -> str:
    return fname.replace("train_cot_", "").replace(".jsonl", "")


records = []
per_file = Counter()
for fname in TRAIN_CATEGORIES:
    fp = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fp):
        print(f"[skip] missing {fname}")
        continue
    cat = category_from_fname(fname)
    with open(fp) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = [m for m in rec.get("messages", []) if m.get("role") != "system"]
            if len(msgs) < 2:
                continue
            user = msgs[0]["content"]
            answer = extract_boxed(msgs[-1]["content"])
            if answer is None:
                continue
            records.append({
                "id": rec.get("id", hashlib.md5(user.encode()).hexdigest()[:8]),
                "category": cat,
                "user": user,
                "answer": answer,
                "weight": CATEGORY_WEIGHTS.get(cat, 1.0),
            })
            per_file[fname] += 1

print("Loaded rows:")
for k, v in per_file.items():
    print(f"  {k:45s} {v:5d}")

# Dedupe identical prompts.
seen = set()
deduped = []
for r in records:
    h = hashlib.md5(r["user"].encode()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    deduped.append(r)
records = deduped

print(f"\nUnique prompts: {len(records)}")
print("Categories:")
for cat, n in Counter(r["category"] for r in records).most_common():
    print(f"  {cat:30s} {n:5d} weight={CATEGORY_WEIGHTS.get(cat, 1.0)}")


# %%
# ============================================================
# 6. LOAD MODEL + TWO ADAPTERS: policy default + frozen ref
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    trust_remote_code=True,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
base_model.config.use_cache = False
try:
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
except Exception as e:
    print(f"[warn] gradient checkpointing not enabled: {e}")
print(f"Base loaded in {time.time() - t0:.0f}s, VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB")

# Enable the same Mamba CUDA fast path as the working SFT notebook.
nemotron_mod = None
for _name, _m in list(sys.modules.items()):
    if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
        nemotron_mod = _m
        break

if nemotron_mod is not None:
    print(f"[info] is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    if FAST_PATH_AVAILABLE:
        try:
            from causal_conv1d import causal_conv1d_fn
            _x = torch.randn(1, 256, 32, device="cuda", dtype=torch.bfloat16)
            _w = torch.randn(256, 4, device="cuda", dtype=torch.bfloat16)
            causal_conv1d_fn(_x, _w, None, activation="silu")
            import mamba_ssm
            print(f"[ok] mamba_ssm v{mamba_ssm.__version__} loaded")
            nemotron_mod.is_fast_path_available = True
            print("[OK] Mamba FAST PATH ENABLED")
        except Exception as e:
            print(f"[warn] fast path kernel check failed: {e}")
            nemotron_mod.is_fast_path_available = False
    else:
        nemotron_mod.is_fast_path_available = False
        print("[warn] Mamba fast path unavailable; GRPO may be slow/OOM")
else:
    print("[warn] modeling_nemotron_h module not found; fast path config skipped")

print("Loading 0.86 adapter as trainable policy adapter: default")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    adapter_name="default",
    is_trainable=True,
)

print("Loading 0.86 adapter again as frozen reference adapter: ref")
model.load_adapter(
    ADAPTER_PATH,
    adapter_name="ref",
    is_trainable=False,
)

for name, p in model.named_parameters():
    p.requires_grad = ("lora_" in name and ".default." in name)
    if p.requires_grad:
        p.data = p.data.float()

model.set_adapter("default")
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable LoRA params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

if not trainable:
    raise RuntimeError("No trainable default adapter LoRA parameters found.")

for name, p in model.named_parameters():
    if p.requires_grad and "lora_B" in name:
        mean_abs = p.detach().abs().mean().item()
        print(f"Sanity trainable {name[:100]} mean_abs={mean_abs:.6f}")
        if mean_abs < 1e-8:
            raise RuntimeError("Loaded adapter appears near-zero; check ADAPTER_PATH.")
        break

print(f"VRAM after adapters: {torch.cuda.memory_allocated()/1e9:.1f} GB")


# %%
# ============================================================
# 7. TOKEN FILTER + SAMPLER
# ============================================================
def apply_chat(user_text: str) -> str:
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )


usable = []
too_long = Counter()
for r in records:
    text = apply_chat(r["user"])
    n_tok = len(tokenizer(text, add_special_tokens=False).input_ids)
    if n_tok <= MAX_PROMPT_TOKENS:
        r["prompt_text"] = text
        r["prompt_tokens"] = n_tok
        usable.append(r)
    else:
        too_long[r["category"]] += 1

records = usable
print(f"Usable prompts after token filter: {len(records)}")
if too_long:
    print("Dropped too-long prompts:")
    for cat, n in too_long.items():
        print(f"  {cat:30s} {n}")

weights = [max(0.01, float(r["weight"])) for r in records]


def sample_prompt():
    return random.choices(records, weights=weights, k=1)[0]


print("Sample prompt:", sample_prompt()["category"], sample_prompt()["id"])


# %%
# ============================================================
# 8. GRPO PRIMITIVES
# ============================================================
def encode_prompt(prompt_text: str) -> torch.Tensor:
    return tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to("cuda")


@torch.no_grad()
def rollout(prompt_ids: torch.Tensor, k: int):
    model.eval()
    model.set_adapter("default")
    batched = prompt_ids.repeat(k, 1)
    out = model.generate(
        batched,
        do_sample=True,
        temperature=ROLLOUT_TEMP,
        top_p=ROLLOUT_TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_len = prompt_ids.shape[1]
    completion = out[:, prompt_len:]
    completion_mask = torch.ones_like(completion, dtype=torch.float32, device=completion.device)
    eos = tokenizer.eos_token_id
    for i in range(completion.shape[0]):
        eos_pos = (completion[i] == eos).nonzero(as_tuple=False)
        if eos_pos.numel():
            first = int(eos_pos[0].item())
            completion_mask[i, first + 1:] = 0.0
    texts = [
        tokenizer.decode(completion[i][completion_mask[i].bool()], skip_special_tokens=True)
        for i in range(k)
    ]
    return out, completion_mask, texts


def mean_log_probs(input_ids: torch.Tensor, prompt_len: int, completion_mask: torch.Tensor,
                   adapter_name: str, grad: bool):
    model.set_adapter(adapter_name)
    with torch.set_grad_enabled(grad):
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        tok_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # shift_labels index prompt_len-1 is the first completion token.
        mask = torch.zeros_like(tok_logp, dtype=torch.float32)
        c_len = completion_mask.shape[1]
        mask[:, prompt_len - 1:prompt_len - 1 + c_len] = completion_mask

        denom = mask.sum(dim=-1).clamp(min=1.0)
        return (tok_logp * mask).sum(dim=-1) / denom


def advantages_from_rewards(rewards: torch.Tensor):
    adv = rewards - rewards.mean()
    if ADV_NORM and rewards.std() > 1e-6:
        adv = adv / (rewards.std() + 1e-6)
    return adv


print("[ok] rollout/logprob helpers ready")


# %%
# ============================================================
# 9. TRAINING LOOP
# ============================================================
optimizer = torch.optim.AdamW(
    trainable,
    lr=LR,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)

metrics = defaultdict(list)
recent_exact = deque(maxlen=40)
best_recent = -1.0
accepted_updates = 0
attempted = 0
t_start = time.time()


def save_selected_adapter(path: str):
    os.makedirs(path, exist_ok=True)
    model.set_adapter("default")
    model.save_pretrained(path, selected_adapters=["default"])
    # PEFT may save non-default metadata; force eval-server base path.
    cfg_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["base_model_name_or_path"] = "metric/nemotron-3-nano-30b-a3b-bf16"
        cfg["lora_dropout"] = 0.0
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


print("=" * 72)
print("Starting GRPO v2")
print("=" * 72)

while accepted_updates < NUM_ACCEPTED_UPDATES and attempted < MAX_ATTEMPTED_PROMPTS:
    attempted += 1
    item = sample_prompt()
    t_step = time.time()

    prompt_ids = encode_prompt(item["prompt_text"])
    prompt_len = prompt_ids.shape[1]

    rollout_ids, completion_mask, texts = rollout(prompt_ids, K_ROLLOUTS)
    rewards_list = [reward_fn(t, item["answer"]) for t in texts]
    exact_list = [
        1.0 if metric_match(item["answer"], extract_boxed(t)) else 0.0
        for t in texts
    ]
    rewards = torch.tensor(rewards_list, device="cuda", dtype=torch.float32)

    # No group-relative signal if all rewards identical.
    if rewards.std() < 1e-6:
        print(f"try {attempted:04d} upd {accepted_updates:03d} "
              f"[{item['category'][:22]:<22}] R={rewards.mean():.3f} "
              f"exact={sum(exact_list):.0f}/{K_ROLLOUTS} skip=no-variance "
              f"{time.time()-t_step:.0f}s")
        continue

    adv = advantages_from_rewards(rewards)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    log_pi = mean_log_probs(
        rollout_ids,
        prompt_len=prompt_len,
        completion_mask=completion_mask,
        adapter_name="default",
        grad=True,
    )
    with torch.no_grad():
        log_ref = mean_log_probs(
            rollout_ids,
            prompt_len=prompt_len,
            completion_mask=completion_mask,
            adapter_name="ref",
            grad=False,
        )
    model.set_adapter("default")

    log_delta = (log_pi - log_ref.detach()).clamp(-2.0, 2.0)
    sampled_kl = log_delta.mean()
    kl_penalty = (log_delta ** 2).mean()
    pg_loss = -(adv.detach() * log_pi).mean()
    loss = pg_loss + KL_COEF * kl_penalty

    if not torch.isfinite(loss):
        print("[halt] non-finite loss")
        break

    # If KL already too high, skip optimizer step and keep the adapter anchored.
    if abs(float(sampled_kl.detach().item())) > MAX_KL_PER_STEP:
        print(f"try {attempted:04d} upd {accepted_updates:03d} "
              f"[{item['category'][:22]:<22}] skip=kl-spike "
              f"KL={sampled_kl.item():+.3f}")
        continue

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
    optimizer.step()
    accepted_updates += 1

    exact_rate = sum(exact_list) / len(exact_list)
    recent_exact.append(exact_rate)
    recent_avg = sum(recent_exact) / len(recent_exact)

    metrics["attempt"].append(attempted)
    metrics["update"].append(accepted_updates)
    metrics["loss"].append(float(loss.detach().item()))
    metrics["reward"].append(float(rewards.mean().detach().item()))
    metrics["exact_rate"].append(exact_rate)
    metrics["sampled_kl"].append(float(sampled_kl.detach().item()))
    metrics["kl_penalty"].append(float(kl_penalty.detach().item()))
    metrics["grad_norm"].append(float(grad_norm))
    metrics["category"].append(item["category"])

    print(f"try {attempted:04d} upd {accepted_updates:03d} "
          f"[{item['category'][:22]:<22}] "
          f"R={rewards.mean().item():.3f} exact={sum(exact_list):.0f}/{K_ROLLOUTS} "
          f"loss={loss.item():+.4f} KL={sampled_kl.item():+.4f} "
          f"|g|={float(grad_norm):.2f} recent={recent_avg:.3f} "
          f"{time.time()-t_step:.0f}s")

    if accepted_updates % SAVE_EVERY_ACCEPTED == 0:
        ckpt = os.path.join(CKPT_DIR, f"update_{accepted_updates:04d}")
        save_selected_adapter(ckpt)
        print(f"[ckpt] {ckpt}")

    # Save best recent reward, but only while KL remains sane.
    if len(recent_exact) >= 20 and recent_avg > best_recent and abs(sampled_kl.item()) <= KL_TARGET * 2:
        best_recent = recent_avg
        save_selected_adapter(BEST_DIR)
        print(f"[best] recent_exact={best_recent:.3f} -> {BEST_DIR}")

    if time.time() - t_start > 11.5 * 3600:
        print("[time] stopping before Kaggle timeout")
        break

print("=" * 72)
print(f"Finished: attempted={attempted}, accepted_updates={accepted_updates}, "
      f"elapsed={(time.time()-t_start)/60:.1f} min")
print("=" * 72)

save_selected_adapter(OUTPUT_DIR)
print(f"Final adapter saved to {OUTPUT_DIR}")


# %%
# ============================================================
# 10. METRICS + ZIP
# ============================================================
print("Accepted update metrics:")
if metrics["update"]:
    print(f"  updates       : {len(metrics['update'])}")
    print(f"  mean reward   : {sum(metrics['reward']) / len(metrics['reward']):.4f}")
    print(f"  mean exact    : {sum(metrics['exact_rate']) / len(metrics['exact_rate']):.4f}")
    print(f"  last exact    : {sum(metrics['exact_rate'][-20:]) / max(1, len(metrics['exact_rate'][-20:])):.4f}")
    print(f"  mean KL       : {sum(metrics['sampled_kl']) / len(metrics['sampled_kl']):+.5f}")
    print("  categories:")
    for cat, n in Counter(metrics["category"]).most_common():
        print(f"    {cat:30s} {n:4d}")
else:
    print("  no accepted updates; keep the original 0.86 adapter")

metrics_path = "/kaggle/working/grpo_metrics.json"
with open(metrics_path, "w") as f:
    json.dump({k: list(v) for k, v in metrics.items()}, f, indent=2)
print(f"[ok] metrics written: {metrics_path}")


def zip_dir(src_dir: str, zip_path: str):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(src_dir)):
            fpath = os.path.join(src_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, arcname=fname)
    print(f"[ok] {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")


zip_dir(OUTPUT_DIR, "/kaggle/working/grpo_adapter.zip")
if os.path.exists(os.path.join(BEST_DIR, "adapter_config.json")):
    zip_dir(BEST_DIR, "/kaggle/working/grpo_best_adapter.zip")
else:
    print("[info] no best adapter saved; use grpo_adapter.zip only")

print("Next:")
print("  1. Submit grpo_best_adapter.zip first if it exists.")
print("  2. If it regresses, submit grpo_adapter.zip or revert to the original 0.86 adapter.")
print("  3. Do not continue RL if accepted updates have near-zero exact reward or KL spikes.")
