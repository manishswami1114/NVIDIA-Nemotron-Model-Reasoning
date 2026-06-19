#!/usr/bin/env python3
"""Emit nemotron_grpo_rlvr.ipynb — GRPO (RLVR) from the 0.86 adapter.

No new dataset: reads train.csv directly, rewards with a programmatic verifier
(+ dense partial-credit shaping), RL-trains only the prompts the 0.86 model
misses, curriculum-ordered. Reuses the env-tested vLLM+PEFT GRPO scaffolding.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "nemotron_grpo_rlvr.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}

cells = []

cells.append(md("""# Nemotron GRPO (RLVR) — continue from the 0.86 adapter

No new dataset. RLVR = GRPO with a **verifiable reward**:
- prompts + gold answers come straight from `train.csv`
- reward = programmatic verifier (exact / numeric / text) **+ dense partial credit**
  (close answers — 1-bit bit-flips, off-by-one numbers, near-miss cryptarithm —
  get smooth reward so RL can climb on sparse puzzles)
- RL only on the **prompts the 0.86 model misses**, curriculum-ordered (easy→hard)
- policy **starts from your 0.86 adapter** and is refined

This is the NVIDIA RLVR recipe (generate → verify → advantage → clipped update),
not SFT imitation. V15 is your gold reference; it is NOT imitated here.
"""))

cells.append(code('''# 1. INSTALL
import subprocess, sys
def pip_install(*p):
    for x in p:
        subprocess.call([sys.executable,"-m","pip","install","-q",x])
try:
    pip_install("trl>=0.17.0","peft>=0.15.0","accelerate","bitsandbytes","vllm>=0.8.0")
except Exception as e:
    print("install note:", e)
print("ok")
'''))

cells.append(code('''# 2. IMPORTS
import os, json, re, csv, time, gc, math, random, difflib
from collections import defaultdict, Counter
from typing import List, Optional
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
random.seed(0); np.random.seed(0); torch.manual_seed(0)
'''))

cells.append(code('''# 3. CONFIG
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
# >>> YOUR 0.86 ADAPTER (set this) <<<
SFT_ADAPTER_PATH = "/kaggle/input/your-086-adapter"   # must contain adapter_config.json

# train.csv (prompts + gold answers) — first existing wins
TRAIN_CSV = next((p for p in [
    "/kaggle/input/nemotron-data/train.csv",
    "/kaggle/input/train/train.csv",
    "data/raw/train.csv",
] if os.path.exists(p)), None)
# optional: 0.86 eval csv to know which prompts are already solved (id,is_correct)
EVAL_086 = next((p for p in [
    "/kaggle/input/nemotron-data/evaluation_results_086.csv",
    "evaluation_results_086.csv",
] if os.path.exists(p)), None)
assert TRAIN_CSV, "train.csv not found"
print("train.csv:", TRAIN_CSV, "| eval_086:", EVAL_086)

OUTPUT_DIR="/kaggle/working/grpo_adapter"; BEST_DIR="/kaggle/working/best_adapter"
os.makedirs(OUTPUT_DIR,exist_ok=True); os.makedirs(BEST_DIR,exist_ok=True)

LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.0
GROUP_SIZE=6; CLIP_EPS=0.2; KL=0.01; GRPO_LR=5e-6
MAX_NEW_TOKENS=4096; TEMP=0.9; TOP_P=0.95
NUM_ROUNDS=12; PROMPTS_PER_ROUND=48; MAX_GRAD_NORM=1.0
VLLM_MAX_LEN=8192; VLLM_UTIL=0.55
'''))

cells.append(code('''# 4. VERIFIER + DENSE SHAPED REWARD
def extract_boxed(t):
    st=[m.start() for m in re.finditer(r"\\\\boxed\\{", t)]
    if not st:
        m=re.findall(r"(?:final answer|answer)\\s*[:=]\\s*(.+)", t, re.I)
        return m[-1].strip() if m else None
    s=st[-1]+len("\\\\boxed{"); last=t.rfind("}")
    return t[s:last].strip() if last>s else t[s:].strip()

def _nt(s): return re.sub(r"\\s+"," ",s.strip().lower())

def matches(pred, gold, verify):
    if pred is None: return False
    pred, gold = pred.strip(), str(gold).strip()
    if verify=="numeric":
        try: return math.isclose(float(pred),float(gold),rel_tol=1e-2,abs_tol=1e-2)
        except: return False
    if verify=="text": return _nt(pred)==_nt(gold)
    return pred==gold

def reward(text, gold, verify):
    pred=extract_boxed(text)
    if pred is None: return -1.0
    if matches(pred, gold, verify): return 1.0
    # dense partial credit (wrong but close) in [0, 0.5]
    if verify=="numeric":
        try:
            p=float(pred); g=float(gold)
            return 0.5*max(0.0, 1-min(1.0, abs(p-g)/(abs(g)+1e-6)))
        except: return -0.5
    return 0.5*difflib.SequenceMatcher(None, pred, str(gold)).ratio()

# self-test
assert reward("\\\\boxed{10010111}","10010111","exact")==1.0
assert 0<reward("\\\\boxed{10010110}","10010111","exact")<1.0   # 1-bit off -> partial
assert reward("no box","x","exact")==-1.0
print("reward ok")
'''))

cells.append(code('''# 5. LOAD PROMPTS (RL only on what 0.86 misses), curriculum-ordered
def categorize(p):
    if "bit manipulation" in p: return "bit_manipulation"
    if "encryption" in p or "decrypt" in p: return "cipher"
    if "gravitational" in p: return "gravity"
    if "numeral system" in p: return "numeral"
    if "unit conversion" in p: return "unit_conversion"
    if "secret set of transformation rules" in p:
        crypto=False
        for l in p.split("\\n"):
            l=l.strip()
            if " = " in l and "determine" not in l.lower():
                left=l.split(" = ",1)[0].strip()
                if len(left)==5 and not any(c.isdigit() for c in left): crypto=True
                break
        qop=None; ex=set()
        for l in p.split("\\n"):
            l=l.strip()
            if "determine the result for" in l.lower():
                q=l.split(":")[-1].strip(); qop=q[2] if len(q)>=3 else None
            elif " = " in l:
                lf=l.split(" = ",1)[0].strip()
                if len(lf)==5: ex.add(lf[2])
        return ("cryptarithm" if crypto else "equation_numeric")+("_deduce" if qop in ex else "_guess")
    return "unknown"

VERIFY={"bit_manipulation":"exact","cipher":"text","gravity":"numeric","numeral":"exact",
        "unit_conversion":"numeric","equation_numeric_deduce":"exact",
        "equation_numeric_guess":"exact","cryptarithm_deduce":"exact","cryptarithm_guess":"exact"}

rows=list(csv.DictReader(open(TRAIN_CSV)))
solved={}
if EVAL_086:
    solved={r["id"]:str(r["is_correct"]).lower()=="true" for r in csv.DictReader(open(EVAL_086))}

cat_rows=defaultdict(list)
for r in rows: cat_rows[categorize(r["prompt"])].append(r)
cat_acc={c:(sum(solved.get(x["id"],False) for x in rs)/len(rs) if solved else 0.5) for c,rs in cat_rows.items()}

samples=[]
for r in rows:
    c=categorize(r["prompt"])
    if solved and solved.get(r["id"],False):   # skip already-solved
        continue
    if not solved and cat_acc.get(c,1.0)>0.95:  # no eval -> skip easy cats
        continue
    samples.append({"id":r["id"],"category":c,"prompt":r["prompt"],
                    "answer":r["answer"],"verify":VERIFY.get(c,"exact"),
                    "diff":cat_acc.get(c,0.5)})
samples.sort(key=lambda x:-x["diff"])   # curriculum: easier (higher pass) first
print("RL prompts:", len(samples), dict(Counter(s["category"] for s in samples)))
'''))

cells.append(code('''# 6. vLLM ROLLOUTS
def fmt(p): return tokenizer.apply_chat_template([{"role":"user","content":p}],tokenize=False,add_generation_prompt=True)
def gen_rollouts(model_path, lora_path, prompts):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    kw=dict(model=model_path,dtype="bfloat16",max_model_len=VLLM_MAX_LEN,trust_remote_code=True,
            gpu_memory_utilization=VLLM_UTIL,enforce_eager=True)
    use=lora_path and os.path.exists(os.path.join(lora_path,"adapter_model.safetensors"))
    if use: kw["enable_lora"]=True; kw["max_lora_rank"]=LORA_RANK
    eng=LLM(**kw)
    sp=SamplingParams(n=GROUP_SIZE,temperature=TEMP,top_p=TOP_P,max_tokens=MAX_NEW_TOKENS,stop=["<|endoftext|>"])
    req=LoRARequest("p",1,lora_path) if use else None
    outs=eng.generate([fmt(p) for p in prompts], sp, lora_request=req)
    res=[[o.text for o in out.outputs] for out in outs]
    del eng; gc.collect(); torch.cuda.empty_cache()
    return res
'''))

cells.append(code('''# 7. HF POLICY (resume from 0.86 adapter)
def targets(m):
    suff=Counter(n.split(".")[-1] for n,mm in m.named_modules() if isinstance(mm,nn.Linear))
    want=["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","up_proj","down_proj"]
    ex={"lm_head","embed_tokens","shared","router","score","classifier"}
    return [w for w in want if w in suff and w not in ex]
def load_policy(model_path, lora_path):
    m=AutoModelForCausalLM.from_pretrained(model_path,device_map={"":0},trust_remote_code=True,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation="eager")
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    if lora_path and os.path.exists(os.path.join(lora_path,"adapter_config.json")):
        m=PeftModel.from_pretrained(m, lora_path, is_trainable=True)
    else:
        m=get_peft_model(m, LoraConfig(r=LORA_RANK,lora_alpha=LORA_ALPHA,target_modules=targets(m),
            lora_dropout=LORA_DROPOUT,bias="none",task_type="CAUSAL_LM"))
    return m
tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
'''))

cells.append(code('''# 8. GRPO CORE
def tok_lp(model, ptxt, rtxt):
    enc=tokenizer(ptxt+rtxt,return_tensors="pt",truncation=True,max_length=VLLM_MAX_LEN).to(model.device)
    pl=tokenizer(ptxt,return_tensors="pt").input_ids.shape[1]; tl=enc.input_ids.shape[1]
    if tl<=pl+1: return torch.zeros(1,device=model.device)
    lg=model(**enc).logits[:,pl-1:tl-1,:]
    lab=enc.input_ids[:,pl:]
    return F.log_softmax(lg.float(),-1).gather(-1,lab.unsqueeze(-1)).squeeze(-1).squeeze(0)
@torch.no_grad()
def ref_old(model, ptxt, rtxt):
    model.eval(); old=tok_lp(model,ptxt,rtxt).detach()
    model.disable_adapter_layers(); ref=tok_lp(model,ptxt,rtxt).detach(); model.enable_adapter_layers()
    return ref, old
def grpo_loss(model, ptxt, rtxt, adv, ref_lp, old_lp):
    new=tok_lp(model,ptxt,rtxt); L=min(len(new),len(old_lp),len(ref_lp))
    new,old_lp,ref_lp=new[:L],old_lp[:L],ref_lp[:L]
    ratio=torch.exp(new-old_lp); a=torch.tensor(adv,device=new.device,dtype=new.dtype)
    pl=-torch.min(ratio*a, torch.clamp(ratio,1-CLIP_EPS,1+CLIP_EPS)*a).mean()
    klp=(torch.exp(ref_lp-new)-(ref_lp-new)-1).mean()
    return pl+KL*klp
'''))

cells.append(code('''# 9. TRAINING LOOP (generate -> reward -> advantage -> update)
best=-1e9; stats=[]
for rnd in range(NUM_ROUNDS):
    t0=time.time(); print(f"\\n=== ROUND {rnd+1}/{NUM_ROUNDS} ===")
    batch=random.sample(samples, min(PROMPTS_PER_ROUND,len(samples)))
    lora_for_gen = BEST_DIR if (rnd>0 and os.path.exists(os.path.join(BEST_DIR,"adapter_config.json"))) else SFT_ADAPTER_PATH
    rollouts=gen_rollouts(MODEL_PATH, lora_for_gen, [b["prompt"] for b in batch])

    model=load_policy(MODEL_PATH, lora_for_gen)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=GRPO_LR)
    flat=[]; rsum=0; corr=0; ntot=0
    for b, group in zip(batch, rollouts):
        ptxt=fmt(b["prompt"]); rs=np.array([reward(g,b["answer"],b["verify"]) for g in group],dtype=np.float32)
        rsum+=rs.mean();
        for g in group:
            ntot+=1; corr+= matches(extract_boxed(g),b["answer"],b["verify"])
        adv=rs-rs.mean();
        if rs.std()>1e-6: adv=adv/(rs.std()+1e-6)
        for g,a in zip(group,adv): flat.append((ptxt,g,float(a)))
    model.train(); random.shuffle(flat); opt.zero_grad(); acc=0
    for ptxt,g,a in flat:
        if abs(a)<1e-6: continue
        rl,ol=ref_old(model,ptxt,g)
        (grpo_loss(model,ptxt,g,a,rl,ol)/4).backward(); acc+=1
        if acc%4==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],MAX_GRAD_NORM)
            opt.step(); opt.zero_grad()
    if acc%4: opt.step(); opt.zero_grad()
    mr=rsum/len(batch); ac=corr/max(ntot,1)
    model.save_pretrained(OUTPUT_DIR)
    if mr>best: best=mr; model.save_pretrained(BEST_DIR); print(f"  new best reward={mr:.3f}")
    stats.append((rnd,mr,ac)); print(f"  reward={mr:.3f} solve_rate={ac:.2%} {(time.time()-t0)/60:.1f}m")
    del model,opt; gc.collect(); torch.cuda.empty_cache()
'''))

cells.append(code('''# 10. SUMMARY + ZIP
import zipfile
print("rnd  reward  solve%");
for r,mr,ac in stats: print(f"{r+1:>3} {mr:>7.3f} {ac:>6.1%}")
print("best reward:", round(best,3))
Z="/kaggle/working/grpo_adapter.zip"
if os.path.exists(Z): os.remove(Z)
with zipfile.ZipFile(Z,"w",zipfile.ZIP_DEFLATED) as zf:
    for f in os.listdir(BEST_DIR):
        fp=os.path.join(BEST_DIR,f)
        if os.path.isfile(fp): zf.write(fp,f)
print("zipped:", Z)
'''))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(nb,indent=1))
print("wrote", OUT, "cells", len(cells))
