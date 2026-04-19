# Quick Start: From 66% → 88% (Bit Manipulation 85%+)

## 📊 The Situation

**Base Model Score:** 66.14%
**Bit Manipulation (Base):** 45% (1602 problems)
**Target:** 85%+ on bit manipulation
**Expected Final Score:** 88%+

---

## ⚡ Three Actions (In Order)

### ACTION 1: Upload Data (5 minutes)

Go to https://kaggle.com/datasets/create

```
Title: Nemotron CoT v5 Merged (9406 Verified Examples)
Files: train_cot_v5_merged.jsonl (8.9 MB)
Slug: nemotron-cot-v5
Visibility: Public
```

---

### ACTION 2: Run v6 SFT Training (2-3 hours)

Create notebook on Kaggle from `nemotron_v5_train_only.py`

**Inputs:**
- `nemotron-cot-v5` (dataset you just created)
- `metric/nemotron-3-nano-30b-a3b-bf16` (base model)
- `nvidia-nemotron-offline-packages` (TRL, PEFT)

**Accelerator:** GPU (T4/L4/P100, any is fine)

**Expected Score:** 0.73-0.84 (+15% from base)

**Monitor:** Loss should drop 20-30% in epoch 1

---

### ACTION 3: Run v7 GRPO RL Training (8-10 hours, optional but crucial)

Implement GRPO from the `STRATEGY_85_PERCENT_BIT_MANIPULATION.md` document.

**Key points:**
- Load v6 checkpoint
- Binary reward (1.0 if correct, 0.0 if wrong)
- Temperature = 1.0 (exploration)
- 4-8 samples per prompt

**Expected Score:** 0.85-0.88 (+6% from v6)

---

## 🎯 Why This Works

| Stage | What We Fix | Bit Accuracy | Overall |
|-------|-----------|--------------|---------|
| Base | Nothing | 45% | 66% |
| v6 SFT | LoRA config (out_proj, alpha, LR) | 70% | 82% |
| v7 RL | Pattern exploration via rewards | 85% | 88% |

**The Key Insight:**
```
SFT learns from data.
RL discovers patterns not in data.
Together → 85%+
```

---

## 📈 Expected Progress

```
Day 1 Morning: Upload data
Day 1 Afternoon: Start v6 training
Day 2 Morning: v6 results (should see ~0.73-0.82)
Day 2-3: Setup & run v7 RL
Day 4: Results (~0.85+)
Day 5: Polish & submit
```

---

## 🚀 Advanced (If You Want 90%+)

1. **Dynamic Sampling:** Only train on prompts where model gets 50% of attempts correct
2. **Multi-Round RL:** After RL plateaus, merge weights and RL again
3. **Temperature Scheduling:** Start high (1.5), decay to low (0.7)
4. **Category Weighting:** Oversample bit_manipulation in RL batches

See `STRATEGY_85_PERCENT_BIT_MANIPULATION.md` for details.

---

## 📋 Success Checklist

- [ ] Data uploaded to Kaggle
- [ ] v6 notebook created & running
- [ ] v6 loss decreasing (not NaN)
- [ ] v6 final score ≥ 0.73
- [ ] v7 RL code implemented
- [ ] v7 RL running without crashes
- [ ] v7 score ≥ 0.85
- [ ] Submit to Kaggle

---

## 🔥 If Something Goes Wrong

**"v6 training crashed"**
→ Check: Is out_proj excluded from LoRA? Run `model.print_trainable_parameters()`, should NOT show out_proj.

**"Loss is NaN"**
→ Check: Is RMSNorm fix applied? (Should be automatic in v5 code)

**"v6 scores < 0.73"**
→ Check: Is LR exactly 1e-5? Is data format correct? Sample first example from notebook.

**"v7 RL not improving"**
→ Check: Can model generate ANY correct answers? If not, temperature might be 0. Set it to 1.0.

---

## 📚 Full Documentation

- `TRAINING_CODE_EXPLAINED.md` — Line-by-line breakdown of training code
- `STRATEGY_85_PERCENT_BIT_MANIPULATION.md` — Deep strategic guide
- `DATA_IMPROVEMENT_STRATEGY.md` — When/how to improve data
- `NOTEBOOK_0.69_WRITEUP.md` — Why 0.69 scored well despite bugs
- `KAGGLE_DISCUSSION.md` — Ready-to-post discussion explaining v6

---

**Time to 85%+: 5-7 days of computation**
**Your effort: ~4 hours setup + monitoring**

**Let's go!** 🚀

