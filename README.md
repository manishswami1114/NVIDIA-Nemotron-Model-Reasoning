# 🧠 NVIDIA Nemotron Model Reasoning Challenge

Welcome to the **NVIDIA Nemotron Reasoning Project**! This repository showcases a cutting-edge approach to training Artificial Intelligence (AI) to solve complex logic and math puzzles deterministically. 

## 🌟 What is this project?

Imagine trying to teach an AI not just to guess an answer, but to "think" through a problem step-by-step, perfectly, every single time. That's what we are doing here. 

Specifically, we are training the **NVIDIA Nemotron AI model** to solve complex logic puzzles and *cryptarithms* (math puzzles where numbers are replaced by letters, like `SEND + MORE = MONEY`). 

### For Non-Technical Readers 🧑‍🏫
Normally, AI models are great at chatting and writing poetry, but they struggle with strict math and logic because they just guess the most likely next word. 
To fix this, we:
1. **Use an infallible "Math Engine"**: We use a powerful mathematical tool called a Z3 solver. Think of it as a super-calculator that never makes logical mistakes.
2. **Generate "Thinking" Examples**: We use the math engine to create thousands of step-by-step solutions (called *Chain-of-Thought*).
3. **Teach the AI**: We feed these perfect step-by-step examples to the NVIDIA Nemotron AI. We are essentially giving the AI an answer key with the work shown, teaching it *how* to think logically instead of just memorizing answers.

By the end, our AI learns to solve highly complex puzzles with an incredible degree of accuracy!

### For Technical Professionals 💻
This repository implements a fine-tuning pipeline for the NVIDIA Nemotron model, focusing on complex reasoning tasks. 
Instead of relying on brute-force algorithms or standard LLM generation which is prone to hallucination, we leverage a **Z3-based SMT (Satisfiability Modulo Theories) solver**. 

The core workflow:
1. **Deterministic Data Generation**: The Z3 solver computes mathematically proven solutions and formats them into high-quality, step-by-step Chain-of-Thought (CoT) trajectories.
2. **Parameter-Efficient Fine-Tuning (PEFT)**: We use LoRA (Low-Rank Adaptation) targeting specific attention modules to inject this reasoning capability into the base Nemotron model efficiently without retraining the entire network.
3. **Evaluation**: Ensuring deterministic correctness aligned with strict evaluation metrics.

---

## 📁 Repository Structure

```text
NVIDIA-Nemotron-Model/
├── src/                    # Source code components
│   ├── data_generation/    # Scripts to generate Z3-verified Chain-of-Thought data
│   ├── solvers/            # Z3 SMT solvers and analytical logic engines
│   └── training/           # Main training and fine-tuning scripts
├── notebooks/              # Jupyter Notebooks for experimentation and EDA
├── data/                   # (Ignored in Git) Raw datasets and processed JSONL files
├── docs/                   # Detailed documentation and strategy explainers
└── README.md               # You are here!
```

## 🚀 How to Run the Pipeline

### 1. Setup Environment
Ensure you have Python 3.10+ installed. Install the necessary machine learning libraries (PyTorch, Transformers, PEFT, TRL, Z3-solver, etc.):
```bash
pip install -r requirements.txt
```

### 2. Generate Verified Training Data
Before training the AI, we need the Z3 solver to generate the "answer keys":
- Use the scripts in `src/data_generation/` to create synthetic data.
- Ensure your raw datasets are in the `data/raw/` folder.

### 3. Train the Model
Run the fine-tuning script to adapt the NVIDIA Nemotron model using LoRA:
```bash
python src/training/nemotron_v8_train.py
```
*(You can also explore the Jupyter notebooks in the `notebooks/` directory for interactive training experiments).*

## 📖 Deep Dive Documentation
Want to know the specifics of our strategy? Check out the `docs/` folder:
- **`TRAINING_CODE_EXPLAINED.md`**: Deep dive into the LoRA configuration and training loop.
- **`README_COMPLETE_STRATEGY.md`**: A comprehensive overview of the competition strategy and architecture.
