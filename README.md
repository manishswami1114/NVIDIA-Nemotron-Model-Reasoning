# NVIDIA Nemotron Model Reasoning Challenge

This repository contains the codebase and methodology for training and running the NVIDIA Nemotron model on the Reasoning Challenge.

The project is structured to generate robust, step-by-step reasoning sequences (Chain-of-Thought) and fine-tuning the model using LoRA. It contains scripts for filtering data, generating synthetic data mathematically, fine-tuning the base LLM, and generating the final submissions.

## Project Structure

```bash
NVIDIA-Nemotron-Model/
├── src/                    # Source code components
│   ├── data_generation/    # Scripts to generate and filter Chain-of-Thought
│   ├── solvers/            # Analytical or handcrafted solvers for data augmentation
│   └── training/           # Main Python training and submission scripts
├── notebooks/              # Jupyter Notebooks for EDA, generation, and tuning experiments
├── data/                   # Data directory (ignored in version control)
│   ├── raw/                # Original train.csv and test.csv
│   └── processed/          # Processed JSONL files (synthetic & merged data)
├── docs/                   # Documentation and strategy writeups
│   └── images/             # Visualizations for documentation
└── README.md               # This file
```

## How to the run the model / pipeline

### 1. Requirements

Ensure you have your environment configured with Python 3.10+ and the required dependencies for LLM fine-tuning:
```bash
pip install -r requirements.txt  # If provided
# Or ensure you have: torch, transformers, peft, accelerate, trl, datasets, pandas, etc.
```

### 2. Data Preparation and Generation
Before training, you must prepare the Chain-of-Thought data.

- **Explore & Filter**: Check `src/data_generation/02_cot_filter_v4.py` or use corresponding notebooks.
- **Generate Synthetic Data**: Use `src/data_generation/generate_synthetic_data_v8.py` to automatically create synthetic augmented inputs based on various permutations.
- Ensure the raw `train.csv` and `test.csv` are placed in `data/raw/`.
- Synthetic data logic heavily depends on rules encoded within generators like `cot_v4_generators.py` and solvers in `src/solvers/`.

### 3. Model Training (Fine-Tuning)
The training logic primarily utilizes LoRA for parameter-efficient fine-tuning on the NVIDIA Nemotron model.

Run the latest version of the training script:
```bash
python src/training/nemotron_v8_train.py
```
*(You may need to configure parameters within the script itself or adapt it to use `argparse` arguments depending on your environment).*

For experimentation with different hyperparameters and merged approaches, refer to the Jupyter notebooks, specifically `notebooks/nemotron_v7_training.ipynb` or `nemotron_training_v4_merged.ipynb`.

### 4. Documentation & Strategies
We highly recommend reviewing the `docs/` directory to understand the iterations of strategy, including:
- **`TRAINING_CODE_EXPLAINED.md`**: Detailed exposition of model training decisions.
- **`STRATEGY_85_PERCENT_BIT_MANIPULATION.md`**: A core strategy document explaining the algorithmic edge.
- **`README_COMPLETE_STRATEGY.md`**: Broad overview of the competition strategy.

## Version Control

This repository is configured to exclude large processed data artifacts, model checkpoints, and standard caches (`__pycache__`, `.ipynb_checkpoints`) to ensure clean version histories.

## Contributing

Feel free to fork this repository, add enhancements to the data generation, or test out new model configurations. All training logs and artifacts should be directed locally away from the tracked source directory.
