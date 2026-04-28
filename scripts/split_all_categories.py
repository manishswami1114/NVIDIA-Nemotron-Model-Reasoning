import json
import os
from pathlib import Path

def split_categories():
    base_dir = Path("/Users/manishswami/developer/NVIDIA-Nemotron Model")
    problems_index = base_dir / "tong_reasoners/problems.jsonl"
    verified_data = base_dir / "data/processed/train_cot_v11_verified.jsonl"
    out_dir = base_dir / "data/processed/all_categories_split"
    weak_solved_dir = base_dir / "data/processed/weak_categories_solved"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Map Prompt to Category
    prompt_to_category = {}
    for p_file in (base_dir / "tong_reasoners/problems").glob("*.jsonl"):
        with open(p_file, "r") as f:
            meta = json.loads(f.read().strip())
            prompt_to_category[meta["prompt"].strip()] = meta["category"]
            
    # 2. Distribute records from the main verified file
    category_files = {}
    try:
        with open(verified_data, "r") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                # Get the user prompt
                user_msg = record["messages"][0]["content"]
                # The prompt has '\nPlease put your final answer inside `\boxed{}`.' appended to it
                clean_prompt = user_msg.replace("\nPlease put your final answer inside `\\boxed{}`.", "").strip()
                
                if clean_prompt not in prompt_to_category:
                    # Fallback matching
                    matched = False
                    for p in prompt_to_category:
                        if p in clean_prompt:
                            clean_prompt = p
                            matched = True
                            break
                    if not matched:
                        continue
                    
                cat = prompt_to_category[clean_prompt]
                # Skip the weak categories from the main file since we have better versions
                if cat in ["equation_numeric_guess", "cryptarithm_deduce", "cryptarithm_guess"]:
                    continue
                    
                if cat not in category_files:
                    category_files[cat] = open(out_dir / f"train_cot_{cat}.jsonl", "w")
                
                category_files[cat].write(json.dumps(record) + "\n")
    finally:
        for f in category_files.values():
            f.close()
            
    # 3. Copy the high-quality weak categories into the final folder
    import shutil
    for fname in os.listdir(weak_solved_dir):
        if fname.endswith(".jsonl"):
            # Determine correct name format. We want them to just be train_cot_{category}.jsonl
            # weak_categories_solved has:
            # train_cot_equation_numeric_guess.jsonl
            # train_cot_cryptarithm_deduce_z3.jsonl -> train_cot_cryptarithm_deduce.jsonl
            # train_cot_cryptarithm_guess_z3.jsonl -> train_cot_cryptarithm_guess.jsonl
            
            src = weak_solved_dir / fname
            if "deduce_z3" in fname:
                dst = out_dir / "train_cot_cryptarithm_deduce.jsonl"
            elif "guess_z3" in fname:
                dst = out_dir / "train_cot_cryptarithm_guess.jsonl"
            else:
                dst = out_dir / fname
                
            shutil.copy2(src, dst)
            print(f"Copied optimized weak category: {dst.name}")
            
    print("Successfully split all categories into:", str(out_dir))

if __name__ == "__main__":
    split_categories()
