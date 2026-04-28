import json
import os
import glob
import re

def main():
    folder = "/Users/manishswami/developer/NVIDIA-Nemotron Model/data/processed/all_categories_split"
    files = glob.glob(os.path.join(folder, "*.jsonl"))
    
    max_chars_all = 0
    max_words_all = 0
    max_file = ""
    
    print(f"Scanning {len(files)} files for max token length...\n")
    print(f"{'Category File':<45} | {'Max Chars':<10} | {'Max Words':<10} | {'Est. Tokens (Chars/3)':<20}")
    print("-" * 90)
    
    for fpath in files:
        fname = os.path.basename(fpath)
        max_chars_in_file = 0
        max_words_in_file = 0
        
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                messages = record.get("messages", [])
                
                # Combine all text in the conversation
                full_text = ""
                for msg in messages:
                    full_text += msg.get("content", "") + " "
                    
                char_len = len(full_text)
                # Words including punctuation splits
                words = len(re.findall(r"\w+|[^\w\s]", full_text))
                
                if char_len > max_chars_in_file: max_chars_in_file = char_len
                if words > max_words_in_file: max_words_in_file = words
                
                if char_len > max_chars_all:
                    max_chars_all = char_len
                    max_words_all = words
                    max_file = fname
                    
        est_tokens = max_chars_in_file // 3
        print(f"{fname:<45} | {max_chars_in_file:<10} | {max_words_in_file:<10} | {est_tokens:<20}")
        
    print("-" * 90)
    print(f"Overall Maximum Found in: {max_file}")
    print(f"Max Characters: {max_chars_all}")
    print(f"Max Words (approx): {max_words_all}")
    print(f"Conservative Token Estimate for max_seq_len: {max_chars_all // 3}")

if __name__ == "__main__":
    main()
