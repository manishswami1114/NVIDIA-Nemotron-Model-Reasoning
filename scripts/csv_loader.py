#!/usr/bin/env python3
"""Custom CSV loader for train.csv that preserves literal "" everywhere.
Rules discovered from the data:
- Rows are separated by newlines that precede an 8-hex-char + comma pattern (id).
- Each row has 3 fields: id,prompt,answer
- Prompt is enclosed in "..." where the OUTER " are delimiters; ALL inner content is literal (no doublequote unescaping; no escape char). The prompt's closing " is the one immediately followed by ',' that starts the answer field.
- Answer field: leading " is a delimiter ONLY if next char isn't also "; trailing " is always stripped (delimiter).
"""
import re
from pathlib import Path

def load_rows(path):
    with open(path) as f:
        txt = f.read()
    parts = re.split(r'\n(?=[0-9a-f]{8},)', txt)
    header = parts[0].split('\n')[0]
    rows_text = []
    first_row_start = parts[0].find('\n') + 1
    if first_row_start < len(parts[0]):
        rows_text.append(parts[0][first_row_start:])
    rows_text.extend(parts[1:])
    return header, rows_text

def parse_row(row):
    """Return (id, prompt, answer) with literal "" preserved."""
    rid = row[:8]
    rest = row[9:]  # skip id + first comma
    # prompt starts with opening " (the very first char after id,)
    if not rest.startswith('"'):
        # unexpected — fallback: no prompt quoting
        last_comma = rest.rfind(',')
        return rid, rest[:last_comma].strip(), rest[last_comma+1:].strip()
    # find closing " of prompt: it's the LAST occurrence of '",' (quote followed by comma)
    # All inner " are LITERAL — so we look for the rightmost '",' near the end
    # But there might be '",' inside prompt too? In practice the prompt content doesn't have ',' in chunky math notation. Safer: take LAST '",' in the row.
    close_idx = rest.rfind('",')
    if close_idx <= 0:
        # no comma after closing quote: prompt may end at last "
        return rid, rest[1:].rstrip('"').strip(), ''
    prompt = rest[1:close_idx]
    # PROMPT field uses CSV-escape: "" represents one literal "
    prompt = prompt.replace('""', '"')
    answer_raw = rest[close_idx+2:].rstrip('\n').rstrip()
    # Now apply the rule for answer:
    #   leading " is a delimiter ONLY if next char isn't also "
    #   trailing " is always a delimiter
    if not answer_raw:
        return rid, prompt, ''
    # Answer rule: strip outer "..." ONLY if both first and last chars are "
    # Treat inner "" as LITERAL (no CSV-unescape inside the answer)
    if len(answer_raw) >= 2 and answer_raw.startswith('"') and answer_raw.endswith('"'):
        answer_raw = answer_raw[1:-1]
    # ANSWER field also uses CSV-unescape: "" → " (same as prompt convention)
    answer_raw = answer_raw.replace('""', '"')
    return rid, prompt, answer_raw

def load_all(path='data/raw/train.csv'):
    _, rows = load_rows(path)
    out = []
    for r in rows:
        rid, p, a = parse_row(r)
        out.append({'id': rid, 'prompt': p, 'answer': a})
    return out

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent
    rows = load_all(BASE/'data/raw/train.csv')
    print(f"Loaded {len(rows)} rows")
    for tid in ['00457d26','b16d6db2','c8b57f1d','4a02017f']:
        r = next((x for x in rows if x['id']==tid), None)
        if not r: print(f"{tid}: missing"); continue
        print(f"\n{tid}: answer={r['answer']!r} (len={len(r['answer'])})")
        # show equation lines
        for ln in r['prompt'].split('\n'):
            ln = ln.strip()
            if (' = ' in ln and len(ln) < 50) or 'determine' in ln.lower():
                print(f"  eq: {ln!r}  (len={len(ln)})")
