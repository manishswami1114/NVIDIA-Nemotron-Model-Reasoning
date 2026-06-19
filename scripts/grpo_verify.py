#!/usr/bin/env python3
"""GRPO reward verifier for V22.

reward(completion_text, sample) -> 1.0 if the rollout's \\boxed{} answer matches
the gold answer under the sample's per-category `verify` rule, else 0.0.

Usage in a GRPO loop:
    from grpo_verify import reward
    r = reward(model_output_text, sample_dict)
"""
import re, math


def extract_boxed(text):
    """Return the last \\boxed{...} content (brace-balanced), or None."""
    starts = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    if not starts:
        # fallback: a bare 'answer:' line
        m = re.findall(r'(?:final answer|answer)\s*[:=]\s*(.+)', text, re.IGNORECASE)
        return m[-1].strip() if m else None
    s = starts[-1] + len(r'\boxed{')
    # Cryptarithm answers can contain '{' and '}' as symbols, which breaks brace
    # balancing. The boxed answer is the final token, so take everything from the
    # last \boxed{ up to the LAST '}' in the text.
    last_brace = text.rfind('}')
    if last_brace > s:
        return text[s:last_brace].strip()
    # fallback: brace-balanced scan
    depth = 1
    i = s
    while i < len(text) and depth:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[s:i].strip()


def _norm_text(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def match(pred, gold, verify):
    if pred is None:
        return False
    pred = pred.strip()
    gold = str(gold).strip()
    if verify == 'numeric':
        try:
            return math.isclose(float(pred), float(gold), rel_tol=1e-2, abs_tol=1e-2)
        except ValueError:
            return False
    if verify == 'text':
        return _norm_text(pred) == _norm_text(gold)
    # exact (bit strings, roman numerals, eq/crypto answers)
    return pred == gold


def reward(completion_text, sample):
    pred = extract_boxed(completion_text)
    return 1.0 if match(pred, sample['answer'], sample.get('verify', 'exact')) else 0.0


if __name__ == "__main__":
    # self-test
    s = {'answer': '10010111', 'verify': 'exact'}
    assert reward("...\\boxed{10010111}", s) == 1.0
    assert reward("...\\boxed{10010110}", s) == 0.0
    s = {'answer': '154.62', 'verify': 'numeric'}
    assert reward("\\boxed{154.63}", s) == 1.0
    s = {'answer': 'cat imagines book', 'verify': 'text'}
    assert reward("\\boxed{Cat  imagines book}", s) == 1.0
    print("grpo_verify self-test passed")
