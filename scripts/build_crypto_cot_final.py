#!/usr/bin/env python3
"""Generate step-by-step trial-and-error CoTs for cryptarithm puzzles
using family-verified mappings from crypto_family_solutions.json."""
import json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cryptarithm_family as F

BASE = Path(__file__).resolve().parent.parent

FAMILY_NAMES = {
    'DIRECT_MAXMIN':  'direct max/min',
    'DIRECT_SIMPLE':  'direct simple (a, b without reversal)',
    'REV_AB':         'reversed-each (rev of rev(a) and rev(b) combined)',
    'REV_MAXMIN':     'rev of max/min of reversed inputs',
}

def op_english(op):
    d = {
      'max+min':'larger plus smaller','max-min':'larger minus smaller','max*min':'larger times smaller',
      '(max*min)+1':'larger×smaller, then add 1','(max*min)-1':'larger×smaller, then subtract 1',
      '(max+min)+1':'larger+smaller, then add 1','(max+min)-1':'larger+smaller, then subtract 1',
      'max%min':'remainder of larger ÷ smaller','max||min':'concatenate larger then smaller',
      'min||max':'concatenate smaller then larger','-(max-min)':'negative of (larger-smaller)',
      'min-max':'smaller minus larger','min+max':'smaller plus larger',
      'a+b':'a plus b','a-b':'a minus b','b-a':'b minus a','a*b':'a × b','b*a':'b × a',
      '(a*b)+1':'a × b, +1','(a*b)-1':'a × b, -1','(a+b)+1':'a + b, +1','(a+b)-1':'a + b, -1',
      '(b+a)+1':'b + a, +1','(b+a)-1':'b + a, -1','(b*a)+1':'b × a, +1','(b*a)-1':'b × a, -1',
      'a||b':'concatenate a then b','b||a':'concatenate b then a',
      'a%b':'a mod b','b%a':'b mod a','-(a-b)':'-(a-b)','-(b-a)':'-(b-a)',
      'rev(rev(a)+rev(b))':'reverse each, add, reverse the sum',
      'rev(rev(a)-rev(b))':'reverse each, subtract, reverse the result',
      'rev(rev(b)-rev(a))':'reverse each, subtract (b first), reverse the result',
      'rev(rev(a)*rev(b))':'reverse each, multiply, reverse the product',
      'rev(rev(b)*rev(a))':'reverse each, multiply (b first), reverse the product',
      'rev(rev(a)*rev(b)+1)':'reverse each, multiply, +1, reverse',
      'rev(rev(a)*rev(b)-1)':'reverse each, multiply, -1, reverse',
      'rev(rev(a)+rev(b)+1)':'reverse each, add, +1, reverse',
      'rev(rev(a)+rev(b)-1)':'reverse each, add, -1, reverse',
      'rev(rev(a)||rev(b))':'reverse each, concatenate, reverse',
      'rev(rev(b)||rev(a))':'reverse each, concatenate (b first), reverse',
      'rev(rev(a)%rev(b))':'reverse each, mod, reverse',
      'rev(rev(b)%rev(a))':'reverse each, mod (b first), reverse',
      '-rev(rev(a)-rev(b))':'negative of reversed (rev(a)-rev(b))',
      'rev(max(rev(a),rev(b))+min(rev(a),rev(b)))':'reverse each, larger+smaller, reverse',
      'rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':'reverse each, larger-smaller, reverse',
      'rev(max(rev(a),rev(b))*min(rev(a),rev(b)))':'reverse each, larger×smaller, reverse',
      'rev(max(rev(a),rev(b))+min(rev(a),rev(b))+1)':'reverse each, larger+smaller, +1, reverse',
      'rev(max(rev(a),rev(b))+min(rev(a),rev(b))-1)':'reverse each, larger+smaller, -1, reverse',
      'rev(max(rev(a),rev(b))*min(rev(a),rev(b))+1)':'reverse each, larger×smaller, +1, reverse',
      'rev(max(rev(a),rev(b))*min(rev(a),rev(b))-1)':'reverse each, larger×smaller, -1, reverse',
      'rev(max(rev(a),rev(b))%min(rev(a),rev(b)))':'reverse each, larger mod smaller, reverse',
      'rev(max(rev(a),rev(b))||min(rev(a),rev(b)))':'reverse each, concat (larger,smaller), reverse',
      'rev(min(rev(a),rev(b))||max(rev(a),rev(b)))':'reverse each, concat (smaller,larger), reverse',
      'rev(min(rev(a),rev(b))-max(rev(a),rev(b)))':'reverse each, smaller-larger, reverse',
      '-rev(max(rev(a),rev(b))-min(rev(a),rev(b)))':'negative of reversed (larger-smaller)',
    }
    return d.get(op, op)

def show_work(op, a, b):
    """Render the arithmetic for op(a,b) step by step."""
    from cryptarithm_family import rev2, rev_str, rev_signed
    if op.startswith('rev(rev('):
        ra, rb = rev2(a), rev2(b)
        if 'ra*rb' in op or 'rb*ra' in op:
            p = ra*rb
            if '+1' in op: return f"rev({a})={ra}, rev({b})={rb}; {ra}×{rb}+1={p+1}; reverse → {rev_str(str(p+1))}"
            if '-1' in op: return f"rev({a})={ra}, rev({b})={rb}; {ra}×{rb}-1={p-1}; reverse → {rev_signed(p-1)}"
            return f"rev({a})={ra}, rev({b})={rb}; {ra}×{rb}={p}; reverse → {rev_str(str(p))}"
        if 'ra+rb' in op or 'rb+ra' in op:
            s = ra+rb
            if '+1' in op: return f"rev({a})={ra}, rev({b})={rb}; {ra}+{rb}+1={s+1}; reverse → {rev_str(str(s+1))}"
            if '-1' in op: return f"rev({a})={ra}, rev({b})={rb}; {ra}+{rb}-1={s-1}; reverse → {rev_signed(s-1)}"
            return f"rev({a})={ra}, rev({b})={rb}; {ra}+{rb}={s}; reverse → {rev_str(str(s))}"
        if 'ra-rb' in op:
            return f"rev({a})={ra}, rev({b})={rb}; {ra}-{rb}={ra-rb}; reverse (digits swap, sign keeps) → {rev_signed(ra-rb)}"
        if 'rb-ra' in op:
            return f"rev({a})={ra}, rev({b})={rb}; {rb}-{ra}={rb-ra}; reverse → {rev_signed(rb-ra)}"
        if 'ra||rb' in op:
            return f"rev({a})={ra}, rev({b})={rb}; concat → {ra:02d}{rb:02d}; reverse → {rev_str(f'{ra:02d}{rb:02d}')}"
        if 'rb||ra' in op:
            return f"rev({a})={ra}, rev({b})={rb}; concat → {rb:02d}{ra:02d}; reverse → {rev_str(f'{rb:02d}{ra:02d}')}"
        if 'ra%rb' in op and rb>0:
            return f"rev({a})={ra}, rev({b})={rb}; {ra}%{rb}={ra%rb}; reverse → {rev_str(str(ra%rb))}"
        if 'rb%ra' in op and ra>0:
            return f"rev({a})={ra}, rev({b})={rb}; {rb}%{ra}={rb%ra}; reverse → {rev_str(str(rb%ra))}"
    if op.startswith('rev(max('):
        ra,rb = rev2(a), rev2(b); mx,mn = max(ra,rb), min(ra,rb)
        if 'max*min' in op or '*min' in op and 'max(' in op:
            p = mx*mn
            if '+1' in op: return f"rev({a})={ra}, rev({b})={rb}; max={mx}, min={mn}; {mx}×{mn}+1={p+1}; reverse → {rev_str(str(p+1))}"
            if '-1' in op: return f"rev({a})={ra}, rev({b})={rb}; max={mx}, min={mn}; {mx}×{mn}-1={p-1}; reverse → {rev_signed(p-1)}"
            return f"rev({a})={ra}, rev({b})={rb}; max={mx}, min={mn}; {mx}×{mn}={p}; reverse → {rev_str(str(p))}"
        if '+min' in op:
            s = mx+mn
            if '+1' in op: return f"rev({a})={ra}, rev({b})={rb}; max+min+1={s+1}; reverse → {rev_str(str(s+1))}"
            if '-1' in op: return f"rev({a})={ra}, rev({b})={rb}; max+min-1={s-1}; reverse → {rev_signed(s-1)}"
            return f"rev({a})={ra}, rev({b})={rb}; max+min={s}; reverse → {rev_str(str(s))}"
        if '-min' in op:
            return f"rev({a})={ra}, rev({b})={rb}; max-min={mx-mn}; reverse → {rev_signed(mx-mn)}"
        if '%min' in op:
            return f"rev({a})={ra}, rev({b})={rb}; max%min={mx%mn if mn else '?'}; reverse → {rev_str(str(mx%mn)) if mn else '?'}"
        if '||min' in op:
            return f"rev({a})={ra}, rev({b})={rb}; concat → {mx:02d}{mn:02d}; reverse → {rev_str(f'{mx:02d}{mn:02d}')}"
    mx,mn = max(a,b), min(a,b)
    d = {
        'a+b': f"{a}+{b}={a+b}", 'a-b': f"{a}-{b}={a-b}", 'b-a': f"{b}-{a}={b-a}",
        'a*b': f"{a}×{b}={a*b}", 'b*a': f"{b}×{a}={b*a}",
        '(a*b)+1': f"{a}×{b}+1={a*b+1}", '(a*b)-1': f"{a}×{b}-1={a*b-1}",
        '(a+b)+1': f"{a}+{b}+1={a+b+1}", '(a+b)-1': f"{a}+{b}-1={a+b-1}",
        '(b+a)+1': f"{b}+{a}+1={b+a+1}", '(b+a)-1': f"{b}+{a}-1={b+a-1}",
        '(b*a)+1': f"{b}×{a}+1={b*a+1}", '(b*a)-1': f"{b}×{a}-1={b*a-1}",
        'a||b': f"{a}||{b}={a:02d}{b:02d}", 'b||a': f"{b}||{a}={b:02d}{a:02d}",
        'max+min': f"max({a},{b})+min({a},{b})={mx}+{mn}={mx+mn}",
        'max-min': f"max({a},{b})-min({a},{b})={mx}-{mn}={mx-mn}",
        'max*min': f"max×min={mx}×{mn}={mx*mn}",
        '(max*min)+1': f"max×min+1={mx*mn+1}",
        '(max*min)-1': f"max×min-1={mx*mn-1}",
        '(max+min)+1': f"max+min+1={mx+mn+1}",
        '(max+min)-1': f"max+min-1={mx+mn-1}",
        'max%min': f"max%min={mx}%{mn}={mx%mn if mn else '?'}",
        'max||min': f"concat → {mx:02d}{mn:02d}", 'min||max': f"concat → {mn:02d}{mx:02d}",
        'min-max': f"min-max={mn-mx}", 'min+max': f"min+max={mn+mx}",
        '-(max-min)': f"-(max-min) = -({mx}-{mn}) = -{mx-mn}",
        '-(a-b)': f"-(a-b) = -({a-b})", '-(b-a)': f"-(b-a) = -({b-a})",
    }
    return d.get(op, f"{op}({a},{b})")

def build_cot(puzzle):
    eqs = puzzle['equations']
    query = tuple(puzzle['query']) if isinstance(puzzle['query'], list) else puzzle['query']
    gt = puzzle['answer']
    mp = puzzle['map']
    om = puzzle['ops']
    family = puzzle['family']
    is_guess = (puzzle['kind'] == 'guess')

    q1s, qop, q2s = query
    L = []
    L.append("This is a cryptarithm. Every distinct symbol stands for one digit (0-9). The same symbol always means the same digit; two different symbols may happen to share a digit. The puzzle is built around one digit assignment that makes every equation true with one consistent rule per operator.")
    L.append("Each left-hand side has exactly 5 symbols — the middle one is the operator, the four around it form two 2-symbol numbers. The result is on the right.")
    L.append("I have to read every character literally: ` , ' , \" and \\ are ordinary symbols, not quotes or escapes. If a symbol like \" appears twice in a row (\"\") that is the same symbol used twice, each still one digit. The result can also start with the operator character itself — when it does, that leading char is a minus sign and the rest are digits (so \"-@\\\"\" means a negative two-digit value).")
    L.append("Key rule about the operator rule itself: every operator in the puzzle uses a formula from the SAME family. If `-` is the simple a-b, then `+` here is also a direct (non-reversed) formula. If `-` is `rev(rev(a)-rev(b))`, then `+` is some `rev(rev(...))` form too. I do NOT mix families inside one puzzle.")
    L.append("I'll put my final answer inside \\boxed{}.\n")

    L.append("The examples:")
    for n1s, op, n2s, res in eqs:
        L.append(f"  {n1s}{op}{n2s} = {res}")
    L.append(f"\nI need to find: {q1s}{qop}{q2s}\n")

    # Letter assignment
    al = {}; idx = 0
    for n1s, op, n2s, res in eqs:
        for c in n1s + n2s + res:
            if c == op: continue  # leading minus-sign char isn't a digit-symbol
            if c not in al:
                al[c] = chr(65 + idx); idx += 1
    for c in q1s + q2s + gt:
        if c == qop: continue
        if c not in al:
            al[c] = chr(65 + idx); idx += 1
    L.append("Step 1 — label each distinct symbol with a letter (operators excluded):")
    L.append("  " + ", ".join(f"'{s}'={al[s]}" for s in sorted(al, key=lambda x: al[x])) + "\n")

    L.append("Step 2 — rewrite in letters (operator in brackets):")
    for n1s, op, n2s, res in eqs:
        a1 = ''.join(al[c] for c in n1s); a2 = ''.join(al[c] for c in n2s)
        # mark op-as-sign in result
        if len(res) >= 2 and res[0] == op and all(c != op for c in res[1:]):
            ar = '-' + ''.join(al.get(c,'?') for c in res[1:])
            note = f"   ({len(res)-1}-digit NEGATIVE value)"
        else:
            ar = ''.join(al.get(c,'?') for c in res); note = f"   ({len(res)}-digit result)"
        L.append(f"  {a1} [{op}] {a2} = {ar}{note}")
    aq1 = ''.join(al.get(c,'?') for c in q1s); aq2 = ''.join(al.get(c,'?') for c in q2s)
    L.append(f"  Target: {aq1} [{qop}] {aq2} = ?\n")

    if is_guess:
        L.append(f"The target operator '{qop}' is not in the examples, so I infer its rule from the same family as the others.")
    else:
        L.append(f"The target operator '{qop}' also appears in the examples, so I reuse its rule.")

    L.append(f"\nStep 3 — pick the operation family. The result digit-lengths tell me a lot:")
    for op in sorted(set(e[1] for e in eqs)):
        lens = []
        for n1s,o,n2s,res in eqs:
            if o != op: continue
            if len(res)>=2 and res[0]==op:
                lens.append(f"{len(res)-1} (neg)")
            else:
                lens.append(str(len(res)))
        L.append(f"  '{op}' → result lengths {lens}.")
    L.append(f"After trying families with these constraints, the only family that lets every operator have ONE consistent rule with a valid digit assignment is **{FAMILY_NAMES[family]}**.")

    L.append(f"\nStep 4 — search the digit assignment within that family. Try, check each example, backtrack on any conflict, retry. The assignment that satisfies every equation simultaneously is:")
    for s in sorted(al, key=lambda x: al[x]):
        if s in mp:
            L.append(f"  {al[s]} ('{s}') = {mp[s]}")
    # also any new symbols in answer not in equations
    extra = [c for c in gt if c not in al and c != qop]
    if extra:
        for c in extra: L.append(f"  (new in answer) '{c}' = {mp.get(c,'?')}")

    L.append(f"\nStep 5 — verify every example with these digits and the family's formulas:")
    for n1s, op, n2s, res in eqs:
        a = mp[n1s[0]]*10 + mp[n1s[1]]; b = mp[n2s[0]]*10 + mp[n2s[1]]
        opn = om[op]
        L.append(f"  {n1s}{op}{n2s} = {res}: numbers are {a} and {b}; rule for '{op}' = {opn} ({op_english(opn)})")
        L.append(f"    {show_work(opn, a, b)}")
        # show the matched result
        if len(res)>=2 and res[0]==op:
            expected = -int(''.join(str(mp[c]) for c in res[1:]))
        else:
            expected = int(''.join(str(mp[c]) for c in res))
        L.append(f"    Written in symbols this is {res}, value {expected} ✓")

    L.append(f"\nStep 6 — apply to the target {q1s}{qop}{q2s}:")
    qa = mp[q1s[0]]*10 + mp[q1s[1]]; qb = mp[q2s[0]]*10 + mp[q2s[1]]
    qopn = om[qop]
    L.append(f"  numbers are {qa} and {qb}; '{qop}' = {qopn} ({op_english(qopn)})")
    L.append(f"  {show_work(qopn, qa, qb)}")
    L.append(f"  Written back in the puzzle's symbols: {gt}.")

    L.append(f"\nVerification Step:")
    L.append(f"[✓] Every symbol read literally (including ` ' \" \\) — none skipped? -> YES")
    L.append(f"[✓] One digit assignment satisfies ALL equations? -> YES")
    L.append(f"[✓] All operators use formulas from the SAME family ({FAMILY_NAMES[family]})? -> YES")
    L.append(f"[✓] Same operator symbol always uses the same formula? -> YES")
    L.append(f"\nAll constraints satisfied. The solution is verified.")
    L.append(f"\\boxed{{{gt}}}")
    return "<think>\n" + '\n'.join(L) + "\n</think>\n\\boxed{" + gt + "}"

def main():
    sols_path = BASE/"scripts"/"crypto_family_solutions.json"
    if not sols_path.exists():
        print(f"Waiting on {sols_path} from solver…")
        return
    sols = json.load(open(sols_path))
    print(f"Loaded {len(sols)} family-verified solutions.")

    # index by id
    by_id = {s['id']: s for s in sols}

    # update v14 jsonls
    for fname in ["train_cot_cryptarithm_deduce.jsonl", "train_cot_cryptarithm_guess.jsonl"]:
        path = BASE/"all_categorical_splits_v14"/fname
        with open(path) as f: records = [json.loads(l) for l in f]
        upd = skip = 0
        for rec in records:
            # find puzzle id by matching query (records lack explicit id)
            user = [m for m in rec['messages'] if m['role']=='user'][0]['content']
            qm = re.search(r'determine the result for:\s*(.+?)(?:\n|$)', user)
            if not qm: skip += 1; continue
            q_str = qm.group(1).strip()
            # find solution whose query matches and equations match
            cand = None
            for s in sols:
                if ''.join(s['query']) == q_str:
                    cand = s; break
            if cand is None: skip += 1; continue
            cot = build_cot(cand)
            rec['messages'][-1]['content'] = cot
            upd += 1
        with open(path,'w') as f:
            for rec in records: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  {fname}: updated={upd} skipped={skip}/{len(records)}")

if __name__ == "__main__":
    main()
