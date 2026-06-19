#!/usr/bin/env python3
"""
Cryptarithm solver v3: GT-guided with strict length filtering.

Strategy:
1. Strict result-length filtering (a+b can't produce 4 digits)
2. For each operator, only try ops that can produce ALL its result lengths
3. Enumerate operation combos (typically 17^2 = 289 max)
4. For each combo, propagate constraints equation-by-equation
5. Verify answer against ground truth from train.csv
"""
import json, re, time, sys
from collections import defaultdict
from itertools import product
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def rev_s(n):
    s = str(abs(n))
    return -int(s[::-1]) if n < 0 else int(s[::-1])

def get_ops(a, b):
    mx, mn = max(a,b), min(a,b)
    ra, rb = rev_s(a), rev_s(b)
    rmx, rmn = max(ra,rb), min(ra,rb)
    R = {}
    R['a+b']=a+b; R['a-b']=a-b; R['b-a']=b-a; R['a*b']=a*b
    R['a||b']=int(f"{a:02d}{b:02d}"); R['b||a']=int(f"{b:02d}{a:02d}")
    R['max+min']=mx+mn; R['max-min']=mx-mn; R['max*min']=mx*mn
    R['(a*b)+1']=a*b+1; R['(a*b)-1']=a*b-1
    R['(a+b)+1']=a+b+1; R['(a+b)-1']=a+b-1
    if mn>0: R['max%min']=mx%mn
    if b>0: R['a%b']=a%b
    if a>0: R['b%a']=b%a
    R['min-max']=mn-mx
    R['max||min']=int(f"{mx:02d}{mn:02d}"); R['min||max']=int(f"{mn:02d}{mx:02d}")
    R['rev(ra+rb)']=rev_s(ra+rb); R['rev(ra-rb)']=rev_s(ra-rb); R['rev(rb-ra)']=rev_s(rb-ra)
    R['rev(ra*rb)']=rev_s(ra*rb)
    R['rev(ra||rb)']=rev_s(int(f"{ra:02d}{rb:02d}")); R['rev(rb||ra)']=rev_s(int(f"{rb:02d}{ra:02d}"))
    R['rev(ra*rb+1)']=rev_s(ra*rb+1); R['rev(ra*rb-1)']=rev_s(ra*rb-1)
    R['rev(ra+rb+1)']=rev_s(ra+rb+1); R['rev(ra+rb-1)']=rev_s(ra+rb-1)
    if rb>0: R['rev(ra%rb)']=rev_s(ra%rb)
    if ra>0: R['rev(rb%ra)']=rev_s(rb%ra)
    R['rev(rmx+rmn)']=rev_s(rmx+rmn); R['rev(rmx-rmn)']=rev_s(rmx-rmn)
    R['rev(rmx*rmn)']=rev_s(rmx*rmn)
    R['rev(rmx*rmn+1)']=rev_s(rmx*rmn+1); R['rev(rmx*rmn-1)']=rev_s(rmx*rmn-1)
    R['rev(rmx+rmn+1)']=rev_s(rmx+rmn+1); R['rev(rmx+rmn-1)']=rev_s(rmx+rmn-1)
    if rmn>0: R['rev(rmx%rmn)']=rev_s(rmx%rmn)
    R['rev(rmx||rmn)']=rev_s(int(f"{rmx:02d}{rmn:02d}"))
    R['-(max-min)']=-(mx-mn); R['-rev(rmx-rmn)']=-rev_s(rmx-rmn)
    return R

OP_NAMES = list(get_ops(12,34).keys())

# Pre-compute which ops can produce each digit count
# Check ALL 100x100 pairs
OPS_CAN_PRODUCE = defaultdict(set)  # digit_count -> set of op_names
for op_name in OP_NAMES:
    for a in range(100):
        for b in range(100):
            ops = get_ops(a, b)
            if op_name in ops:
                rv = ops[op_name]
                if rv >= 0:
                    OPS_CAN_PRODUCE[len(str(rv))].add(op_name)

def extend_multi(mapping, pairs):
    m = dict(mapping)
    for sym, digit in pairs:
        if sym in m:
            if m[sym] != digit: return None
        else: m[sym] = digit
    return m

def parse_puzzle(prompt):
    equations = []; query = None
    for line in prompt.split('\n'):
        line = line.strip()
        if not line or any(kw in line.lower() for kw in ['alice','secret','each letter','please','boxed','example','final']): continue
        if 'determine' in line.lower():
            m = re.search(r':\s*(.+?)$', line)
            if m:
                q = m.group(1).strip()
                if len(q) >= 5: query = (q[0:2], q[2], q[3:5])
            continue
        if ' = ' in line:
            parts = line.split(' = ', 1)
            left = parts[0].strip(); right = parts[1].strip()
            if len(left) == 5: equations.append((left[0:2], left[2], left[3:5], right))
    return equations, query

def order_equations(equations):
    if not equations: return []
    remaining = list(range(len(equations))); ordered = []; known = set()
    while remaining:
        best = max(remaining, key=lambda i: (
            len(set(equations[i][0]+equations[i][2]+equations[i][3]) & known)*100
            + (10-len(equations[i][3]))))
        ordered.append(best)
        known.update(equations[best][0]+equations[best][2]+equations[best][3])
        remaining.remove(best)
    return ordered

def solve_eq(n1s, n2s, res_s, op_name, mapping):
    """Solve one equation. Returns list of valid extended mappings."""
    res_len = len(res_s)
    results = []
    
    if n1s[0] in mapping and n1s[1] in mapping:
        n1r = [mapping[n1s[0]]*10+mapping[n1s[1]]]
    elif n1s[0] in mapping:
        d=mapping[n1s[0]]; n1r=range(d*10,d*10+10)
    elif n1s[1] in mapping:
        d=mapping[n1s[1]]; n1r=range(d,100,10)
    elif n1s[0]==n1s[1]:
        n1r=[d*11 for d in range(10)]
    else: n1r=range(100)
    
    for num1 in n1r:
        d0,d1=num1//10,num1%10
        if n1s[0]==n1s[1] and d0!=d1: continue
        m1=extend_multi(mapping,[(n1s[0],d0),(n1s[1],d1)])
        if not m1: continue
        
        if n2s[0] in m1 and n2s[1] in m1:
            n2r=[m1[n2s[0]]*10+m1[n2s[1]]]
        elif n2s[0] in m1:
            d=m1[n2s[0]]; n2r=range(d*10,d*10+10)
        elif n2s[1] in m1:
            d=m1[n2s[1]]; n2r=range(d,100,10)
        elif n2s[0]==n2s[1]:
            n2r=[d*11 for d in range(10)]
        else: n2r=range(100)
        
        for num2 in n2r:
            d2,d3=num2//10,num2%10
            if n2s[0]==n2s[1] and d2!=d3: continue
            m2=extend_multi(m1,[(n2s[0],d2),(n2s[1],d3)])
            if not m2: continue
            ops=get_ops(num1,num2)
            if op_name not in ops: continue
            rv=ops[op_name]
            if rv<0: continue
            rd=[int(c) for c in str(rv)]
            if len(rd)!=res_len: continue
            m3=extend_multi(m2,[(res_s[j],rd[j]) for j in range(res_len)])
            if m3: results.append(m3)
    return results

def get_valid_ops_for_operator(equations, op_char):
    """Get operations that CAN produce correct result lengths for ALL equations with this operator."""
    eqs = [(n1,n2,res) for n1,op,n2,res in equations if op==op_char]
    result_lens = [len(res) for _,_,res in eqs]
    
    # Strict filter: op must be able to produce EACH result length
    valid = []
    for op_name in OP_NAMES:
        ok = True
        for rl in result_lens:
            if op_name not in OPS_CAN_PRODUCE.get(rl, set()):
                ok = False; break
        if ok:
            valid.append(op_name)
    return valid

def solve_puzzle(equations, query, gt_answer=None, timeout=30):
    """Solve with strict filtering + GT verification."""
    start = time.time()
    
    # Get unique operators and their valid operations
    op_groups = defaultdict(list)
    for i, (n1,op,n2,res) in enumerate(equations):
        op_groups[op].append(i)
    
    unique_ops = list(op_groups.keys())
    op_valid = {op: get_valid_ops_for_operator(equations, op) for op in unique_ops}
    
    # Order equations for constraint propagation
    order = order_equations(equations)
    
    # Try operation combos
    combo_lists = [[(op, on) for on in op_valid[op]] for op in unique_ops]
    if not all(combo_lists): return None
    
    total_combos = 1
    for cl in combo_lists: total_combos *= len(cl)
    
    for combo in product(*combo_lists):
        if time.time()-start > timeout: break
        
        op_map = {oc: on for oc, on in combo}
        
        # Propagate constraints equation by equation
        candidates = [{}]
        
        for eq_idx in order:
            if time.time()-start > timeout: break
            n1s, op, n2s, res_s = equations[eq_idx]
            next_cands = []
            
            for mapping in candidates:
                if time.time()-start > timeout: break
                valid = solve_eq(n1s, n2s, res_s, op_map[op], mapping)
                next_cands.extend(valid)
                if len(next_cands) > 5000: break
            
            # Dedup
            if len(next_cands) > 500:
                seen = set(); deduped = []
                for m in next_cands:
                    key = tuple(sorted(m.items()))
                    if key not in seen: seen.add(key); deduped.append(m)
                next_cands = deduped[:5000]
            
            candidates = next_cands
            if not candidates: break
        
        if not candidates: continue
        
        # Try to solve query with these candidates
        if not query: continue
        q1s, qop, q2s = query
        
        for mapping in candidates[:100]:
            if not all(s in mapping for s in q1s+q2s): continue
            q1 = mapping[q1s[0]]*10+mapping[q1s[1]]
            q2 = mapping[q2s[0]]*10+mapping[q2s[1]]
            
            d2s = {}
            for sym, dig in mapping.items():
                if dig not in d2s: d2s[dig] = sym
            
            ops_try = [op_map[qop]] if qop in op_map else list(op_valid.get(qop, OP_NAMES))
            
            for op_name in ops_try:
                ops = get_ops(q1, q2)
                if op_name not in ops: continue
                rv = ops[op_name]
                
                if rv < 0:
                    rd = [int(c) for c in str(abs(rv))]
                    if all(d in d2s for d in rd):
                        answer = qop + ''.join(d2s[d] for d in rd)
                        if gt_answer is None or answer == gt_answer:
                            return mapping, op_map, answer, len(candidates), time.time()-start
                    continue
                
                rd = [int(c) for c in str(rv)]
                if all(d in d2s for d in rd):
                    answer = ''.join(d2s[d] for d in rd)
                    if gt_answer is None or answer == gt_answer:
                        return mapping, op_map, answer, len(candidates), time.time()-start
    
    return None

def load_ground_truth():
    """Load GT answers from train.csv."""
    gt = {}
    with open(BASE / "data" / "raw" / "train.csv") as f:
        for line in f:
            # Find query in line, answer is after last comma
            m = re.search(r'determine.*?:\s*(.{5,}?)"?\s*,\s*"?([^"\n]+)"?\s*$', line)
            if m:
                query = m.group(1).strip().strip('"')
                answer = m.group(2).strip().strip('"')
                if query and answer:
                    gt[query] = answer
    return gt

def main():
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} entries")
    
    total_solved = total_matched = total_failed = total_records = 0
    
    for fname in ["train_cot_cryptarithm_deduce.jsonl", "train_cot_cryptarithm_guess.jsonl"]:
        path = BASE / "all_categorical_splits_v14" / fname
        with open(path) as f:
            records = [json.loads(l) for l in f]
        
        cat = "DEDUCE" if "deduce" in fname else "GUESS"
        n = len(records)
        total_records += n
        print(f"\n{'='*60}\n{cat}: {n} records (first 50)")
        
        solved = matched = failed = 0
        
        for i in range(min(50, n)):
            user = [m for m in records[i]['messages'] if m['role']=='user'][0]
            equations, query = parse_puzzle(user['content'])
            if not equations or not query:
                failed += 1; continue
            
            # Find GT
            query_str = query[0] + query[1] + query[2]
            gt_ans = gt.get(query_str)
            if not gt_ans:
                # Try from existing boxed
                existing = records[i]['messages'][-1]['content']
                boxed = re.findall(r'\\boxed\{([^}]*)\}', existing)
                gt_ans = boxed[-1] if boxed else None
            
            # Solve WITH GT guidance (verify answer matches)
            result = solve_puzzle(equations, query, gt_answer=gt_ans, timeout=20)
            
            if result:
                mapping, op_map, answer, nc, elapsed = result
                mk = "✓"
                matched += 1; solved += 1
                if i < 15 or True:
                    print(f"  [{i:2d}] ✓ ans='{answer}' ({elapsed:.1f}s {nc}c) ops={op_map}")
            else:
                # Try without GT
                result2 = solve_puzzle(equations, query, gt_answer=None, timeout=10)
                if result2:
                    mapping, op_map, answer, nc, elapsed = result2
                    mk = "?" 
                    solved += 1
                    if i < 15:
                        print(f"  [{i:2d}] ? ans='{answer}' gt='{gt_ans}' ({elapsed:.1f}s)")
                else:
                    failed += 1
                    if i < 15:
                        print(f"  [{i:2d}] FAIL gt='{gt_ans}'")
        
        print(f"\n  Matched={matched} Solved={solved} Failed={failed}/50")
        total_solved += solved; total_matched += matched; total_failed += failed
    
    print(f"\n{'='*60}")
    print(f"TOTAL: Matched={total_matched} Solved={total_solved} Failed={total_failed}")

if __name__ == "__main__":
    main()
