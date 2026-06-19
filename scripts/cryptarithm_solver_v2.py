#!/usr/bin/env python3
"""Restored working backward solver (815/823) + strict per-candidate verification."""
import json, re, time
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

def rev_s(n):
    s=str(abs(n));return -int(s[::-1]) if n<0 else int(s[::-1])

def get_ops(a,b):
    mx,mn=max(a,b),min(a,b);ra,rb=rev_s(a),rev_s(b);rmx,rmn=max(ra,rb),min(ra,rb)
    R={}
    R['a+b']=a+b;R['a-b']=a-b;R['b-a']=b-a;R['a*b']=a*b
    R['a||b']=int(f"{a:02d}{b:02d}");R['b||a']=int(f"{b:02d}{a:02d}")
    R['max+min']=mx+mn;R['max-min']=mx-mn;R['max*min']=mx*mn
    R['(a*b)+1']=a*b+1;R['(a*b)-1']=a*b-1;R['(a+b)+1']=a+b+1;R['(a+b)-1']=a+b-1
    R['(b*a)+1']=b*a+1;R['(b*a)-1']=b*a-1;R['(b+a)+1']=b+a+1;R['(b+a)-1']=b+a-1
    if mn>0:R['max%min']=mx%mn
    if b>0:R['a%b']=a%b
    if a>0:R['b%a']=b%a
    R['min-max']=mn-mx;R['max||min']=int(f"{mx:02d}{mn:02d}");R['min||max']=int(f"{mn:02d}{mx:02d}")
    R['rev(ra+rb)']=rev_s(ra+rb);R['rev(ra-rb)']=rev_s(ra-rb);R['rev(rb-ra)']=rev_s(rb-ra)
    R['rev(ra*rb)']=rev_s(ra*rb);R['rev(rb*ra)']=rev_s(rb*ra);R['rev(rb+ra)']=rev_s(rb+ra)
    R['rev(ra||rb)']=rev_s(int(f"{ra:02d}{rb:02d}"));R['rev(rb||ra)']=rev_s(int(f"{rb:02d}{ra:02d}"))
    R['rev(ra*rb+1)']=rev_s(ra*rb+1);R['rev(ra*rb-1)']=rev_s(ra*rb-1)
    R['rev(ra+rb+1)']=rev_s(ra+rb+1);R['rev(ra+rb-1)']=rev_s(ra+rb-1)
    R['rev(rb*ra+1)']=rev_s(rb*ra+1);R['rev(rb+ra+1)']=rev_s(rb+ra+1);R['rev(rb+ra-1)']=rev_s(rb+ra-1)
    if rb>0:R['rev(ra%rb)']=rev_s(ra%rb)
    if ra>0:R['rev(rb%ra)']=rev_s(rb%ra)
    R['rev(rmx+rmn)']=rev_s(rmx+rmn);R['rev(rmx-rmn)']=rev_s(rmx-rmn)
    R['rev(rmx*rmn)']=rev_s(rmx*rmn)
    R['rev(rmx*rmn+1)']=rev_s(rmx*rmn+1);R['rev(rmx*rmn-1)']=rev_s(rmx*rmn-1)
    R['rev(rmx+rmn+1)']=rev_s(rmx+rmn+1);R['rev(rmx+rmn-1)']=rev_s(rmx+rmn-1)
    if rmn>0:R['rev(rmx%rmn)']=rev_s(rmx%rmn)
    R['rev(rmx||rmn)']=rev_s(int(f"{rmx:02d}{rmn:02d}"))
    R['-(max-min)']=-(mx-mn);R['-(b-a)']=-(b-a);R['-(a-b)']=-(a-b)
    R['-rev(rmx-rmn)']=-rev_s(rmx-rmn);R['-rev(ra-rb)']=-rev_s(ra-rb)
    return R

def extend_multi(m,pairs):
    m=dict(m)
    for s,d in pairs:
        if s in m:
            if m[s]!=d:return None
        else:m[s]=d
    return m

def parse_puzzle(prompt):
    equations=[];query=None
    for line in prompt.split('\n'):
        line=line.strip()
        if not line or any(kw in line.lower() for kw in ['alice','secret','each letter','please','boxed','example','final']):continue
        if 'determine' in line.lower():
            m=re.search(r':\s*(.+?)$',line)
            if m:
                q=m.group(1).strip()
                if len(q)>=5:query=(q[0:2],q[2],q[3:5])
            continue
        if ' = ' in line:
            parts=line.split(' = ',1);left=parts[0].strip();right=parts[1].strip()
            if len(left)==5:equations.append((left[0:2],left[2],left[3:5],right))
    return equations,query

def _verify_known(equations, mapping, eq_op_map):
    """All FULLY-KNOWN equations must hold. Returns False if any contradicts."""
    for n1s,op,n2s,res_s in equations:
        if all(s in mapping for s in n1s+n2s+res_s) and op in eq_op_map:
            n1=mapping[n1s[0]]*10+mapping[n1s[1]];n2=mapping[n2s[0]]*10+mapping[n2s[1]]
            rv=int(''.join(str(mapping[s]) for s in res_s))
            ops=get_ops(n1,n2)
            if ops.get(eq_op_map[op])!=rv:return False
    return True

def solve_backward(equations, query, gt_answer, timeout=15):
    start=time.time()
    q1s,qop,q2s=query
    gt_syms=list(gt_answer)
    eq_ops_present=set(op for _,op,_,_ in equations)
    for q1_val in range(100):
        if time.time()-start>timeout:return None
        m0=extend_multi({},[(q1s[0],q1_val//10),(q1s[1],q1_val%10)])
        if not m0:continue
        for q2_val in range(100):
            m1=extend_multi(m0,[(q2s[0],q2_val//10),(q2s[1],q2_val%10)])
            if not m1:continue
            for op_name,rv in get_ops(q1_val,q2_val).items():
                if rv<0:
                    if len(gt_answer)>=2 and gt_answer[0]==qop:
                        rd=[int(c) for c in str(abs(rv))]
                        if len(rd)!=len(gt_answer)-1:continue
                        m2=extend_multi(m1,[(gt_answer[j+1],rd[j]) for j in range(len(rd))])
                    else:continue
                else:
                    rd=[int(c) for c in str(rv)]
                    if len(rd)!=len(gt_answer):continue
                    m2=extend_multi(m1,[(gt_syms[j],rd[j]) for j in range(len(rd))])
                if not m2:continue
                eq_op_map={}
                if qop in eq_ops_present:eq_op_map[qop]=op_name
                mapping=dict(m2)
                # propagate (a few passes)
                for _ in range(4):
                    progressed=False
                    for n1s,op,n2s,res_s in equations:
                        kn_lr=all(s in mapping for s in n1s+n2s)
                        kn_res=all(s in mapping for s in res_s)
                        if kn_lr and op in eq_op_map:
                            n1=mapping[n1s[0]]*10+mapping[n1s[1]];n2=mapping[n2s[0]]*10+mapping[n2s[1]]
                            ops=get_ops(n1,n2);erv=ops.get(eq_op_map[op])
                            if erv is not None and erv>=0:
                                erd=[int(c) for c in str(erv)]
                                if len(erd)==len(res_s):
                                    nm=extend_multi(mapping,[(res_s[j],erd[j]) for j in range(len(erd))])
                                    if nm and nm!=mapping:mapping=nm;progressed=True
                        elif kn_lr and kn_res and op not in eq_op_map:
                            n1=mapping[n1s[0]]*10+mapping[n1s[1]];n2=mapping[n2s[0]]*10+mapping[n2s[1]]
                            expected=int(''.join(str(mapping[s]) for s in res_s))
                            for en,ev in get_ops(n1,n2).items():
                                if ev==expected:eq_op_map[op]=en;progressed=True;break
                    if not progressed:break
                # brute-fill remaining equations (op known): solve unknown number symbols
                for n1s,op,n2s,res_s in equations:
                    if all(s in mapping for s in n1s+n2s+res_s):continue
                    if op not in eq_op_map:continue
                    target_known=all(s in mapping for s in res_s)
                    filled=False
                    for a in range(100):
                        ta_=extend_multi(mapping,[(n1s[0],a//10),(n1s[1],a%10)])
                        if not ta_:continue
                        for b in range(100):
                            tb=extend_multi(ta_,[(n2s[0],b//10),(n2s[1],b%10)])
                            if not tb:continue
                            ops=get_ops(a,b);erv=ops.get(eq_op_map[op])
                            if erv is None or erv<0:continue
                            erd=[int(c) for c in str(erv)]
                            if len(erd)!=len(res_s):continue
                            tc=extend_multi(tb,[(res_s[j],erd[j]) for j in range(len(erd))])
                            if tc:mapping=tc;filled=True;break
                        if filled:break
                # STRICT verify all fully-known equations
                if not _verify_known(equations, mapping, eq_op_map):continue
                # require: every equation has its operator assigned and (ideally) all symbols
                if any(op not in eq_op_map for _,op,_,_ in equations):continue
                eq_op_map[qop]=op_name
                return mapping,eq_op_map
    return None

solve=solve_backward

def load_gt():
    gt={}
    with open(BASE/"data"/"raw"/"train.csv") as f:
        for line in f:
            m=re.search(r'determine.*?:\s*(.{5,}?)"?\s*,\s*"?([^"\n]+)"?\s*$',line)
            if m:
                q=m.group(1).strip().strip('"');a=m.group(2).strip().strip('"')
                if q and a:gt[q]=a
    return gt

if __name__=="__main__":
    gt=load_gt();t0=time.time();ts=tf=tn=tinc=0
    for fname in ["train_cot_cryptarithm_deduce.jsonl","train_cot_cryptarithm_guess.jsonl"]:
        with open(BASE/"all_categorical_splits_v14"/fname) as f:records=[json.loads(l) for l in f]
        cat="GUESS" if "guess" in fname else "DEDUCE";s=fl=ng=inc=0
        for i in range(len(records)):
            user=[m for m in records[i]['messages'] if m['role']=='user'][0]
            eqs,q=parse_puzzle(user['content'])
            if not eqs or not q:fl+=1;continue
            qs=q[0]+q[1]+q[2];ga=gt.get(qs)
            if not ga:
                bx=re.findall(r'\\boxed\{([^}]*)\}',records[i]['messages'][-1]['content']);ga=bx[-1] if bx else None
            if not ga:ng+=1;continue
            r=solve_backward(eqs,q,ga,timeout=10)
            if r:
                mp,om=r
                need=set()
                for n1s,op,n2s,res_s in eqs:need.update(n1s+n2s+res_s)
                need.update(q[0]+q[2])
                if need.issubset(mp):s+=1
                else:inc+=1
            else:fl+=1
            if (i+1)%150==0:print(f"  {cat} {i+1}: full={s} incomplete={inc} fail={fl} [{time.time()-t0:.0f}s]",flush=True)
        print(f"{cat}: full={s} incomplete={inc} failed={fl} nogt={ng}/{len(records)}",flush=True)
        ts+=s;tf+=fl;tn+=ng;tinc+=inc
    print(f"TOTAL full={ts} incomplete={tinc} failed={tf} nogt={tn} ({time.time()-t0:.0f}s)",flush=True)
