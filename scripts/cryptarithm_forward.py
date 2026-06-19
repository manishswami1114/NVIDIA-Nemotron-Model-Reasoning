#!/usr/bin/env python3
"""Forward cryptarithm solver: find ONE mapping satisfying ALL equations, then query.
No tight timeout, no aggressive cap. Most-constrained equation first."""
import json, re, time
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent

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

# ops by result length (positive results)
from collections import defaultdict as dd
OPS_BY_LEN=dd(set)
for _a in range(100):
    for _b in range(100):
        for _n,_v in get_ops(_a,_b).items():
            if _v>=0:OPS_BY_LEN[len(str(_v))].add(_n)
OPS_BY_LEN={k:sorted(v) for k,v in OPS_BY_LEN.items()}

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

def eq_extend(n1s,n2s,res_s,op_name,m):
    """All extended maps satisfying this equation with op_name, given partial m.
    Respects known symbols + internal repetition. Uses result-inversion when result known."""
    res_len=len(res_s);out=[]
    res_known=all(s in m for s in res_s)
    R=int(''.join(str(m[s]) for s in res_s)) if res_known else None
    # num1 candidates
    if n1s[0] in m and n1s[1] in m:n1r=[m[n1s[0]]*10+m[n1s[1]]]
    elif n1s[0] in m:d=m[n1s[0]];n1r=range(d*10,d*10+10)
    elif n1s[1] in m:d=m[n1s[1]];n1r=range(d,100,10)
    elif n1s[0]==n1s[1]:n1r=[d*11 for d in range(10)]
    else:n1r=range(100)
    for num1 in n1r:
        d0,d1=num1//10,num1%10
        if n1s[0]==n1s[1] and d0!=d1:continue
        m1=extend_multi(m,[(n1s[0],d0),(n1s[1],d1)])
        if not m1:continue
        if n2s[0] in m1 and n2s[1] in m1:n2r=[m1[n2s[0]]*10+m1[n2s[1]]]
        elif n2s[0] in m1:d=m1[n2s[0]];n2r=range(d*10,d*10+10)
        elif n2s[1] in m1:d=m1[n2s[1]];n2r=range(d,100,10)
        elif n2s[0]==n2s[1]:n2r=[d*11 for d in range(10)]
        else:n2r=range(100)
        for num2 in n2r:
            d2,d3=num2//10,num2%10
            if n2s[0]==n2s[1] and d2!=d3:continue
            m2=extend_multi(m1,[(n2s[0],d2),(n2s[1],d3)])
            if not m2:continue
            ops=get_ops(num1,num2)
            if op_name not in ops:continue
            rv=ops[op_name]
            if rv<0:continue
            srv=str(rv)
            if len(srv)>res_len:continue
            srv=srv.zfill(res_len)   # allow leading-zero results
            rd=[int(c) for c in srv]
            m3=extend_multi(m2,[(res_s[j],rd[j]) for j in range(res_len)])
            if m3:out.append(m3)
    return out

def forward_solve(equations, query, gt_answer, timeout=60, hard_cap=200000):
    """Find ALL consistent (map,op_map) over ALL equations, then pick one matching GT query."""
    start=time.time();ta=start+timeout
    n=len(equations)
    # candidates: list of (map, op_map)
    cands=[({},{})]
    done=[False]*n
    def internal_rep(eq):
        s=eq[0]+eq[2]+eq[3];return len(s)-len(set(s))
    for step in range(n):
        if time.time()>ta:return None
        # pick next eq: most known symbols (rep map), then internal repetition, then op-known
        rep_map,rep_om=cands[0]
        best=-1;bs=None
        for i in range(n):
            if done[i]:continue
            eq=equations[i]
            kc=sum(1 for s in eq[0]+eq[2]+eq[3] if s in rep_map)
            # prioritize: op already known, more known symbols, SHORTER result, more internal repetition
            sc=(eq[1] in rep_om, kc, -len(eq[3]), internal_rep(eq))
            if bs is None or sc>bs:bs=sc;best=i
        done[best]=True
        n1s,op,n2s,res_s=equations[best]
        nc=[]
        for m,om in cands:
            if time.time()>ta:return None
            ops_try=[om[op]] if op in om else OPS_BY_LEN.get(len(res_s),list(get_ops(12,34)))
            for opn in ops_try:
                for nm in eq_extend(n1s,n2s,res_s,opn,m):
                    no=om if op in om else {**om,op:opn}
                    nc.append((nm,no))
            if len(nc)>hard_cap:break
        # dedup
        seen=set();dd2=[]
        for m,om in nc:
            k=(tuple(sorted(m.items())),tuple(sorted(om.items())))
            if k not in seen:seen.add(k);dd2.append((m,om))
        cands=dd2
        if not cands:return None
    # All equations satisfied by every candidate. Now match query to GT.
    q1s,qop,q2s=query
    qop_in=qop in set(o for _,o,_,_ in equations)
    for m,om in cands:
        if not all(s in m for s in q1s+q2s):
            # query symbol not constrained by equations: enumerate it
            pass
        # fill query symbols if missing by trying digits consistent
        base=m
        # gather query num values (may need enumerate unknown query-only symbols)
        unk=[s for s in set(q1s+q2s) if s not in base]
        # small enumerate
        from itertools import product as _p
        for combo in _p(range(10),repeat=len(unk)):
            mm=dict(base)
            ok=True
            for s,d in zip(unk,combo):
                if s in mm and mm[s]!=d:ok=False;break
                mm[s]=d
            if not ok:continue
            q1=mm[q1s[0]]*10+mm[q1s[1]];q2=mm[q2s[0]]*10+mm[q2s[1]]
            d2s={}
            for s,d in mm.items():
                if d not in d2s:d2s[d]=s
            ops=get_ops(q1,q2)
            qops=[om[qop]] if (qop_in and qop in om) else OPS_BY_LEN.get(len(gt_answer),list(get_ops(12,34)))
            for on in qops:
                if on not in ops:continue
                rv=ops[on]
                if rv<0:
                    if len(gt_answer)>=2 and gt_answer[0]==qop:
                        rd=str(abs(rv))
                        if len(rd)>len(gt_answer)-1:continue
                        rd=rd.zfill(len(gt_answer)-1)
                        if all(int(c) in d2s for c in rd) and qop+''.join(d2s[int(c)] for c in rd)==gt_answer:
                            fo=dict(om);fo[qop]=on;return mm,fo
                    continue
                srv=str(rv)
                if len(srv)>len(gt_answer):continue
                srv=srv.zfill(len(gt_answer))
                if all(int(c) in d2s for c in srv) and ''.join(d2s[int(c)] for c in srv)==gt_answer:
                    fo=dict(om);fo[qop]=on;return mm,fo
    return None

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
    import sys
    gt=load_gt()
    with open(BASE/"all_categorical_splits_v14"/"train_cot_cryptarithm_deduce.jsonl") as f:
        records=[json.loads(l) for l in f]
    # test puzzle 0 and a couple others, verify global consistency
    for idx in [0,1,2,3,5]:
        user=[m for m in records[idx]['messages'] if m['role']=='user'][0]
        eqs,q=parse_puzzle(user['content']);qs=q[0]+q[1]+q[2];ga=gt.get(qs)
        if not ga:print(f"P{idx}: no gt");continue
        t=time.time();r=forward_solve(eqs,q,ga,timeout=60);el=time.time()-t
        if r:
            mp,om=r
            allok=all(get_ops(mp[n1[0]]*10+mp[n1[1]],mp[n2[0]]*10+mp[n2[1]]).get(om[op])==int(''.join(str(mp[s]) for s in res)) for n1,op,n2,res in eqs)
            # check no degenerate (count distinct digits used)
            print(f"P{idx}: {el:.1f}s ops={om} all_eqs_ok={allok} answer_gt={ga}")
            print(f"      map={dict(sorted(mp.items()))}")
        else:
            print(f"P{idx}: FAIL ({el:.1f}s) gt={ga}")
