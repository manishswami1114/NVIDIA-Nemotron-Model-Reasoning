#!/usr/bin/env python3
"""Cryptarithm solver: operation-combo + DFS-backtracking over equations.
For each op-per-operator combo (freq+length ordered), precompute allowed (n1,n2,res) triples
per equation, then DFS assigning most-constrained equation first."""
import json, re, time
from pathlib import Path
from itertools import product
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

OP_FREQ=['a||b','b||a','rev(ra*rb)','rev(rb*ra)','rev(ra+rb)','rev(rb+ra)','rev(ra||rb)','rev(rb||ra)',
    '(a*b)+1','(a*b)-1','a*b','a+b','a-b','b-a','max-min','max+min','max*min',
    'rev(ra-rb)','rev(rb-ra)','rev(ra*rb+1)','rev(ra*rb-1)','rev(ra+rb+1)','rev(ra+rb-1)',
    'rev(rb*ra+1)','rev(rb+ra+1)','rev(rb+ra-1)','(a+b)+1','(a+b)-1','(b*a)+1','(b*a)-1',
    '(b+a)+1','(b+a)-1','max%min','a%b','b%a','min-max','max||min','min||max',
    'rev(rmx+rmn)','rev(rmx-rmn)','rev(rmx*rmn)','rev(rmx*rmn+1)','rev(rmx*rmn-1)',
    'rev(rmx+rmn+1)','rev(rmx+rmn-1)','rev(rmx%rmn)','rev(rmx||rmn)','rev(ra%rb)','rev(rb%ra)',
    '-(max-min)','-(b-a)','-(a-b)','-rev(rmx-rmn)','-rev(ra-rb)']
ALLOPS=[o for o in OP_FREQ if o in get_ops(12,34)]
for o in get_ops(12,34):
    if o not in ALLOPS:ALLOPS.append(o)

# Precompute, per op, list of (n1,n2,res_value) for all 100x100 (positive results only)
OP_TRIPLES={}
for op in ALLOPS:
    lst=[]
    for a in range(100):
        for b in range(100):
            v=get_ops(a,b).get(op)
            if v is not None and v>=0:lst.append((a,b,v))
    OP_TRIPLES[op]=lst

OP_TRIPLES_ALL={}
for op in ALLOPS:
    lst=[]
    for a in range(100):
        for b in range(100):
            v=get_ops(a,b).get(op)
            if v is not None:lst.append((a,b,v))
    OP_TRIPLES_ALL[op]=lst

# ops by result length
from collections import defaultdict as dd
OPS_BY_LEN=dd(list)
for op in ALLOPS:
    lens=set(len(str(v)) for _,_,v in OP_TRIPLES[op])
    for L in lens:OPS_BY_LEN[L].append(op)


def _op_result_str(op, a, b):
    """Natural result string for op (preserves leading zeros for concatenation)."""
    ra,rb=rev_s(a),rev_s(b)
    if op=='a||b':return f"{a:02d}{b:02d}"
    if op=='b||a':return f"{b:02d}{a:02d}"
    if op=='max||min':
        mx,mn=max(a,b),min(a,b);return f"{mx:02d}{mn:02d}"
    if op=='min||max':
        mx,mn=max(a,b),min(a,b);return f"{mn:02d}{mx:02d}"
    if op=='rev(ra||rb)':return str(rev_s(int(f"{ra:02d}{rb:02d}")))
    if op=='rev(rb||ra)':return str(rev_s(int(f"{rb:02d}{ra:02d}")))
    if op=='rev(rmx||rmn)':
        rmx,rmn=max(ra,rb),min(ra,rb);return str(rev_s(int(f"{rmx:02d}{rmn:02d}")))
    v=get_ops(a,b).get(op)
    if v is None:return None
    return str(v)  # may be negative

def extend(m,pairs):
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

def assign_triple(m, n1s,n2s,res_s, n1,n2,resv):
    """Try assigning num1=n1, num2=n2, result=resv (zero-padded to len(res_s)) into mapping."""
    srv=str(resv)
    if len(srv)>len(res_s):return None
    srv=srv.zfill(len(res_s))
    pairs=[(n1s[0],n1//10),(n1s[1],n1%10),(n2s[0],n2//10),(n2s[1],n2%10)]
    pairs+=[(res_s[j],int(srv[j])) for j in range(len(res_s))]
    return extend(m,pairs)

def _consistent_options(eq, m, om):
    """Return list of (op,a,b,v) consistent with current digit-map m and op-map om for this equation."""
    n1s,op,n2s,res_s=eq
    L=len(res_s)
    if op in om: ops=[om[op]]
    else: ops=EQ_CANDIDATE_OPS.get((n1s,op,n2s,tuple(res_s)),OPS_BY_LEN.get(L,ALLOPS))
    out=[]
    k1a=m.get(n1s[0]);k1b=m.get(n1s[1]);k2a=m.get(n2s[0]);k2b=m.get(n2s[1])
    res_known=all(s in m for s in res_s)
    Rval=int(''.join(str(m[s]) for s in res_s)) if res_known else None
    for opn in ops:
        for (a,b,v) in OP_TRIPLES[opn]:
            if len(str(v))>L:continue
            if k1a is not None and a//10!=k1a:continue
            if k1b is not None and a%10!=k1b:continue
            if k2a is not None and b//10!=k2a:continue
            if k2b is not None and b%10!=k2b:continue
            if Rval is not None and v!=Rval and str(v).zfill(L)!=str(Rval).zfill(L):continue
            out.append((opn,a,b,v))
    return out

EQ_CANDIDATE_OPS={}

def solve_full(equations, ta):
    """Unified DFS: discover ops + digits together. MRV equation selection."""
    n=len(equations)
    holder=[None]
    def dfs(m, om, done):
        if time.time()>ta:return False
        if len(done)==n:
            holder[0]=(dict(m),dict(om));return True
        # MRV: pick unassigned eq with fewest consistent options
        best=-1;best_opts=None
        for i in range(n):
            if i in done:continue
            opts=_consistent_options(equations[i],m,om)
            if best_opts is None or len(opts)<len(best_opts):
                best_opts=opts;best=i
                if len(opts)==0:break
        if not best_opts:return False
        n1s,op,n2s,res_s=equations[best]
        nd=done|{best}
        for (opn,a,b,v) in best_opts:
            if time.time()>ta:return False
            srv=str(v).zfill(len(res_s))
            pairs=[(n1s[0],a//10),(n1s[1],a%10),(n2s[0],b//10),(n2s[1],b%10)]
            pairs+=[(res_s[j],int(srv[j])) for j in range(len(res_s))]
            m2=extend(m,pairs)
            if not m2:continue
            om2=om if op in om else {**om,op:opn}
            if dfs(m2,om2,nd):return True
        return False
    if dfs({},{},frozenset()):return holder[0]
    return None



# Index triples by num1 for fast lookup; store natural result string
OP_INDEX={}
for op in ALLOPS:
    by_a={}
    for a in range(100):
        row=[]
        for b in range(100):
            rs=_op_result_str(op,a,b)
            if rs is None:continue
            row.append((b,rs))
        by_a[a]=row
    OP_INDEX[op]=by_a

CONCAT_OPS={'a||b','b||a','max||min','min||max','rev(ra||rb)','rev(rb||ra)','rev(rmx||rmn)'}

def solve_full_q(equations, ta, neg_query=None, min_distinct=4):
    n=len(equations)
    holder=[None]
    def options(eq,m,om):
        n1s,op,n2s,res_s=eq
        L=len(res_s)
        is_neg = neg_query is not None and (n1s,op,n2s,tuple(res_s))==neg_query
        if op in om: ops=[om[op]]
        else: ops=OPS_BY_LEN.get(L,ALLOPS) if not is_neg else ALLOPS
        out=[]
        k1a=m.get(n1s[0]);k1b=m.get(n1s[1]);k2a=m.get(n2s[0]);k2b=m.get(n2s[1])
        rk=all(s in m for s in res_s)
        Rstr=''.join(str(m[s]) for s in res_s) if rk else None
        # candidate num1 values respecting known digits
        if k1a is not None and k1b is not None:a_vals=[k1a*10+k1b]
        elif k1a is not None:a_vals=range(k1a*10,k1a*10+10)
        elif k1b is not None:a_vals=range(k1b,100,10)
        elif n1s[0]==n1s[1]:a_vals=[d*11 for d in range(10)]
        else:a_vals=range(100)
        for opn in ops:
            concat=opn in CONCAT_OPS
            byA=OP_INDEX[opn]
            for a in a_vals:
                if n1s[0]==n1s[1] and a//10!=a%10:continue
                for (b,rs) in byA[a]:
                    if k2a is not None and b//10!=k2a:continue
                    if k2b is not None and b%10!=k2b:continue
                    if n2s[0]==n2s[1] and b//10!=b%10:continue
                    r=rs
                    if is_neg:
                        if not r.startswith('-'):continue
                        r=r[1:]
                    elif r.startswith('-'):continue
                    if concat:
                        if len(r)!=L:continue
                    else:
                        if len(r)>L:continue
                        r=r.zfill(L)
                    if Rstr is not None and r!=Rstr:continue
                    out.append((opn,a,b,r))
        out.sort(key=lambda t:-len(set(f"{t[1]:02d}{t[2]:02d}{t[3]}")))
        return out
    def dfs(m,om,done):
        if time.time()>ta:return False
        if len(done)==n:
            if len(set(m.values()))<min_distinct:return False
            holder[0]=(dict(m),dict(om));return True
        best=-1;bo=None
        for i in range(n):
            if i in done:continue
            o=options(equations[i],m,om)
            if bo is None or len(o)<len(bo):
                bo=o;best=i
                if not o:break
        if not bo:return False
        n1s,op,n2s,res_s=equations[best];nd=done|{best}
        for (opn,a,b,rs) in bo:
            if time.time()>ta:return False
            pairs=[(n1s[0],a//10),(n1s[1],a%10),(n2s[0],b//10),(n2s[1],b%10)]
            pairs+=[(res_s[j],int(rs[j])) for j in range(len(res_s))]
            m2=extend(m,pairs)
            if not m2:continue
            om2=om if op in om else {**om,op:opn}
            if dfs(m2,om2,nd):return True
        return False
    if dfs({},{},frozenset()):return holder[0]
    return None

def solve(equations, query, gt_answer, timeout=30):
    start=time.time();ta=start+timeout
    global EQ_CANDIDATE_OPS
    q1s,qop,q2s=query
    # Build query-as-equation. Two forms:
    #  normal: result symbols = gt_answer  (op produces positive value)
    #  op-prefix (negative): gt_answer[0]==qop and rest are symbols -> op produces negative
    neg_prefix = len(gt_answer)>=2 and gt_answer[0]==qop and all(c!=qop for c in gt_answer[1:])
    qres = gt_answer[1:] if neg_prefix else gt_answer
    # if gt_answer still contains operator chars (other special forms) -> cannot model cleanly
    bad = any(not (c.isprintable()) for c in qres)
    eqs_plus = list(equations) + [(q1s, qop, q2s, qres)]

    EQ_CANDIDATE_OPS={}
    for n1s,op,n2s,res_s in eqs_plus:
        L=len(res_s)
        # for the query eq in neg_prefix mode, the op yields negative -> allow neg ops
        EQ_CANDIDATE_OPS[(n1s,op,n2s,tuple(res_s))]=OPS_BY_LEN.get(L,ALLOPS)

    # If neg_prefix, we must allow the query op to be negative. Patch _consistent_options via flag.
    res=solve_full_q(eqs_plus, ta, neg_query=( (q1s,qop,q2s,tuple(qres)) if neg_prefix else None))
    if not res:return None
    m,om=res
    # sanity: verify all original equations hold
    for n1s,op,n2s,res_s in equations:
        n1=m[n1s[0]]*10+m[n1s[1]];n2=m[n2s[0]]*10+m[n2s[1]]
        rv=int(''.join(str(m[s]) for s in res_s))
        if get_ops(n1,n2).get(om.get(op))!=rv:return None
    return m,om

solve_backward=solve

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
    gt=load_gt()
    with open(BASE/"all_categorical_splits_v14"/"train_cot_cryptarithm_deduce.jsonl") as f:
        records=[json.loads(l) for l in f]
    ok=fail=0;t0=time.time()
    for i in range(15):
        user=[m for m in records[i]['messages'] if m['role']=='user'][0]
        eqs,q=parse_puzzle(user['content']);qs=q[0]+q[1]+q[2];ga=gt.get(qs)
        if not ga:continue
        t=time.time();r=solve(eqs,q,ga,timeout=20);el=time.time()-t
        if r:
            mp,om=r
            allok=all(get_ops(mp[n1[0]]*10+mp[n1[1]],mp[n2[0]]*10+mp[n2[1]]).get(om[op])==int(''.join(str(mp[s]) for s in res)) for n1,op,n2,res in eqs)
            ok+=1;print(f"  P{i}: OK {el:.2f}s eqs_ok={allok}",flush=True)
        else:
            fail+=1;print(f"  P{i}: FAIL {el:.2f}s gt={ga}",flush=True)
    print(f"15: ok={ok} fail={fail} ({time.time()-t0:.0f}s)",flush=True)
