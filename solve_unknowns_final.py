#!/usr/bin/env python3
"""Final disambiguation for 'unknown logic' lines.
Group unknown lines by (block, operator). The shared rule must satisfy EVERY
line in that group, so intersect candidate formula-sets across the group.
Also pull in any same-operator sibling in the block that already has a formula."""
import re
from collections import defaultdict

def rev_int(s): return int(s[::-1])
CAND=['a-b','a+b','a*b','b-a','abs(a-b)','-abs(a-b)','a+b-1','a+b+1','(a*b)+1','(a*b)-1',
 'rev(a)+rev(b)','rev(a)-rev(b)','rev(b)-rev(a)','rev(a)*rev(b)','rev(a)+rev(b)+1',
 'rev(a)+rev(b)-1','(rev(a)*rev(b))+1','(rev(a)*rev(b))-1','abs(rev(a)-rev(b))',
 '-abs(rev(a)-rev(b))','max(a,b)%min(a,b)','max(rev(a),rev(b))%min(rev(a),rev(b))',
 'a||b','b||a','rev(a)||rev(b)']

def calc(expr,a_str,b_str):
    a,b=int(a_str),int(b_str); ra,rb=rev_int(a_str),rev_int(b_str)
    def smod(x,y): return x%y if y else None
    if expr=='a||b':           return ('STR',a_str+b_str)
    if expr=='b||a':           return ('STR',b_str+a_str)
    if expr=='rev(a)||rev(b)': return ('STR',a_str[::-1]+b_str[::-1])
    t={'a-b':a-b,'a+b':a+b,'a*b':a*b,'b-a':b-a,'abs(a-b)':abs(a-b),'-abs(a-b)':-abs(a-b),
       'a+b-1':a+b-1,'a+b+1':a+b+1,'(a*b)+1':a*b+1,'(a*b)-1':a*b-1,'rev(a)+rev(b)':ra+rb,
       'rev(a)-rev(b)':ra-rb,'rev(b)-rev(a)':rb-ra,'rev(a)*rev(b)':ra*rb,'rev(a)+rev(b)+1':ra+rb+1,
       'rev(a)+rev(b)-1':ra+rb-1,'(rev(a)*rev(b))+1':ra*rb+1,'(rev(a)*rev(b))-1':ra*rb-1,
       'abs(rev(a)-rev(b))':abs(ra-rb),'-abs(rev(a)-rev(b))':-abs(ra-rb),
       'max(a,b)%min(a,b)':smod(max(a,b),min(a,b)),
       'max(rev(a),rev(b))%min(rev(a),rev(b))':smod(max(ra,rb),min(ra,rb))}
    return ('NUM',t[expr])

def clean(op,res):
    s=res; sign=1
    if s.startswith('-'): sign=-1; s=s[1:]
    elif op and s.startswith(op): sign=-1; s=s[len(op):]
    if op and s.endswith(op): s=s[:-len(op)]
    return sign,s

def match(op,kv,res,target):
    sign,mag=clean(op,res)
    if mag=='' or not mag.isdigit(): return False
    if kv[0]=='STR': return kv[1]==mag or int(kv[1])==int(mag)
    val=kv[1]
    if val is None: return False
    mi=int(mag)
    if target=='c': return val==sign*mi
    av=abs(val)
    return ((int(mag[::-1])==av) or (int(str(av)[::-1])==mi)) and ((sign==(-1 if val<0 else 1)) or val==0)

def cand_set(op,a,b,res):
    out=set()
    for e in CAND:
        for tg in ('c','rev(c)'):
            if match(op,calc(e,a,b),res,tg): out.add(f"{e} = {tg}")
    return out

with open('puzzles_guess_logic.md') as f: raw=f.readlines()
lre=re.compile(r'^\s*(\d+)(.)(\d+)\s*=\s*(\S+)\s*\(a.b\)\s*=\s*(.+?)\s*$')
blocks=[]; cur=[]
for l in raw:
    if l.strip()=='':
        if cur: blocks.append(cur); cur=[]
        continue
    m=lre.match(l.strip())
    if m: cur.append(m.groups())   # (ln?, ) -> groups are (a,op,b,res,formula); need line no
# redo with line numbers
blocks=[]; cur=[]
for i,l in enumerate(raw,1):
    if l.strip()=='':
        if cur: blocks.append(cur); cur=[]
        continue
    m=lre.match(l.strip())
    if m: cur.append((i,)+m.groups())
if cur: blocks.append(cur)

print("="*92)
print("FINAL formulas for 'unknown logic' lines (intersected over same-operator block siblings)")
print("="*92)
final={}
for blk in blocks:
    byop=defaultdict(list)
    for (ln,a,op,b,res,formula) in blk:
        byop[op].append((ln,a,b,res,formula))
    for op,items in byop.items():
        unknowns=[it for it in items if it[4].strip()=='unknown logic']
        if not unknowns: continue
        # intersect candidate sets across all unknown same-op lines in this block
        inter=None
        for (ln,a,b,res,_) in unknowns:
            cs=cand_set(op,a,b,res)
            inter = cs if inter is None else (inter & cs)
        # also constrain by known same-op sibling formula if present
        known=[it for it in items if it[4].strip()!='unknown logic']
        sib=None
        if known:
            f=known[0][4]; p=f.rsplit('=',1)
            sib=(p[0].strip()+' = '+p[1].strip()) if len(p)==2 else f+' = c'
            if sib in (inter or set()): inter={sib}
        for (ln,a,b,res,_) in unknowns:
            chosen = sorted(inter)[0] if inter else "(ambiguous)"
            final[ln]=chosen

for ln in sorted(final):
    # recover line text
    for blk in blocks:
        for (i,a,op,b,res,formula) in blk:
            if i==ln:
                print(f"Line {ln:3d}: {a}{op}{b} = {res:>6}   ->  {final[ln]}")
print("="*92)
print(f"Resolved {len(final)} unknown-logic lines.")
