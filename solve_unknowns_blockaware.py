#!/usr/bin/env python3
"""Disambiguate each 'unknown logic' line using its block's shared rule.
A block = consecutive non-empty lines separated by blank lines. Within a block
the same operator/rule is used, so we read the formula from sibling lines that
already have one; if siblings are all unknown, we fall back to the candidate set."""
import re
from collections import Counter

def rev_int(s): return int(s[::-1])

CANDIDATES = [
    'a-b','a+b','a*b','b-a',
    'abs(a-b)','-abs(a-b)','a+b-1','a+b+1','(a*b)+1','(a*b)-1',
    'rev(a)+rev(b)','rev(a)-rev(b)','rev(b)-rev(a)','rev(a)*rev(b)',
    'rev(a)+rev(b)+1','rev(a)+rev(b)-1','(rev(a)*rev(b))+1','(rev(a)*rev(b))-1',
    'abs(rev(a)-rev(b))','-abs(rev(a)-rev(b))',
    'max(a,b)%min(a,b)','max(rev(a),rev(b))%min(rev(a),rev(b))',
    'a||b','b||a','rev(a)||rev(b)',
]

def calc(expr,a_str,b_str):
    a,b=int(a_str),int(b_str); ra,rb=rev_int(a_str),rev_int(b_str)
    def smod(x,y): return x%y if y else None
    if expr=='a||b':           return ('STR',a_str+b_str)
    if expr=='b||a':           return ('STR',b_str+a_str)
    if expr=='rev(a)||rev(b)': return ('STR',a_str[::-1]+b_str[::-1])
    t={'a-b':a-b,'a+b':a+b,'a*b':a*b,'b-a':b-a,'abs(a-b)':abs(a-b),'-abs(a-b)':-abs(a-b),
       'a+b-1':a+b-1,'a+b+1':a+b+1,'(a*b)+1':a*b+1,'(a*b)-1':a*b-1,
       'rev(a)+rev(b)':ra+rb,'rev(a)-rev(b)':ra-rb,'rev(b)-rev(a)':rb-ra,'rev(a)*rev(b)':ra*rb,
       'rev(a)+rev(b)+1':ra+rb+1,'rev(a)+rev(b)-1':ra+rb-1,'(rev(a)*rev(b))+1':ra*rb+1,
       '(rev(a)*rev(b))-1':ra*rb-1,'abs(rev(a)-rev(b))':abs(ra-rb),'-abs(rev(a)-rev(b))':-abs(ra-rb),
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
    if kv[0]=='STR':
        return kv[1]==mag or int(kv[1])==int(mag)
    val=kv[1]
    if val is None: return False
    mi=int(mag)
    if target=='c': return val==sign*mi
    av=abs(val)
    return ((int(mag[::-1])==av) or (int(str(av)[::-1])==mi)) and ((sign==(-1 if val<0 else 1)) or val==0)

# parse file into blocks
with open('puzzles_guess_logic.md') as f:
    raw=f.readlines()

line_re=re.compile(r'^\s*(\d+)(.)(\d+)\s*=\s*(\S+)\s*\(a.b\)\s*=\s*(.+?)\s*$')
blocks=[]; cur=[]
for i,l in enumerate(raw,1):
    if l.strip()=='':
        if cur: blocks.append(cur); cur=[]
        continue
    m=line_re.match(l.strip())
    if m: cur.append((i,)+m.groups())
if cur: blocks.append(cur)

def known_formula_of_block(blk):
    """Return the shared written formula (expr, target) from non-unknown siblings."""
    forms=[]
    for (_,a,op,b,res,formula) in blk:
        if formula.strip()=='unknown logic': continue
        parts=formula.rsplit('=',1)
        if len(parts)==2:
            forms.append((parts[0].strip(),parts[1].strip()))
        else:
            forms.append((formula.strip(),'c'))
    if not forms: return None
    return Counter(forms).most_common(1)[0][0]

print("="*92)
print("FINAL: formula for each unknown-logic line (disambiguated by its block)")
print("="*92)
results=[]
for blk in blocks:
    shared=known_formula_of_block(blk)
    for (ln,a,op,b,res,formula) in blk:
        if formula.strip()!='unknown logic': continue
        chosen=None; how=None
        # 1) try the block's shared formula
        if shared:
            expr,target=shared
            if expr in CANDIDATES and match(op,calc(expr,a,b),res,target):
                chosen=f"{expr} = {target}"; how="block-rule"
        # 2) else brute force, prefer rev(c) forms / first candidate
        if not chosen:
            cands=[]
            for expr in CANDIDATES:
                for target in ('c','rev(c)'):
                    if match(op,calc(expr,a,b),res,target):
                        cands.append(f"{expr} = {target}")
            chosen = cands[0] if cands else "??? no match"
            how = "brute(first of %d)"%len(cands) if cands else "none"
        results.append((ln,a,op,b,res,chosen,how,shared))
        sh = f"{shared[0]} = {shared[1]}" if shared else "(block all-unknown)"
        print(f"Line {ln:3d}: {a}{op}{b} = {res:>6}   ->  {chosen:35s} [{how}; block rule: {sh}]")
print("="*92)
print(f"Total unknown-logic lines resolved: {len(results)}")
