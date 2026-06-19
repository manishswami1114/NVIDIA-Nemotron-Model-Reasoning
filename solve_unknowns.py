#!/usr/bin/env python3
"""For each 'unknown logic' line, brute-force the known formula vocabulary
(both '= c' and '= rev(c)' conventions) and report any formula that reproduces
the shown result, honoring the dataset's sign/padding/symbol conventions."""
import re

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

def calc(expr, a_str, b_str):
    a,b = int(a_str),int(b_str); ra,rb = rev_int(a_str),rev_int(b_str)
    def smod(x,y): return x%y if y else None
    if expr=='a||b':            return ('STR', a_str+b_str)
    if expr=='b||a':            return ('STR', b_str+a_str)
    if expr=='rev(a)||rev(b)':  return ('STR', a_str[::-1]+b_str[::-1])
    table = {
        'a-b':a-b,'a+b':a+b,'a*b':a*b,'b-a':b-a,
        'abs(a-b)':abs(a-b),'-abs(a-b)':-abs(a-b),
        'a+b-1':a+b-1,'a+b+1':a+b+1,'(a*b)+1':a*b+1,'(a*b)-1':a*b-1,
        'rev(a)+rev(b)':ra+rb,'rev(a)-rev(b)':ra-rb,'rev(b)-rev(a)':rb-ra,
        'rev(a)*rev(b)':ra*rb,'rev(a)+rev(b)+1':ra+rb+1,'rev(a)+rev(b)-1':ra+rb-1,
        '(rev(a)*rev(b))+1':ra*rb+1,'(rev(a)*rev(b))-1':ra*rb-1,
        'abs(rev(a)-rev(b))':abs(ra-rb),'-abs(rev(a)-rev(b))':-abs(ra-rb),
        'max(a,b)%min(a,b)':smod(max(a,b),min(a,b)),
        'max(rev(a),rev(b))%min(rev(a),rev(b))':smod(max(ra,rb),min(ra,rb)),
    }
    return ('NUM', table[expr])

def clean(op,res):
    s=res; sign=1
    if s.startswith('-'): sign=-1; s=s[1:]
    elif op and s.startswith(op): sign=-1; s=s[len(op):]
    if op and s.endswith(op): s=s[:-len(op)]
    return sign,s

def matches(op, kind_val, result_str, target):
    sign,mag = clean(op,result_str)
    if mag=='' or not mag.isdigit(): return False
    kind = kind_val[0]
    if kind=='STR':
        got=kind_val[1]
        return got==mag or int(got)==int(mag)
    val=kind_val[1]
    if val is None: return False
    mag_int=int(mag)
    if target=='c':
        return val==sign*mag_int
    else:
        absval=abs(val)
        ok=(int(mag[::-1])==absval) or (int(str(absval)[::-1])==mag_int)
        sgn=(sign==(-1 if val<0 else 1)) or val==0
        return ok and sgn

with open('puzzles_guess_logic.md') as f:
    lines=f.readlines()

pat=re.compile(r'^\s*(\d+)(.)(\d+)\s*=\s*(\S+)\s*\(a.b\)\s*=\s*unknown logic\s*$')
print("="*92)
print("UNKNOWN-LOGIC LINES — candidate formulas that reproduce the shown result")
print("="*92)
solved=0; still=0
for i,raw in enumerate(lines,1):
    m=pat.match(raw.strip())
    if not m: continue
    a_str,op,b_str,res = m.group(1),m.group(2),m.group(3),m.group(4)
    found=[]
    for expr in CANDIDATES:
        kv=calc(expr,a_str,b_str)
        for target in ('c','rev(c)'):
            if matches(op,kv,res,target):
                tag=expr+(' = rev(c)' if target=='rev(c)' else ' = c')
                if tag not in found: found.append(tag)
    if found:
        solved+=1
        print(f"Line {i:3d}: {a_str}{op}{b_str} = {res}")
        for t in found: print(f"            -> {t}")
    else:
        still+=1
        print(f"Line {i:3d}: {a_str}{op}{b_str} = {res}    [NO known formula matches]")
print("="*92)
print(f"Solved {solved} / {solved+still} unknown-logic lines from known vocabulary; {still} still unexplained")
