import json, z3, time
p = json.loads(open('/Users/manishswami/developer/NVIDIA-Nemotron Model/tong_reasoners/problems/00c032a8.jsonl').read())
e = p['examples']
si = 2
syms = sorted(set("}`]?(#<)\\!&@"))

def to_z3(s, zv):
    val = z3.IntVal(0)
    for i, c in enumerate(reversed(s)): val += zv[c] * (10**i)
    return val

def add_op(slv, a, b, o, op):
    if op=="addition": slv.add(a+b==o)
    elif op=="abs_diff": slv.add(z3.If(a>=b,a-b,b-a)==o)
    elif op=="multiplication": slv.add(a*b==o)

am = {"]": "addition", "<": "abs_diff", "!": "multiplication"}
slv = z3.Solver()
zv = {s: z3.Int(s) for s in syms}
for v in zv.values(): slv.add(v>=0, v<10)
slv.add(z3.Distinct(*list(zv.values())))

for ex in e:
    inp, out = ex["input_value"], ex["output_value"]
    la, lb, o_s = inp[:si], inp[si+1:], out
    add_op(slv, to_z3(la, zv), to_z3(lb, zv), to_z3(o_s, zv), am[inp[si]])

t0 = time.time()
print("Check:", slv.check())
print("Time:", time.time() - t0)
if slv.check() == z3.sat:
    m = slv.model()
    print({s: m[zv[s]].as_long() for s in syms})
