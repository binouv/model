import hashlib,json,random,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]/"g5s_trace"/"src"
sys.path.insert(0,str(BASE))
from model import ByteTokenizer
from data import FAMS,canonical,make,row,solve

def generate(seed_base,denied=None):
    tok=ByteTokenizer(); occ=set() if denied is None else set(denied); out={}
    spec=[("train",8192,seed_base,0,0),("validation",400,seed_base+1,0,0),("test_iid",160,seed_base+2,0,0),("test_surface",80,seed_base+3,0,1),("test_extrapolation",80,seed_base+4,1,1)]
    for split,n,seed,ood,v in spec:
        r=random.Random(seed); rows=[]
        while len(rows)<n:
            c=make(r,FAMS[len(rows)%5],bool(ood)); g=canonical(c)
            if g in occ: continue
            z=row(c,v,tok)
            if len(z["tokens"])>256: continue
            rows.append(z); occ.add(g)
        out[split]=rows
    r=random.Random(seed_base+5); rows=[]; pairs=0
    while pairs<40:
        f="arithmetic" if pairs<20 else "memory_update"; c=make(r,f); g=canonical(c)
        if g in occ: continue
        c2=json.loads(json.dumps(c))
        if f=="arithmetic": c2["op"]=r.choice([x for x in "+-*" if x!=c["op"]])
        else:
            ks=sorted({k for k,_ in c["writes"]}); c2["query"]=r.choice([k for k in ks if k!=c["query"]])
        if solve(c)==solve(c2): continue
        pid=f"cf{pairs:03d}"; a,b=row(c,0,tok,pid),row(c2,0,tok,pid)
        if max(len(a["tokens"]),len(b["tokens"]))>256: continue
        rows += [a,b]; occ.add(g); pairs += 1
    out["test_counterfactual"]=rows
    return out,occ

def build(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    _,denied=generate(76011); assert len(denied)==8952
    fresh,all_groups=generate(78011,denied); assert len(all_groups)-len(denied)==8952
    man={}
    for split,rows in fresh.items():
        p=root/f"{split}.jsonl"
        p.write_text("".join(json.dumps(z,separators=(",",":"))+"\n" for z in rows))
        man[split]={"n":len(rows),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    seen={}
    for split,rows in fresh.items():
        for z in rows:
            assert str(solve(z["case"]))==z["answer"]
            if split!="test_counterfactual": assert z["canonical"] not in seen
            elif z["canonical"] in seen: assert seen[z["canonical"]]=="test_counterfactual"
            seen[z["canonical"]]=split
    assert not set(seen).intersection(denied)
    m={"status":"completed","schema":"G7R_data_v1","prior_denied_groups":8952,"fresh_groups":len(seen),"seed_base":78011,"splits":man}
    (root/"manifest.json").write_text(json.dumps(m,indent=2))
    return m

if __name__=="__main__":
    import sys
    print(json.dumps(build(sys.argv[1]),indent=2))
