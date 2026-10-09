"""Fixed training cases and fresh semantically grouped held evaluation."""
from __future__ import annotations
import sys,json,random,hashlib,copy,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
import parent_data as old
from tokenizer import Tokenizer
from pointer_model import atomic_json,sha
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def pack(z,tok):
    p=z['prompt'];pt=tok.encode(p,bos=True)
    target='F='+z['answer']
    ids=pt+tok.encode(target,eos=True)
    return {**z,'tokens':ids,'prompt_tokens':len(pt),'target':target}
def build():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    forbidden={z['group'] for p in (ROOT/'data').glob('reasoning_*.jsonl') for z in read(p)}
    occupied=set(forbidden);info={}
    for split,n,seed,variant,ood in [('validation',100,92101,0,False),('iid',400,92102,0,False),
                                     ('surface',200,92103,1,False),('extrapolation',200,92104,1,True)]:
        r=random.Random(seed);out=[];attempt=0
        while len(out)<n:
            attempt+=1
            c=old.make_case(r,old.FAMILIES[len(out)%5],ood)
            g=old.canonical(c)
            if g in occupied:continue
            z=old.make_row(c,tok,r,variant=variant)
            if len(pack(z,tok)['tokens'])>192:continue
            assert int(z['answer'])==old.independent_trace(c)[1]
            out.append(z);occupied.add(g)
        p=ROOT/'data'/f'{split}.jsonl'
        p.write_text(''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in out))
        info[split]={'n':n,'attempts':attempt,'sha256':sha(p)}
    r=random.Random(92105);out=[]
    while len(out)<200:
        c=old.make_case(r,'memory_update',False);g=old.canonical(c)
        if g in occupied:continue
        alt=copy.deepcopy(c);alt['query']=r.choice([k for k in 'abcd' if k!=c['query']])
        if old.solve(c)==old.solve(alt):continue
        lang=r.choice(['ru','en']);pair='l100p_'+str(len(out)//2)
        zz=[old.make_row(v,tok,r,variant=0,lang=lang,pair=pair) for v in [c,alt]]
        if max(len(pack(z,tok)['tokens']) for z in zz)>192:continue
        out.extend(zz);occupied.add(g)
    p=ROOT/'data/counterfactual.jsonl';p.write_text(''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in out))
    info['counterfactual']={'n':200,'pairs':100,'sha256':sha(p)}
    atomic_json(ROOT/'data/MANIFEST.json',{'status':'complete_data_only','held_seeds':[92101,92102,92103,92104,92105],
       'blocked_ancestor_groups':len(forbidden),'cross_split_group_overlap':0,
       'original_train_sha256':sha(ROOT/'data/train_parent.jsonl'),'splits':info,
       'scope':'Synthetic RU/EN; fresh against all available L100M cases. No broad Qwen benchmark.'})
def training_rows():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    return [pack(z,tok) for z in read(ROOT/'data/train_parent.jsonl')]
def batch(rr):
    import torch
    n=max(len(z['tokens']) for z in rr)-1
    x=torch.zeros((len(rr),n),dtype=torch.long);y=torch.full_like(x,-100)
    p=torch.tensor([z['prompt_tokens'] for z in rr])
    for i,z in enumerate(rr):
        t=torch.tensor(z['tokens']);end=len(t)-1
        x[i,:end]=t[:-1];y[i,p[i]-1:end]=t[p[i]:]
    return x,y,p
if __name__=='__main__':build()
