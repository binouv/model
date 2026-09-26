import collections,hashlib,random,torch
from common import ROOT,Model,Config,ex,rows,mat

def prepare(arm,seed):
    out=ROOT/"metrics"/f"{arm}_s{seed}.json"
    if out.exists(): raise FileExistsError(str(out))
    torch.manual_seed(seed); random.seed(seed)
    m=Model(Config())
    assert sum(p.numel() for p in m.parameters())==469648
    sem=rows("train"); pools={}
    for tr in (False,True):
        p=collections.defaultdict(list)
        for z in sem:
            q=mat(z,tr); p[(q["family"],q["prompt_tokens"])].append(q)
        pools[tr]=p
    keys=[k for k,v in pools[False].items() if len(v)>=4 and len(pools[True][k])>=4]
    rng=random.Random(78311)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01)
    return out,m,pools,keys,rng,opt,[],hashlib.sha256(),ex.state_hash(m)
