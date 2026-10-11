"""The existing100M model, with training-only weighted semantic-token NLL."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from common import *

def batch(rr,device='cpu'):
    n=max(len(r['tokens'])-1 for r in rr)
    x=torch.zeros((len(rr),n),dtype=torch.long);y=torch.full_like(x,-100)
    w=torch.zeros_like(x,dtype=torch.float32)
    for i,r in enumerate(rr):
        ts=r['tokens'];j=r['prefix']-1;end=len(ts)-1
        x[i,:end]=torch.tensor(ts[:-1]);y[i,j:end]=torch.tensor(ts[j+1:]);w[i,j:end]=torch.tensor(r['weights'])
    return x.to(device),y.to(device),w.to(device)

def loss_forward(m,x,y,w):
    h=m.embed(x)
    for block in m.blocks:
        if m.training and m.config.checkpoint_blocks:h=checkpoint(lambda z,b=block:b(z)[0],h,use_reentrant=False)
        else:h=block(h)[0]
    h=m.norm(h);keep=y.ne(-100)
    if not bool(keep.any()):raise ValueError('No supervised tokens')
    logits=F.linear(h[keep],m.embed.weight).float()
    ce=F.cross_entropy(logits,y[keep],reduction='none');ww=w[keep]
    if not bool((ww>0).all()):raise ValueError('Invalid supervised weight')
    return (ce*ww).sum()/ww.sum(),ce.detach().mean()

@torch.inference_mode()
def generate(m,tok,prompts,max_new_tokens=128,device='cpu'):
    # The inference interface cannot accept cases, labels, oracle states or candidates.
    if not all(isinstance(p,str) for p in prompts):raise TypeError('Only prompt strings allowed')
    ids=[tok.encode(p,bos=True) for p in prompts]
    if len({len(z) for z in ids})!=1:raise ValueError('Equal-length batch required')
    m.eval()
    with torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):out=m.generate(torch.tensor(ids,device=device),max_new_tokens=max_new_tokens)
    result=[]
    for r in out.tolist():
        eos=tok.EOS in r;r=r[:r.index(tok.EOS)] if eos else r
        result.append({'text':tok.decode(r),'token_ids':r,'eos':eos})
    return result
