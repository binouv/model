"""G3 architecture screen: causal generative probes, not a 310M weight continuation.

full: four independent attention blocks.
shared: two attention blocks applied twice (four executed blocks, not four unique).
hybrid: two full-attention and two persistent gated-delta blocks.
Delta has parallel causal training and a mathematically matched streaming state.
No solver, memory-key parser, ground truth, or restricted output vocabulary here.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

@dataclass
class Config:
    variant:str='full'
    vocab:int=8192
    dim:int=128
    heads:int=4
    layers:int=4
    hidden:int=384
    key_dim:int=16
    max_context:int=1024

class Norm(nn.Module):
    def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,True)+1e-6)).to(x.dtype)*self.weight

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.hd=c.dim//c.heads
        self.qkv=nn.Linear(c.dim,3*c.dim,bias=False);self.out=nn.Linear(c.dim,c.dim,bias=False)
        self.register_buffer('freq',1/(10000**(torch.arange(0,self.hd,2).float()/self.hd)),persistent=False)
    def rope(self,x,start):
        t=torch.arange(start,start+x.shape[2],device=x.device);a=t[:,None]*self.freq[None,:]
        co,si=a.cos()[None,None],a.sin()[None,None];p,q=x[...,::2],x[...,1::2]
        return torch.stack((p*co-q*si,p*si+q*co),-1).flatten(-2)
    def forward(self,x,cache=None,reset_memory=False):
        B,T,D=x.shape;H=self.c.heads;hd=self.hd
        q,k,v=self.qkv(x).view(B,T,3,H,hd).permute(2,0,3,1,4).unbind(0)
        start=0 if cache is None else cache[0].shape[2]
        q,k=self.rope(q,start),self.rope(k,start)
        if cache is not None:k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2)
        if start==0:y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        elif T==1:y=F.scaled_dot_product_attention(q,k,v)
        else:
            mask=torch.arange(k.shape[2],device=x.device)[None,:]<=torch.arange(start,start+T,device=x.device)[:,None]
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        return self.out(y.transpose(1,2).reshape(B,T,D)),(k,v)

def parallel_delta(q,k,v,g,beta,state=None,reset_memory=False):
    """[B,H,T,K/V], g<=0; stable lower-triangular linear solve.
    u_t=beta_t(v_t-k_t^T exp(g_t) S_{t-1}); S_t=exp(g_t)S_{t-1}+k_t u_t^T.
    FP32 state, no future inputs. Caller resets at document boundaries.
    """
    q,k,v,g,beta=(z.float() for z in (q,k,v,g,beta));B,H,T,K=k.shape;V=v.shape[-1]
    if state is None:state=torch.zeros((B,H,K,V),device=q.device,dtype=q.dtype)
    if reset_memory:
        u=beta[...,None]*v;y=(q*k).sum(-1,True)*u
        last=k[:,:,-1,:,None]*u[:,:,-1,None,:]
        return y,last
    if T==1:
        decayed=state*g[:,:,0,None,None].exp()
        u=beta[:,:,0,None]*(v[:,:,0]-(k[:,:,0,:,None]*decayed).sum(-2))
        last=decayed+k[:,:,0,:,None]*u[:,:,None,:]
        y=(q[:,:,0,:,None]*last).sum(-2).unsqueeze(2)
        return y,last
    lp=g.cumsum(-1)
    ratio=(lp[...,None]-lp[...,None,:]).clamp_max(0).exp().tril()
    gram=k@k.transpose(-1,-2)
    lower=(gram*ratio*beta[...,None]).tril(-1)
    ident=torch.eye(T,device=q.device,dtype=q.dtype)
    rhs=beta[...,None]*(v-(k@state)*lp.exp()[...,None])
    u=torch.linalg.solve_triangular(ident+lower,rhs,upper=False,unitriangular=True)
    y=(q@state)*lp.exp()[...,None]+((q@k.transpose(-1,-2))*ratio).tril()@u
    last=state*lp[:,:,-1,None,None].exp()+k.transpose(-1,-2)@(u*ratio[:,:,-1,:,None])
    return y,last

class Delta(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;H=c.heads;K=c.key_dim;V=c.dim//H
        self.q=nn.Linear(c.dim,H*K,bias=False);self.k=nn.Linear(c.dim,H*K,bias=False)
        self.v=nn.Linear(c.dim,H*V,bias=False);self.gates=nn.Linear(c.dim,2*H)
        self.read=nn.Linear(c.dim,c.dim,bias=False);self.out=nn.Linear(c.dim,c.dim,bias=False)
    def forward(self,x,cache=None,reset_memory=False):
        B,T,D=x.shape;H=self.c.heads;K=self.c.key_dim;V=D//H
        q=F.normalize(self.q(x).view(B,T,H,K).transpose(1,2),dim=-1,eps=1e-6)
        k=F.normalize(self.k(x).view(B,T,H,K).transpose(1,2),dim=-1,eps=1e-6)
        v=self.v(x).view(B,T,H,V).transpose(1,2)
        gg,bb=self.gates(x).view(B,T,2,H).permute(2,0,3,1).unbind(0)
        g=-F.softplus(gg-4.0);beta=bb.sigmoid()
        y,state=parallel_delta(q,k,v,g,beta,cache,reset_memory)
        y=y.transpose(1,2).reshape(B,T,D)*self.read(x).sigmoid()
        return self.out(y),state

class Block(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.n1=Norm(c.dim);self.n2=Norm(c.dim)
        self.mix=Attention(c) if kind=='attention' else Delta(c)
        self.uv=nn.Linear(c.dim,2*c.hidden,bias=False);self.down=nn.Linear(c.hidden,c.dim,bias=False)
    def forward(self,x,cache=None,reset_memory=False):
        y,new=self.mix(self.n1(x),cache,reset_memory);x=x+y
        u,v=self.uv(self.n2(x)).chunk(2,-1)
        return x+self.down(F.silu(u)*v),new

class Model(nn.Module):
    def __init__(self,c:Config):
        super().__init__();self.config=c
        if c.variant not in ('full','shared','hybrid'):raise ValueError('unknown architecture')
        if c.layers!=4:raise ValueError('this experiment fixes four executed blocks')
        self.embed=nn.Embedding(c.vocab,c.dim)
        kinds=['attention']*4 if c.variant=='full' else ['attention']*2 if c.variant=='shared' else ['delta','attention','delta','attention']
        self.blocks=nn.ModuleList([Block(c,k) for k in kinds]);self.norm=Norm(c.dim)
        self.apply(self.init)
    def init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if getattr(m,'bias',None) is not None:nn.init.zeros_(m.bias)
    def forward(self,ids,targets=None,cache=None,last_only=False,reset_memory=False):
        x=self.embed(ids);new=[]
        for i,j in enumerate([0,1,0,1] if self.config.variant=='shared' else range(4)):
            x,s=self.blocks[j](x,None if cache is None else cache[i],reset_memory);new.append(s)
        x=self.norm(x)
        if targets is not None:
            valid=targets.ne(-100);selected=x[valid];labels=targets[valid]
            if not labels.numel():raise ValueError('empty supervision')
            logits=F.linear(selected,self.embed.weight)
            return F.cross_entropy(logits,labels)
        logits=F.linear(x[:,-1:] if last_only else x,self.embed.weight)
        return logits,new
    @torch.inference_mode()
    def generate(self,ids,max_new=16,reset_memory=False):
        self.eval();logits,cache=self(ids,last_only=True,reset_memory=reset_memory)
        finished=torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device);out=[]
        for i in range(max_new):
            z=logits[:,-1].argmax(-1);z=torch.where(finished,torch.full_like(z,2),z)
            out.append(z);finished|=z.eq(2)
            if bool(finished.all()):break
            logits,cache=self(z[:,None],cache=cache,last_only=True,reset_memory=reset_memory)
        return torch.stack(out,1)

def report(m):
    return {'unique_parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'unique_blocks':len(m.blocks),'executed_blocks':4,'variant':m.config.variant,'persistent_delta_state_per_token':m.config.variant=='hybrid','biological_connectome':False,'full_vocabulary_generation':True}
