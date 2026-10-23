"""FlyGraph L100M: causal GQA / gated-delta hybrid, all parameters trainable.
G8C's mathematical delta update, scaled to 14 blocks with GQA attention.
Starts from scratch: G8C/G1 checkpoint weights have incompatible dimensions.
No symbolic solver or teacher is used in forward/generate.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class Config:
    vocab_size: int = 8192
    dim: int = 768
    layers: int = 14
    heads: int = 12
    kv_heads: int = 4
    hidden: int = 2048
    delta_key_dim: int = 32
    max_context: int = 2048
    checkpoint_blocks: bool = True
    rope_theta: float = 10000.

class RMSNorm(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim))
    def forward(self,x):
        z=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        return z.to(x.dtype)*self.weight

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; self.hd=c.dim//c.heads
        assert c.dim%c.heads==0 and c.heads%c.kv_heads==0 and self.hd%2==0
        self.q=nn.Linear(c.dim,c.dim,bias=False)
        self.k=nn.Linear(c.dim,c.kv_heads*self.hd,bias=False)
        self.v=nn.Linear(c.dim,c.kv_heads*self.hd,bias=False)
        self.out=nn.Linear(c.dim,c.dim,bias=False)
        self.register_buffer('freq',1.0/(c.rope_theta**(torch.arange(0,self.hd,2).float()/self.hd)),persistent=False)
    def rope(self,x,start):
        angle=torch.outer(torch.arange(start,start+x.shape[2],device=x.device).float(),self.freq.float())
        co=angle.cos()[None,None].to(x.dtype); si=angle.sin()[None,None].to(x.dtype)
        a,b=x[...,::2],x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),-1).flatten(-2)
    def forward(self,x,cache=None):
        B,T,D=x.shape; c=self.c
        q=self.q(x).view(B,T,c.heads,self.hd).transpose(1,2)
        k=self.k(x).view(B,T,c.kv_heads,self.hd).transpose(1,2)
        v=self.v(x).view(B,T,c.kv_heads,self.hd).transpose(1,2)
        start=0 if cache is None else cache[0].shape[2]
        q,k=self.rope(q,start),self.rope(k,start)
        if cache is not None:k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2)
        kk=k.repeat_interleave(c.heads//c.kv_heads,1);vv=v.repeat_interleave(c.heads//c.kv_heads,1)
        if start==0:y=F.scaled_dot_product_attention(q,kk,vv,is_causal=True)
        elif T==1:y=F.scaled_dot_product_attention(q,kk,vv,is_causal=False)
        else:
            allowed=torch.arange(k.shape[2],device=x.device)[None,:]<=torch.arange(start,start+T,device=x.device)[:,None]
            y=F.scaled_dot_product_attention(q,kk,vv,attn_mask=allowed)
        return self.out(y.transpose(1,2).reshape(B,T,D)),(k,v)

def delta_rule(q,k,v,log_decay,beta,state=None):
    """FP32 causal delta update, equivalent to sequential state propagation.
    S_t=exp(g_t)S_(t-1)+k_t beta_t(v_t-k_t^T exp(g_t)S_(t-1)).
    Training samples have separate states; no cross-document persistent state.
    """
    out_dtype=v.dtype
    with torch.autocast(device_type=q.device.type,enabled=False):
        q,k,v,g,beta=(a.float() for a in (q,k,v,log_decay,beta))
        B,H,T,K=k.shape; V=v.shape[-1]
        state=torch.zeros(B,H,K,V,device=q.device,dtype=torch.float32) if state is None else state.float()
        if T==1:
            dec=state*g[:,:,0,None,None].exp()
            residual=beta[:,:,0,None]*(v[:,:,0]-(k[:,:,0,:,None]*dec).sum(-2))
            final=dec+k[:,:,0,:,None]*residual[:,:,None,:]
            return (q[:,:,0,:,None]*final).sum(-2).unsqueeze(2).to(out_dtype),final
        lp=g.cumsum(-1); ratio=(lp[...,None]-lp[...,None,:]).clamp_max(0).exp().tril()
        lower=((k@k.transpose(-1,-2))*ratio*beta[...,None]).tril(-1)
        eye=torch.eye(T,device=q.device,dtype=torch.float32)
        rhs=beta[...,None]*(v-(k@state)*lp.exp()[...,None])
        u=torch.linalg.solve_triangular(eye+lower,rhs,upper=False,unitriangular=True)
        y=(q@state)*lp.exp()[...,None]+((q@k.transpose(-1,-2))*ratio).tril()@u
        final=state*lp[:,:,-1,None,None].exp()+k.transpose(-1,-2)@(u*ratio[:,:,-1,:,None])
        return y.to(out_dtype),final

class Delta(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.q=nn.Linear(c.dim,c.heads*c.delta_key_dim,bias=False)
        self.k=nn.Linear(c.dim,c.heads*c.delta_key_dim,bias=False)
        self.v=nn.Linear(c.dim,c.dim,bias=False)
        self.gates=nn.Linear(c.dim,2*c.heads)
        self.read=nn.Linear(c.dim,c.dim,bias=False)
        self.out=nn.Linear(c.dim,c.dim,bias=False)
    def forward(self,x,cache=None):
        B,T,D=x.shape;H=self.c.heads;K=self.c.delta_key_dim;V=D//H
        q=F.normalize(self.q(x).view(B,T,H,K).transpose(1,2).float(),dim=-1)
        k=F.normalize(self.k(x).view(B,T,H,K).transpose(1,2).float(),dim=-1)
        v=self.v(x).view(B,T,H,V).transpose(1,2)
        g,b=self.gates(x).view(B,T,2,H).permute(2,0,3,1).unbind(0)
        y,state=delta_rule(q,k,v,-F.softplus(g.float()-4),b.float().sigmoid(),cache)
        return self.out(y.transpose(1,2).reshape(B,T,D)*self.read(x).sigmoid()),state

class Block(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.kind=kind;self.norm1=RMSNorm(c.dim);self.norm2=RMSNorm(c.dim)
        self.mix=Attention(c) if kind=='a' else Delta(c)
        self.uv=nn.Linear(c.dim,2*c.hidden,bias=False);self.down=nn.Linear(c.hidden,c.dim,bias=False)
    def forward(self,x,cache=None):
        y,new=self.mix(self.norm1(x),cache);x=x+y
        u,v=self.uv(self.norm2(x)).chunk(2,-1)
        return x+self.down(F.silu(u)*v),new

class Model(nn.Module):
    def __init__(self,c:Config):
        super().__init__();self.config=c;assert c.layers%2==0
        self.embed=nn.Embedding(c.vocab_size,c.dim)
        self.blocks=nn.ModuleList([Block(c,k) for k in 'da'*(c.layers//2)])
        self.norm=RMSNorm(c.dim);self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.down.weight,std=.02/math.sqrt(2*c.layers))
            nn.init.normal_(b.mix.out.weight,std=.02/math.sqrt(2*c.layers))
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if getattr(m,'bias',None) is not None:nn.init.zeros_(m.bias)
    def forward(self,ids,targets=None,cache=None,last_only=False):
        past=0 if cache is None else cache[1][0].shape[2]
        if ids.shape[1]+past>self.config.max_context:raise ValueError('context limit exceeded')
        x=self.embed(ids);new=[]
        for i,b in enumerate(self.blocks):
            if self.training and self.config.checkpoint_blocks and cache is None:
                x=checkpoint(lambda z,bb=b: bb(z)[0],x,use_reentrant=False);st=None
            else:x,st=b(x,None if cache is None else cache[i])
            new.append(st)
        x=self.norm(x)
        if targets is not None:
            valid=targets.ne(-100)
            if not bool(valid.any()):raise ValueError('no supervised tokens')
            return F.cross_entropy(F.linear(x[valid],self.embed.weight).float(),targets[valid])
        return F.linear(x[:,-1:] if last_only else x,self.embed.weight),new
    @torch.inference_mode()
    def generate(self,ids,max_new_tokens=48,eos_id=2):
        self.eval();lg,cache=self(ids,last_only=True)
        done=torch.zeros(ids.shape[0],device=ids.device,dtype=torch.bool);seq=[]
        for _ in range(min(max_new_tokens,self.config.max_context-ids.shape[1])):
            nxt=lg[:,-1].argmax(-1);nxt=torch.where(done,torch.full_like(nxt,eos_id),nxt)
            seq.append(nxt);done|=nxt.eq(eos_id)
            if bool(done.all()):break
            lg,cache=self(nxt[:,None],cache=cache,last_only=True)
        return torch.stack(seq,1) if seq else ids.new_empty((ids.shape[0],0))

def parameter_report(m):
    by={}
    for name,p in m.named_parameters():
        key='blocks' if name.startswith('blocks.') else name.split('.')[0]
        by[key]=by.get(key,0)+p.numel()
    n=sum(p.numel() for p in m.parameters())
    return {'total_parameters':n,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'active_parameters':n,'components':by,'block_pattern':'da'*(m.config.layers//2),'tied_embedding_and_output':True,'persistent_intertoken_state':True,'symbolic_solver_in_forward':False}
