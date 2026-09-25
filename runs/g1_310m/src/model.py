"""FlyGraph G1: causal LM backbone + a sparse recurrent graph workspace.

The workspace graph is SYNTHETIC, not an imported biological connectome.
All backbone and workspace weights are trained. Addresses are not averaged:
external immutable-record memory is a separate, explicitly exposed component.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class Config:
    vocab_size: int = 8192
    d_model: int = 1536
    n_layers: int = 12
    n_heads: int = 24
    n_kv_heads: int = 4
    d_ff: int = 4096
    graph_width: int = 128
    graph_rounds: int = 2
    max_context: int = 2048
    rope_theta: float = 10000.0
    checkpoint_blocks: bool = True
    init_seed: int = 7101

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype) * self.weight

class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__(); self.c=c; self.hd=c.d_model//c.n_heads
        assert self.hd%2==0 and c.n_heads%c.n_kv_heads==0
        self.q=nn.Linear(c.d_model,c.d_model,bias=False)
        self.k=nn.Linear(c.d_model,c.n_kv_heads*self.hd,bias=False)
        self.v=nn.Linear(c.d_model,c.n_kv_heads*self.hd,bias=False)
        self.o=nn.Linear(c.d_model,c.d_model,bias=False)
        freq=1.0/(c.rope_theta**(torch.arange(0,self.hd,2).float()/self.hd))
        self.register_buffer('freq',freq,persistent=False)
    def rope(self,x,start):
        p=torch.arange(start,start+x.shape[-2],device=x.device,dtype=torch.float32)
        angle=torch.outer(p,self.freq.float())
        co=angle.cos()[None,None,:,:].to(x.dtype); si=angle.sin()[None,None,:,:].to(x.dtype)
        a,b=x[...,::2],x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),dim=-1).flatten(-2)
    def forward(self,x,past=None,use_cache=False):
        B,T,D=x.shape; c=self.c
        q=self.q(x).view(B,T,c.n_heads,self.hd).transpose(1,2)
        k=self.k(x).view(B,T,c.n_kv_heads,self.hd).transpose(1,2)
        v=self.v(x).view(B,T,c.n_kv_heads,self.hd).transpose(1,2)
        start=0 if past is None else past[0].shape[-2]
        q=self.rope(q,start); k=self.rope(k,start)
        if past is not None: k=torch.cat((past[0],k),-2); v=torch.cat((past[1],v),-2)
        new=(k,v) if use_cache else None
        kk=k.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1)
        vv=v.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1)
        if past is None:
            y=F.scaled_dot_product_attention(q,kk,vv,is_causal=True)
        elif T==1:
            y=F.scaled_dot_product_attention(q,kk,vv,is_causal=False)
        else:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:] <= torch.arange(start,start+T,device=x.device)[:,None]
            y=F.scaled_dot_product_attention(q,kk,vv,attn_mask=mask)
        y=y.transpose(1,2).contiguous().view(B,T,D)
        return self.o(y),new

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); self.an=RMSNorm(c.d_model); self.fn=RMSNorm(c.d_model)
        self.attn=Attention(c)
        self.up=nn.Linear(c.d_model,c.d_ff,bias=False)
        self.gate=nn.Linear(c.d_model,c.d_ff,bias=False)
        self.down=nn.Linear(c.d_ff,c.d_model,bias=False)
    def forward(self,x,past=None,use_cache=False):
        a,new=self.attn(self.an(x),past,use_cache); x=x+a
        z=self.fn(x); x=x+self.down(F.silu(self.gate(z))*self.up(z))
        return x,new

class SparseGraphWorkspace(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        self.projections=nn.ModuleList([nn.Linear(c.d_model,c.graph_width,bias=False) for _ in range(c.n_layers)])
        # Three directed incoming edges per node, including cycles. No biological claim.
        edges=torch.tensor([[(i-1)%c.n_layers,(i-3)%c.n_layers,(i+5)%c.n_layers] for i in range(c.n_layers)])
        self.register_buffer('edges',edges)
        self.edge_logits=nn.Parameter(torch.zeros(c.n_layers,3))
        self.cell=nn.GRUCell(c.graph_width,c.graph_width)
        self.out=nn.Linear(c.n_layers*c.graph_width,c.d_model,bias=False)
        self.norm=RMSNorm(c.d_model)
        self.gain=nn.Parameter(torch.tensor(0.10))
    def forward(self,states,disable_edges=False,rounds=None):
        z=torch.stack([torch.tanh(p(x)) for p,x in zip(self.projections,states)],dim=-2)
        B,T,N,W=z.shape; weights=self.edge_logits.softmax(-1)
        for _ in range(self.c.graph_rounds if rounds is None else rounds):
            if disable_edges:
                msg=torch.zeros_like(z)
            else:
                msg=(z[:,:,self.edges,:]*weights[None,None,:,:,None]).sum(-2)
            z=self.cell(msg.reshape(-1,W),z.reshape(-1,W)).view(B,T,N,W)
        return torch.tanh(self.gain)*self.norm(self.out(z.flatten(-2)))

class FlyGraphLM(nn.Module):
    def __init__(self,c:Config):
        super().__init__(); self.config=c
        self.embed=nn.Embedding(c.vocab_size,c.d_model)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.graph=SparseGraphWorkspace(c)
        self.final_norm=RMSNorm(c.d_model)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.down.weight,std=0.02/math.sqrt(2*c.n_layers))
            nn.init.normal_(b.attn.o.weight,std=0.02/math.sqrt(2*c.n_layers))
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0,std=0.02)
            if getattr(m,'bias',None) is not None: nn.init.zeros_(m.bias)
    def forward(self,ids,targets=None,past=None,use_cache=False,disable_graph_edges=False,loss_mask=None):
        if ids.shape[1]+(0 if past is None else past[0][0].shape[-2])>self.config.max_context:
            raise ValueError('context exceeds configured max_context; no silent truncation')
        x=self.embed(ids); states=[]; cache=[]
        for i,b in enumerate(self.blocks):
            old=None if past is None else past[i]
            if self.training and self.config.checkpoint_blocks and not use_cache:
                x=checkpoint(lambda xx,bb=b: bb(xx)[0],x,use_reentrant=False)
                new=None
            else: x,new=b(x,old,use_cache)
            states.append(x); cache.append(new)
        x=self.final_norm(x+self.graph(states,disable_edges=disable_graph_edges))
        # Tied embedding/output matrix; counted once and actually optimized.
        logits=F.linear(x,self.embed.weight)
        if targets is None: return (logits,cache) if use_cache else logits
        losses=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),reduction='none',ignore_index=-100).view_as(targets)
        valid=targets.ne(-100).float()
        if loss_mask is not None: valid=valid*loss_mask
        loss=(losses*valid).sum()/valid.sum().clamp_min(1)
        return loss,logits
    @torch.no_grad()
    def generate(self,ids,max_new_tokens=64,temperature=0.0,eos_id=2,disable_graph_edges=False):
        self.eval(); logits,cache=self(ids,use_cache=True,disable_graph_edges=disable_graph_edges)
        tokens=ids; finished=torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device)
        for _ in range(max_new_tokens):
            if temperature<=0: nxt=logits[:,-1].argmax(-1)
            else: nxt=torch.multinomial((logits[:,-1]/temperature).softmax(-1),1).squeeze(-1)
            nxt=torch.where(finished,torch.full_like(nxt,eos_id),nxt)
            tokens=torch.cat((tokens,nxt[:,None]),1); finished|=nxt.eq(eos_id)
            if bool(finished.all()) or tokens.shape[1]>=self.config.max_context: break
            logits,cache=self(nxt[:,None],past=cache,use_cache=True,disable_graph_edges=disable_graph_edges)
        return tokens

def parameter_report(model):
    by={}
    for n,p in model.named_parameters():
        root='blocks' if n.startswith('blocks.') else n.split('.')[0]
        by[root]=by.get(root,0)+p.numel()
    return {'unique_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'components':by,'graph':'synthetic directed cyclic graph; NOT FlyWire','output':'unrestricted autoregressive tokenizer vocabulary'}
