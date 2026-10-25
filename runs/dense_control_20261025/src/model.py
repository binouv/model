"""Untrained dense/deep-thin control; local FlyGraph component adaptation.

Independent GQA blocks, pre-RMSNorm, SwiGLU, RoPE, tied input/output weights.
This architecture is NOT compatible with the trained FlyGraph hybrid checkpoint.
No external model implementation or weights are imported.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ARCHITECTURE='flygraph_dense_deepthin_v1'

@dataclass(frozen=True)
class Config:
    architecture: str=ARCHITECTURE
    vocab_size: int=8192
    dim: int=576
    layers: int=30
    heads: int=9
    kv_heads: int=3
    hidden: int=1536
    max_context: int=2048
    rope_theta: float=10000.
    norm_eps: float=1e-6
    initializer_std: float=.02
    checkpoint_blocks: bool=True
    def __post_init__(self):
        if self.architecture!=ARCHITECTURE:raise ValueError('different architecture: no implicit checkpoint conversion')
        if min(self.vocab_size,self.dim,self.layers,self.heads,self.kv_heads,self.hidden,self.max_context)<=0:raise ValueError('dimensions must be positive')
        if self.dim%self.heads or self.heads%self.kv_heads or (self.dim//self.heads)%2:raise ValueError('invalid GQA/RoPE dimensions')
        if self.norm_eps<=0 or self.rope_theta<=0 or self.initializer_std<=0:raise ValueError('invalid numerical configuration')

@dataclass(frozen=True)
class KVCache:
    layers: tuple[tuple[Tensor,Tensor],...]
    position: int

class RMSNorm(nn.Module):
    def __init__(self,dim,eps):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
    def forward(self,x):
        z=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return z.to(x.dtype)*self.weight

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;self.head_dim=c.dim//c.heads
        self.q=nn.Linear(c.dim,c.dim,bias=False)
        self.k=nn.Linear(c.dim,c.kv_heads*self.head_dim,bias=False)
        self.v=nn.Linear(c.dim,c.kv_heads*self.head_dim,bias=False)
        self.out=nn.Linear(c.dim,c.dim,bias=False)
        frequency=1/(c.rope_theta**(torch.arange(0,self.head_dim,2).float()/self.head_dim))
        self.register_buffer('frequency',frequency,persistent=False)
    def rope(self,x,position):
        positions=torch.arange(position,position+x.shape[2],device=x.device).float()
        angle=torch.outer(positions,self.frequency.float())
        co,si=angle.cos()[None,None].to(x.dtype),angle.sin()[None,None].to(x.dtype)
        a,b=x[...,::2],x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),-1).flatten(-2)
    def forward(self,x,past=None,position=0,use_cache=False):
        B,T,D=x.shape;c=self.c;hd=self.head_dim
        q=self.q(x).view(B,T,c.heads,hd).transpose(1,2)
        k=self.k(x).view(B,T,c.kv_heads,hd).transpose(1,2)
        v=self.v(x).view(B,T,c.kv_heads,hd).transpose(1,2)
        q,k=self.rope(q,position),self.rope(k,position)
        if past is not None:k=torch.cat((past[0],k),dim=2);v=torch.cat((past[1],v),dim=2)
        # Portable CPU GQA path inherited from the local hybrid; repeated KV is
        # temporary. Cache retains only kv_heads, not heads.
        kk=k.repeat_interleave(c.heads//c.kv_heads,1);vv=v.repeat_interleave(c.heads//c.kv_heads,1)
        if position==0:y=F.scaled_dot_product_attention(q,kk,vv,is_causal=True)
        elif T==1:y=F.scaled_dot_product_attention(q,kk,vv,is_causal=False)
        else:
            allowed=torch.arange(k.shape[2],device=x.device)[None,:]<=torch.arange(position,position+T,device=x.device)[:,None]
            y=F.scaled_dot_product_attention(q,kk,vv,attn_mask=allowed)
        return self.out(y.transpose(1,2).reshape(B,T,D)),(k,v) if use_cache else None

class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm1=RMSNorm(c.dim,c.norm_eps);self.norm2=RMSNorm(c.dim,c.norm_eps)
        self.attention=Attention(c);self.uv=nn.Linear(c.dim,2*c.hidden,bias=False);self.down=nn.Linear(c.hidden,c.dim,bias=False)
    def forward(self,x,past=None,position=0,use_cache=False):
        y,cache=self.attention(self.norm1(x),past,position,use_cache);x=x+y
        gate,value=self.uv(self.norm2(x)).chunk(2,-1)
        return x+self.down(F.silu(gate)*value),cache

class Model(nn.Module):
    def __init__(self,c:Config):
        super().__init__();self.config=c;self.embed=nn.Embedding(c.vocab_size,c.dim)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)]);self.norm=RMSNorm(c.dim,c.norm_eps)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.down.weight,std=c.initializer_std/math.sqrt(2*c.layers))
            nn.init.normal_(b.attention.out.weight,std=c.initializer_std/math.sqrt(2*c.layers))
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=self.config.initializer_std)
    @property
    def output_weight(self):
        """The actual output projection uses this same Parameter/storage."""
        return self.embed.weight
    def _validate_cache(self,cache,ids):
        if cache is None:return 0
        c=self.config
        if not isinstance(cache,KVCache) or len(cache.layers)!=c.layers or cache.position<0:raise ValueError('invalid cache metadata')
        shape=(ids.shape[0],c.kv_heads,cache.position,c.dim//c.heads)
        for k,v in cache.layers:
            if tuple(k.shape)!=shape or tuple(v.shape)!=shape or k.device!=ids.device or v.device!=ids.device:raise ValueError('inconsistent cache shape, batch, device or position')
        return cache.position
    def forward(self,ids,targets=None,cache:KVCache|None=None,last_only=False,use_cache=True):
        if ids.ndim!=2 or ids.dtype!=torch.long or ids.shape[1]==0:raise ValueError('ids must be a nonempty [batch,time] int64 tensor')
        past=self._validate_cache(cache,ids)
        if past+ids.shape[1]>self.config.max_context:raise ValueError('context limit exceeded')
        if targets is not None and (targets.shape!=ids.shape or targets.dtype!=torch.long):raise ValueError('targets must be aligned int64 token labels with ids shape')
        if self.training and cache is not None:raise ValueError('cached training is unsupported; use full sequences')
        build_cache=use_cache and targets is None
        x=self.embed(ids);next_layers=[]
        for i,block in enumerate(self.blocks):
            if self.training and self.config.checkpoint_blocks:
                if build_cache:raise ValueError('activation-checkpointed training requires use_cache=False or targets')
                x=checkpoint(lambda z,b=block:b(z)[0],x,use_reentrant=False)
            else:
                x,new=block(x,None if cache is None else cache.layers[i],past,build_cache)
                if build_cache:next_layers.append(new)
        x=self.norm(x)
        if targets is not None:
            valid=targets.ne(-100)
            if not bool(valid.any()):raise ValueError('no supervised targets')
            return F.cross_entropy(F.linear(x[valid],self.output_weight).float(),targets[valid])
        logits=F.linear(x[:,-1:] if last_only else x,self.output_weight)
        return logits,KVCache(tuple(next_layers),past+ids.shape[1]) if build_cache else None
    @torch.inference_mode()
    def generate(self,ids,max_new_tokens=16,eos_id=2):
        if max_new_tokens<0:raise ValueError('negative output budget')
        self.eval();remaining=min(max_new_tokens,self.config.max_context-ids.shape[1])
        if remaining<=0:
            if ids.shape[1]>self.config.max_context:raise ValueError('context limit exceeded')
            return ids.new_empty((ids.shape[0],0))
        logits,cache=self(ids,last_only=True);done=torch.zeros(ids.shape[0],device=ids.device,dtype=torch.bool);tokens=[]
        for step in range(remaining):
            nxt=logits[:,-1].argmax(-1)
            if eos_id is not None:nxt=torch.where(done,torch.full_like(nxt,eos_id),nxt);done|=nxt.eq(eos_id)
            tokens.append(nxt)
            if bool(done.all()) or step+1==remaining:break
            logits,cache=self(nxt[:,None],cache=cache,last_only=True)
        return torch.stack(tokens,1)

def parameter_report(m):
    c=m.config;d=c.dim;h=c.heads;kv=c.kv_heads;head=d//h
    theoretical=c.vocab_size*d+c.layers*(2*d*d+2*d*kv*head+3*d*c.hidden+2*d)+d
    actual=sum(p.numel() for p in m.parameters());assert actual==theoretical
    return dict(architecture=ARCHITECTURE,parameters=actual,formula_parameters=theoretical,tensor_count=len(list(m.parameters())),embedding_parameters=c.vocab_size*d,block_parameters=actual-c.vocab_size*d-d,final_norm_parameters=d,tied_embeddings=True,unique_blocks=c.layers,executed_blocks=c.layers,all_attention=True,has_moe=False,trained=False,config=asdict(c))
