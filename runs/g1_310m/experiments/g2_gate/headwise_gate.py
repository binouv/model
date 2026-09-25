"""Experimental headwise output gating, inspired by gated attention literature.

Original implementation for FlyGraph; no Qwen or DeepSeek weights are imported.
Unlike Qwen3.5's elementwise sigmoid gate, this uses one 2*sigmoid gate per
attention head. Zero initialization is exactly function-preserving. This is an
engineering prototype, NOT part of the G1 training run and NOT a proven gain.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class GatedAttention(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.c, self.hd = base.c, base.hd
        self.q, self.k, self.v, self.o = base.q, base.k, base.v, base.o
        self.register_buffer('freq', base.freq, persistent=False)
        self.head_gate = nn.Linear(self.c.d_model, self.c.n_heads, bias=True,
                                   device=base.q.weight.device, dtype=base.q.weight.dtype)
        nn.init.zeros_(self.head_gate.weight)
        nn.init.zeros_(self.head_gate.bias)
        self.reset_gate = False  # sole paired control; still compute the gate

    def rope(self, x, start):
        pos = torch.arange(start, start+x.shape[-2], device=x.device, dtype=torch.float32)
        angle = torch.outer(pos, self.freq.float())
        co, si = angle.cos()[None,None].to(x.dtype), angle.sin()[None,None].to(x.dtype)
        a,b = x[...,::2],x[...,1::2]
        return torch.stack((a*co-b*si,a*si+b*co),dim=-1).flatten(-2)

    def forward(self, x, past=None, use_cache=False):
        B,T,D=x.shape; c=self.c
        q=self.q(x).view(B,T,c.n_heads,self.hd).transpose(1,2)
        k=self.k(x).view(B,T,c.n_kv_heads,self.hd).transpose(1,2)
        v=self.v(x).view(B,T,c.n_kv_heads,self.hd).transpose(1,2)
        start=0 if past is None else past[0].shape[-2]
        q,k=self.rope(q,start),self.rope(k,start)
        if past is not None:
            k,v=torch.cat((past[0],k),-2),torch.cat((past[1],v),-2)
        cache=(k,v) if use_cache else None
        kk=k.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1)
        vv=v.repeat_interleave(c.n_heads//c.n_kv_heads,dim=1)
        if past is None:
            y=F.scaled_dot_product_attention(q,kk,vv,is_causal=True)
        elif T==1:
            y=F.scaled_dot_product_attention(q,kk,vv,is_causal=False)
        else:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:] <= torch.arange(start,start+T,device=x.device)[:,None]
            y=F.scaled_dot_product_attention(q,kk,vv,attn_mask=mask)
        gate=2*torch.sigmoid(self.head_gate(x))
        if self.reset_gate:
            gate=gate*0+1  # control computes same projection/nonlinearity
        y=y.transpose(1,2)*gate[...,None]
        return self.o(y.contiguous().view(B,T,D)),cache

def install_headwise_gates(model):
    """Call after loading a G1 checkpoint and BEFORE constructing the optimizer.

    Existing backbone weights are reused, not copied or reinitialized. A gated
    state_dict needs the patch installed before load_state_dict; the unchanged
    G1 checkpoint loader must not silently accept this different architecture.
    """
    for block in model.blocks:
        if isinstance(block.attn, GatedAttention):
            raise ValueError('gates already installed')
        block.attn=GatedAttention(block.attn)
    return model
