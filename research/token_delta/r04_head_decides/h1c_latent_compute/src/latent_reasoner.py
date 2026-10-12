"""Small explicitly compute-bearing recurrent head, NOT passive hidden-state probe.

Frozen Qwen outputs a single [B, 2560] pre-answer state. This module
performs trainable additional compute T=0/2/8 with *tied* parameters,
then yields two digits and per-step confidence to calibrate a stop gate.
It is deliberately independent of transformers and CUDA for CPU unit tests.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentReasoner(nn.Module):
    def __init__(self,input_width:int=2560,hidden:int=96):
        super().__init__()
        if input_width<1 or hidden<1:
            raise ValueError("invalid dimensions")
        self.input_width=input_width
        self.hidden=hidden
        self.norm=nn.LayerNorm(input_width)
        self.encoder=nn.Linear(input_width,hidden)
        self.recurrent=nn.GRUCell(hidden,hidden)
        self.readout=nn.Linear(hidden,20)

    def forward(self,x:torch.Tensor,steps:int=8,save_trajectory:bool=False):
        if x.ndim!=2 or x.shape[1]!=self.input_width:
            raise ValueError(f"expected [batch, {self.input_width}] features")
        if steps<0 or steps>32:
            raise ValueError("invalid number of internal steps")
        context=F.gelu(self.encoder(self.norm(x.float())))
        state=context
        trajectory=[]
        for t in range(steps+1):
            if save_trajectory:
                trajectory.append(self.readout(state).reshape(-1,2,10))
            if t<steps:
                state=self.recurrent(context,state)
        final=self.readout(state).reshape(-1,2,10)
        return (final,torch.stack(trajectory,dim=1)) if save_trajectory else final

    @staticmethod
    @torch.no_grad()
    def confidence(logits:torch.Tensor)->torch.Tensor:
        """Conservative minimum max-softmax over the two output positions."""
        if logits.shape[-2:]!=(2,10):
            raise ValueError("expected (..., 2, 10) logits")
        return logits.float().softmax(-1).max(-1).values.min(-1).values

    @torch.no_grad()
    def adaptive(self,x:torch.Tensor,threshold:float,checkpoints=(0,2,4,8)):
        """Real early exit: recurrent work runs ONLY on still-undecided rows.

        Unlike evaluating a full 8-step trajectory and selecting an earlier
        output after the fact, this routine skips future GRUCell calls for
        completed examples. The logits and selected depths should match the
        simulated full-trajectory policy within floating point tolerance.
        """
        if not 0 < threshold < 1:
            raise ValueError("invalid stop confidence threshold")
        times=tuple(checkpoints)
        if not times or times[0]!=0 or times[-1]!=8 or sorted(set(times))!=list(times):
            raise ValueError("checkpoints must be ordered including zero and eight")
        if x.ndim!=2 or x.shape[1]!=self.input_width or len(x)==0:
            raise ValueError(f"expected a nonempty [batch, {self.input_width}] feature tensor")

        active_ids=torch.arange(len(x),device=x.device)
        active_context=F.gelu(self.encoder(self.norm(x.float())))
        active_state=active_context
        chosen_logits=active_context.new_empty((len(x),2,10))
        chosen_steps=torch.full((len(x),),8,dtype=torch.long,device=x.device)
        prev=0
        for t in times:
            for _ in range(t-prev):
                active_state=self.recurrent(active_context,active_state)
            logits=self.readout(active_state).reshape(-1,2,10)
            ready=(torch.ones(len(active_ids),dtype=torch.bool,device=x.device)
                   if t==times[-1] else self.confidence(logits)>=threshold)
            chosen_logits[active_ids[ready]]=logits[ready]
            chosen_steps[active_ids[ready]]=t
            if t==times[-1] or bool(ready.all()):
                break
            still=~ready
            active_ids=active_ids[still]
            active_context=active_context[still]
            active_state=active_state[still]
            prev=t
        return chosen_logits,chosen_steps


def two_digit_loss(logits:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    if logits.ndim!=3 or logits.shape[-2:]!=(2,10) or target.shape!=logits.shape[:2]:
        raise ValueError("expected logits [B,2,10] and targets [B,2]")
    return F.cross_entropy(logits.reshape(-1,10),target.reshape(-1))