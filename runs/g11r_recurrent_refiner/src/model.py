from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]/'g5s_trace'/'src'/'model.py'
spec=importlib.util.spec_from_file_location('g5_base_model',BASE);g5=importlib.util.module_from_spec(spec);spec.loader.exec_module(g5)
PAD,BOS,EOS,VOCAB=g5.PAD,g5.BOS,g5.EOS,g5.VOCAB
ByteTokenizer,Norm,Block=g5.ByteTokenizer,g5.Norm,g5.Block

@dataclass
class Config:
    vocab:int=VOCAB;dim:int=96;heads:int=4;hidden:int=256;key_dim:int=12;max_context:int=256
    refine_steps:int=1
    variant:str='causal_hybrid'

class Model(nn.Module):
    def __init__(self,c:Config):
        super().__init__();self.config=c
        assert c.refine_steps in (1,4)
        self.embed=nn.Embedding(c.vocab,c.dim)
        self.blocks=nn.ModuleList([Block(c,k) for k in 'dada'])
        self.refiner=Block(c,'a')
        self.norm=Norm(c.dim)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if getattr(m,'bias',None) is not None:nn.init.zeros_(m.bias)
    def hidden_states(self,ids):
        x=self.embed(ids)
        for b in self.blocks:x,_=b(x)
        states=[]
        for _ in range(self.config.refine_steps):
            x,_=self.refiner(x)
            states.append(self.norm(x))
        return states
    def forward(self,ids,targets=None):
        states=self.hidden_states(ids)
        if targets is not None:
            keep=targets.ne(-100);loss=[]
            for x in states:loss.append(F.cross_entropy(F.linear(x[keep],self.embed.weight),targets[keep]))
            return torch.stack(loss).mean()
        return F.linear(states[-1],self.embed.weight)
    @torch.inference_mode()
    def generate(self,ids,max_new=32):
        self.eval();seq=ids.clone();outs=[];done=torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device)
        for _ in range(max_new):
            logits=self(seq)[:,-1];z=logits.argmax(-1);z=torch.where(done,torch.full_like(z,EOS),z);outs.append(z);done|=z.eq(EOS);seq=torch.cat((seq,z[:,None]),1)
            if bool(done.all()):break
        return torch.stack(outs,1)

def report(m):
    n=sum(p.numel() for p in m.parameters())
    return {'total_parameters':n,'active_trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'refine_steps':m.config.refine_steps,'shared_refiner_parameters':sum(p.numel() for p in m.refiner.parameters()),'base_blocks':4,'persistent_intertoken_delta_state_in_backbone':True,'generation_recomputes_full_prefix':True}
