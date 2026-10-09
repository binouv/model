"""Learned prompt-copy distribution over exact tokenizer IDs.
Original FlyGraph implementation inspired by the pointer-generator concept.
No arithmetic parser, latest-key solver, candidate answers or gold in inference.
"""
from __future__ import annotations
import sys, math, json, hashlib, random, os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from model import Config, Model
from tokenizer import Tokenizer

class PointerModel(Model):
    def __init__(self,c:Config,copy_enabled:bool=True,rank:int=64):
        super().__init__(c)
        self.copy_enabled=bool(copy_enabled);self.rank=rank
        self.copy_query=nn.Linear(c.dim,rank,bias=False)
        self.copy_key=nn.Linear(c.dim,rank,bias=False)
        self.copy_gate=nn.Linear(c.dim,1,bias=True)
        nn.init.normal_(self.copy_query.weight,std=.02)
        nn.init.normal_(self.copy_key.weight,std=.02)
        nn.init.zeros_(self.copy_gate.weight)
        nn.init.constant_(self.copy_gate.bias,-3.)
    def encode(self,ids,cache=None):
        old=None if cache is None else cache['blocks']
        past=0 if old is None else old[1][0].shape[2]
        if past+ids.shape[1]>self.config.max_context:raise ValueError('context exceeded')
        x=self.embed(ids);states=[]
        for i,b in enumerate(self.blocks):
            x,s=b(x,None if old is None else old[i]);states.append(s)
        return self.norm(x),states,past
    def distribution(self,h,key_bank,source_ids,allowed):
        # h [N,D]; key_bank [N,S,R], allowed [N,S]. All math below FP32.
        with torch.autocast(device_type=h.device.type,enabled=False):
            logits=F.linear(h.float(),self.embed.weight.float())
            pv=logits.softmax(-1)
            q=F.linear(h.float(),self.copy_query.weight.float())
            scores=(key_bank.float()*q[:,None,:]).sum(-1)/math.sqrt(self.rank)
            valid=allowed.any(-1)
            safe=allowed.clone()
            safe[~valid,0]=True
            attention=scores.masked_fill(~safe,-1e9).softmax(-1)
            attention=attention*allowed.float()
            pc=torch.zeros_like(pv).scatter_add(1,source_ids,attention)
            gate=F.linear(h.float(),self.copy_gate.weight.float(),self.copy_gate.bias.float()).sigmoid()
            gate=gate*valid[:,None]*float(self.copy_enabled)
            mixture=(1-gate)*pv+gate*pc
            return mixture,gate
    def forward(self,ids,targets=None,prefix_lens=None,cache=None,last_only=False,return_gate=False):
        h,blocks,past=self.encode(ids,cache)
        b,t,d=h.shape
        if cache is None:
            if prefix_lens is None:
                prefix_lens=torch.full((b,),t,dtype=torch.long,device=ids.device)
            if prefix_lens.shape!=(b,) or bool((prefix_lens<1).any()) or bool((prefix_lens>t).any()):
                raise ValueError('invalid prompt boundaries')
            key_bank=self.copy_key(h)
            source_ids=ids
            positions=torch.arange(t,device=ids.device)
            source_mask=(positions[None,:]<prefix_lens[:,None])&ids.ge(3)
        else:
            key_bank,source_ids,source_mask=cache['keys'],cache['ids'],cache['source_mask']
            positions=torch.arange(source_ids.shape[1],device=ids.device)
        if targets is not None:
            keep=targets.ne(-100)
            if not bool(keep.any()):raise ValueError('no target tokens')
            bi,ti=keep.nonzero(as_tuple=True);hh=h[bi,ti]
            allowed=source_mask[bi]&(positions[None,:]<=(ti+past)[:,None])
            prob,gate=self.distribution(hh,key_bank[bi],source_ids[bi],allowed)
            loss=F.nll_loss(prob.clamp_min(1e-30).log(),targets[bi,ti])
            return (loss,gate.mean().detach()) if return_gate else loss
        if last_only:
            bi=torch.arange(b,device=ids.device);ti=torch.full((b,),t-1,device=ids.device)
        else:
            bi=torch.arange(b,device=ids.device).repeat_interleave(t)
            ti=torch.arange(t,device=ids.device).repeat(b)
        allowed=source_mask[bi]&(positions[None,:]<=(ti+past)[:,None])
        prob,gate=self.distribution(h[bi,ti],key_bank[bi],source_ids[bi],allowed)
        out=prob.clamp_min(1e-30).log().view(b,1 if last_only else t,-1)
        state={'blocks':blocks,'keys':key_bank,'ids':source_ids,'source_mask':source_mask}
        return out,state
    @torch.inference_mode()
    def generate(self,ids,max_new_tokens=32,eos_id=2):
        self.eval();lg,cache=self(ids,last_only=True)
        done=torch.zeros(ids.shape[0],dtype=torch.bool,device=ids.device);out=[]
        for _ in range(min(max_new_tokens,self.config.max_context-ids.shape[1])):
            nxt=lg[:,-1].argmax(-1);nxt=torch.where(done,torch.full_like(nxt,eos_id),nxt)
            out.append(nxt);done|=nxt.eq(eos_id)
            if bool(done.all()):break
            lg,cache=self(nxt[:,None],cache=cache,last_only=True)
        return torch.stack(out,1) if out else ids.new_empty((ids.shape[0],0))

def sha(path):
    hh=hashlib.sha256()
    with Path(path).open('rb') as f:
        for bb in iter(lambda:f.read(4<<20),b''):hh.update(bb)
    return hh.hexdigest()
def state_hash(m):
    hh=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):
        hh.update(k.encode());hh.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return hh.hexdigest()
def atomic_json(path,z):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False))
    os.replace(tmp,path)
def init_parent(path,enabled=True):
    path=Path(path)
    if sha(path/'model.safetensors')!='b2112e21dcc653780da5c6f36137c0deab3f2fcc01bb04662eb518cf4a07b3ec':
        raise ValueError('incorrect ancestor bytes')
    torch.manual_seed(9201)
    c=Config(**json.loads((path/'config.json').read_text()))
    c.checkpoint_blocks=False
    m=PointerModel(c,enabled)
    old=load_file(str(path/'model.safetensors'))
    missing,unexpected=m.load_state_dict(old,strict=False)
    assert set(missing)=={'copy_query.weight','copy_key.weight','copy_gate.weight','copy_gate.bias'}
    assert not unexpected
    return m.float()

def save_checkpoint(m,path,step,opt=None,rng=None,counts=None,dtype=None):
    from dataclasses import asdict
    path=Path(path)
    if path.exists():raise FileExistsError('refuse overwrite '+str(path))
    path.mkdir(parents=True)
    cfg={'backbone':asdict(m.config),'copy_enabled':m.copy_enabled,'rank':m.rank}
    atomic_json(path/'config.json',cfg)
    sd={k:v.detach().cpu().contiguous().to(dtype) if dtype and v.is_floating_point() else v.detach().cpu().contiguous() for k,v in m.state_dict().items()}
    save_file(sd,str(path/'model.safetensors'))
    if opt is not None:
        torch.save({'schema':'L100P_v1','step':step,'optimizer':opt.state_dict(),
            'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
            'sampler_rng':rng.getstate(),'counts':counts},path/'training_state.pt')
    names=['model.safetensors','config.json']+(['training_state.pt'] if opt is not None else [])
    atomic_json(path/'MANIFEST.json',{'step':step,'counts':counts,'parameters':sum(p.numel() for p in m.parameters()),
        'files':{n:{'bytes':(path/n).stat().st_size,'sha256':sha(path/n)} for n in names},
        'copy_enabled':m.copy_enabled,'state_dtype':str(next(iter(sd.values())).dtype),
        'optimizer_rng_saved':opt is not None,'model_state_hash':state_hash(m) if dtype is None else None})
    (path/'COMPLETE').write_text('complete\n')
    return path
def load_checkpoint(path):
    path=Path(path)
    if not (path/'COMPLETE').exists():raise ValueError('incomplete checkpoint')
    mf=json.loads((path/'MANIFEST.json').read_text())
    for n,v in mf['files'].items():
        if sha(path/n)!=v['sha256']:raise ValueError('bad checkpoint SHA256')
    cfg=json.loads((path/'config.json').read_text())
    with torch.device('meta'):
        m=PointerModel(Config(**cfg['backbone']),cfg['copy_enabled'],cfg['rank'])
    m.load_state_dict(load_file(str(path/'model.safetensors')),assign=True)
    for b in m.blocks:
        if b.kind=='a':
            b.mix.freq=1/(m.config.rope_theta**(torch.arange(0,b.mix.hd,2).float()/b.mix.hd))
    return m.float(),mf
