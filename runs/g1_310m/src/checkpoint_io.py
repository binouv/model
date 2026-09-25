from __future__ import annotations
import dataclasses,hashlib,json,os,random
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from model import Config,FlyGraphLM

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def save_checkpoint(model,path,optimizer=None,step=0,rng_state=None,training=None,dtype=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    (path/'config.json').write_text(json.dumps(dataclasses.asdict(model.config),indent=2))
    state=model.state_dict(); group={};size=0;index={};files=[]
    def flush():
        nonlocal group,size
        if not group:return
        name=f'model-{len(files):03d}.safetensors';tmp=path/(name+'.tmp')
        save_file(group,str(tmp));os.replace(tmp,path/name)
        files.append({'file':name,'bytes':(path/name).stat().st_size,'sha256':digest(path/name)})
        for k in group:index[k]=name
        group={};size=0
    for key,value in state.items():
        v=value.detach().cpu().contiguous()
        if dtype is not None and v.is_floating_point():v=v.to(dtype)
        n=v.numel()*v.element_size()
        if group and size+n>64*1024**2:flush()
        group[key]=v;size+=n
    flush()
    metadata={'format':'safetensors_sharded_v1','step':step,'weight_map':index,'files':files,'training':training or {},'dtype':str(dtype or next(model.parameters()).dtype)}
    if optimizer is not None:
        tmp=path/'training_state.pt.tmp'
        # Locally generated state only; never load untrusted pickle checkpoints.
        torch.save({'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'python_rng':random.getstate(),'sampler_rng':rng_state},tmp)
        os.replace(tmp,path/'training_state.pt');metadata['training_state_sha256']=digest(path/'training_state.pt')
    (path/'index.json').write_text(json.dumps(metadata,indent=2));(path/'COMPLETE').write_text('checkpoint fully saved and hashed\n')
    return metadata

def load_checkpoint(path,device='cpu',dtype=None):
    path=Path(path)
    if not (path/'COMPLETE').exists():raise ValueError('incomplete checkpoint')
    c=Config(**json.loads((path/'config.json').read_text()))
    # No double allocation of a random 310M model and loaded weights.
    with torch.device('meta'):model=FlyGraphLM(c)
    idx=json.loads((path/'index.json').read_text()); state={}
    for f in idx['files']:
        pp=path/f['file']
        if digest(pp)!=f['sha256']:raise ValueError('checkpoint SHA256 mismatch')
        state.update(load_file(str(pp),device=str(device)))
    model.load_state_dict(state,assign=True)
    # Nonpersistent rotary buffers are not in state_dict and must be reconstructed.
    for b in model.blocks:
        b.attn.freq=1.0/(c.rope_theta**(torch.arange(0,b.attn.hd,2,device=device).float()/b.attn.hd))
    if dtype is not None:model.to(dtype=dtype)
    return model,idx
