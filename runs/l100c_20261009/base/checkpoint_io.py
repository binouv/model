"""Atomic sharded-free local checkpoints; weights and resumable state are separate.
Only load trusted, hash-verified optimizer pickle files created by this project.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,os,random
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from model import Config,Model,Attention

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False));os.replace(tmp,path)
def state_hash(m):
    h=hashlib.sha256()
    for n,p in sorted(m.state_dict().items()):
        h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def save(m,path,*,step,optimizer=None,sampler=None,run_config=None,counts=None,dtype=None):
    path=Path(path)
    if path.exists():raise FileExistsError('Refuse to overwrite a checkpoint: '+str(path))
    path.mkdir(parents=True)
    tensors={k:(v.detach().cpu().contiguous().to(dtype) if dtype is not None and v.is_floating_point() else v.detach().cpu().contiguous()) for k,v in m.state_dict().items()}
    save_file(tensors,str(path/'model.safetensors.tmp'));os.replace(path/'model.safetensors.tmp',path/'model.safetensors')
    put(path/'config.json',asdict(m.config));files=['model.safetensors','config.json']
    if optimizer is not None:
        st={'schema':'FlyGraph-L100M-v1','step':step,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'sampler_state':sampler.state(),'run_config':run_config,'counts':counts}
        torch.save(st,path/'training_state.pt.tmp');os.replace(path/'training_state.pt.tmp',path/'training_state.pt');files.append('training_state.pt')
    manifest={'schema':'FlyGraph-L100M-v1','step':step,'parameters':sum(p.numel() for p in m.parameters()),'saved_dtype':str(dtype or next(m.parameters()).dtype),
        'files':{n:{'bytes':(path/n).stat().st_size,'sha256':sha(path/n)} for n in files},'optimizer_rng_present':optimizer is not None,'counts':counts or {}}
    put(path/'MANIFEST.json',manifest);(path/'COMPLETE').write_text('Atomic files saved and hashes verified\n')
    return manifest

def load(path,device='cpu'):
    path=Path(path)
    if not (path/'COMPLETE').exists():raise ValueError('Incomplete checkpoint')
    mf=json.loads((path/'MANIFEST.json').read_text())
    for n,rec in mf['files'].items():
        if Path(n).name!=n:raise ValueError('Unsafe filename')
        p=path/n
        if p.stat().st_size!=rec['bytes'] or sha(p)!=rec['sha256']:raise ValueError('Checkpoint hash mismatch: '+n)
    c=Config(**json.loads((path/'config.json').read_text()))
    with torch.device('meta'):m=Model(c)
    state=load_file(str(path/'model.safetensors'),device=str(device));m.load_state_dict(state,assign=True)
    for b in m.blocks:
        if isinstance(b.mix,Attention):b.mix.freq=1.0/(c.rope_theta**(torch.arange(0,b.mix.hd,2,device=device).float()/b.mix.hd))
    return m,mf
