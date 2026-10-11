"""I/O and exact checkpoint handling for a finite local continuation."""
from __future__ import annotations
import hashlib, json, os, random, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from model import Config, Model
from tokenizer import Tokenizer

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def put(p,z):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,p)

def rows(p):
    with Path(p).open() as f:return [json.loads(s) for s in f if s.strip()]

def write_rows(p,rr):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in rr))

def load_model(path,config=None,device='cpu'):
    path=Path(path)
    if path.is_dir():
        cfg=json.loads((path/'config.json').read_text());path=path/'model.safetensors'
    else:cfg=json.loads(Path(config).read_text())
    with torch.device('meta'):m=Model(Config(**cfg))
    m.load_state_dict(load_file(str(path)),strict=True,assign=True)
    # RoPE frequencies are intentionally non-persistent buffers.
    for b in m.blocks:
        if b.kind=='a':b.mix.freq=1/(m.config.rope_theta**(torch.arange(0,b.mix.hd,2).float()/b.mix.hd))
    return m.float().to(device)

def tensor_hashes(m):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in m.state_dict().items()}

def save_checkpoint(m,opt,step,sampler,counts,protocol_hash,device):
    from dataclasses import asdict
    path=ROOT/'checkpoints'/f'step_{step:04d}'
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial');tmp.mkdir(parents=True,exist_ok=False)
    save_file({k:v.detach().cpu().contiguous() for k,v in m.state_dict().items()},str(tmp/'model.safetensors'))
    put(tmp/'config.json',asdict(m.config))
    state={'schema':'L100V_v1','step':step,'optimizer':opt.state_dict(),'python_rng':random.getstate(),
      'torch_rng':torch.get_rng_state(),'sampler_rng':sampler.getstate(),'counts':counts,
      'protocol_sha256':protocol_hash,'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
    torch.save(state,tmp/'training_state.pt')
    put(tmp/'MANIFEST.json',{'schema':'L100V_v1','step':step,'parameters':sum(p.numel() for p in m.parameters()),
     'counts':counts,'protocol_sha256':protocol_hash,'files':{n:{'bytes':(tmp/n).stat().st_size,'sha256':sha(tmp/n)} for n in ['model.safetensors','config.json','training_state.pt']}})
    (tmp/'COMPLETE').write_text('complete\n');os.replace(tmp,path)
    put(ROOT/'reports/NEXT_STATE.json',{'status':'checkpoint_not_final_result','step':step,'checkpoint':str(path.relative_to(ROOT)),'goal_achieved':False})
    return path

def verify_checkpoint(p):
    p=Path(p)
    if not (p/'COMPLETE').exists():raise ValueError('Incomplete checkpoint')
    z=json.loads((p/'MANIFEST.json').read_text())
    for n,info in z['files'].items():
        if (p/n).stat().st_size!=info['bytes'] or sha(p/n)!=info['sha256']:raise ValueError('Checkpoint mismatch '+n)
    return z
