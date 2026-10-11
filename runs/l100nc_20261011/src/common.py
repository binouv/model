from __future__ import annotations
import dataclasses,hashlib,json,os,random,sys,time
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from model import Model,Config,parameter_report
from tokenizer import Tokenizer
PARENT_SHA='07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)

def load_model(path=None):
    path=Path(path) if path else ROOT/'parent'
    weight=path/'model.safetensors';cfg=json.loads((path/'config.json').read_text())
    if path==ROOT/'parent' and sha(weight)!=PARENT_SHA:raise ValueError('wrong ancestor')
    if (path/'MANIFEST.json').exists():
        spec=json.loads((path/'MANIFEST.json').read_text())
        for name,s in spec['files'].items():
            if sha(path/name)!=s['sha256']:raise ValueError('SHA256 mismatch '+name)
    m=Model(Config(**cfg));m.load_state_dict(load_file(str(weight)),strict=True)
    assert sum(p.numel() for p in m.parameters())==100028328
    return m

def rows(split):
    return [json.loads(l) for l in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]

def tokenize(rr):
    t=Tokenizer.load(ROOT/'data/tokenizer.json');out=[]
    for r in rr:
        p=t.encode(r['prompt'],bos=True);target=t.encode(r['target'],eos=True)
        out.append({**r,'ids':p+target,'plen':len(p)})
    return out

def batch(rr):
    length=max(len(r['ids']) for r in rr)-1
    x=torch.zeros(len(rr),length,dtype=torch.long);y=torch.full_like(x,-100)
    for i,r in enumerate(rr):
        t=torch.tensor(r['ids']);n=len(t)-1;x[i,:n]=t[:-1];y[i,r['plen']-1:n]=t[r['plen']:]
    return x,y

def save(m,opt,rng,step,records,parent_sha=PARENT_SHA,dtype=None):
    path=ROOT/'checkpoints'/f'step_{step:04d}' if dtype is None else ROOT/'export_bf16'
    path.mkdir(parents=True,exist_ok=False)
    state={k:v.detach().cpu().contiguous().to(dtype=dtype or v.dtype) for k,v in m.state_dict().items()}
    tmp=path/'model.safetensors.tmp';save_file(state,str(tmp));os.replace(tmp,path/'model.safetensors')
    put(path/'config.json',dataclasses.asdict(m.config))
    if opt is not None:
        st={'schema':'L100NC-v1','step':step,'optimizer':opt.state_dict(),'sampler_rng':rng.getstate(),
            'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'records':records,
            'parent_sha256':parent_sha,'new_optimizer_at_stage_start':True,
            'protocol_sha256':sha(ROOT/'configs/preregistered.json')}
        torch.save(st,path/'training_state.pt')
    put(path/'MANIFEST.json',{'step':step,'schema':'L100NC-v1','files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in path.iterdir() if p.is_file()},'exact_ancestor_resume':False,'parameters':100028328})
    (path/'COMPLETE').write_text('all files saved and hashed\n');return path

@torch.inference_mode()
def generate(m,tok,prompts,limit=96):
    if not all(isinstance(p,str) for p in prompts):raise TypeError('prompt text only')
    tt=[tok.encode(p,bos=True) for p in prompts]
    if len({len(t) for t in tt})!=1:raise ValueError('equal length inference batches')
    with torch.autocast('cpu',dtype=torch.bfloat16):out=m.generate(torch.tensor(tt),max_new_tokens=limit)
    result=[]
    for seq in out.tolist():
        eos=tok.EOS in seq;seq=seq[:seq.index(tok.EOS)] if eos else seq
        result.append({'text':tok.decode(seq),'ids':seq,'eos':eos})
    return result
