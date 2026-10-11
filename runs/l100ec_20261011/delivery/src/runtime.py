"""Verified inference only; restores exact inherited model code, not EC trainer."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from model import Config,Model
from tokenizer import Tokenizer

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()

def checked(root:Path,relative:str,expected:str)->Path:
    root=root.resolve();p=(root/relative).resolve()
    if not p.is_relative_to(root):raise ValueError('path escapes bundle')
    if not p.is_file() or digest(p)!=expected:raise ValueError('missing file or SHA256 mismatch: '+relative)
    return p

def load_model(root:Path,arm:str,device:str='cpu'):
    root=Path(root).resolve();spec=json.loads((root/'ASSETS.json').read_text())
    if spec['format']!='L100EC-recovered-inference-v1':raise ValueError('unknown bundle format')
    if arm not in spec['models']:raise ValueError('unknown model arm')
    for name,expected in spec['base_source'].items():checked(root,'base/'+name,expected)
    cfg=spec['models'][arm]
    weights=checked(root,cfg['weights_path'],cfg['sha256'])
    if weights.stat().st_size!=cfg['bytes']:raise ValueError('wrong weight length')
    config=checked(root,cfg['config_path'],cfg['config_sha256'])
    tokenizer=checked(root,spec['tokenizer']['path'],spec['tokenizer']['sha256'])
    # BF16 bytes are loaded into FP32 parameters; unavailable FP32 precision is NOT recovered.
    m=Model(Config(**json.loads(config.read_text())))
    m.load_state_dict(load_file(str(weights)),strict=True)
    if sum(p.numel() for p in m.parameters())!=cfg['parameters']:raise ValueError('parameter count mismatch')
    m.eval().requires_grad_(False).to(device)
    return m,Tokenizer.load(tokenizer),spec

def generate(model,tokenizer,prompt:str,max_new_tokens:int=96,bf16_matmul:bool=True):
    if not isinstance(prompt,str):raise TypeError('prompt must be text')
    if not 1<=max_new_tokens<=256:raise ValueError('generation limit must be 1..256')
    device=next(model.parameters()).device
    ids=torch.tensor([tokenizer.encode(prompt,bos=True)],device=device)
    if ids.shape[1]+max_new_tokens>model.config.max_context:raise ValueError('prompt plus output budget exceeds context')
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16_matmul):
        seq=model.generate(ids,max_new_tokens=max_new_tokens)[0].tolist()
    eos=tokenizer.EOS in seq
    if eos:seq=seq[:seq.index(tokenizer.EOS)]
    return {'prompt':prompt,'completion':tokenizer.decode(seq),'token_ids':seq,'eos':eos,'generated_tokens':len(seq),'tools_used':False,'new_quality_measurement':False}
