"""Verify reconstructed native state and deterministic one-update continuation."""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from common import ROOT,setup,load,rows,encode,atomic_json,sha
from train import train_update
def fingerprint(m):
    h=hashlib.sha256()
    for name,p in m.state_dict().items():h.update(name.encode());h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def one(weights,state_path):
    m,tok=load(weights);st=torch.load(state_path,weights_only=True);cfg=st['schedule_config'];step=st['step']
    assert st['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
    assert st['sampler_cursor']==step*4
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False);opt.load_state_dict(st['optimizer']);torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);nr=st['numpy_rng'];np.random.set_state((nr['kind'],np.array(nr['keys'],dtype=np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']));del st
    data=rows('train.jsonl');cursor=(step*4)%len(data);batch=[encode(tok,r) for r in data[cursor:cursor+4]];assert len(batch)==4
    result=train_update(m,opt,batch,step+1,cfg);fp=fingerprint(m)
    return {'checkpoint_step':step,'fixture_ids':[r['id'] for r in data[cursor:cursor+4]],'loss':result['loss'],'grad_norm':result['grad_norm'],'after_fixture_parameter_fingerprint':fp}
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();setup()
    first=one(a.weights,a.state);second=one(a.weights,a.state);assert first==second
    atomic_json(a.output,{'status':'passed','weights_sha256':sha(a.weights),'native_state_sha256':sha(a.state),'two_independent_native_loads_replay_identically':True,'fixture':first,'saved_files_not_modified':True,'scope':'engineering continuation on already published training records only; not an additional quality run'})
    print('Remote reconstructed native checkpoint continuation verified',flush=True)
if __name__=='__main__':main()
