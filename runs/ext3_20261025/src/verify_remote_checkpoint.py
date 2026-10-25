import argparse,hashlib,json
from pathlib import Path
import torch
from common import ROOT,setup,load,rows,sha,atomic_json
from objective import configure
from verify_resume import restore_rng
from train import train_update

def one(w,sp):
 m,tok=load(w);params=configure(m);s=torch.load(sp,weights_only=True);cfg=s['schedule_config'];step=s['step'];assert s['sampler_cursor']==3*step and s['data_manifest_sha256']==sha(ROOT/'data/manifest.json');opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False);opt.load_state_dict(s['optimizer']);restore_rng(s);del s
 data=rows('train.jsonl');cursor=(3*step)%len(data);v=train_update(m,opt,data[cursor:cursor+3],step+1,cfg,tok);h=hashlib.sha256()
 for name,p in m.state_dict().items():h.update(name.encode());h.update(p.detach().contiguous().numpy().tobytes())
 return dict(checkpoint_step=step,loss=v['loss'],grad_norm=v['grad_norm'],after_fixture_parameter_fingerprint=h.hexdigest(),fixture_ids=[r['id'] for r in data[cursor:cursor+3]])
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();setup();x=one(a.weights,a.state);y=one(a.weights,a.state);assert x==y
 atomic_json(a.output,dict(status='passed',weights_sha256=sha(a.weights),native_state_sha256=sha(a.state),two_independent_native_loads_replay_identically=True,fixture=x,saved_files_not_modified=True))
 print('Remote reconstructed native checkpoint replay passed',flush=True)
if __name__=='__main__':main()
