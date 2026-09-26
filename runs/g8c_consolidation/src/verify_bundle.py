"""Verify G8C bundle bytes, replay native optimizer/RNG and rescore saved predictions.
Loads only manifest-verified native checkpoints created by this project.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base')]
import study
from model import Model,Config

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 manifest=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())
 for r in manifest['files']:
  p=(ROOT/r['path']).resolve()
  if not p.is_relative_to(ROOT.resolve()) or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('bundle file mismatch: '+r['path'])
 scores=0;replays=0;updates=0
 for seed in study.SEEDS:
  for arm in study.ARMS:
   name=f'{arm}_s{seed}';cp=ROOT/'checkpoints'/name/'step2000'
   state=load_file(str(cp/'model.safetensors'));m=Model(Config(**json.loads((cp/'config.json').read_text())));m.load_state_dict(state)
   native=torch.load(cp/'resume.pt',weights_only=False,map_location='cpu')
   assert native['step']==2000 and all(torch.equal(v,native['model'][k]) for k,v in state.items())
   assert all(torch.isfinite(p).all() for p in m.parameters())
   saved=json.loads((ROOT/f'metrics/{name}.json').read_text())
   assert study.state_hash(m)==saved['final_weight_state_sha256']
   for split in study.SPLITS:
    gold={z['id']:z for z in study.read_rows(ROOT/f'data/fresh/{split}.jsonl')}
    for stage in ['before','tests']:
     rows=saved[stage][split]['rows'];assert set(gold)=={z['id'] for z in rows}
     for z in rows:
      val=study.parse(z['text']);ans=study.answer(gold[z['id']]['case']);ok=z['eos'] and val is not None and val==ans
      assert ok==z['exact'] and ans==int(z['gold'])
      scores+=1
   check=[study.pack(z) for z in study.read_rows(ROOT/'data/fresh/iid.jsonl')[:16]]
   result=study.evaluate(m,check)['rows'];old={z['id']:z for z in saved['tests']['iid']['rows']}
   for z in result:
    assert all(z[k]==old[z['id']][k] for k in ['text','ids','eos','exact']);replays+=1
   keys,pools=study.pools_for_training();hashes=[]
   for _ in range(2):
    mm,opt,rng=study.restore(native);key=keys[rng.randrange(len(keys))];batch=rng.choices(pools[key],k=16)
    study.update(mm,opt,batch,.00006);hashes.append(study.state_hash(mm));updates+=1
   assert hashes[0]==hashes[1]
 print(json.dumps({'status':'passed','files_verified':len(manifest['files']),'raw_predictions_rescored':scores,'reloaded_generation_cases':replays,'optimizer_replay_checks':updates//2,'engineering_updates_discarded':updates,'not_new_training_runs':True},indent=2))
if __name__=='__main__':main()
