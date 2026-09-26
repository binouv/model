import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base')]
import torch
import experiment as e
from model import Model,Config
def setup_module():
 if not (ROOT/'data/train.jsonl').exists():e.build_data()
def test_split_pair_integrity_and_no_overlap():
 denied=e.denied_groups();seen=set()
 for sp in ['train','validation','held_iid','held_surface','held_extrapolation']:
  rows=e.read(ROOT/f'data/{sp}.jsonl');d={}
  for z in rows:d.setdefault(z['canonical'],[]).append(z)
  assert all(len(v)==2 for v in d.values());assert not (set(d)&denied) and not (set(d)&seen);seen|=set(d)
  for p in d.values():assert p[0]['answer']==p[1]['negative_answer'] and p[1]['answer']==p[0]['negative_answer']
def test_candidate_batch_masks_only_completion():
 rows=e.read(ROOT/'data/train.jsonl')[:4];x,y=e.candidate_batch(rows,False)
 for i,z in enumerate(rows):assert (y[i,:z['prompt_tokens']-1]==-100).all() and (y[i,z['prompt_tokens']-1:]!=-100).any()
def test_contrastive_has_expected_direction_on_constructed_logits():
 pos=torch.tensor([0.2,0.4]);neg=torch.tensor([1.0,0.9]);good=torch.nn.functional.softplus(0.5+pos-neg).mean();bad=torch.nn.functional.softplus(0.5+neg-pos).mean();assert good<bad
def test_model_parameter_count_and_forward():
 m=Model(Config());assert sum(p.numel() for p in m.parameters())==469648
 rows=e.read(ROOT/'data/train.jsonl')[:2];x,y=e.candidate_batch(rows,False);n=e.seq_nll(m,x,y);assert n.shape==(2,) and torch.isfinite(n).all()
def test_free_generation_api_only_takes_prompts():
 m=Model(Config());rows=e.read(ROOT/'data/held_iid.jsonl')[:2];out,_=e.generate(m,[z['prompt'] for z in rows],max_new=2);assert len(out)==2 and all(set(z)=={'text','ids','eos'} for z in out)
def test_parent_checkpoints_are_real_and_distinct():
 hs=[]
 for s in e.SEEDS:
  p,st,m,opt=e.restore_parent(s);assert p.stat().st_size>1_000_000 and st['step']==2000;hs.append(e.state_hash(m))
 assert len(set(hs))==3
