import hashlib,json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent/'g5s_trace'/'src')]
from model import Model,Config,report
import train

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_data_hashes_and_freshness():
 cfg=json.loads((ROOT/'configs/preregistered.json').read_text());m=json.loads((ROOT/'data/manifest.json').read_text());assert m['prior_denied_groups']==17904 and m['fresh_groups']==9492 and m['cross_split_canonical_overlap']==0
 for sp,h in cfg['data_sha256'].items():assert sha(ROOT/'data'/f'{sp}.jsonl')==h
def test_parameter_match_and_refiner_shared():
 torch.manual_seed(8401);a=Model(Config(refine_steps=4));torch.manual_seed(8401);b=Model(Config(refine_steps=1));assert report(a)['total_parameters']==report(b)['total_parameters'];assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
def test_same_batch_sequence():
 p,keys=train.pool();seq=[]
 for arm in train.ARMS:
  r=random.Random(84311);seq.append([[z['id'] for z in r.choices(p[keys[r.randrange(len(keys))]],k=16)] for _ in range(50)])
 assert seq[0]==seq[1]
def test_no_gold_case_in_prompt_and_strict_target():
 for z in train.rows('test_iid')[:50]:
  q=train.mat(z);assert 'F=<integer>' in q['prompt'] and q['answer'] not in q['prompt'][-25:];assert len(q['tokens'])<=256
def test_family_balance():
 d={}
 for z in train.rows('train'):d[z['family']]=d.get(z['family'],0)+1
 assert max(d.values())-min(d.values())<=1 and len(d)==5
def test_counterfactual_pairs():
 d={}
 for z in train.rows('test_counterfactual'):d.setdefault(z['pair_id'],[]).append(z)
 assert len(d)==100 and all(len(v)==2 and v[0]['answer']!=v[1]['answer'] for v in d.values())
def test_forward_backward_both_arms():
 p,keys=train.pool()
 for arm in train.ARMS:
  torch.manual_seed(1);m=Model(train.cfg_for(arm));r=random.Random(1);bb=r.choices(p[keys[0]],k=4);x,y,_=train.batch(bb);loss=m(x,y);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
def test_recur4_same_block_reused_not_untied():
 m=Model(Config(refine_steps=4));assert sum(1 for n,_ in m.named_modules() if n=='refiner')==1
def test_lr_endpoints():assert 0<train.lr_for(1)<.001 and 0<train.lr_for(2000)<=.000151
