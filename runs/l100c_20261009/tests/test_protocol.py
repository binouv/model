from pathlib import Path
import copy,importlib.util,json,random,sys
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import *
import data
import train
from evaluate import predict,parse
from checkpoint_io import state_hash

def tiny():
 torch.manual_seed(9311)
 return Model(Config(vocab_size=8192,dim=32,layers=2,heads=4,kv_heads=2,hidden=64,delta_key_dim=4,checkpoint_blocks=False))

def test_exact_count_meta():
 with torch.device('meta'):m=Model(Config())
 assert sum(p.numel() for p in m.parameters())==100028328

def test_all_splits_disjoint_and_valid():
 seen={}
 for split in ['train','validation','test_iid','test_surface','test_extrapolation']:
  for pp in train.pair_list(split):
   a,b=pp;assert a['group']==b['group'] and a['answer']!=b['answer']
   assert a['group'] not in seen;seen[a['group']]=split
   assert a['prompt']!=b['prompt'] and a['lang']==b['lang']
   assert all(data.independent(z['case'])==int(z['answer'])==data.old.solve(z['case']) for z in pp)

def test_memory_chronology_group():
 a={'family':'memory_update','writes':[['a',1],['b',2],['a',3]],'query':'a'}
 b={'family':'memory_update','writes':[['b',2],['a',1],['a',3]],'query':'b'}
 assert data.group(a)==data.group(b)
 assert data.independent(a)==3 and data.independent(b)==2
 b['writes']=[['a',3],['b',2],['a',1]]
 assert data.group(a)!=data.group(b)

def test_commutative_and_program_groups():
 a={'family':'arithmetic','values':[3,7],'op':'+'};b={'family':'arithmetic','values':[7,3],'op':'*'}
 assert data.group(a)==data.group(b)
 a={'family':'code_trace','start':3,'program':[['+',2],['-',4]]};b={'family':'code_trace','start':3,'program':[['+',4],['+',2]]}
 assert data.group(a)==data.group(b)

def test_masks_swaps_identical_lengths():
 tok=Tokenizer.load(ROOT/'data/tokenizer.json')
 for pp in train.pair_list('validation'):
  x,y,ids=train.make_batch([pp],tok)
  assert torch.equal(y.ne(-100).sum(1)[::2],y.ne(-100).sum(1)[1::2])
  for j,z in enumerate(pp):
   p=z['prompt_tokens'];assert (y[2*j,:p-1]==-100).all()
   assert tok.decode(y[2*j][y[2*j]!=-100].tolist())=='F='+z['answer']
   assert tok.decode(y[2*j+1][y[2*j+1]!=-100].tolist())=='F='+pp[1-j]['answer']

def test_control_exact_positive_nll():
 m=tiny();tok=Tokenizer.load(ROOT/'data/tokenizer.json');x,y,_=train.make_batch(train.pair_list('validation')[:1],tok)
 loss,nll,_,_,_=train.loss_components(m,x,y,0,.2)
 base=m(x[::2],y[::2]);assert torch.allclose(loss,base,atol=2e-6)
 assert torch.equal(loss,nll)

def test_rank_prefers_correct_and_ties_not_success():
 import torch.nn.functional as F
 s=torch.tensor([-.1,-1.,-.1,-1.]);correct=F.softplus(.2+s[1::2]-s[::2]).mean()
 wrong=F.softplus(.2+s[::2]-s[1::2]).mean();assert correct<wrong
 s=torch.tensor([-.3,-.3]);assert not bool((s[::2]>s[1::2]).any())

def test_parser_no_gold_dependent_last_number():
 assert parse('F=12')==12 and parse('T=2+4=6;F=6')==6
 for bad in ['12','F=12;F=7','F=12 blah','T=F=2;F=2']:assert parse(bad) is None

def test_prompt_only_interface():
 tok=Tokenizer.load(ROOT/'data/tokenizer.json')
 class Spy:
  def generate(self,ids,max_new_tokens):
   assert max_new_tokens==64
   assert ids.tolist()==[tok.encode('Hello',bos=True)]
   return torch.tensor([tok.encode('F=7',eos=True)])
 assert predict(Spy(),tok,['Hello'])[0]['text']=='F=7'

def test_causality_and_cache():
 m=tiny().eval();x=torch.randint(3,300,(1,16));xx=x.clone();xx[:,8:]=torch.randint(3,300,(1,8))
 with torch.no_grad():
  a=m(x)[0];b=m(xx)[0];assert torch.allclose(a[:,:8],b[:,:8],atol=1e-6)
  _,state=m(x[:,:7]);b,_=m(x[:,7:],cache=state);assert torch.allclose(a[:,7:],b,atol=2e-5)

def test_gradients_all_trainable_tensors():
 m=tiny();tok=Tokenizer.load(ROOT/'data/tokenizer.json');x,y,_=train.make_batch(train.pair_list('validation')[:1],tok)
 loss,*_=train.loss_components(m,x,y,.5,.2);loss.backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in m.parameters())

def test_native_sampler_optimizer_replay():
 cfg=json.loads((ROOT/'configs/preregistered.json').read_text());m=tiny();sam=train.Sampler(cfg);tok=Tokenizer.load(ROOT/'data/tokenizer.json')
 opt=torch.optim.Adafactor(m.parameters(),lr=.006,foreach=False)
 train.update(m,opt,sam.next(),tok,.006,.5,.2)
 state=copy.deepcopy(m.state_dict());op=copy.deepcopy(opt.state_dict());rng=sam.state();trng=torch.get_rng_state();hashes=[]
 for _ in range(2):
  n=tiny();n.load_state_dict(state);o=torch.optim.Adafactor(n.parameters(),lr=.006,foreach=False);o.load_state_dict(copy.deepcopy(op));s=train.Sampler(cfg);s.restore(rng);torch.set_rng_state(trng)
  train.update(n,o,s.next(),tok,.006,.5,.2);hashes.append(state_hash(n))
 assert hashes[0]==hashes[1]
