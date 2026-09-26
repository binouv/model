import copy,json,random,sys,tempfile
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base')]
import study as s
from model import Config,Model,ByteTokenizer,parallel_delta
import data

def test_source_matches_registered_parent():
 import hashlib
 for name,digest in [('model.py','217e14d62f7cd40b16971c562ac208978354e1d1'),('data.py','edb79ee84eba4e06c0997becb3d0280f3bbb5bb6')]:
  b=(ROOT/'base'/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==digest

def test_full_parameters():
 assert sum(p.numel() for p in Model(Config()).parameters())==469648

def test_unrestricted_loss_mask():
 r=s.pack(data.row({'family':'arithmetic','values':[4,7],'op':'+'},0,ByteTokenizer()))
 x,y,p=s.batch([r]);assert (y[0,:p[0]-1]==-100).all();assert y[0,p[0]-1:].tolist()==ByteTokenizer().encode('F=11',eos=True)

def test_parser_no_arbitrary_last_integer():
 assert s.parse('T=3+4=7;F=7')==7 and s.parse('F=-5')==-5
 for bad in ['7','F=7;F=4','T=F=7;F=4','F=7 extra','x=7']:
  assert s.parse(bad) is None

def test_semantic_canonicalization():
 assert data.canonical({'family':'arithmetic','values':[4,7],'op':'+'})==data.canonical({'family':'arithmetic','values':[7,4],'op':'*'})
 a={'family':'memory_update','writes':[['a',1],['b',2],['a',3]],'query':'a'}
 b={'family':'memory_update','writes':[['b',2],['a',1],['a',3]],'query':'b'}
 assert data.canonical(a)==data.canonical(b)
 assert s.answer(a)==3 and s.answer(b)==2

def test_actual_parent_and_fresh_data():
 s.prepare();audit=json.loads((ROOT/'reports/DATA_AUDIT.json').read_text());assert audit['parent_overlap']==0 and audit['fresh_cases']==1600
 seen={}
 for p in (ROOT/'data/fresh').glob('*.jsonl'):
  for z in s.read_rows(p):
   assert int(z['answer'])==s.answer(z['case'])
   assert z['canonical'] not in seen or seen[z['canonical']]==p.stem=='counterfactual'
   seen[z['canonical']]=p.stem

def test_nontrivial_counterfactuals():
 pairs={}
 for z in s.read_rows(ROOT/'data/fresh/counterfactual.jsonl'):pairs.setdefault(z['pair_id'],[]).append(z)
 assert len(pairs)==200 and all(len(v)==2 and v[0]['answer']!=v[1]['answer'] for v in pairs.values())

def tiny():
 torch.manual_seed(33);return Model(Config(dim=32,heads=4,hidden=64,key_dim=4)).eval()

def test_causality_and_cache():
 m=tiny();x=torch.randint(3,259,(1,20));xx=x.clone();xx[:,10:]=torch.randint(3,259,(1,10))
 with torch.no_grad():
  a,_=m(x);b,_=m(xx);assert torch.allclose(a[:,:10],b[:,:10],atol=2e-6)
  _,cache=m(x[:,:9]);b,_=m(x[:,9:],cache=cache);assert torch.allclose(a[:,9:],b,atol=2e-5)

def test_delta_batch_equals_stream():
 torch.manual_seed(1);q=torch.randn(2,2,11,3);k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1);v=torch.randn(2,2,11,4);g=-torch.rand(2,2,11);beta=torch.rand(2,2,11)
 full,last=parallel_delta(q,k,v,g,beta);state=None;outs=[]
 for t in range(11):
  out,state=parallel_delta(q[:,:,t:t+1],k[:,:,t:t+1],v[:,:,t:t+1],g[:,:,t:t+1],beta[:,:,t:t+1],state);outs.append(out)
 assert torch.allclose(full,torch.cat(outs,2),atol=2e-5) and torch.allclose(last,state,atol=2e-5)

def test_real_optimizer_rng_next_update():
 m=tiny();opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95));r=random.Random(19)
 z=s.pack(data.row({'family':'arithmetic','values':[4,7],'op':'+'},0,ByteTokenizer()));s.update(m,opt,[z],.001)
 st={'schema':'G8C_v1','config':m.config.__dict__,'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'sampler_rng':r.getstate(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state()}
 a,oa,ra=s.restore(st);b,ob,rb=s.restore(st);assert ra.random()==rb.random()
 s.update(a,oa,[z],.0003);s.update(b,ob,[z],.0003);assert s.state_hash(a)==s.state_hash(b)

def test_prompt_only_interface():
 class Spy:
  def generate(self,ids,pl,max_new):
   assert max_new==128 and ids.shape==(2,3);return torch.tensor([[73,64,58,2],[73,64,58,2]])
 out,_=s.predictions(Spy(),['ab','ab']);assert all(z['text']=='F=7' and z['eos'] for z in out)

def test_no_partial_aggregate(tmp_path,monkeypatch):
 monkeypatch.setattr(s,'ROOT',tmp_path)
 with pytest.raises(FileNotFoundError):s.aggregate()
