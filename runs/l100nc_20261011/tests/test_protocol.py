import copy,json,sys,random
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base')]
import data_nc as d
from judge import execute,validate
from common import batch,tokenize,rows,generate,Tokenizer
from model import Model,Config,delta_rule

def test_same_function_different_surface_grouped():
    a={'family':'pipeline','ops':[['add',4],['add',5]]}
    b={'family':'pipeline','ops':[['add',6],['add',3]]}
    assert d.semantic_key(a)==d.semantic_key(b)
    assert d.semantic_key(a)==d.semantic_key({**a,'ops':list(reversed(a['ops']))})

def test_disjoint_and_balanced():
    seen={}
    for s in ['train','validation','iid','surface','length','counterfactual']:
        rr=rows(s);g={r['semantic'] for r in rr}
        assert not (set(seen)&g)
        seen.update({k:s for k in g})
        assert sum(r['lang']=='ru' for r in rr)==len(rr)//2
        assert len({r['id'] for r in rr})==len(rr)

def test_counterfactuals():
    g={}
    for r in rows('counterfactual'):g.setdefault(r['pair_id'],[]).append(r)
    assert len(g)==80 and all(len(x)==2 for x in g.values())
    for a,b in g.values():
        assert a['prompt']!=b['prompt'] and a['semantic']!=b['semantic']
        assert any(d.expected(a['case'],x)!=d.expected(b['case'],x) for x in d.test_inputs(a['case']))

def test_gold_executes_all_families():
    for i,f in enumerate(d.FAMILIES):
        c=d.make_case(random.Random(712+i),f);xx=d.test_inputs(c)
        result=execute(d.solution(c),xx)
        assert result['error'] is None and result['outputs']==[d.expected(c,x) for x in xx]

def test_reject_code_escape():
    for c in ["import os", "def solve(x):\n return x.__class__", "def solve(x):\n return open('x')", "def solve(x):\n return __import__('os')", "def solve(x):\n return solve(x)", "def solve(x):\n while True: pass", "def solve(x):\n return 2 ** 100000000"]:
        assert not execute(c,[1])['valid']

def test_gold_code_not_in_prompt():
    for r in rows('train'):
        assert '\n    return ' not in r['prompt'] and 'def solve' not in r['prompt']
        assert 'x *' not in r['prompt'] and 'x +' not in r['prompt']

def test_loss_masks():
    rr=tokenize(rows('train')[:3]);x,y=batch(rr)
    for i,z in enumerate(rr):
        assert (y[i,:z['plen']-1]==-100).all()
        assert y[i,z['plen']-1:len(z['ids'])-1].tolist()==z['ids'][z['plen']:]

def tiny():
    torch.manual_seed(127)
    return Model(Config(dim=32,layers=2,heads=4,kv_heads=2,hidden=64,delta_key_dim=4,checkpoint_blocks=False)).eval()

def test_causal_and_cache():
    m=tiny();x=torch.randint(3,8192,(1,8));xx=x.clone();xx[:,4:]=torch.randint(3,8192,(1,4))
    with torch.no_grad():
        a,_=m(x);b,_=m(xx);assert torch.allclose(a[:,:4],b[:,:4],atol=2e-6)
        _,c=m(x[:,:5]);b,_=m(x[:,5:],cache=c);assert torch.allclose(a[:,5:],b,atol=2e-5)

def test_no_parameter_padding():
    with torch.device('meta'):m=Model(Config())
    assert sum(p.numel() for p in m.parameters())==100028328
    assert all(p.requires_grad for p in m.parameters())

def test_gradient_all_tensors():
    m=tiny().train();x=torch.randint(3,8192,(2,8));loss=m(x[:,:-1],x[:,1:]);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_tokenizer_roundtrip():
    t=Tokenizer.load(ROOT/'data/tokenizer.json')
    for x in ['Привет. def solve(x):\n    return x + 3\n','025 = -12','🙂 日本語']:
        assert t.decode(t.encode(x,bos=True,eos=True))==x

def test_prompt_only_interface():
    t=Tokenizer.load(ROOT/'data/tokenizer.json')
    class Spy:
        def generate(self,ids,max_new_tokens):
            assert max_new_tokens==96 and ids.shape[0]==1
            return torch.tensor([[2]])
    assert generate(Spy(),t,['ab'])==[{'text':'','ids':[],'eos':True}]
    with pytest.raises(TypeError):generate(Spy(),t,[{'gold':3}])
