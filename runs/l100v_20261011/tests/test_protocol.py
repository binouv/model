from pathlib import Path
import copy,json,random,sys
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import Config,Model,ROOT,Tokenizer,rows
from data import canonical,solve,make_row,pack
from engine import batch,loss_forward,generate
from evaluate import execute_code,expected_code,numeric,score

def tiny():
    torch.manual_seed(17);torch.set_num_threads(2)
    return Model(Config(vocab_size=8192,dim=32,layers=2,heads=4,kv_heads=2,hidden=64,delta_key_dim=8,max_context=256,checkpoint_blocks=False))

def test_canonical_commutative_and_subtraction():
    a={'family':'arithmetic','a':2,'b':7,'op':'+'};b={**a,'a':7,'b':2}
    assert canonical(a)==canonical(b)
    assert canonical({**a,'op':'-'})!=canonical({**b,'op':'-'})

def test_memory_groups_ignore_query_and_interleaving():
    a={'family':'memory_update','writes':[['a',1],['b',3],['a',8]],'query':'a'}
    b={**a,'writes':[['b',3],['a',1],['a',8]],'query':'b'}
    assert canonical(a)==canonical(b)
    assert canonical(a)!=canonical({**a,'writes':[['a',8],['b',3],['a',1]]})
    assert solve(a)[0]=='8'

def test_affine_equivalence_group():
    a={'family':'code_generation','ops':[['+',2],['*',3]]}
    b={'family':'code_generation','ops':[['*',3],['+',6]]}
    assert canonical(a)==canonical(b)

@pytest.mark.parametrize('case',[
 {'family':'arithmetic','a':47,'b':65,'op':'+'},
 {'family':'arithmetic','a':7,'b':16,'op':'-'},
 {'family':'conditional','a':8,'threshold':8,'delta':3},
 {'family':'code_trace','start':3,'ops':[['+',2],['-',8]]},
 {'family':'list_reasoning','values':[2,3,4]},
 {'family':'memory_update','writes':[['a',2],['b',7],['a',6]],'query':'a'},
 {'family':'code_generation','ops':[['+',3],['*',4],['-',1]]}])
def test_target_oracle_and_pack(case):
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');r=make_row(case,'ru',0,1)
    ok,_=score(r['completion'],True,r);assert ok
    x,y,w=batch([pack(r,tok)])
    assert y.ne(-100).sum()==len(tok.encode(r['completion'],eos=True))
    assert (w[y==-100]==0).all()
    assert (w[y!=-100]>=1).all()
    if case['family']!='code_generation':assert (w==8).any()
    assert not score(r['completion'],False,r)[0]

def test_no_labels_in_generator():
    with pytest.raises(TypeError):generate(None,None,[{'prompt':'p','answer':'1'}])

@pytest.mark.parametrize('text',["import os\ndef solve(x): return 1",'def solve(x):\n while True: pass','def solve(x):\n return eval("1")','def solve(x):\n return x**999999','def solve(x):\n return x.__class__'])
def test_reject_unsafe_code(text):
    with pytest.raises((ValueError,SyntaxError)):execute_code(text)

def test_correct_code_and_no_repair():
    c={'family':'code_generation','ops':[['+',4],['*',3]]}
    _,s,_=solve(c);assert execute_code(s)==expected_code(c)
    with pytest.raises((ValueError,SyntaxError)):execute_code('return x+1')

@pytest.mark.parametrize('s,expected',[('T=1,2;F=2',2),('F=-14',-14),('F=2 F=3',None),('the answer is 2',None),('F=2 garbage',None)])
def test_numeric_parser(s,expected):assert numeric(s)==expected

def test_uniform_loss_matches_existing_model():
    m=tiny();x=torch.randint(3,100,(2,12));y=x.clone();y[:,:5]=-100;w=y.ne(-100).float()
    a=m(x,y);b,_=loss_forward(m,x,y,w);torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

def test_weighted_loss_finite_all_grads():
    m=tiny();x=torch.randint(3,100,(2,12));y=x.clone();w=torch.ones_like(x).float();w[:,6:9]=8
    a,_=loss_forward(m,x,y,w);a.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_checkpointing_gradient_equivalence():
    m=tiny();b=copy.deepcopy(m);b.config.checkpoint_blocks=True
    x=torch.randint(3,100,(2,12));y=x.clone();w=torch.ones_like(x).float()
    a,_=loss_forward(m,x,y,w);c,_=loss_forward(b,x,y,w);a.backward();c.backward()
    torch.testing.assert_close(a,c,atol=1e-6,rtol=1e-6)
    for p,q in zip(m.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-6)

def test_cached_equals_full_and_causal():
    m=tiny().eval();x=torch.randint(3,100,(1,12));z=x.clone();z[:,8:]=77
    with torch.no_grad():
        full,_=m(x);changed,_=m(z);_,cache=m(x[:,:8]);tail,_=m(x[:,8:],cache=cache)
    torch.testing.assert_close(full[:,:8],changed[:,:8],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(full[:,8:],tail,atol=1e-5,rtol=1e-5)

def test_new_partitions_do_not_overlap():
    mf=json.loads((ROOT/'data/MANIFEST.json').read_text());assert not any(mf['intersections'].values())
    tr={r['group'] for split in ('elementary','composed') for r in rows(ROOT/'data'/f'{split}.jsonl')}
    for split in ('iid','surface','ood','counterfactual'):
        rr=rows(ROOT/'data'/f'{split}.jsonl');assert tr.isdisjoint(r['group'] for r in rr)
    cf=rows(ROOT/'data/counterfactual.jsonl')
    for a,b in zip(cf[::2],cf[1::2]):assert a['group']==b['group'] and a['answer']!=b['answer'] and a['pair_id']==b['pair_id']

def test_tokens_reversible():
    t=Tokenizer.load(ROOT/'data/tokenizer.json')
    s='Привет, x=-17; y=5.\n日本語';assert t.decode(t.encode(s))==s
