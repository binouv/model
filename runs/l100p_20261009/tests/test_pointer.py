import sys,copy,json,random
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pointer_model import PointerModel,Config,Model,state_hash
from data import read,batch,pack
from tokenizer import Tokenizer
torch.set_num_threads(2)
def tiny(on=True):
    torch.manual_seed(6)
    return PointerModel(Config(vocab_size=300,dim=32,layers=2,heads=4,kv_heads=2,hidden=64,delta_key_dim=4,checkpoint_blocks=False),on,rank=8).eval()
def test_probabilities_sum():
    m=tiny();x=torch.randint(3,300,(2,9))
    with torch.no_grad():lp,_=m(x)
    assert torch.allclose(lp.exp().sum(-1),torch.ones(2,9),atol=2e-6)
def test_future_causality():
    m=tiny();x=torch.randint(3,300,(2,9));xx=x.clone();xx[:,5:]=torch.randint(3,300,(2,4))
    with torch.no_grad():a,_=m(x);b,_=m(xx)
    assert torch.allclose(a[:,:5],b[:,:5],atol=1e-6)
def test_targets_do_not_reach_prompt_keys():
    m=tiny();x=torch.randint(3,300,(1,9));xx=x.clone();xx[:,6:]=99;p=torch.tensor([5])
    with torch.no_grad():a,_=m(x,prefix_lens=p);b,_=m(xx,prefix_lens=p)
    assert torch.equal(a[:,:6],b[:,:6])
def test_cached_continuation():
    m=tiny();x=torch.randint(3,300,(1,10));p=torch.tensor([6])
    with torch.no_grad():
        full,_=m(x,prefix_lens=p)
        _,ca=m(x[:,:6]);later,cb=m(x[:,6:],cache=ca)
    assert torch.allclose(full[:,6:],later,atol=1e-5)
    assert cb['ids'].shape[1]==6
def test_no_copy_equals_base():
    m=tiny(False);base=Model(m.config);sd={k:v for k,v in m.state_dict().items() if not k.startswith('copy_')}
    base.load_state_dict(sd);base.eval();x=torch.randint(3,300,(1,9))
    with torch.no_grad():a,_=m(x);b,_=base(x)
    assert torch.allclose(a,b.log_softmax(-1),atol=2e-6)
def test_copy_never_invents_source_ids():
    m=tiny();h=torch.randn(1,32);keys=torch.randn(1,4,8);ids=torch.tensor([[10,10,11,12]])
    with torch.no_grad():
        m.copy_gate.weight.zero_();m.copy_gate.bias.fill_(30.)
        p,g=m.distribution(h,keys,ids,torch.tensor([[True,True,False,False]]))
    assert abs(float(p[0,10])-1)<1e-6 and float(p[0,11])<1e-6
def test_padding_no_copy_mass():
    m=tiny();x=torch.tensor([[1,0,0]])
    with torch.no_grad():lp,_=m(x)
    assert torch.isfinite(lp).all() and torch.allclose(lp.exp().sum(-1),torch.ones(1,3),atol=1e-6)
def test_added_params():
    with torch.device('meta'):m=PointerModel(Config())
    assert sum(p.numel() for p in m.parameters())==100127401
def test_copy_parameters_gradients():
    m=tiny().train();x=torch.randint(3,300,(2,10));y=x.clone()
    y[:,5:]=x[:,:5];y[:,:5]=-100;loss=m(x,y,prefix_lens=torch.tensor([5,5]));loss.backward()
    for n,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),n
    assert m.copy_key.weight.grad.abs().max()>0
def test_splits_and_semantics():
    import parent_data as old
    blocked={z['group'] for p in (ROOT/'data').glob('reasoning_*.jsonl') for z in read(p)}
    seen={}
    for s in ['validation','iid','surface','extrapolation','counterfactual']:
        for z in read(ROOT/'data'/f'{s}.jsonl'):
            assert z['group'] not in blocked
            assert z['group'] not in seen or seen[z['group']]==s=='counterfactual'
            seen[z['group']]=s
            assert int(z['answer'])==old.independent_trace(z['case'])[1]
def test_cf_changed_answer_same_context():
    rr=read(ROOT/'data/counterfactual.jsonl')
    for a,b in zip(rr[::2],rr[1::2]):
        assert a['pair_id']==b['pair_id']
        assert a['answer']!=b['answer']
        assert a['case']['writes']==b['case']['writes']
def test_masks():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');rr=[pack(z,tok) for z in read(ROOT/'data/iid.jsonl')[:4]]
    x,y,p=batch(rr)
    for i,z in enumerate(rr):
        assert (y[i,:p[i]-1]==-100).all()
        assert y[i,p[i]-1:len(z['tokens'])-1].tolist()==tok.encode('F='+z['answer'],eos=True)
