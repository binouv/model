import sys,copy,json
from pathlib import Path
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from model import Model,Config,parallel_delta,report
from build_data import canonical
from tokenizer import Tokenizer

def small(variant):
    torch.manual_seed(31)
    return Model(Config(variant=variant,vocab=300,dim=32,heads=2,hidden=64,key_dim=8)).eval()
@pytest.mark.parametrize('variant',['full','shared','hybrid'])
def test_causal_and_cached(variant):
    m=small(variant);x=torch.randint(3,300,(2,14));xx=x.clone();xx[:,7:]=torch.randint(3,300,(2,7))
    with torch.no_grad():
        a,_=m(x);b,_=m(xx);assert torch.allclose(a[:,:7],b[:,:7],atol=3e-6)
        _,cache=m(x[:,:5]);parts=[]
        for t in range(5,14):
            o,cache=m(x[:,t:t+1],cache=cache);parts.append(o)
        assert torch.allclose(a[:,5:],torch.cat(parts,1),atol=3e-5)
@pytest.mark.parametrize('variant',['full','shared','hybrid'])
def test_every_trainable_has_finite_gradient_and_reload(variant):
    m=small(variant).train();x=torch.randint(3,300,(2,9));y=x.roll(-1,1);y[:,:4]=-100
    loss=m(x,y);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    n=small(variant);n.load_state_dict(m.state_dict());m.eval()
    with torch.no_grad():assert torch.equal(m(x)[0],n(x)[0])
@pytest.mark.parametrize('variant',['full','shared','hybrid'])
def test_sparse_loss_matches_full_vocab_dense(variant):
    m=small(variant);x=torch.randint(3,300,(2,9));y=x.roll(-1,1);y[:,:4]=-100
    a=m(x,y);lg,_=m(x)
    b=torch.nn.functional.cross_entropy(lg.flatten(0,1),y.flatten(),ignore_index=-100)
    assert torch.allclose(a,b,atol=1e-6)
def test_delta_parallel_matches_sequential_with_nonzero_state_and_gradients():
    torch.manual_seed(11);B,H,T,K,V=2,2,11,4,6
    k=torch.nn.functional.normalize(torch.randn(B,H,T,K),dim=-1).requires_grad_();q=torch.randn_like(k).requires_grad_()
    v=torch.randn(B,H,T,V,requires_grad=True);g=(-torch.rand(B,H,T)).requires_grad_();beta=torch.rand(B,H,T,requires_grad=True);s=torch.randn(B,H,K,V,requires_grad=True)
    y,ss=parallel_delta(q,k,v,g,beta,s);outs=[];state=s
    for t in range(T):
        yy,state=parallel_delta(q[:,:,t:t+1],k[:,:,t:t+1],v[:,:,t:t+1],g[:,:,t:t+1],beta[:,:,t:t+1],state)
        outs.append(yy)
    yy=torch.cat(outs,2)
    assert torch.allclose(y,yy,atol=2e-6) and torch.allclose(ss,state,atol=2e-6)
    aa=torch.autograd.grad(y.square().sum()+ss.square().sum(),(q,k,v,g,beta,s),retain_graph=True)
    bb=torch.autograd.grad(yy.square().sum()+state.square().sum(),(q,k,v,g,beta,s))
    assert all(torch.allclose(a,b,atol=2e-5,rtol=3e-5) for a,b in zip(aa,bb))
def test_delta_reset_not_noop():
    m=small('hybrid');x=torch.randint(3,300,(1,13))
    with torch.no_grad():assert (m(x)[0]-m(x,reset_memory=True)[0]).abs().max()>1e-6
def test_data_canonical_no_leaks():
    train=[json.loads(s) for s in (ROOT/'data/train.jsonl').read_text().splitlines()]
    gs={canonical(z['case']) for z in train};val={canonical(json.loads(s)['case']) for s in (ROOT/'data/validation.jsonl').read_text().splitlines()};assert not gs&val
    for p in (ROOT/'data').glob('test_*.jsonl'):
        tt={canonical(json.loads(s)['case']) for s in p.read_text().splitlines()};assert not(gs|val)&tt
    assert canonical({'family':'arithmetic','values':[2,3],'op':'+'})==canonical({'family':'arithmetic','values':[3,2],'op':'*'})
def test_tokenizer_and_output_budget():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');assert tok.vocab_size==8192
    for p in (ROOT/'data').glob('test_*.jsonl'):
        for z in map(json.loads,p.read_text().splitlines()):
            assert tok.decode(tok.encode(z['prompt']))==z['prompt']
            assert len(z['tokens'])-z['prompt_tokens']<=16
