import copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import Config,FlyGraphLM,parameter_report
from headwise_gate import install_headwise_gates

def tiny():
    torch.manual_seed(7171)
    return FlyGraphLM(Config(vocab_size=300,d_model=64,n_layers=4,n_heads=4,n_kv_heads=2,d_ff=128,graph_width=16,checkpoint_blocks=False)).eval()

def test_initial_function_preservation():
    m=tiny();n=copy.deepcopy(m);install_headwise_gates(n);x=torch.randint(3,300,(1,12))
    with torch.no_grad(): a,b=m(x),n(x)
    assert torch.equal(a,b)

def test_gate_gradients_exist():
    m=install_headwise_gates(tiny()).train();x=torch.randint(3,300,(2,12))
    loss,_=m(x[:,:-1],x[:,1:]);loss.backward()
    for b in m.blocks:
        assert torch.isfinite(b.attn.head_gate.weight.grad).all()
        assert b.attn.head_gate.weight.grad.abs().max()>0

def test_causality_and_cache_after_nonzero_gate():
    m=install_headwise_gates(tiny())
    with torch.no_grad():
        for b in m.blocks: b.attn.head_gate.weight.normal_(0,.03)
        x=torch.randint(3,300,(1,12));xx=x.clone();xx[:,6:]=torch.randint(3,300,(1,6))
        a=m(x);b=m(xx);assert torch.allclose(a[:,:6],b[:,:6],atol=1e-6)
        _,c=m(x[:,:8],use_cache=True);y,_=m(x[:,8:],past=c,use_cache=True)
        assert torch.allclose(a[:,8:],y,atol=2e-5)

def test_gated_state_dict_reload():
    a=install_headwise_gates(tiny());b=install_headwise_gates(tiny())
    with torch.no_grad(): a.blocks[0].attn.head_gate.bias.fill_(.2)
    b.load_state_dict(a.state_dict());x=torch.randint(3,300,(1,10))
    with torch.no_grad(): assert torch.equal(a(x),b(x))

def test_real_added_parameter_count():
    with torch.device('meta'):
        m=FlyGraphLM(Config());n0=parameter_report(m)['unique_parameters']
        install_headwise_gates(m);n1=parameter_report(m)['unique_parameters']
    assert n1-n0==12*(1536*24+24)==442656
    assert n1==310435909

def test_control_is_not_empty():
    m=install_headwise_gates(tiny());x=torch.randint(3,300,(1,10))
    with torch.no_grad():
        for b in m.blocks: b.attn.head_gate.bias.fill_(.5)
        a=m(x)
        for b in m.blocks: b.attn.reset_gate=True
        b=m(x)
    assert (a-b).abs().max()>1e-6
