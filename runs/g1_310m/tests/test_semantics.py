import json,sys,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model import Config,FlyGraphLM,parameter_report
from tokenizer import Tokenizer
from protected_memory import ProtectedMemory

def tiny():
    torch.manual_seed(44)
    return FlyGraphLM(Config(vocab_size=300,d_model=64,n_layers=4,n_heads=4,n_kv_heads=2,d_ff=128,graph_width=16,checkpoint_blocks=False)).eval()
def test_causality():
    m=tiny();x=torch.randint(3,300,(1,16));y=x.clone();y[:,8:]=torch.randint(3,300,(1,8))
    with torch.no_grad():a=m(x);b=m(y)
    assert torch.allclose(a[:,:8],b[:,:8],atol=1e-6)
def test_cached_generation_equivalence():
    m=tiny();x=torch.randint(3,300,(1,16));cache=None;outs=[]
    with torch.no_grad():
        full=m(x)
        for j in range(16):
            o,cache=m(x[:,j:j+1],past=cache,use_cache=True);outs.append(o)
    assert torch.allclose(full,torch.cat(outs,1),atol=2e-5)
def test_graph_causal_and_active():
    m=tiny();x=torch.randint(3,300,(1,12))
    with torch.no_grad():a=m(x);b=m(x,disable_graph_edges=True)
    assert (a-b).abs().max()>1e-6
def test_all_tensors_get_gradients():
    m=tiny().train();x=torch.randint(3,300,(1,16));loss,_=m(x[:,:-1],x[:,1:]);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_parameter_count():
    with torch.device('meta'):m=FlyGraphLM(Config())
    assert parameter_report(m)['unique_parameters']==309993253
    assert all(p.requires_grad for p in m.parameters())
def test_memory_latest_revision_persists():
    m=ProtectedMemory();m.write('a','old');m.write('b','stay');m.write('a','new')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'m.json';m.save(p);n=ProtectedMemory.load(p)
    assert n.read('a').value=='new' and n.read('b').value=='stay' and n.read('none') is None
def test_tokenizer_lossless_and_fixed():
    t=Tokenizer.load(Path(__file__).resolve().parents[1]/'data/tokenizer.json')
    for text in ['Привет, мир! print("ё")\n\t0012','Hello world.','∀x∈ℕ, 日本語 🙂','abc_42 = x[0]  \n']:
        assert t.decode(t.encode(text,bos=True,eos=True))==text
    assert t.vocab_size==8192
def test_no_split_overlap():
    root=Path(__file__).resolve().parents[1]/'data'
    sets={s:{json.loads(l)['id'] for l in (root/f'{s}.jsonl').read_text().splitlines()} for s in ['train','validation','test']}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
