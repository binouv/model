import hashlib,json,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base')]
from runtime import checked,generate,load_model
from model import Config,Model,delta_rule
from tokenizer import Tokenizer

def tiny():
    torch.manual_seed(53)
    return Model(Config(vocab_size=512,dim=32,layers=2,heads=4,kv_heads=2,hidden=64,delta_key_dim=4,max_context=64,checkpoint_blocks=False)).eval()

def test_causal():
    m=tiny();x=torch.randint(3,512,(1,12));y=x.clone();y[:,6:]=torch.randint(3,512,(1,6))
    with torch.inference_mode():a,_=m(x);b,_=m(y)
    assert torch.allclose(a[:,:6],b[:,:6],atol=2e-6)

def test_cache():
    m=tiny();x=torch.randint(3,512,(1,12))
    with torch.inference_mode():a,_=m(x);_,cache=m(x[:,:7]);b,_=m(x[:,7:],cache=cache)
    assert torch.allclose(a[:,7:],b,atol=2e-5)

def test_delta_stream():
    torch.manual_seed(5);q=torch.randn(1,2,9,4);k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1);v=torch.randn(1,2,9,5);g=-torch.rand(1,2,9);beta=torch.rand(1,2,9)
    a,sa=delta_rule(q,k,v,g,beta);s=None;out=[]
    for i in range(9):
        y,s=delta_rule(q[:,:,i:i+1],k[:,:,i:i+1],v[:,:,i:i+1],g[:,:,i:i+1],beta[:,:,i:i+1],s);out.append(y)
    assert torch.allclose(a,torch.cat(out,2),atol=2e-5) and torch.allclose(sa,s,atol=2e-5)

def test_tokenizer_roundtrip():
    t=Tokenizer.load(ROOT/'data/tokenizer.json')
    for s in ['Привет! Ёж\n 0012','print("x")\n\t','日本語🙂']:assert t.decode(t.encode(s,bos=True,eos=True))==s
    assert t.vocab_size==8192

def test_corruption_rejected(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'ok');h=hashlib.sha256(b'ok').hexdigest();checked(tmp_path,'x',h);p.write_bytes(b'bad')
    with pytest.raises(ValueError):checked(tmp_path,'x',h)

def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):checked(tmp_path,'../x','x')

def test_missing_weight_rejected(tmp_path):
    s=json.loads((ROOT/'ASSETS.json').read_text());(tmp_path/'ASSETS.json').write_text(json.dumps(s))
    with pytest.raises(ValueError):load_model(tmp_path,'curriculum')

def test_context_rejected():
    with pytest.raises(ValueError):generate(tiny(),Tokenizer([]),'x'*63,8)

def test_prompt_type_rejected():
    with pytest.raises(TypeError):generate(tiny(),Tokenizer([]),{'gold':7})

def test_source_exact_ancestor():
    for n,h in json.loads((ROOT/'ASSETS.json').read_text())['base_source'].items():assert hashlib.sha256((ROOT/'base'/n).read_bytes()).hexdigest()==h

def test_model_parameter_count():
    with torch.device('meta'):m=Model(Config())
    assert sum(p.numel() for p in m.parameters())==100028328

def test_runtime_has_no_pickle_loader():
    import inspect,runtime
    assert 'torch.load(' not in inspect.getsource(runtime)
