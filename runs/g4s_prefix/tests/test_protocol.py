import json,tempfile,torch
from pathlib import Path
from model import *
from data import build,solve
def test_params_and_tokenizer():
 t=ByteTokenizer();assert t.decode(t.encode('ё Python 123'))=='ё Python 123'
 for v in ('causal_hybrid','prefix_hybrid'):assert report(Model(Config(variant=v)))['total_parameters']==469648
def test_answer_causality():
 for v in ('causal_hybrid','prefix_hybrid'):
  torch.manual_seed(1);m=Model(Config(variant=v));x=torch.randint(3,VOCAB,(2,12));p=torch.tensor([8,9]);a,_=m(x,prefix_lens=p);xx=x.clone();xx[0,10]=3+(int(xx[0,10])-2)%256;b,_=m(xx,prefix_lens=p);assert torch.allclose(a[0,9],b[0,9],atol=1e-5)
def test_data_semantics():
 with tempfile.TemporaryDirectory() as d:
  build(Path(d));seen={}
  for f in Path(d).glob('*.jsonl'):
   for z in map(json.loads,f.read_text().splitlines()):assert str(solve(z['case']))==z['answer']
