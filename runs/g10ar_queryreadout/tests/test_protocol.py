import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"src"),str(ROOT/"base")]
import experiment as e
from model import Model,Config
def test_adapter_parameter_match():
 assert sum(p.numel() for p in e.QueryReadout().parameters())==36865
 assert sum(p.numel() for p in e.TokenMLP().parameters())==36865
def test_flat_initialization_match():
 a=e.QueryReadout();b=e.TokenMLP();e.init_adapter(a,7601);e.init_adapter(b,7601)
 assert torch.equal(torch.cat([p.detach().view(-1) for p in a.parameters()]),torch.cat([p.detach().view(-1) for p in b.parameters()]))
def test_gate_zero_identity():
 x=torch.randn(2,7,96)
 for a in (e.QueryReadout(),e.TokenMLP()):assert torch.equal(a(x)[0],x)
def test_canonical_cross_key_permutation():
 a=[["w",1],["x",5],["w",2],["y",7],["z",9]];b=[["x",5],["y",7],["w",1],["z",9],["w",2]]
 assert e.canonical_history(a)==e.canonical_history(b)
def test_canonical_keeps_per_key_chronology():
 a=[["w",1],["x",5],["w",2],["y",7],["z",9]];b=[["w",2],["x",5],["w",1],["y",7],["z",9]]
 assert e.canonical_history(a)!=e.canonical_history(b)
def test_group_answers_differ():
 r=__import__("random").Random(1)
 for _ in range(50):
  g=e.make_group(r,False)
  if g:assert g["answers"][0]!=g["answers"][1] and set(k for k,_ in g["writes"])==set(e.KEYS)
def test_strict_parser():
 assert e.strict_parse("F=12")==12 and e.strict_parse(" F=-3\n")==-3
 assert e.strict_parse("T=x;F=12") is None and e.strict_parse("answer 12") is None
def test_base_frozen_counts():
 m=e.AdapterLM(Model(Config()),e.ARMS[0],7601);r=e.report_model(m)
 assert (r["base_parameters"],r["adapter_parameters"],r["total_parameters"],r["active_trainable_parameters"])==(469648,36865,506513,36865)
 assert not any(p.requires_grad for p in m.base.parameters())
def test_initial_logits_equal_parent():
 torch.manual_seed(1);base=Model(Config());ids=torch.randint(0,259,(2,20))
 with torch.no_grad():parent,_=base(ids)
 m=e.AdapterLM(base,e.ARMS[0],7601)
 with torch.no_grad():got,_,_=m.forward_logits(ids)
 assert torch.equal(parent,got)
def test_mixed_count():
 z=e.make_mixed(__import__("random").Random(92306),200);assert len(z)==200 and len({x["canonical"] for x in z})==200
def test_preregistered_parent_hashes():
 cfg=json.loads((ROOT/"configs/preregistered.json").read_text());assert cfg["parent"]["native_checkpoint_sha256"]=={str(k):v for k,v in e.PARENT_SHA.items()}
def test_attention_incremental():
 a=e.QueryReadout();y,c=a(torch.randn(2,5,96));z,c2=a(torch.randn(2,1,96),c);assert y.shape==(2,5,96) and z.shape==(2,1,96) and c2[0].shape[2]==6
