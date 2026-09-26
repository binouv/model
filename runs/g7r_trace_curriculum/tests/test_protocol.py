import hashlib,json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src";sys.path.insert(0,str(SRC))
from common import ex,rows,mat
from train_setup import prepare
from train import lr_for

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def test_registered_data_hashes():
    cfg=json.loads((ROOT/"configs/preregistered.json").read_text())
    for split,h in cfg["data_sha256"].items():assert sha(ROOT/"data"/f"{split}.jsonl")==h

def test_model_size_and_same_initialization_within_seed():
    for seed in (7801,7802,7803):
        _,a,_,_,_,_,_,_,ha=prepare("trace_then_final",seed)
        _,b,_,_,_,_,_,_,hb=prepare("final_only",seed)
        assert sum(p.numel() for p in a.parameters())==469648 and ha==hb

def test_semantic_splits_and_solver():
    seen={}
    for split in ("train","validation","test_iid","test_surface","test_extrapolation","test_counterfactual"):
        for z in rows(split):
            assert str(ex.independent_answer(z["case"]))==z["answer"]
            g=z["canonical"]
            if split!="test_counterfactual":assert g not in seen
            elif g in seen:assert seen[g]=="test_counterfactual"
            seen[g]=split

def test_trace_and_final_prompts_share_prompt_tokens():
    for z in rows("train")[:100]:
        a,b=mat(z,False),mat(z,True)
        assert a["prompt"]==b["prompt"] and a["prompt_tokens"]==b["prompt_tokens"] and a["id"]==b["id"]

def test_sampler_semantic_sequence_matches_arms():
    seq=[]
    for arm in ("trace_then_final","final_only"):
        _,_,pools,keys,rng,_,_,_,_=prepare(arm,7801);ids=[]
        for step in range(1,51):
            tr=arm=="trace_then_final" and step<=1000
            key=keys[rng.randrange(len(keys))];ids.append([z["id"] for z in rng.choices(pools[tr][key],k=16)])
        seq.append(ids)
    assert seq[0]==seq[1]

def test_lr_finite_and_endpoints():
    assert 0<lr_for(1)<.001 and 0<lr_for(2000)<=.000151
    assert all(lr_for(x)>0 for x in (1,100,1000,2000))

def test_held_not_training_and_counts():
    assert len(rows("train"))==8192 and len(rows("validation"))==400
    assert len(rows("test_iid"))==160 and len(rows("test_surface"))==80 and len(rows("test_extrapolation"))==80 and len(rows("test_counterfactual"))==80
    train={z["canonical"] for z in rows("train")}
    for sp in ("validation","test_iid","test_surface","test_extrapolation","test_counterfactual"):assert train.isdisjoint({z["canonical"] for z in rows(sp)})

def test_counterfactual_pairs_are_two_and_gold_changes():
    d={}
    for z in rows("test_counterfactual"):d.setdefault(z["pair_id"],[]).append(z)
    assert len(d)==40 and all(len(v)==2 for v in d.values()) and all(v[0]["answer"]!=v[1]["answer"] for v in d.values())

def test_targets_fit_context_and_trace_is_verified():
    for z in rows("train")[:250]:
        for tr in (False,True):
            q=mat(z,tr);assert len(q["tokens"])<=256
            if tr:
                t,a=ex.trace(z["case"]);assert a==int(z["answer"]) and t

def test_no_gold_field_in_model_prompt_suffix():
    for z in rows("test_iid")[:50]:
        q=mat(z,False);assert "F=<integer>" in q["prompt"]

def test_same_family_balance():
    fam={}
    for z in rows("train"):fam[z["family"]]=fam.get(z["family"],0)+1
    assert set(fam)=={"arithmetic","conditional","code_trace","list_reasoning","memory_update"} and max(fam.values())-min(fam.values())<=1

def test_tiny_forward_backward():
    _,m,pools,keys,rng,opt,_,_,_=prepare("final_only",7801)
    key=keys[rng.randrange(len(keys))];bb=rng.choices(pools[False][key],k=4)
    x,y,p=ex.batch(bb);loss=m(x,y,p);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(g.grad).all() for g in m.parameters() if g.grad is not None)
