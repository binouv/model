import json
from pathlib import Path
import r55_protected_branch_reservation as m

def test_seed_split(): assert set(m.CAL_SEEDS).isdisjoint(m.HELD_SEEDS)
def test_one_ablation(): assert m.TOPK==4 and m.GRACE==2
def test_selected_quota():
 d=json.loads(Path("r55_calibration.json").read_text()); assert d["selected_quota"]==2
def test_prune_keeps_global_best_as_first():
 c=[(10,1,0,False,frozenset(),0,0,0,0,False),(5,2,0,False,frozenset({1}),0,0,0,2,True)]
 k,_=m.prune_with_reservation(c,2,1); assert k[0][0]==10
def test_completed_metrics_have_no_semantic_mismatch():
 d=json.loads(Path("r55_metrics.json").read_text()); assert sum(d["semantic_mismatch_by_condition"].values())==0