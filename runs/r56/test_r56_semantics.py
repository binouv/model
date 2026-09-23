import json
from pathlib import Path
import r56_address_diversity_allocation as m

def test_seed_split_disjoint():
    assert set(m.CAL_SEEDS).isdisjoint(set(m.HELD_SEEDS))

def test_exactly_one_ablation():
    assert m.CAP_GRID == (1,2,4,8)

def test_selected_cap():
    z=json.loads(Path('r56_calibration.json').read_text())
    assert z['selected_cap']==8

def test_no_semantic_mismatch():
    z=json.loads(Path('r56_metrics.json').read_text())
    assert all(v['semantic_mismatch']==0 for v in z['held_out_completed_only'].values())

def test_diverse_prune_returns_global_score_order():
    c=[(1.0,1,0,True,frozenset(),0,0,0,7),(3.0,2,0,False,frozenset(),0,0,0,7),(2.0,3,0,False,frozenset(),0,0,0,8)]
    k,_=m.diverse_prune(c,2,1)
    assert [x[0] for x in k]==sorted([x[0] for x in k],reverse=True)