import json
from pathlib import Path
import r57_state_conditional_diversity as m

def test_seed_split_disjoint():
    assert set(m.CAL_SEEDS).isdisjoint(set(m.HELD_SEEDS))

def test_joint_group_distinguishes_reasoning_state():
    x1=(1.0,3,2,True,frozenset(),0.0,0.0,0.0,7)
    x2=(0.9,4,5,False,frozenset(),0.0,0.0,0.0,7)
    assert m._group_key(x1)==(7,2)
    assert m._group_key(x2)==(7,5)
    assert m._group_key(x1)!=m._group_key(x2)

def test_calibration_selected_cap_is_frozen():
    d=json.loads(Path('r57_calibration.json').read_text())
    assert d['selected_cap']==1
    assert d['seeds']==[5721,5722]

def test_all_held_semantics_clean():
    for cn in ['id8','ood32','ood64','ood128']:
        d=json.loads(Path(f'held_{cn}.json').read_text())
        assert d['semantic_mismatch']==0

def test_completed_128_has_no_generation_loss_but_pruning_loss():
    d=json.loads(Path('held_ood128.json').read_text())['results']['r57_joint_state_diverse']
    assert d['n_episodes']==6
    assert d['true_generation_losses']['mean']==0.0
    assert d['true_pruning_losses']['mean']>0.0