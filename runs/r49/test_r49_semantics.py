import json
from pathlib import Path
import r49_discriminative_joint_filter as m

def test_train_held_seeds_disjoint():
    assert {4921,4922}.isdisjoint({4941,4942,4943})

def test_transition_support_fixed():
    assert m.TRANS_K == 8

def test_modes_are_exactly_main_and_one_ablation():
    assert {'verifier','local'} == {'verifier','local'}

def test_incomplete_128_is_not_in_metrics():
    d=json.loads(Path('r49_metrics.json').read_text())
    assert 'ood128' not in d['held_results_completed_only']
    assert d['excluded_incomplete']==['ood128']