import json, numpy as np
from pathlib import Path
p=Path(__file__).parent
m=json.loads((p/'r61_metrics.json').read_text())
def test_completed(): assert m['status']=='completed'
def test_no_semantic_mismatch(): assert m['semantic_validation_mismatches']==0
def test_seed_disjoint():
 c=m['protocol']; assert set(c['train_seeds']).isdisjoint(c['calibration_seeds']); assert set(c['train_seeds']).isdisjoint(c['held_seeds']); assert set(c['calibration_seeds']).isdisjoint(c['held_seeds'])
def test_one_ablation(): assert 'single_ablation' in m['protocol']
def test_compute_pairing():
 for cn,z in m['held_results'].items(): assert z['r61_top8_gate']['n_episodes']==z['ablation_frozen_top4']['n_episodes']
