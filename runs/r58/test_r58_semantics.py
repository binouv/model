import inspect,json,copy
from pathlib import Path
import r58_shallow_lookahead as m
import r52_base as r52

def test_seed_split_disjoint(): assert set(m.CAL_SEEDS).isdisjoint(set(m.HELD_SEEDS))
def test_calibration_frozen():
 d=json.loads(Path('r58_calibration.json').read_text()); assert d['selected_horizon']==1; assert d['seeds']==[5821,5822]
def test_rollout_has_no_oracle_reads():
 s=inspect.getsource(m.rollout_value); assert "['target']" not in s and "['idx']" not in s
def test_all_held_semantics_clean():
 for cn in ['id8','ood32','ood64','ood128']:
  d=json.loads(Path(f'held_{cn}.json').read_text()); assert d['semantic_mismatch']==0
def test_completed_counts():
 d=json.loads(Path('../output/r58_run_integrity.json').read_text()); assert d['held_completed']=={'id8':18,'ood32':12,'ood64':12,'ood128':6}; assert not d['incomplete_or_timed_out_included']
def test_r58_improves_prune_boundary_on_long_horizon():
 d=json.loads(Path('../output/r58_metrics.json').read_text())['held_results']
 for cn in ['ood64','ood128']:
  assert d[cn]['r58_lookahead']['true_pruning_losses']['mean'] <= d[cn]['ablation_r54_global_top32']['true_pruning_losses']['mean']
