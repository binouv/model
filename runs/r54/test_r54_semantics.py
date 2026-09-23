import json
from pathlib import Path
import r54_adaptive_ecc_branching as m

def test_seed_splits_disjoint():
    assert set(m.CAL_SEEDS).isdisjoint(set(m.HELD_SEEDS))

def test_exactly_one_ablation():
    assert m.TOPK == 4
    d=json.loads(Path("r54_metrics.json").read_text())
    assert d["one_primary_hypothesis"] is True
    assert d["one_confirming_ablation"] == "always-on R53 top-4 branching"

def test_calibrated_trigger_nontrivial_and_bounded():
    d=json.loads(Path("r54_calibration.json").read_text())
    assert d["selected"]["threshold"] > 0
    assert 0 < d["selected"]["trigger_rate"] <= d["max_trigger_rate"]

def test_semantics_clean_and_128_complete():
    d=json.loads(Path("r54_metrics.json").read_text())
    assert sum(d["semantic_validation_mismatches"].values()) == 0
    assert d["held_results"]["ood128"]["r54_adaptive_branch"]["n_episodes"] == 6

def test_no_workspace_scaling():
    assert m.WIDTH == 32 and m.EXPAND == 8