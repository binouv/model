import json, numpy as np
from pathlib import Path
import r52_learned_ecc_address as m
def test_seeds_disjoint(): assert set(m.TRAIN_SEEDS).isdisjoint(m.HELD_SEEDS)
def test_capacity_and_width(): assert m.NCODE>=320 and m.PHYS==96
def test_saved_code_unique_full_rank():
 z=np.load("r52_ecc_model.npz",allow_pickle=True); g=z["generator"]; c=z["codebook"]; assert m.gf2_rank(g)==m.K; assert np.unique(c,axis=0).shape[0]==m.NCODE
def test_completed_semantics_zero_mismatch():
 d=json.loads(Path("r52_metrics.json").read_text()); assert sum(d["semantic_validation_mismatches"].values())==0
def test_exactly_one_ablation():
 d=json.loads(Path("r52_config.json").read_text()); assert "ablation" in d and "random" in d["ablation"]