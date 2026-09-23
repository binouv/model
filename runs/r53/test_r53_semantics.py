import json
from pathlib import Path
import numpy as np

def test_protocol():
    d=json.loads(Path("r53_config.json").read_text())
    assert d["main"]["top_k"]==4 and abs(d["main"]["decoder_prior_weight_beta"]-.10)<1e-12
    assert d["held_seeds"]==[5351,5352,5353]

def test_one_ablation():
    d=json.loads(Path("r53_config.json").read_text())
    assert "hard nearest-code" in d["ablation"]

def test_completed_counts():
    d=json.loads(Path("r53_metrics.json").read_text())["held_results_completed_only"]
    assert d["id8"]["r53_topk_decoder"]["n_episodes"]==18
    assert d["ood32"]["r53_topk_decoder"]["n_episodes"]==12
    assert d["ood64"]["r53_topk_decoder"]["n_episodes"]==12
    assert d["ood128"]["r53_topk_decoder"]["n_episodes"]==6

def test_semantics_clean():
    d=json.loads(Path("r53_metrics.json").read_text())
    assert sum(d["semantic_validation_mismatches"].values())==0

def test_ecc_artifact():
    z=np.load("r52_ecc_model.npz",allow_pickle=False)
    assert z["generator"].shape==(9,96) and z["codebook"].shape==(512,96)