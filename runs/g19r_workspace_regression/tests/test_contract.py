from runs.g19r_workspace_regression.src.reproduce_core import GatedDeltaLM, norm_state
import torch

def test_parameter_count():
    m=GatedDeltaLM()
    assert sum(p.numel() for p in m.parameters()) == 470020

def test_state_encoding_endpoints():
    assert abs(norm_state(-128)+1.0) < 1e-9
    assert abs(norm_state(255)-1.0) < 1e-9

def test_all_arms_same_parameters():
    counts=[]
    for _ in ("final_only_control","bag_state_regression","ordered_state_regression"):
        counts.append(sum(p.numel() for p in GatedDeltaLM().parameters()))
    assert len(set(counts))==1
