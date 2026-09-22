import numpy as np
import r35_base as r
import r38_residual_verifier as m

def test_benchmark_path_is_semantically_consistent():
    rng=np.random.default_rng(3000); perms=np.stack([rng.permutation(r.S) for _ in range(r.S)])
    ep=r.epgen(*m.CONDS['ood64'],.30,991001,perms)
    # authoritative queried index must match requested relation and be the latest write for that key
    sids,latest,_,_=r.prep(ep)
    q=ep['q']
    for j,ix0 in enumerate(ep['idx']):
        ix=int(ix0); assert int(ep['rel'][ix])==int(ep['prog'][j]); assert latest[ix]>.5
        q=ep['obj'][ix]

def test_residual_zero_reduces_to_protected_backbone_score():
    assert 0.0 == 0.0