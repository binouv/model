from __future__ import annotations
from pathlib import Path
import copy, importlib.util, numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r36', HERE / 'r36_adaptive_verifier_beam.py')
r36 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r36)
model, alpha, perms = r36.load_base(HERE)
r = r36.r

# 1) Generated queried edges are latest writes and relation-consistent.
for hops, args in [(8,r36.R36_CONDS['id8']),(64,r36.R36_CONDS['ood64']),(128,r36.R36_CONDS['ood128'])]:
    for k in range(4):
        ep = r.epgen(*args,.30,880000+hops*100+k,perms)
        sids, latest, _, _ = r.prep(ep)
        for j,ix in enumerate(ep['idx']):
            assert latest[int(ix)] == 1.0
            assert int(ep['rel'][int(ix)]) == int(ep['prog'][j])

# 2) Never-trigger adaptive policy is exactly fixed width8 in selected state.
for k in range(5):
    ep = r.epgen(*r36.R36_CONDS['ood64'],.30,881000+k,perms)
    a = r36.fixed_verifier(ep,perms,model,alpha,width=8,expand=8)
    b = r36.adaptive_verifier(ep,perms,model,alpha,threshold=-1e9,signal='verifier')
    assert a['state'] == b['state']

# 3) Forced-trigger at step0 is exactly fixed width32 in selected state.
for k in range(5):
    ep = r.epgen(*r36.R36_CONDS['ood64'],.30,882000+k,perms)
    a = r36.fixed_verifier(ep,perms,model,alpha,width=32,expand=8)
    b = r36.adaptive_verifier(ep,perms,model,alpha,threshold=1e9,signal='verifier')
    assert a['state'] == b['state']

# 4) Oracle idx can change tracking metrics but not inference-selected state/trigger/eval count.
for k in range(4):
    ep = r.epgen(*r36.R36_CONDS['ood32'],.30,883000+k,perms)
    z1 = r36.adaptive_verifier(ep,perms,model,alpha,threshold=.10,signal='verifier')
    ep2 = copy.deepcopy(ep)
    rng = np.random.default_rng(9900+k)
    ep2['idx'] = rng.integers(0,len(ep2['rel']),size=len(ep2['idx']),dtype=np.int64)
    z2 = r36.adaptive_verifier(ep2,perms,model,alpha,threshold=.10,signal='verifier')
    assert z1['state'] == z2['state']
    assert z1['trigger_step'] == z2['trigger_step']
    assert z1['verifier_candidate_evals'] == z2['verifier_candidate_evals']

print('R36 semantic/policy tests: PASS')