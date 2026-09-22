from pathlib import Path
import inspect
import numpy as np
import r35_base as r
import r37_offpolicy_prefix_verifier as z

ROOT=Path(__file__).resolve().parent
base,alpha,perms=z.load_base(ROOT)

def load_prefix(name):
    w=np.load(ROOT/name,allow_pickle=True)
    return {'mean':w['mean'],'scale':w['scale'],'coef':w['coef'],'intercept':float(w['intercept'][0])}


def test_generator_latest_key_semantics():
    for hops,cn in [(8,'id8'),(32,'ood32'),(64,'ood64'),(128,'ood128')]:
        for i in range(8):
            ep=r.epgen(*z.R37_CONDS[cn],.30,880000+hops*100+i,perms)
            sids,latest,_,_=r.prep(ep)
            path_sids=[]
            for j,ix0 in enumerate(ep['idx']):
                ix=int(ix0)
                assert int(ep['rel'][ix])==int(ep['prog'][j])
                assert latest[ix] > .5
                path_sids.append(int(sids[ix]))
            assert len(path_sids)==len(set(path_sids))


def test_prefix_score_oracle_independence():
    mdl=load_prefix('r37_offpolicy_model.npz')
    ep=r.epgen(*z.R37_CONDS['ood32'],.30,991122,perms); pr=r.prep(ep)
    a=z.prefix_beam(ep,pr,perms,base,alpha,mdl,.20)
    ep2=dict(ep); ep2['idx']=ep['idx'][::-1].copy()
    b=z.prefix_beam(ep2,pr,perms,base,alpha,mdl,.20)
    # Oracle trace is allowed to change evaluation flags only, never selected inference state/compute path.
    assert a['state']==b['state']
    assert a['candidate_evals']==b['candidate_evals']
    assert a['trigger_step']==b['trigger_step']


def test_feature_shape_and_finite():
    mdl=load_prefix('r37_offpolicy_model.npz')
    assert len(mdl['coef'])==len(z.PREFIX_FEATURE_NAMES)
    ep=r.epgen(*z.R37_CONDS['id8'],.30,123456,perms); pr=r.prep(ep)
    ec=z.edge_candidates(ep,0,ep['q'],frozenset(),0.,0.,0.,pr,base,alpha,8)
    ix,f,er,key,margin=ec[0]; s=z.update_stats(z.init_stats(),er,f)
    x=z.make_prefix_feature(f,er,0,8,z.init_stats(),s,0,len(ep['prog']))
    assert x.shape==(len(z.PREFIX_FEATURE_NAMES),)
    assert np.isfinite(x).all()
    assert np.isfinite(z.prefix_logit(mdl,x))

if __name__=='__main__':
    test_generator_latest_key_semantics(); test_prefix_score_oracle_independence(); test_feature_shape_and_finite(); print('R37 semantic tests: PASS')