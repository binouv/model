import importlib.util, json, numpy as np
from pathlib import Path

SRC=Path(__file__).with_name('r35_consistency_verifier.py')
spec=importlib.util.spec_from_file_location('r35',SRC)
r35=importlib.util.module_from_spec(spec); spec.loader.exec_module(r35)

def run():
    rng=np.random.default_rng(3000)
    perms=np.stack([rng.permutation(r35.S) for _ in range(r35.S)])
    ep=r35.epgen(*r35.CONDS['ood32'],.30,991337,perms)
    sids,latest,fut,fm=r35.prep(ep)
    assert len(ep['prog'])==32
    assert np.all(np.isfinite(ep['sub'])) and np.all(np.isfinite(ep['obj']))
    for j,ix in enumerate(ep['idx']):
        assert int(ep['rel'][ix])==int(ep['prog'][j])
        assert latest[ix] == 1.0, 'oracle queried write must be the latest version for its exact key'
    q=ep['q']; used=set(); clp=cm=cf=0.0
    for j,r in enumerate(ep['prog'][:4]):
        logits=10*(ep['sub']@q)+3*(ep['rel']==r)+2*ep['ts']
        p,order,margin,ent=r35.softstats(logits)
        ix=int(order[0])
        f=r35.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut,fm,used,clp,cm,cf)
        assert f.shape==(len(r35.FEATURE_NAMES),)
        assert np.all(np.isfinite(f))
        used.add((int(sids[ix]),int(ep['rel'][ix])))
        clp+=f[3]; cm+=margin; cf+=f[8]; q=ep['obj'][ix]
    metrics=json.loads(Path(__file__).with_name('r35_metrics.json').read_text())
    assert metrics['status']=='completed'
    assert metrics['held_results']['ood64']['beam8_verifier']['answer']['mean'] > metrics['held_results']['ood64']['beam8_local']['answer']['mean']
    print('R35 semantic/tests: PASS')

if __name__=='__main__': run()