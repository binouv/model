from __future__ import annotations
import numpy as np
import r35_base as r
import r52_base as r52
import r58_base as t
import r59_value_distillation as s

CAL_SEEDS=(6041,6042)
HELD_SEEDS=(6051,6052,6053)
EXT_CONDS={'ood256ecc':(256,480,900,.50)}
TRIGGER_QUANTILE=.25
TRIGGER_THRESHOLD=0.0214364938712599

def build_ext(cn,seed,perms,cb):
    cond=EXT_CONDS[cn] if cn in EXT_CONDS else r.CONDS[cn]
    st=r52.make_structure(*cond,seed,perms)
    ep=r52.episode_from_structure(st,'optimized_linear',cb*np.sqrt(r.PHYS),cb*np.sqrt(r.PHYS))
    ep['true_code_ids']=np.asarray(st['code_ids'][st['path'][:-1]],dtype=np.int64)
    return ep

def select_step(ep,j,rec,pr,assets,model,threshold=TRIGGER_THRESHOLD,enable_fallback=True):
    base,alpha,perms,cb,_=assets
    rec.sort(key=lambda z:z[0][0],reverse=True); scores=[float(z[0][0]) for z in rec]
    if j+1>=len(ep['prog']): return [z[0] for z in rec[:t.WIDTH]],0,False,99.0
    pool=rec[:min(t.RERANK_POOL,len(rec))]
    X=np.stack([s.candidate_feature(ep,j,ch,inc,f,pr,rk,scores) for rk,(ch,inc,f) in enumerate(pool)])
    pv=s.predict_student(model,X)
    sr=[(float(ch[0]+t.LOOKAHEAD_WEIGHT*v),ch) for (ch,inc,f),v in zip(pool,pv)]
    sr.sort(key=lambda z:z[0],reverse=True)
    gap=float(sr[t.WIDTH-1][0]-sr[t.WIDTH][0]) if len(sr)>t.WIDTH else 99.0
    if not enable_fallback or gap>threshold: return [z[1] for z in sr[:t.WIDTH]],0,False,gap
    er=[]; evals=0
    for ch,inc,f in pool:
        v,ee=t.rollout_value(ep,j+1,ch,1,perms,base,alpha,pr); evals+=ee
        er.append((float(ch[0]+t.LOOKAHEAD_WEIGHT*v),ch))
    er.sort(key=lambda z:z[0],reverse=True)
    return [z[1] for z in er[:t.WIDTH]],evals,True,gap
