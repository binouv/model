from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import r35_base as r
import r52_base as r52
import r53_syndrome_multihyp_decoder as r53
import r54_adaptive_ecc_branching as r54

NOISE=0.30
WIDTH=32
EXPAND=8
TOPK=4
BETA=0.10
TRIGGER_THRESHOLD=1.694594383239746  # frozen from R54 calibration
GRACE=2  # number of pruning decisions for which a new ECC-only branch gets protected status
CAL_SEEDS=(5521,5522)
HELD_SEEDS=(5551,5552,5553)
QUOTA_GRID=(2,4,8)
HELD_COUNTS={'id8':6,'ood32':4,'ood64':4,'ood128':2}


def load_assets(root: Path):
    return r53.load_assets(root)


def build_episode(cn:str, seed:int, perms, cb):
    return r53.build_episode(cn,seed,perms,cb)


def prune_with_reservation(cand, width:int, quota:int):
    """Keep a bounded quota of live protected descendants, then fill remaining slots globally.

    Candidate tuple begins with (score, ..., ttl, origin_alt). Only ttl>0 is protected.
    Selection is fully score-based within each stratum; no oracle labels are used.
    """
    if not cand:
        return [],0
    cand_sorted=sorted(cand,key=lambda x:x[0],reverse=True)
    prot=[x for x in cand_sorted if int(x[-2])>0]
    keep=[]; seen=set()
    q=min(int(quota),width,len(prot))
    for x in prot[:q]:
        # Unique concrete trajectory identity at this step: edge, reasoning state, used-key set, ttl.
        key=(int(x[1]),int(x[2]),x[4])
        if key not in seen:
            keep.append(x); seen.add(key)
    for x in cand_sorted:
        if len(keep)>=width: break
        key=(int(x[1]),int(x[2]),x[4])
        if key in seen: continue
        keep.append(x); seen.add(key)
    keep=sorted(keep[:width],key=lambda x:x[0],reverse=True)
    return keep,min(q,len(keep))


def reserved_branch_beam(ep, perms, base_model, alpha, cb, quota:int,
                         threshold:float=TRIGGER_THRESHOLD, grace:int=GRACE,
                         k_decode:int=TOPK, beta:float=BETA):
    """R55: R54 candidate support + bounded survival reservation for ECC-only alternatives.

    The baseline raw support is always generated. Under low decoder margin, top-k ECC support is added.
    Only ECC-induced edges absent from the parent's raw top-EXPAND support start a protected lineage.
    A protected lineage carries a short TTL; at pruning, up to `quota` of WIDTH slots are reserved for
    the highest-scoring live protected descendants. Remaining slots use the ordinary global score.
    """
    sids,latest,fut_tab,fm_tab=r.prep(ep)
    dtop,dlp=r53.precompute_decoder(ep,cb,r53.MAX_K)
    eorders=r53.precompute_edge_orders(ep,cb,EXPAND)
    # score, qsrc(-1=query else edge), state, prefix_ok, used, clp, cm, cf, ttl, origin_alt
    hyps=[(0.0,-1,int(ep['start']),True,frozenset(),0.0,0.0,0.0,0,False)]
    surv=[]; evals=0; trigger_hyp=0; hyp_evals=0; trigger_steps=0
    reserved_used=[]; protected_frac=[]; true_reserved=[]
    true_top1=[]; true_topk=[]; true_trigger=[]
    for j,rel0 in enumerate(ep['prog']):
        rel=int(rel0); cand=[]; step_trigger=False
        true_src=0 if j==0 else int(ep['idx'][j-1])+1
        tid=int(ep['true_code_ids'][j])
        true_top1.append(float(tid==int(dtop[true_src,0])))
        true_topk.append(float(tid in set(map(int,dtop[true_src,:k_decode]))))
        dm_true=float(dlp[true_src,0]-dlp[true_src,1])
        true_trigger.append(float(dm_true <= threshold))
        for score,qsrc,st,pok,used,clp,cm,cf,ttl,parent_alt in hyps:
            hyp_evals+=1
            src=0 if qsrc<0 else qsrc+1
            q=ep['q'] if qsrc<0 else ep['obj'][qsrc]
            raw_logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']
            p,raw_order,margin,ent=r.softstats(raw_logits)
            dmargin=float(dlp[src,0]-dlp[src,1])
            uncertain=bool(dmargin <= threshold)
            if uncertain:
                trigger_hyp+=1; step_trigger=True
            raw_set=set(int(ix) for ix in raw_order[:EXPAND])
            edge_best={}
            # Exact R52/R35 baseline support.
            for ix0 in raw_order[:EXPAND]:
                ix=int(ix0); evals+=1
                f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                ns=score+r.verifier_score(base_model,f)+alpha*f[3]
                # Existing protected lineage decays but remains eligible for reservation.
                child_ttl=max(0,int(ttl)-1)
                edge_best[ix]=(ns,f,child_ttl,bool(parent_alt and child_ttl>0),False)
            # Low-margin pointer: add discrete ECC alternatives.
            if uncertain:
                for kk in range(k_decode):
                    cid=int(dtop[src,kk]); prior=beta*float(dlp[src,kk])
                    for ix0 in eorders[rel][cid]:
                        ix=int(ix0); evals+=1
                        f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                        ns=score+r.verifier_score(base_model,f)+alpha*f[3]+prior
                        is_new_alt=ix not in raw_set
                        if is_new_alt:
                            child_ttl=int(grace)
                            lineage=True
                        else:
                            child_ttl=max(0,int(ttl)-1)
                            lineage=bool(parent_alt and child_ttl>0)
                        old=edge_best.get(ix)
                        # If same edge has two derivations, keep best score; on exact tie prefer protected provenance.
                        if old is None or ns>old[0]+1e-12 or (abs(ns-old[0])<=1e-12 and child_ttl>old[2]):
                            edge_best[ix]=(ns,f,child_ttl,lineage,is_new_alt)
            for ix,(ns,f,child_ttl,lineage,is_new_alt) in edge_best.items():
                key=(int(sids[ix]),int(ep['rel'][ix]))
                cand.append((ns,ix,int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j]),
                             used|{key},clp+f[3],cm+margin,cf+f[8],child_ttl,lineage))
        if step_trigger: trigger_steps += 1
        selected,nres=prune_with_reservation(cand,WIDTH,quota)
        hyps=[tuple(x) for x in selected]
        reserved_used.append(float(nres))
        protected_frac.append(float(np.mean([int(h[-2])>0 for h in hyps])) if hyps else 0.0)
        surv.append(any(h[3] for h in hyps))
        true_h=[h for h in hyps if h[3]]
        true_reserved.append(float(any(int(h[-2])>0 for h in true_h)))
    best=hyps[0]
    return {
        'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),
        'selected_true_path':float(best[3]),'candidate_evals':int(evals),
        'trigger_hyp_fraction':float(trigger_hyp/max(1,hyp_evals)),
        'trigger_step_fraction':float(trigger_steps/max(1,len(ep['prog']))),
        'true_source_trigger_rate':float(np.mean(true_trigger)),
        'decoder_true_top1':float(np.mean(true_top1)),'decoder_true_topk':float(np.mean(true_topk)),
        'decoder_final_true_topk':float(true_topk[-1]),
        'reserved_slots_used_mean':float(np.mean(reserved_used)),
        'protected_beam_fraction':float(np.mean(protected_frac)),
        'true_prefix_reserved_fraction':float(np.mean(true_reserved)),
    }


def ablation_r54(ep,perms,base_model,alpha,cb):
    return r54.adaptive_branch_beam(ep,perms,base_model,alpha,cb,TRIGGER_THRESHOLD,TOPK,BETA)


def baseline_r52(ep,perms,base_model,alpha):
    return r53.baseline_r52(ep,perms,base_model,alpha)


def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals',
          'trigger_hyp_fraction','trigger_step_fraction','true_source_trigger_rate','decoder_true_top1',
          'decoder_true_topk','decoder_final_true_topk','reserved_slots_used_mean','protected_beam_fraction',
          'true_prefix_reserved_fraction']
    out={'n_episodes':len(rows)}
    for key in keys:
        vals=[float(x[key]) for x in rows if key in x]
        if vals:
            a=np.asarray(vals,float); out[key]={'mean':float(a.mean()),'std':float(a.std()),'values':a.tolist()}
    out['seeds']=[]
    for sd in sorted(set(int(x['seed']) for x in rows)):
        rr=[x for x in rows if int(x['seed'])==sd]; z={'seed':sd,'n':len(rr)}
        for key in keys:
            vals=[float(x[key]) for x in rr if key in x]
            if vals:
                a=np.asarray(vals,float); z[key]={'mean':float(a.mean()),'std':float(a.std())}
        out['seeds'].append(z)
    return out


def evaluate_condition(cn,seeds,n,assets,cb,quota, include_baseline=True):
    base,alpha,perms,_=assets
    names=['r55_reserved','ablation_r54_global_prune']
    if include_baseline: names=['r52_ecc_r35']+names
    rows={m:[] for m in names}; semantic_mismatch=0
    for sd in seeds:
        for i in range(n):
            ep=build_episode(cn,sd*100000+i,perms,cb)
            semantic_mismatch += int(not r52.semantic_validate(ep,perms))
            if include_baseline:
                z=baseline_r52(ep,perms,base,alpha); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['r52_ecc_r35'].append(z)
            z=reserved_branch_beam(ep,perms,base,alpha,cb,quota); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['r55_reserved'].append(z)
            z=ablation_r54(ep,perms,base,alpha,cb); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['ablation_r54_global_prune'].append(z)
    return {m:summarize(v) for m,v in rows.items()},semantic_mismatch


def calibrate_quota(assets,cb,seeds=CAL_SEEDS):
    """Choose only the reservation quota on separate seeds. R54 trigger/top-k/beta stay frozen."""
    base,alpha,perms,_=assets
    # Keep calibration compact but include short and long horizon.
    episodes=[]
    for cn in ['id8','ood64','ood128']:
        for sd in seeds:
            ep=build_episode(cn,sd*100000,perms,cb)
            episodes.append((cn,sd,ep))
    trials=[]
    for quota in QUOTA_GRID:
        by={cn:[] for cn in ['id8','ood64','ood128']}
        for cn,sd,ep in episodes:
            z=reserved_branch_beam(ep,perms,base,alpha,cb,quota)
            z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; by[cn].append(z)
        sm={cn:summarize(v) for cn,v in by.items()}
        # Long horizon dominates; 8-hop is explicitly protected. Path survival breaks sparse answer ties.
        obj=(2.0*sm['ood128']['answer']['mean'] + 1.0*sm['ood64']['answer']['mean'] +
             0.5*sm['ood128']['final_true_path_in_beam']['mean'] + 0.25*sm['ood64']['final_true_path_in_beam']['mean'] +
             1.0*sm['id8']['answer']['mean'])
        trials.append({'quota':int(quota),'objective':float(obj),'summary':sm})
    # maximize objective; ties prefer smaller reservation footprint
    trials.sort(key=lambda x:(x['objective'],-x['quota']),reverse=True)
    return {
        'seeds':list(seeds),'conditions':['id8','ood64','ood128'],'n_per_condition_seed':1,
        'quota_grid':list(QUOTA_GRID),'fixed_grace':GRACE,'frozen_trigger_threshold':TRIGGER_THRESHOLD,
        'selection_rule':'maximize 2*answer128 + answer64 + 0.5*finalPath128 + 0.25*finalPath64 + answer8; ties prefer smaller quota',
        'selected_quota':int(trials[0]['quota']),'trials':trials
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--mode',choices=['calibrate','held','pilot'],default='pilot')
    ap.add_argument('--condition',default='ood64')
    ap.add_argument('--n',type=int,default=1)
    ap.add_argument('--quota',type=int,default=None)
    ap.add_argument('--seed',type=int,default=None)
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.root); assets=load_assets(root); cb=assets[3]
    if args.mode=='calibrate':
        z=calibrate_quota(assets,cb); txt=json.dumps(z,indent=2)
    else:
        quota=args.quota
        if quota is None: quota=int(calibrate_quota(assets,cb)['selected_quota'])
        seeds=[args.seed] if args.seed is not None else ([5551] if args.mode=='pilot' else list(HELD_SEEDS))
        z,sem=evaluate_condition(args.condition,seeds,args.n,assets,cb,quota,include_baseline=True)
        txt=json.dumps({'condition':args.condition,'quota':quota,'results':z,'semantic_mismatch':sem},indent=2)
    if args.out: Path(args.out).write_text(txt,encoding='utf-8')
    print(txt)

if __name__=='__main__': main()