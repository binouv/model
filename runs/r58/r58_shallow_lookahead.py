from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, time, zipfile
from pathlib import Path
import numpy as np
import r35_base as r
import r37_base as r37
import r52_base as r52
import r53_base as r53

NOISE=0.30
WIDTH=32
EXPAND=8
TOPK=4
BETA=0.10
TRIGGER_THRESHOLD=1.694594383239746  # frozen R54
RERANK_POOL=48
LOOKAHEAD_WEIGHT=0.50
MARGIN_WEIGHT=0.15
DISCOUNT=0.80
HORIZON_GRID=(1,2,4)
CAL_SEEDS=(5821,5822)
HELD_SEEDS=(5851,5852,5853)
HELD_COUNTS={'id8':6,'ood32':4,'ood64':4,'ood128':2}


def load_assets(root:Path):
    model,alpha,perms=r37.load_base(root)
    model_path=root/'r52_ecc_model.npz'
    if model_path.exists():
        z=np.load(model_path,allow_pickle=False); cb=np.asarray(z['codebook'],np.float32); meta={'source':'npz'}
    else:
        G,cb,meta=r52.reproduce_r52_codebook()
        np.savez_compressed(model_path,generator=G,codebook=cb,meta_json=json.dumps(meta))
        meta={'source':'reproduced',**meta}
    cb=r.norm(cb)
    return model,alpha,perms,cb,meta


def build_episode(cn,seed,perms,cb):
    return r53.build_episode(cn,seed,perms,cb)


def prep_episode(ep,cb):
    sids,latest,fut_tab,fm_tab=r.prep(ep)
    dtop,dlp=r53.precompute_decoder(ep,cb,r53.MAX_K)
    eorders=r53.precompute_edge_orders(ep,cb,EXPAND)
    return (sids,latest,fut_tab,fm_tab,dtop,dlp,eorders,{})


def raw_stats(ep,qsrc,rel,cache):
    src=0 if qsrc<0 else qsrc+1
    key=(src,int(rel))
    z=cache.get(key)
    if z is not None: return z
    q=ep['q'] if qsrc<0 else ep['obj'][qsrc]
    logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']
    p,order,margin,ent=r.softstats(logits)
    z=(q,p,order,float(margin),float(ent)); cache[key]=z
    return z


def expand_hyp(ep,j,rel,hyp,perms,base_model,alpha,pr,track_truth=True):
    # hyp: score,qsrc,state,prefix_ok,used,clp,cm,cf
    score,qsrc,st,pok,used,clp,cm,cf=hyp
    sids,latest,fut_tab,fm_tab,dtop,dlp,eorders,cache=pr
    src=0 if qsrc<0 else qsrc+1
    q,p,raw_order,margin,ent=raw_stats(ep,qsrc,rel,cache)
    dmargin=float(dlp[src,0]-dlp[src,1]); uncertain=bool(dmargin<=TRIGGER_THRESHOLD)
    edge_best={}; evals=0
    for ix0 in raw_order[:EXPAND]:
        ix=int(ix0); evals+=1
        f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
        inc=float(r.verifier_score(base_model,f)+alpha*f[3])
        edge_best[ix]=(score+inc,f,inc)
    if uncertain:
        for kk in range(TOPK):
            cid=int(dtop[src,kk]); prior=BETA*float(dlp[src,kk])
            for ix0 in eorders[rel][cid]:
                ix=int(ix0); evals+=1
                f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                inc=float(r.verifier_score(base_model,f)+alpha*f[3]+prior)
                total=score+inc
                old=edge_best.get(ix)
                if old is None or total>old[0]: edge_best[ix]=(total,f,inc)
    out=[]
    true_ix=int(ep['idx'][j]) if track_truth else None
    for ix,(ns,f,inc) in edge_best.items():
        key=(int(sids[ix]),int(ep['rel'][ix]))
        child=(float(ns),ix,int(perms[int(ep['op'][ix]),st]),bool(track_truth and pok and ix==true_ix),
               used|{key},float(clp+f[3]),float(cm+margin),float(cf+f[8]))
        out.append((child,inc))
    return out,evals,uncertain


def rollout_value(ep,start_j,cand_hyp,horizon,perms,base_model,alpha,pr):
    """Oracle-free single-trajectory shallow value rollout.

    Uses only frozen R35/R54 inference quantities and known future relation program.
    `prefix_ok`, target, and true indices are never read for ranking; prefix flag is carried only
    because the common hypothesis tuple includes it.
    """
    if horizon<=0 or start_j>=len(ep['prog']): return 0.0,0
    # erase diagnostic prefix flag so rollout cannot accidentally use it downstream
    sc,qsrc,st,_pok,used,clp,cm,cf=cand_hyp
    h=(float(sc),int(qsrc),int(st),False,used,float(clp),float(cm),float(cf))
    vals=[]; weights=[]; evals=0
    for depth,j in enumerate(range(start_j,min(len(ep['prog']),start_j+horizon))):
        rel=int(ep['prog'][j])
        children,ee,_=expand_hyp(ep,j,rel,h,perms,base_model,alpha,pr,track_truth=False)
        evals+=ee
        if not children: break
        # Future consistency = strong best edge plus separation from runner-up.
        ranked=sorted(children,key=lambda z:z[1],reverse=True)
        best_h,best_inc=ranked[0]
        second_inc=ranked[1][1] if len(ranked)>1 else best_inc
        gap=float(best_inc-second_inc)
        step_value=float(best_inc + MARGIN_WEIGHT*gap)
        w=DISCOUNT**depth; vals.append(step_value*w); weights.append(w)
        h=best_h
        # keep diagnostic flag erased at every simulated step
        h=(h[0],h[1],h[2],False,h[4],h[5],h[6],h[7])
    return (float(sum(vals)/max(1e-9,sum(weights))) if vals else 0.0),evals


def run_beam(ep,perms,base_model,alpha,cb,horizon:int,lookahead:bool):
    pr=prep_episode(ep,cb)
    dtop,dlp=pr[4],pr[5]
    hyps=[(0.0,-1,int(ep['start']),True,frozenset(),0.0,0.0,0.0)]
    surv=[]; evals=0; la_evals=0; trigger_steps=0; triggered_hyp=0; hyp_evals=0
    true_topk=[]; true_pruning_losses=0; true_generation_losses=0; true_extension_opportunities=0
    true_immediate_ranks=[]; true_rerank_ranks=[]; rescue_events=0; harm_events=0; changed_steps=0
    for j,rel0 in enumerate(ep['prog']):
        rel=int(rel0); cand=[]; step_trigger=False
        true_src=0 if j==0 else int(ep['idx'][j-1])+1
        tid=int(ep['true_code_ids'][j]); true_topk.append(float(tid in set(map(int,dtop[true_src,:TOPK]))))
        for h in hyps:
            hyp_evals+=1
            children,ee,unc=expand_hyp(ep,j,rel,h,perms,base_model,alpha,pr)
            evals+=ee
            if unc: triggered_hyp+=1;step_trigger=True
            cand.extend([x[0] for x in children])
        if step_trigger: trigger_steps+=1
        prev_true=any(h[3] for h in hyps); true_cand=any(x[3] for x in cand)
        if prev_true:
            true_extension_opportunities+=1
            if not true_cand: true_generation_losses+=1
        ranked=sorted(cand,key=lambda x:x[0],reverse=True)
        immediate_top=ranked[:WIDTH]
        immediate_ids={(x[1],x[2],x[4]) for x in immediate_top}
        if true_cand:
            ranks=[k+1 for k,x in enumerate(ranked) if x[3]]
            true_immediate_ranks.append(float(min(ranks)))
        if lookahead and j+1<len(ep['prog']):
            pool=ranked[:min(RERANK_POOL,len(ranked))]
            scored=[]
            for x in pool:
                v,lee=rollout_value(ep,j+1,x,int(horizon),perms,base_model,alpha,pr)
                la_evals+=lee
                scored.append((float(x[0]+LOOKAHEAD_WEIGHT*v),x,v))
            scored.sort(key=lambda z:z[0],reverse=True)
            selected=[z[1] for z in scored[:WIDTH]]
            if true_cand:
                rr=[k+1 for k,z in enumerate(scored) if z[1][3]]
                if rr: true_rerank_ranks.append(float(min(rr)))
                imm=any(x[3] for x in immediate_top); new=any(x[3] for x in selected)
                rescue_events+=int((not imm) and new); harm_events+=int(imm and (not new))
            selected_ids={(x[1],x[2],x[4]) for x in selected}
            changed_steps+=int(selected_ids!=immediate_ids)
        else:
            selected=immediate_top
            if true_cand:
                rr=[k+1 for k,x in enumerate(ranked) if x[3]]; true_rerank_ranks.append(float(min(rr)))
        hyps=selected
        now_true=any(h[3] for h in hyps)
        if prev_true and true_cand and not now_true: true_pruning_losses+=1
        surv.append(now_true)
    best=max(hyps,key=lambda x:x[0])
    return {
      'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),
      'selected_true_path':float(best[3]),'candidate_evals':int(evals),'lookahead_edge_evals':int(la_evals),
      'total_edge_evals':int(evals+la_evals),'trigger_hyp_fraction':float(triggered_hyp/max(1,hyp_evals)),
      'trigger_step_fraction':float(trigger_steps/max(1,len(ep['prog']))),'decoder_true_topk':float(np.mean(true_topk)),
      'true_extension_opportunities':int(true_extension_opportunities),'true_generation_losses':int(true_generation_losses),
      'true_pruning_losses':int(true_pruning_losses),'mean_true_immediate_rank':float(np.mean(true_immediate_ranks)) if true_immediate_ranks else math.nan,
      'mean_true_rerank_rank':float(np.mean(true_rerank_ranks)) if true_rerank_ranks else math.nan,
      'lookahead_rescue_events':int(rescue_events),'lookahead_harm_events':int(harm_events),
      'lookahead_changed_step_fraction':float(changed_steps/max(1,len(ep['prog']))),
    }


def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals','lookahead_edge_evals','total_edge_evals',
          'trigger_hyp_fraction','trigger_step_fraction','decoder_true_topk','true_extension_opportunities','true_generation_losses','true_pruning_losses',
          'mean_true_immediate_rank','mean_true_rerank_rank','lookahead_rescue_events','lookahead_harm_events','lookahead_changed_step_fraction']
    out={'n_episodes':len(rows)}
    for k in keys:
        vals=[float(x[k]) for x in rows if k in x and np.isfinite(float(x[k]))]
        if vals:
            a=np.asarray(vals,float);out[k]={'mean':float(a.mean()),'std':float(a.std()),'values':a.tolist()}
    out['seeds']=[]
    for sd in sorted(set(int(x['seed']) for x in rows)):
        rr=[x for x in rows if int(x['seed'])==sd]; z={'seed':sd,'n':len(rr)}
        for k in keys:
            vals=[float(x[k]) for x in rr if k in x and np.isfinite(float(x[k]))]
            if vals:
                a=np.asarray(vals,float);z[k]={'mean':float(a.mean()),'std':float(a.std())}
        out['seeds'].append(z)
    return out


def evaluate_condition(cn,seeds,n,assets,horizon,run_ablation=True):
    base,alpha,perms,cb,_=assets
    rows={'r58_lookahead':[],'ablation_r54_global_top32':[]}; sem=0
    for sd in seeds:
        for i in range(n):
            ep=build_episode(cn,sd*100000+i,perms,cb); sem+=int(not r52.semantic_validate(ep,perms))
            z=run_beam(ep,perms,base,alpha,cb,horizon,True); z['answer']=float(z['state']==int(ep['target']));z['seed']=sd;rows['r58_lookahead'].append(z)
            if run_ablation:
                z=run_beam(ep,perms,base,alpha,cb,0,False);z['answer']=float(z['state']==int(ep['target']));z['seed']=sd;rows['ablation_r54_global_top32'].append(z)
    out={'r58_lookahead':summarize(rows['r58_lookahead'])}
    if run_ablation: out['ablation_r54_global_top32']=summarize(rows['ablation_r54_global_top32'])
    return out,sem


def calibrate(assets):
    trials=[]
    for h in HORIZON_GRID:
        by={}; sem=0
        for cn in ['id8','ood64']:
            z,s=evaluate_condition(cn,CAL_SEEDS,1,assets,h,False); sem+=s; by[cn]=z['r58_lookahead']
        obj=(2.0*by['ood64']['answer']['mean']+0.75*by['ood64']['final_true_path_in_beam']['mean']+
             0.25*by['ood64']['path_survival']['mean']+by['id8']['answer']['mean'])
        trials.append({'horizon':int(h),'objective':float(obj),'summary':by,'semantic_mismatch':sem})
    trials.sort(key=lambda x:(x['objective'],-x['horizon']),reverse=True)
    return {'seeds':list(CAL_SEEDS),'conditions':['id8','ood64'],'n_per_condition_seed':1,'horizon_grid':list(HORIZON_GRID),
            'lookahead_weight':LOOKAHEAD_WEIGHT,'margin_weight':MARGIN_WEIGHT,'discount':DISCOUNT,'rerank_pool':RERANK_POOL,
            'selection_rule':'maximize 2*answer64 + .75*finalPath64 + .25*pathSurvival64 + answer8; ties prefer shorter horizon',
            'selected_horizon':int(trials[0]['horizon']),'trials':trials}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--mode',choices=['calibrate','held'],default='calibrate')
    ap.add_argument('--condition',default='ood64');ap.add_argument('--seed',type=int,default=None);ap.add_argument('--n',type=int,default=None);ap.add_argument('--horizon',type=int,default=None);ap.add_argument('--out',default=None)
    args=ap.parse_args();root=Path(args.root);assets=load_assets(root)
    if args.mode=='calibrate': z=calibrate(assets)
    else:
        calp=root/'r58_calibration.json'
        h=int(args.horizon if args.horizon is not None else json.loads(calp.read_text())['selected_horizon'])
        seeds=[args.seed] if args.seed is not None else list(HELD_SEEDS);n=int(args.n if args.n is not None else HELD_COUNTS[args.condition])
        zz,sem=evaluate_condition(args.condition,seeds,n,assets,h,True);z={'condition':args.condition,'horizon':h,'results':zz,'semantic_mismatch':sem}
    txt=json.dumps(z,indent=2)
    if args.out: Path(args.out).write_text(txt,encoding='utf-8')
    print(txt)

if __name__=='__main__': main()
