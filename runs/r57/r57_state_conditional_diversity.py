from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import r35_base as r
import r52_base as r52
import r53_syndrome_multihyp_decoder as r53

NOISE=0.30
WIDTH=32
EXPAND=8
TOPK=4
BETA=0.10
TRIGGER_THRESHOLD=1.694594383239746  # frozen from R54 calibration
CAL_SEEDS=(5721,5722)
HELD_SEEDS=(5751,5752,5753)
CAP_GRID=(1,2,4,8)
HELD_COUNTS={'id8':6,'ood32':4,'ood64':4,'ood128':2}


def load_assets(root: Path):
    return r53.load_assets(root)


def build_episode(cn:str, seed:int, perms, cb):
    return r53.build_episode(cn,seed,perms,cb)


def _group_key(x):
    # candidate tuple ends in decoded protected child address; x[2] is next reasoning state
    return (int(x[-1]), int(x[2]))


def state_diverse_prune(cand, width:int, cap:int):
    """Bound occupancy per actionable latent state = (decoded address, reasoning state).

    Candidates are globally ranked first. Up to `cap` members per joint state are admitted,
    then any remaining beam slots are globally backfilled. Final output is globally re-sorted.
    No oracle information is used.
    """
    if not cand:
        return [], {'unique_candidate_groups':0,'unique_beam_groups':0,'max_group_fraction':0.0,'changed':0.0}
    ranked=sorted(cand,key=lambda x:x[0],reverse=True)
    counts=Counter(); keep=[]; ids=set()
    for idx,x in enumerate(ranked):
        gid=_group_key(x)
        if counts[gid] < int(cap):
            keep.append(x); counts[gid]+=1; ids.add(idx)
            if len(keep)>=width: break
    if len(keep)<width:
        for idx,x in enumerate(ranked):
            if idx in ids: continue
            keep.append(x)
            if len(keep)>=width: break
    keep=sorted(keep[:width],key=lambda x:x[0],reverse=True)
    cg=Counter(_group_key(x) for x in ranked)
    bg=Counter(_group_key(x) for x in keep)
    baseline_ids=[(int(x[1]),int(x[2]),x[4]) for x in ranked[:width]]
    keep_ids=[(int(x[1]),int(x[2]),x[4]) for x in keep]
    return keep, {
        'unique_candidate_groups':float(len(cg)),
        'unique_beam_groups':float(len(bg)),
        'max_group_fraction':float(max(bg.values())/max(1,len(keep))) if bg else 0.0,
        'changed':float(set(keep_ids)!=set(baseline_ids)),
    }


def global_prune(cand,width:int):
    ranked=sorted(cand,key=lambda x:x[0],reverse=True)[:width]
    bg=Counter(_group_key(x) for x in ranked)
    cg=Counter(_group_key(x) for x in cand)
    return ranked, {
        'unique_candidate_groups':float(len(cg)),
        'unique_beam_groups':float(len(bg)),
        'max_group_fraction':float(max(bg.values())/max(1,len(ranked))) if bg else 0.0,
        'changed':0.0,
    }


def run_beam(ep, perms, base_model, alpha, cb, cap:int|None):
    """R54 candidate support/scoring; only pruning differs.

    Main (cap integer): diversity allocation over joint (protected address, reasoning state).
    Ablation (cap None): ordinary global top-32 pruning.
    """
    sids,latest,fut_tab,fm_tab=r.prep(ep)
    dtop,dlp=r53.precompute_decoder(ep,cb,r53.MAX_K)
    eorders=r53.precompute_edge_orders(ep,cb,EXPAND)
    hyps=[(0.0,-1,int(ep['start']),True,frozenset(),0.0,0.0,0.0)]
    surv=[]; evals=0; triggered_hyp=0; hyp_evals=0; trigger_steps=0
    true_top1=[]; true_topk=[]; true_trigger=[]
    uniq_c=[]; uniq_b=[]; maxgf=[]; changed=[]; true_group_occup=[]
    true_extension_opportunities=0; true_generation_losses=0; true_pruning_losses=0
    for j,rel0 in enumerate(ep['prog']):
        rel=int(rel0); cand=[]; step_trigger=False
        true_src=0 if j==0 else int(ep['idx'][j-1])+1
        tid=int(ep['true_code_ids'][j])
        true_top1.append(float(tid==int(dtop[true_src,0])))
        true_topk.append(float(tid in set(map(int,dtop[true_src,:TOPK]))))
        dm_true=float(dlp[true_src,0]-dlp[true_src,1])
        true_trigger.append(float(dm_true <= TRIGGER_THRESHOLD))
        for score,qsrc,st,pok,used,clp,cm,cf in hyps:
            hyp_evals+=1
            src=0 if qsrc<0 else qsrc+1
            q=ep['q'] if qsrc<0 else ep['obj'][qsrc]
            raw_logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']
            p,raw_order,margin,ent=r.softstats(raw_logits)
            dmargin=float(dlp[src,0]-dlp[src,1])
            uncertain=bool(dmargin <= TRIGGER_THRESHOLD)
            if uncertain:
                triggered_hyp+=1; step_trigger=True
            edge_best={}
            for ix0 in raw_order[:EXPAND]:
                ix=int(ix0); evals+=1
                f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                ns=score+r.verifier_score(base_model,f)+alpha*f[3]
                edge_best[ix]=(ns,f)
            if uncertain:
                for kk in range(TOPK):
                    cid=int(dtop[src,kk]); prior=BETA*float(dlp[src,kk])
                    for ix0 in eorders[rel][cid]:
                        ix=int(ix0); evals+=1
                        f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                        ns=score+r.verifier_score(base_model,f)+alpha*f[3]+prior
                        old=edge_best.get(ix)
                        if old is None or ns>old[0]: edge_best[ix]=(ns,f)
            for ix,(ns,f) in edge_best.items():
                key=(int(sids[ix]),int(ep['rel'][ix]))
                next_state=int(perms[int(ep['op'][ix]),st])
                child_addr=int(dtop[ix+1,0])
                cand.append((ns,ix,next_state,pok and ix==int(ep['idx'][j]),
                             used|{key},clp+f[3],cm+margin,cf+f[8],child_addr))
        if step_trigger: trigger_steps += 1
        prev_true=any(h[3] for h in hyps)
        true_cand=any(bool(x[3]) for x in cand)
        if prev_true:
            true_extension_opportunities += 1
            if not true_cand:
                true_generation_losses += 1
        if cap is None:
            selected,diag=global_prune(cand,WIDTH)
        else:
            selected,diag=state_diverse_prune(cand,WIDTH,int(cap))
        hyps=[(sc,ix,st,pok,used,clp,cm,cf) for sc,ix,st,pok,used,clp,cm,cf,gid in selected]
        now_true=any(h[3] for h in hyps)
        if prev_true and true_cand and not now_true:
            true_pruning_losses += 1
        surv.append(now_true)
        uniq_c.append(diag['unique_candidate_groups']); uniq_b.append(diag['unique_beam_groups'])
        maxgf.append(diag['max_group_fraction']); changed.append(diag['changed'])
        true_selected=[x for x in selected if bool(x[3])]
        if true_selected:
            tg=_group_key(true_selected[0])
            true_group_occup.append(sum(_group_key(x)==tg for x in selected)/max(1,len(selected)))
        else:
            true_group_occup.append(0.0)
    best=hyps[0]
    return {
        'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),
        'selected_true_path':float(best[3]),'candidate_evals':int(evals),
        'trigger_hyp_fraction':float(triggered_hyp/max(1,hyp_evals)),
        'trigger_step_fraction':float(trigger_steps/max(1,len(ep['prog']))),
        'true_source_trigger_rate':float(np.mean(true_trigger)),
        'decoder_true_top1':float(np.mean(true_top1)),'decoder_true_topk':float(np.mean(true_topk)),
        'decoder_final_true_topk':float(true_topk[-1]),
        'unique_candidate_joint_groups':float(np.mean(uniq_c)),
        'unique_beam_joint_groups':float(np.mean(uniq_b)),
        'max_joint_group_fraction':float(np.mean(maxgf)),
        'diversity_prune_changed_fraction':float(np.mean(changed)),
        'true_prefix_joint_group_fraction':float(np.mean(true_group_occup)),
        'true_extension_opportunities':int(true_extension_opportunities),
        'true_generation_losses':int(true_generation_losses),
        'true_pruning_losses':int(true_pruning_losses),
    }


def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals',
          'trigger_hyp_fraction','trigger_step_fraction','true_source_trigger_rate','decoder_true_top1',
          'decoder_true_topk','decoder_final_true_topk','unique_candidate_joint_groups',
          'unique_beam_joint_groups','max_joint_group_fraction','diversity_prune_changed_fraction',
          'true_prefix_joint_group_fraction','true_extension_opportunities','true_generation_losses','true_pruning_losses']
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


def evaluate_condition(cn,seeds,n,assets,cb,cap):
    base,alpha,perms,_=assets
    rows={'r57_joint_state_diverse':[],'ablation_r54_global_top32':[]}; semantic_mismatch=0
    for sd in seeds:
        for i in range(n):
            ep=build_episode(cn,sd*100000+i,perms,cb)
            semantic_mismatch += int(not r52.semantic_validate(ep,perms))
            z=run_beam(ep,perms,base,alpha,cb,cap); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['r57_joint_state_diverse'].append(z)
            z=run_beam(ep,perms,base,alpha,cb,None); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['ablation_r54_global_top32'].append(z)
    return {m:summarize(v) for m,v in rows.items()},semantic_mismatch


def calibrate_cap(assets,cb,seeds=CAL_SEEDS):
    base,alpha,perms,_=assets
    episodes=[]
    for cn in ['id8','ood64']:
        for sd in seeds:
            ep=build_episode(cn,sd*100000,perms,cb)
            episodes.append((cn,sd,ep))
    trials=[]
    for cap in CAP_GRID:
        by={'id8':[],'ood64':[]}
        for cn,sd,ep in episodes:
            z=run_beam(ep,perms,base,alpha,cb,int(cap)); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; by[cn].append(z)
        sm={cn:summarize(v) for cn,v in by.items()}
        obj=(2.0*sm['ood64']['answer']['mean'] + 0.75*sm['ood64']['final_true_path_in_beam']['mean'] +
             0.25*sm['ood64']['path_survival']['mean'] + 1.0*sm['id8']['answer']['mean'])
        trials.append({'cap':int(cap),'objective':float(obj),'summary':sm})
    trials.sort(key=lambda x:(x['objective'],-x['cap']),reverse=True)
    return {
        'seeds':list(seeds),'conditions':['id8','ood64'],'n_per_condition_seed':1,
        'cap_grid':list(CAP_GRID),'frozen_trigger_threshold':TRIGGER_THRESHOLD,'topk':TOPK,'beta':BETA,
        'selection_rule':'maximize 2*answer64 + 0.75*finalPath64 + 0.25*pathSurvival64 + answer8; ties prefer smaller per-joint-state cap',
        'selected_cap':int(trials[0]['cap']),'trials':trials
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--mode',choices=['calibrate','held','pilot'],default='pilot')
    ap.add_argument('--condition',default='ood64')
    ap.add_argument('--n',type=int,default=1)
    ap.add_argument('--cap',type=int,default=None)
    ap.add_argument('--seed',type=int,default=None)
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.root); assets=load_assets(root); cb=assets[3]
    if args.mode=='calibrate':
        z=calibrate_cap(assets,cb); txt=json.dumps(z,indent=2)
    else:
        cap=args.cap
        if cap is None: cap=int(calibrate_cap(assets,cb)['selected_cap'])
        seeds=[args.seed] if args.seed is not None else ([5751] if args.mode=='pilot' else list(HELD_SEEDS))
        z,sem=evaluate_condition(args.condition,seeds,args.n,assets,cb,cap)
        txt=json.dumps({'condition':args.condition,'cap':cap,'results':z,'semantic_mismatch':sem},indent=2)
    if args.out: Path(args.out).write_text(txt,encoding='utf-8')
    print(txt)

if __name__=='__main__':
    main()