from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import r35_base as r
import r37_base as r37
import r52_base as r52
import r53_syndrome_multihyp_decoder as r53

NOISE=0.30
WIDTH=32
EXPAND=8
TOPK=4
BETA=0.10
CAL_SEEDS=(5421,5422)
HELD_SEEDS=(5451,5452,5453)
HELD_COUNTS={'id8':6,'ood32':4,'ood64':4,'ood128':2}


def load_assets(root: Path):
    return r53.load_assets(root)


def build_episode(cn:str, seed:int, perms, cb):
    return r53.build_episode(cn,seed,perms,cb)


def calibrate_margin_trigger(root:Path, assets, cb, seeds=CAL_SEEDS, n_per_cond_seed=2, max_trigger_rate=.35):
    """Calibrate one scalar uncertainty trigger using only separate calibration episodes.

    Target is whether hard ECC top-1 is wrong at the true-path pointer. Threshold maximizes
    F1 for detecting those errors subject to a bounded trigger rate, so calibration does not
    optimize held-out answer accuracy or use held-out seeds.
    """
    _,_,perms,_=assets
    margins=[]; wrong=[]; labels=[]
    per_condition={}
    for cn in ['id8','ood32','ood64','ood128']:
        cm=[]; cy=[]
        for sd in seeds:
            for i in range(n_per_cond_seed):
                ep=build_episode(cn,sd*100000+i,perms,cb)
                top,lp=r53.precompute_decoder(ep,cb,r53.MAX_K)
                for j in range(len(ep['prog'])):
                    src=0 if j==0 else int(ep['idx'][j-1])+1
                    m=float(lp[src,0]-lp[src,1])
                    y=int(int(top[src,0]) != int(ep['true_code_ids'][j]))
                    margins.append(m); wrong.append(y); labels.append(cn); cm.append(m); cy.append(y)
        per_condition[cn]={'steps':len(cy),'top1_error_rate':float(np.mean(cy)) if cy else 0.0,
                           'margin_median':float(np.median(cm)) if cm else None}
    margins=np.asarray(margins,dtype=float); wrong=np.asarray(wrong,dtype=np.int64)
    # Candidate thresholds are the distinct observed margins; this is deterministic.
    candidates=np.unique(margins)
    rows=[]
    for th in candidates:
        pred=margins <= th
        rate=float(pred.mean())
        if rate > max_trigger_rate + 1e-12:
            continue
        tp=int(np.sum(pred & (wrong==1))); fp=int(np.sum(pred & (wrong==0))); fn=int(np.sum((~pred)&(wrong==1)))
        precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
        f1=2*precision*recall/max(1e-12,precision+recall)
        rows.append({'threshold':float(th),'f1':float(f1),'precision':float(precision),'recall':float(recall),
                     'trigger_rate':rate,'tp':tp,'fp':fp,'fn':fn})
    if not rows:
        raise RuntimeError('no calibration threshold satisfies trigger cap')
    # Maximize error-detection F1; ties -> less compute, then lower threshold.
    rows.sort(key=lambda z:(z['f1'],-z['trigger_rate'],-z['threshold']),reverse=True)
    best=rows[0]
    return {
        'seeds':list(seeds),'n_per_condition_seed':int(n_per_cond_seed),'max_trigger_rate':float(max_trigger_rate),
        'total_steps':int(len(wrong)),'overall_top1_error_rate':float(wrong.mean()),
        'per_condition':per_condition,'selection_rule':'max F1 for detecting hard top-1 ECC decoder errors, subject to trigger_rate<=0.35; ties prefer lower trigger rate',
        'selected':best,'top_candidates':rows[:12]
    }


def adaptive_branch_beam(ep, perms, base_model, alpha, cb, threshold:float, k_decode:int=TOPK, beta:float=BETA):
    """R52/R35 baseline support is preserved; ECC alternatives are added only under decoder uncertainty.

    Confident hypothesis-step: exact R35 candidate generation from the original noisy pointer.
    Uncertain hypothesis-step: same baseline candidates PLUS concrete candidates induced by top-k
    ECC codewords. No address/reasoning averaging is performed.
    """
    sids,latest,fut_tab,fm_tab=r.prep(ep)
    dtop,dlp=r53.precompute_decoder(ep,cb,r53.MAX_K)
    eorders=r53.precompute_edge_orders(ep,cb,EXPAND)
    # score, qsrc(-1=query else chosen edge), state, prefix_ok, used, clp, cm, cf
    hyps=[(0.0,-1,int(ep['start']),True,frozenset(),0.0,0.0,0.0)]
    surv=[]; evals=0; triggered_hyp=0; hyp_evals=0; trigger_steps=0
    true_top1=[]; true_topk=[]; true_trigger=[]
    for j,rel0 in enumerate(ep['prog']):
        rel=int(rel0); cand=[]; step_trigger=False
        # Diagnostics on the true-path source only.
        true_src=0 if j==0 else int(ep['idx'][j-1])+1
        tid=int(ep['true_code_ids'][j])
        true_top1.append(float(tid==int(dtop[true_src,0])))
        true_topk.append(float(tid in set(map(int,dtop[true_src,:k_decode]))))
        dm_true=float(dlp[true_src,0]-dlp[true_src,1])
        true_trigger.append(float(dm_true <= threshold))
        for score,qsrc,st,pok,used,clp,cm,cf in hyps:
            hyp_evals+=1
            src=0 if qsrc<0 else qsrc+1
            q=ep['q'] if qsrc<0 else ep['obj'][qsrc]
            raw_logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']
            p,raw_order,margin,ent=r.softstats(raw_logits)
            dmargin=float(dlp[src,0]-dlp[src,1])
            uncertain=bool(dmargin <= threshold)
            if uncertain:
                triggered_hyp+=1; step_trigger=True
            edge_best={}
            # Always preserve exact baseline R35 support and score.
            for ix0 in raw_order[:EXPAND]:
                ix=int(ix0); evals+=1
                f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                ns=score+r.verifier_score(base_model,f)+alpha*f[3]
                edge_best[ix]=(ns,f,'raw')
            # Only uncertain pointers open extra discrete ECC hypotheses.
            if uncertain:
                for kk in range(k_decode):
                    cid=int(dtop[src,kk]); prior=beta*float(dlp[src,kk])
                    for ix0 in eorders[rel][cid]:
                        ix=int(ix0); evals+=1
                        f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                        ns=score+r.verifier_score(base_model,f)+alpha*f[3]+prior
                        old=edge_best.get(ix)
                        if old is None or ns>old[0]: edge_best[ix]=(ns,f,'ecc')
            for ix,(ns,f,src_kind) in edge_best.items():
                key=(int(sids[ix]),int(ep['rel'][ix]))
                cand.append((ns,ix,int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j]),
                             used|{key},clp+f[3],cm+margin,cf+f[8]))
        if step_trigger: trigger_steps += 1
        cand.sort(key=lambda x:x[0],reverse=True)
        hyps=[(sc,ix,st,pok,used,clp,cm,cf) for sc,ix,st,pok,used,clp,cm,cf in cand[:WIDTH]]
        surv.append(any(h[3] for h in hyps))
    best=hyps[0]
    return {
        'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),
        'selected_true_path':float(best[3]),'candidate_evals':int(evals),
        'trigger_hyp_fraction':float(triggered_hyp/max(1,hyp_evals)),
        'trigger_step_fraction':float(trigger_steps/max(1,len(ep['prog']))),
        'true_source_trigger_rate':float(np.mean(true_trigger)),
        'decoder_true_top1':float(np.mean(true_top1)),'decoder_true_topk':float(np.mean(true_topk)),
        'decoder_final_true_topk':float(true_topk[-1]),
    }


def baseline_r52(ep,perms,base_model,alpha):
    return r53.baseline_r52(ep,perms,base_model,alpha)


def always_on_r53(ep,perms,base_model,alpha,cb):
    return r53.multi_hyp_decode_beam(ep,perms,base_model,alpha,cb,TOPK,BETA)


def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals',
          'trigger_hyp_fraction','trigger_step_fraction','true_source_trigger_rate','decoder_true_top1','decoder_true_topk','decoder_final_true_topk']
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


def evaluate_condition(cn,seeds,n,assets,cb,threshold):
    base,alpha,perms,_=assets
    rows={m:[] for m in ['r52_ecc_r35','r54_adaptive_branch','ablation_r53_always_top4']}
    semantic_mismatch=0
    for sd in seeds:
        for i in range(n):
            ep=build_episode(cn,sd*100000+i,perms,cb)
            semantic_mismatch += int(not r52.semantic_validate(ep,perms))
            z=baseline_r52(ep,perms,base,alpha); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['r52_ecc_r35'].append(z)
            z=adaptive_branch_beam(ep,perms,base,alpha,cb,threshold); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['r54_adaptive_branch'].append(z)
            z=always_on_r53(ep,perms,base,alpha,cb); z['answer']=float(z['state']==int(ep['target'])); z['seed']=sd; rows['ablation_r53_always_top4'].append(z)
    return {m:summarize(v) for m,v in rows.items()},semantic_mismatch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--mode',choices=['calibrate','held','pilot'],default='pilot')
    ap.add_argument('--condition',default='ood64')
    ap.add_argument('--n',type=int,default=1)
    ap.add_argument('--threshold',type=float,default=None)
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    root=Path(args.root); assets=load_assets(root); cb=assets[3]
    if args.mode=='calibrate':
        z=calibrate_margin_trigger(root,assets,cb)
        txt=json.dumps(z,indent=2)
    else:
        th=args.threshold
        if th is None:
            cal=calibrate_margin_trigger(root,assets,cb); th=float(cal['selected']['threshold'])
        seeds=[5451] if args.mode=='pilot' else list(HELD_SEEDS)
        z,sem=evaluate_condition(args.condition,seeds,args.n,assets,cb,th)
        txt=json.dumps({'condition':args.condition,'threshold':th,'results':z,'semantic_mismatch':sem},indent=2)
    if args.out: Path(args.out).write_text(txt,encoding='utf-8')
    print(txt)

if __name__=='__main__':
    main()