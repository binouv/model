from __future__ import annotations
import csv, json, math, time, hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import r35_base as r

R37_CONDS = {**r.CONDS, 'ood256': (256,576,1024,.50)}
# Prefix feature = current edge features + current R35 edge score + parent/trajectory summaries.
# All inference features are oracle-free; prefix_ok is carried only for training labels/evaluation.
PREFIX_FEATURE_NAMES = (
    [f'edge_{x}' for x in r.FEATURE_NAMES] +
    ['edge_r35_score','edge_rank_norm','parent_r35_mean','parent_r35_min','parent_subject_sim_mean',
     'parent_subject_sim_min','parent_future_compat_mean','parent_future_compat_min','parent_latest_mean',
     'parent_repeat_rate','parent_entropy_mean','parent_margin_mean','parent_local_lp_mean',
     'cand_r35_mean','cand_r35_min','cand_subject_sim_mean','cand_subject_sim_min',
     'cand_future_compat_mean','cand_future_compat_min','cand_latest_mean','cand_repeat_rate',
     'cand_entropy_mean','cand_margin_mean','cand_local_lp_mean','progress']
)


def load_base(asset_dir: Path):
    w=np.load(asset_dir/'r35_verifier_weights.npz',allow_pickle=True)
    model={'cols':w['cols'],'mean':w['mean'],'scale':w['scale'],'coef':w['coef'],'intercept':float(w['intercept'][0])}
    alpha=float(w['local_alpha'][0])
    rng=np.random.default_rng(3000); perms=np.stack([rng.permutation(r.S) for _ in range(r.S)])
    return model,alpha,perms


def fit_logreg(X,y,C=0.35):
    sc=StandardScaler().fit(X)
    Z=sc.transform(X)
    lr=LogisticRegression(max_iter=500,C=C,class_weight='balanced',solver='lbfgs',n_jobs=1).fit(Z,y)
    return {'mean':sc.mean_.copy(),'scale':sc.scale_.copy(),'coef':lr.coef_[0].copy(),'intercept':float(lr.intercept_[0])}


def prefix_logit(model,x):
    z=(x-model['mean'])/model['scale']
    return float(model['intercept'] + z@model['coef'])


def init_stats():
    # count, sum_r35,min_r35,sum_sim,min_sim,sum_future,min_future,sum_latest,sum_repeat,sum_ent,sum_margin,sum_lp
    return (0,0.0,1e9,0.0,1e9,0.0,1e9,0.0,0.0,0.0,0.0,0.0)


def update_stats(st,edge_r35,f):
    n,sr,mr,ss,ms,sf,mf,sl,srep,se,sm,slp=st; n2=n+1
    return (n2,sr+edge_r35,min(mr,edge_r35),ss+float(f[0]),min(ms,float(f[0])),
            sf+float(f[8]),min(mf,float(f[8])),sl+float(f[6]),srep+float(f[7]),
            se+float(f[5]),sm+float(f[4]),slp+float(f[3]))


def stats_summary(st):
    n,sr,mr,ss,ms,sf,mf,sl,srep,se,sm,slp=st
    if n<=0: return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    d=float(n)
    return [sr/d,mr,ss/d,ms,sf/d,mf,sl/d,srep/d,se/d,sm/d,slp/d]


def make_prefix_feature(f,edge_r35,edge_rank,expand,parent_stats,cand_stats,j,hops):
    ps=stats_summary(parent_stats); cs=stats_summary(cand_stats)
    return np.asarray(list(f)+[edge_r35,edge_rank/max(1,expand-1)]+ps+cs+[j/max(1,hops-1)],dtype=np.float64)


def edge_candidates(ep,j,q,used,clp,cm,cf,prepared,base_model,alpha,expand):
    sids,latest,fut_tab,fm_tab=prepared; rel=int(ep['prog'][j])
    logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']; p,order,margin,ent=r.softstats(logits)
    out=[]
    for rk,ix0 in enumerate(order[:expand]):
        ix=int(ix0); f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
        edge_r35=r.verifier_score(base_model,f)+alpha*f[3]
        key=(int(sids[ix]),int(ep['rel'][ix]))
        out.append((ix,f,float(edge_r35),key,float(margin)))
    return out


def collect_teacher_forced(ep,prepared,perms,base_model,alpha,expand=8):
    X=[]; y=[]; q=ep['q']; used=frozenset(); clp=cm=cf=0.0; pst=init_stats()
    for j in range(len(ep['prog'])):
        c=edge_candidates(ep,j,q,used,clp,cm,cf,prepared,base_model,alpha,expand)
        for rk,(ix,f,er,key,margin) in enumerate(c):
            cst=update_stats(pst,er,f); X.append(make_prefix_feature(f,er,rk,expand,pst,cst,j,len(ep['prog'])))
            y.append(int(ix==int(ep['idx'][j])))
        tx=int(ep['idx'][j])
        # advance true prefix; compute true feature even if it falls outside top expand
        sids,latest,fut_tab,fm_tab=prepared; rel=int(ep['prog'][j])
        logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']; p,order,margin,ent=r.softstats(logits)
        tf=r.feat(ep,j,q,tx,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
        ter=r.verifier_score(base_model,tf)+alpha*tf[3]; key=(int(sids[tx]),int(ep['rel'][tx]))
        pst=update_stats(pst,ter,tf); used=used|{key}; clp+=tf[3]; cm+=margin; cf+=tf[8]; q=ep['obj'][tx]
    return X,y


def collect_offpolicy(ep,prepared,perms,base_model,alpha,width=16,expand=6,max_rows_per_step=96):
    # tuple: base_score,q,state,prefix_ok,used,clp,cm,cf,stats
    hyps=[(0.0,ep['q'],int(ep['start']),True,frozenset(),0.0,0.0,0.0,init_stats())]
    X=[]; y=[]
    rng=np.random.default_rng(int(len(ep['prog'])*1009 + len(ep['sub'])))
    for j in range(len(ep['prog'])):
        cand=[]; step_rows=[]
        for parent_rank,h in enumerate(hyps):
            bscore,q,st,pok,used,clp,cm,cf,pst=h
            ec=edge_candidates(ep,j,q,used,clp,cm,cf,prepared,base_model,alpha,expand)
            for rk,(ix,f,er,key,margin) in enumerate(ec):
                cst=update_stats(pst,er,f); ok=bool(pok and ix==int(ep['idx'][j]))
                x=make_prefix_feature(f,er,rk,expand,pst,cst,j,len(ep['prog']))
                step_rows.append((x,int(ok)))
                nb=bscore+er
                cand.append((nb,ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),ok,used|{key},clp+f[3],cm+margin,cf+f[8],cst))
        # Keep all positives plus hardest negatives by R35 cumulative score; cap for stable training size.
        pos=[z for z in step_rows if z[1]==1]; neg=[z for z in step_rows if z[1]==0]
        if len(neg)>max_rows_per_step-len(pos):
            # step_rows and cand share generation order, but hardest negative approximation: use first cap after parent beam ordering.
            neg=neg[:max(0,max_rows_per_step-len(pos))]
        for z in pos+neg: X.append(z[0]); y.append(z[1])
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:width]
    return X,y


def make_training(perms,base_model,alpha,seeds,counts,mode):
    X=[]; y=[]; per_cond={}
    for cn,n in counts.items():
        cX=[]; cy=[]
        for sd in seeds:
            for i in range(n):
                ep=r.epgen(*R37_CONDS[cn],.30,sd*100000+i,perms); pr=r.prep(ep)
                if mode=='offpolicy': a,b=collect_offpolicy(ep,pr,perms,base_model,alpha)
                elif mode=='teacher': a,b=collect_teacher_forced(ep,pr,perms,base_model,alpha)
                else: raise ValueError(mode)
                cX.extend(a); cy.extend(b)
        X.extend(cX); y.extend(cy); per_cond[cn]={'rows':len(cy),'positive':int(sum(cy))}
    return np.stack(X),np.asarray(y,dtype=np.int64),per_cond


def baseline_fixed(ep,prepared,perms,base_model,alpha,width=32,expand=8):
    sids,latest,fut_tab,fm_tab=prepared
    hyps=[(0.,ep['q'],int(ep['start']),True,frozenset(),0.,0.,0.)]; surv=[]; evals=0
    for j,rel in enumerate(ep['prog']):
        cand=[]
        for score,q,st,pok,used,clp,cm,cf in hyps:
            logits=10*(ep['sub']@q)+3*(ep['rel']==rel)+2*ep['ts']; p,order,margin,ent=r.softstats(logits)
            for ix0 in order[:expand]:
                evals+=1; ix=int(ix0); f=r.feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                ns=score+r.verifier_score(base_model,f)+alpha*f[3]; key=(int(sids[ix]),int(ep['rel'][ix]))
                cand.append((ns,ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j]),used|{key},clp+f[3],cm+margin,cf+f[8]))
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:width]; surv.append(any(h[3] for h in hyps))
    best=hyps[0]
    return {'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),'selected_true_path':float(best[3]),'candidate_evals':evals,'triggered':False,'trigger_step':None}


def prefix_beam(ep,prepared,perms,base_model,alpha,pmodel,threshold,width0=8,width1=32,expand=8):
    # Rank by direct learned P(prefix-correct), not sum of local scores.
    hyps=[(0.0,ep['q'],int(ep['start']),True,frozenset(),0.,0.,0.,init_stats())]
    surv=[]; width=width0; trig=None; evals=0
    for j in range(len(ep['prog'])):
        cand=[]
        for _,q,st,pok,used,clp,cm,cf,pst in hyps:
            ec=edge_candidates(ep,j,q,used,clp,cm,cf,prepared,base_model,alpha,expand)
            for rk,(ix,f,er,key,margin) in enumerate(ec):
                evals+=1; cst=update_stats(pst,er,f); x=make_prefix_feature(f,er,rk,expand,pst,cst,j,len(ep['prog']))
                sc=prefix_logit(pmodel,x)
                cand.append((sc,ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j]),used|{key},clp+f[3],cm+margin,cf+f[8],cst))
        cand.sort(key=lambda x:x[0],reverse=True)
        gap=float(cand[0][0]-cand[1][0]) if len(cand)>1 else 99.0
        if width==width0 and gap<threshold:
            width=width1; trig=j
        hyps=cand[:width]; surv.append(any(h[3] for h in hyps))
    best=hyps[0]
    return {'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),'selected_true_path':float(best[3]),'candidate_evals':evals,'triggered':trig is not None,'trigger_step':trig}


def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals']
    out={'n_episodes':len(rows)}
    for k in keys:
        a=np.asarray([x[k] for x in rows],float); out[k]={'mean':float(a.mean()),'std':float(a.std()),'values':[float(v) for v in a]}
    tr=np.asarray([x['triggered'] for x in rows],float); out['trigger_rate']=float(tr.mean())
    ss=[x['trigger_step'] for x in rows if x['trigger_step'] is not None]; out['trigger_step_mean_when_triggered']=float(np.mean(ss)) if ss else None
    out['seeds']=[]
    for sd in sorted(set(int(x['seed']) for x in rows)):
        rr=[x for x in rows if int(x['seed'])==sd]; z={'seed':sd,'n':len(rr)}
        for k in keys:
            a=np.asarray([x[k] for x in rr],float); z[k]={'mean':float(a.mean()),'std':float(a.std())}
        out['seeds'].append(z)
    return out


def evaluate_condition(cn,seeds,n,perms,base_model,alpha,offmodel,teachmodel,threshold):
    rows={m:[] for m in ['greedy','r35_fixed32','r37_offpolicy','ablation_teacher_forced']}
    for sd in seeds:
        for i in range(n):
            ep=r.epgen(*R37_CONDS[cn],.30,sd*100000+i,perms); pr=r.prep(ep)
            st,addr,full=r.greedy(ep,perms); z={'state':int(st),'path_survival':float(addr),'final_true_path_in_beam':float(full),'selected_true_path':float(full),'candidate_evals':0,'triggered':False,'trigger_step':None}; z['answer']=float(st==ep['target']); z['seed']=sd; rows['greedy'].append(z)
            z=baseline_fixed(ep,pr,perms,base_model,alpha,32,8); z['answer']=float(z['state']==ep['target']); z['seed']=sd; rows['r35_fixed32'].append(z)
            for name,mdl in [('r37_offpolicy',offmodel),('ablation_teacher_forced',teachmodel)]:
                z=prefix_beam(ep,pr,perms,base_model,alpha,mdl,threshold); z['answer']=float(z['state']==ep['target']); z['seed']=sd; rows[name].append(z)
    return {m:summarize(v) for m,v in rows.items()}


def calibration_score(model,items,perms,base_model,alpha,threshold):
    vals={}
    for cn,eps in items.items():
        rr=[]
        for sd,ep,pr in eps:
            z=prefix_beam(ep,pr,perms,base_model,alpha,model,threshold); z['answer']=float(z['state']==ep['target']); z['seed']=sd; rr.append(z)
        vals[cn]=summarize(rr)
    # Long-horizon objective plus short protection and compute penalty normalized approximately to fixed32.
    obj=(2*vals['id8']['answer']['mean']+vals['ood64']['answer']['mean']+2*vals['ood128']['answer']['mean']+
         0.75*vals['ood128']['final_true_path_in_beam']['mean']-0.0000015*vals['ood128']['candidate_evals']['mean'])
    return float(obj),vals


def calibrate(model,seeds,n,perms,base_model,alpha,grid):
    items={}
    for cn in ('id8','ood64','ood128'):
        a=[]
        for sd in seeds:
            for i in range(n):
                ep=r.epgen(*R37_CONDS[cn],.30,sd*100000+i,perms); a.append((sd,ep,r.prep(ep)))
        items[cn]=a
    rows=[]
    for th in grid:
        obj,vals=calibration_score(model,items,perms,base_model,alpha,th); rows.append({'threshold':float(th),'objective':obj,'metrics':vals})
    best=max(rows,key=lambda x:x['objective']); return rows,float(best['threshold'])


def save_model(path,model):
    np.savez_compressed(path,mean=model['mean'],scale=model['scale'],coef=model['coef'],intercept=np.array([model['intercept']]),feature_names=np.asarray(PREFIX_FEATURE_NAMES,dtype=object))


def make_report(payload):
    H=payload['held_results']; E=payload['extension_256']; v=payload['verdict']
    pct=lambda x:f'{100*x:.2f}%'
    L=['# FlyGraph R37 — off-policy hard-negative prefix verifier','',
       '**Статус:** completed  ','**Приоритет:** cognition — memory/reasoning; audio/video не затрагивались.','',
       '## Точная гипотеза','',payload['hypothesis'],'',
       'R35/R36 verifier был обучен в основном на teacher-forced правильных prefix states. R37 обучает тот же тип oracle-free prefix score на ошибочных prefix states, реально порождённых beam search, и ранжирует candidate prefixes напрямую по вероятности согласованности trajectory.','',
       '## Протокол','',
       f"- Noise `p={payload['protocol']['noise']}`; train seeds {payload['protocol']['train_seeds']}; calibration seeds {payload['protocol']['calibration_seeds']}; held seeds {payload['protocol']['held_seeds']}.",
       f"- Train episode counts/seed: {payload['protocol']['train_counts']}.",
       f"- Held counts/seed: {payload['protocol']['held_counts']}.",
       '- Main: off-policy beam-generated hard negatives; adaptive width 8→32.',
       '- Единственная абляция: идентичные features/model/inference, но verifier обучен только на teacher-forced prefixes.','',
       '## Held-out memory/reasoning','',
       '| Hops | N | Greedy answer | R35 fixed32 | **R37 off-policy** | Teacher-forced ablation | R37 final true path | Ablation final true path |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        n=H[cn]['r37_offpolicy']['n_episodes']; L.append(f"| {h} | {n} | {pct(H[cn]['greedy']['answer']['mean'])} | {pct(H[cn]['r35_fixed32']['answer']['mean'])} | **{pct(H[cn]['r37_offpolicy']['answer']['mean'])}** | {pct(H[cn]['ablation_teacher_forced']['answer']['mean'])} | {pct(H[cn]['r37_offpolicy']['final_true_path_in_beam']['mean'])} | {pct(H[cn]['ablation_teacher_forced']['final_true_path_in_beam']['mean'])} |")
    L += ['', '## Adaptive compute / selection metrics','',
          '| Hops | Method | Path survival | Final true path | Selected exact path | Candidate evals | Trigger rate | Trigger step |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        for m,label in [('r37_offpolicy','R37 off-policy'),('ablation_teacher_forced','Teacher-forced ablation')]:
            z=H[cn][m]; ts=z['trigger_step_mean_when_triggered']; L.append(f"| {h} | {label} | {pct(z['path_survival']['mean'])} | {pct(z['final_true_path_in_beam']['mean'])} | {pct(z['selected_true_path']['mean'])} | {z['candidate_evals']['mean']:.0f} | {pct(z['trigger_rate'])} | {'—' if ts is None else f'{ts:.1f}'} |")
    L += ['', '## 256-hop extension','',
          f"Завершено {E['r37_offpolicy']['n_episodes']} held-out эпизодов.",'',
          '| Method | Answer | Path survival | Final true path | Selected exact path |','|---|---:|---:|---:|---:|']
    for m,label in [('greedy','Greedy'),('r35_fixed32','R35 fixed32'),('r37_offpolicy','R37 off-policy'),('ablation_teacher_forced','Teacher-forced ablation')]:
        z=E[m]; L.append(f"| {label} | {pct(z['answer']['mean'])} | {pct(z['path_survival']['mean'])} | {pct(z['final_true_path_in_beam']['mean'])} | {pct(z['selected_true_path']['mean'])} |")
    L += ['', '## Вердикт','',f"**{v['status']}** — {v['summary']}",'',
          f"128-hop gain vs R35 fixed32: {v['gain128_vs_r35_fixed32_pp']:+.2f} п.п.; off-policy gain vs teacher-forced ablation at 128: {v['gain128_vs_teacher_pp']:+.2f} п.п.; 8-hop regression vs greedy: {v['short_regression_pp']:+.2f} п.п.; 256 final exact path: {pct(E['r37_offpolicy']['final_true_path_in_beam']['mean'])}.",'',
          '## Следующий шаг','',payload['next_step'],'',
          '## Артефакты','', '`r37_metrics.json`, `r37_metrics.csv`, `r37_offpolicy_prefix_verifier.py`, `r37_config.json`, `r37_offpolicy_model.npz`, `r37_teacher_model.npz`, `test_r37_semantics.py`; frozen dependency snapshot: `r35_base.py`, `r35_verifier_weights.npz`.']
    return '\n'.join(L)+'\n'


def main(outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); t0=time.time()
    base,alpha,perms=load_base(outdir)
    cfg={'experiment':'FlyGraph R37 off-policy hard-negative prefix verifier','noise':.30,
         'train_seeds':[3711,3712,3713],'calibration_seeds':[3721,3722],'held_seeds':[3731,3732,3733],
         'train_counts':{'id8':10,'ood32':6,'ood64':5,'ood128':3},
         'held_counts':{'id8':12,'ood32':8,'ood64':8,'ood128':6,'ood256':3},
         'threshold_grid':[0.20,0.50,1.00,2.00], 'beam':{'base':8,'max':32,'expand':8},
         'success_criterion':'Improve held 128-hop answer over R35 fixed32 AND beat teacher-forced ablation at 128; keep 8-hop answer within 2pp of greedy. 256 exact-path survival is an extension target, not required for confirmation.'}
    (outdir/'r37_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    Xo,yo,sto=make_training(perms,base,alpha,cfg['train_seeds'],cfg['train_counts'],'offpolicy')
    Xt,yt,stt=make_training(perms,base,alpha,cfg['train_seeds'],cfg['train_counts'],'teacher')
    om=fit_logreg(Xo,yo); tm=fit_logreg(Xt,yt); save_model(outdir/'r37_offpolicy_model.npz',om); save_model(outdir/'r37_teacher_model.npz',tm)
    cal,th=calibrate(om,cfg['calibration_seeds'],3,perms,base,alpha,cfg['threshold_grid'])
    held={}
    for cn in ('id8','ood32','ood64','ood128'):
        held[cn]=evaluate_condition(cn,cfg['held_seeds'],cfg['held_counts'][cn],perms,base,alpha,om,tm,th)
    ext=evaluate_condition('ood256',cfg['held_seeds'],cfg['held_counts']['ood256'],perms,base,alpha,om,tm,th)
    g128=100*(held['ood128']['r37_offpolicy']['answer']['mean']-held['ood128']['r35_fixed32']['answer']['mean'])
    gt=100*(held['ood128']['r37_offpolicy']['answer']['mean']-held['ood128']['ablation_teacher_forced']['answer']['mean'])
    sr=100*(held['id8']['r37_offpolicy']['answer']['mean']-held['id8']['greedy']['answer']['mean'])
    ok=(g128>0 and gt>0 and sr>=-2)
    status='CONFIRMED' if ok else 'REFUTED / MIXED'
    summary=('beam-generated off-policy hard negatives materially improve long-horizon prefix ranking under the predeclared criterion.' if ok else 'off-policy hard-negative training did not satisfy all predeclared gains over both the R35 scorer and the teacher-forced control.')
    next_step=('R38: keep the off-policy verifier frozen and test bounded backtracking/checkpoint recovery only if 256 exact-path survival remains low; do not scale 100M/200M/300M until 256 trajectory survival is materially above zero.' if ext['r37_offpolicy']['final_true_path_in_beam']['mean']<.20 else 'R38: stress-test the off-policy verifier at higher noise and 256/512 hops with adaptive halt/width before any parameter scaling.')
    payload={'experiment':cfg['experiment'],'status':'completed','hypothesis':'A prefix/trajectory verifier trained on beam-generated off-policy hard negatives will correct the remaining long-horizon ranking drift better than the same verifier trained only on teacher-forced prefixes, improving 128-hop noisy-memory accuracy while preserving 8-hop performance.',
             'protocol':{**cfg,'calibration_n_per_seed':3,'ablation':'same prefix features, LogisticRegression and adaptive inference, but teacher-forced-only training'},
             'training':{'offpolicy':{'shape':list(Xo.shape),'positive_rate':float(yo.mean()),'per_condition':sto},'teacher':{'shape':list(Xt.shape),'positive_rate':float(yt.mean()),'per_condition':stt},'feature_names':PREFIX_FEATURE_NAMES},
             'calibration':cal,'selected_threshold':th,'held_results':held,'extension_256':ext,
             'verdict':{'status':status,'summary':summary,'gain128_vs_r35_fixed32_pp':g128,'gain128_vs_teacher_pp':gt,'short_regression_pp':sr,'criterion_met':bool(ok)},
             'next_step':next_step,'elapsed_sec':time.time()-t0}
    (outdir/'r37_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (outdir/'r37_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['condition','method','n','answer_mean','answer_std','path_survival_mean','final_true_path','selected_true_path','candidate_evals_mean','trigger_rate','trigger_step_mean'])
        for cn,md in list(held.items())+[('ood256',ext)]:
            for m,z in md.items(): w.writerow([cn,m,z['n_episodes'],z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['candidate_evals']['mean'],z['trigger_rate'],z['trigger_step_mean_when_triggered']])
    (outdir/'RESEARCH_REPORT_R37_RU.md').write_text(make_report(payload),encoding='utf-8')
    print(json.dumps({'train_offpolicy':payload['training']['offpolicy'],'train_teacher':payload['training']['teacher'],'threshold':th,'held_answer':{cn:{m:round(z['answer']['mean'],4) for m,z in md.items()} for cn,md in held.items()},'ext256':{m:round(z['answer']['mean'],4) for m,z in ext.items()},'ext256_final':{m:round(z['final_true_path_in_beam']['mean'],4) for m,z in ext.items()},'verdict':payload['verdict'],'elapsed_sec':payload['elapsed_sec']},indent=2))

if __name__=='__main__':
    import sys; main(sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parent))