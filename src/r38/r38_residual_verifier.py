from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import r35_base as r
import r37_base as q

CONDS=q.R37_CONDS

def fit_residual(X,y,C=.35):
    sc=StandardScaler().fit(X); Z=sc.transform(X)
    lr=LogisticRegression(max_iter=500,C=C,class_weight='balanced',solver='lbfgs',n_jobs=1).fit(Z,y)
    return {'mean':sc.mean_.copy(),'scale':sc.scale_.copy(),'coef':lr.coef_[0].copy(),'intercept':float(lr.intercept_[0])}

def raw_logit(m,x):
    z=(x-m['mean'])/m['scale']; return float(m['intercept']+z@m['coef'])

def residual_beam(ep,prepared,perms,base_model,alpha,resmodel,beta,gate_thr,width=32,expand=8,gated=True):
    # Protected backbone score is cumulative R35 score. Learned off-policy prefix score is residual only.
    hyps=[(0.0,ep['q'],int(ep['start']),True,frozenset(),0.,0.,0.,q.init_stats())]
    surv=[]; evals=0; gate_count=0; gate_total=0
    for j in range(len(ep['prog'])):
        cand=[]
        for bscore,qq,st,pok,used,clp,cm,cf,pst in hyps:
            ec=q.edge_candidates(ep,j,qq,used,clp,cm,cf,prepared,base_model,alpha,expand)
            for rk,(ix,f,edge_r35,key,margin) in enumerate(ec):
                evals+=1; cst=q.update_stats(pst,edge_r35,f)
                x=q.make_prefix_feature(f,edge_r35,rk,expand,pst,cst,j,len(ep['prog']))
                z=raw_logit(resmodel,x); conf=abs(1.0/(1.0+math.exp(-max(-30,min(30,z))))-.5)*2.0
                gate=(conf>=gate_thr) if gated else True
                gate_count+=int(gate); gate_total+=1
                # Clip prevents the residual from replacing the protected cumulative score.
                corr=beta*max(-4.0,min(4.0,z)) if gate else 0.0
                ns=bscore+edge_r35+corr
                cand.append((ns,ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j]),used|{key},clp+f[3],cm+margin,cf+f[8],cst))
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:width]; surv.append(any(h[3] for h in hyps))
    best=hyps[0]
    return {'state':int(best[2]),'path_survival':float(np.mean(surv)),'final_true_path_in_beam':float(surv[-1]),'selected_true_path':float(best[3]),'candidate_evals':evals,'gate_rate':gate_count/max(1,gate_total)}

def summarize(rows):
    out={'n_episodes':len(rows)}
    for k in ['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals','gate_rate']:
        a=np.asarray([x[k] for x in rows],float); out[k]={'mean':float(a.mean()),'std':float(a.std()),'values':[float(v) for v in a]}
    out['seeds']=[]
    for sd in sorted(set(int(x['seed']) for x in rows)):
        rr=[x for x in rows if int(x['seed'])==sd]; z={'seed':sd,'n':len(rr)}
        for k in ['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals','gate_rate']:
            a=np.asarray([x[k] for x in rr],float); z[k]={'mean':float(a.mean()),'std':float(a.std())}
        out['seeds'].append(z)
    return out

def evaluate(cn,seeds,n,perms,base_model,alpha,resmodel,beta,gate_thr):
    rows={m:[] for m in ['greedy','r35_fixed32','r38_gated_residual','ablation_ungated_residual']}
    for sd in seeds:
        for i in range(n):
            ep=r.epgen(*CONDS[cn],.30,sd*100000+i,perms); pr=r.prep(ep)
            st,addr,full=r.greedy(ep,perms)
            z={'state':int(st),'path_survival':float(addr),'final_true_path_in_beam':float(full),'selected_true_path':float(full),'candidate_evals':0,'gate_rate':0.,'answer':float(st==ep['target']),'seed':sd}; rows['greedy'].append(z)
            z=q.baseline_fixed(ep,pr,perms,base_model,alpha,32,8); z['gate_rate']=0.; z['answer']=float(z['state']==ep['target']); z['seed']=sd; rows['r35_fixed32'].append(z)
            for name,gated in [('r38_gated_residual',True),('ablation_ungated_residual',False)]:
                z=residual_beam(ep,pr,perms,base_model,alpha,resmodel,beta,gate_thr,32,8,gated); z['answer']=float(z['state']==ep['target']); z['seed']=sd; rows[name].append(z)
    return {m:summarize(v) for m,v in rows.items()}

def calibrate(perms,base_model,alpha,resmodel,seeds,n):
    grid=[]
    for beta in [0.10,0.20]:
      for gate in [0.70,0.85]:
        vals={}
        for cn in ['id8','ood64','ood128']:
            rr=[]
            for sd in seeds:
                for i in range(n):
                    ep=r.epgen(*CONDS[cn],.30,sd*100000+i,perms); pr=r.prep(ep)
                    z=residual_beam(ep,pr,perms,base_model,alpha,resmodel,beta,gate,32,8,True); rr.append(float(z['state']==ep['target']))
            vals[cn]=float(np.mean(rr))
        # protect short performance, favor 128 then 64
        obj=2.5*vals['id8']+1.0*vals['ood64']+2.5*vals['ood128']
        grid.append({'beta':beta,'gate_threshold':gate,'objective':obj,'answers':vals})
    best=max(grid,key=lambda x:x['objective']); return grid,float(best['beta']),float(best['gate_threshold'])

def main(outdir):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); t0=time.time()
    base_model,alpha,perms=q.load_base(Path(__file__).parent)
    train_seeds=[3811,3812,3813]; cal_seeds=[3821,3822]; held_seeds=[3831,3832,3833]
    # Freeze the already-completed R37 off-policy verifier; R38 tests scoring architecture only.
    w=np.load(Path(__file__).parent/'r37_offpolicy_model.npz',allow_pickle=True)
    model={'mean':w['mean'],'scale':w['scale'],'coef':w['coef'],'intercept':float(w['intercept'][0])}
    train_meta={'source':'frozen R37 off-policy model','new_training_rows':0}
    y=np.zeros(0,dtype=np.int64)
    grid,beta,gate=calibrate(perms,base_model,alpha,model,cal_seeds,1)
    held_counts={'id8':4,'ood32':4,'ood64':4,'ood128':4}
    held={cn:evaluate(cn,held_seeds,n,perms,base_model,alpha,model,beta,gate) for cn,n in held_counts.items()}
    # Extension only if primary 128 criterion is at least matched.
    extension=None
    if held['ood128']['r38_gated_residual']['answer']['mean'] >= held['ood128']['r35_fixed32']['answer']['mean']:
        extension=evaluate('ood256',held_seeds,3,perms,base_model,alpha,model,beta,gate)
    short_delta=held['id8']['r38_gated_residual']['answer']['mean']-held['id8']['greedy']['answer']['mean']
    d64=held['ood64']['r38_gated_residual']['answer']['mean']-held['ood64']['r35_fixed32']['answer']['mean']
    d128=held['ood128']['r38_gated_residual']['answer']['mean']-held['ood128']['r35_fixed32']['answer']['mean']
    success=(short_delta>=-.02 and d128>=0 and held['ood64']['r38_gated_residual']['answer']['mean']>=held['ood64']['r35_fixed32']['answer']['mean'])
    payload={'experiment':'FlyGraph R38 protected cumulative + off-policy residual verifier','status':'completed','hypothesis':'An off-policy verifier should act as a confidence-gated residual correction on top of the protected cumulative R35 trajectory score, preserving long-horizon evidence aggregation while correcting hard ranking errors.','protocol':{'noise':.30,'train_seeds':train_seeds,'calibration_seeds':cal_seeds,'held_seeds':held_seeds,'train_counts_per_seed':{'source':'frozen R37 model'},'held_counts_per_seed':held_counts,'beam_width':32,'expand':8,'calibration_grid':{'beta':[.10,.20],'gate_threshold':[.70,.85]}},'training':{'rows':0,'positive_fraction':None,'meta':train_meta},'selected':{'beta':beta,'gate_threshold':gate},'calibration':grid,'held_results':held,'extension_256':extension,'success_criterion':{'id8_delta_vs_greedy':short_delta,'ood64_delta_vs_r35_fixed32':d64,'ood128_delta_vs_r35_fixed32':d128,'passed':bool(success)},'verdict':'CONFIRMED' if success else 'REFUTED/MIXED','elapsed_sec':time.time()-t0}
    (out/'r38_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (out/'r38_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['condition','method','answer_mean','answer_std','path_survival','final_true_path','selected_true_path','candidate_evals','gate_rate'])
        for cn,md in held.items():
            for m,z in md.items(): w.writerow([cn,m,z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['candidate_evals']['mean'],z['gate_rate']['mean']])
        if extension:
            for m,z in extension.items(): w.writerow(['ood256',m,z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['candidate_evals']['mean'],z['gate_rate']['mean']])
    np.savez_compressed(out/'r38_residual_model.npz',mean=model['mean'],scale=model['scale'],coef=model['coef'],intercept=np.array([model['intercept']]),beta=np.array([beta]),gate_threshold=np.array([gate]))
    pct=lambda x:f'{100*x:.2f}%'
    L=['# FlyGraph R38 — protected cumulative score + off-policy residual verifier','',f'**Статус:** {payload["verdict"]}  ','**Приоритет:** cognition / memory / reasoning. Audio/video не затрагивались.','', '## Гипотеза','',payload['hypothesis'],'','## Протокол','',f'- train seeds: {train_seeds}; calibration: {cal_seeds}; held-out: {held_seeds}.',f'- noise p=0.30; beam=32; expand=8; selected beta={beta}, gate={gate}.','- Protected backbone: cumulative R35 verifier score.','- Main: confidence-gated off-policy residual.','- Единственная абляция: тот же residual всегда включён, без confidence gate.','', '## Held-out results','', '| Hops | N | Greedy | R35 fixed32 | **R38 gated residual** | Ungated ablation |','|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        md=held[cn]; n=md['r38_gated_residual']['n_episodes']; L.append(f"| {h} | {n} | {pct(md['greedy']['answer']['mean'])} | {pct(md['r35_fixed32']['answer']['mean'])} | **{pct(md['r38_gated_residual']['answer']['mean'])}** | {pct(md['ablation_ungated_residual']['answer']['mean'])} |")
    L += ['','## Memory trajectory metrics','', '| Hops | Method | Path survival | Final true path | Selected exact path | Gate rate |','|---:|---|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        for m,label in [('r35_fixed32','R35 fixed32'),('r38_gated_residual','R38 gated'),('ablation_ungated_residual','Ungated')]:
            z=held[cn][m]; L.append(f"| {h} | {label} | {pct(z['path_survival']['mean'])} | {pct(z['final_true_path_in_beam']['mean'])} | {pct(z['selected_true_path']['mean'])} | {pct(z['gate_rate']['mean'])} |")
    L += ['','## Verdict','',f"**{payload['verdict']}**. 8-hop delta vs greedy = {100*short_delta:+.2f} pp; 64-hop delta vs R35 fixed32 = {100*d64:+.2f} pp; 128-hop delta = {100*d128:+.2f} pp."]
    if extension:
        L += ['','## 256-hop extension','',f"R38 answer: {pct(extension['r38_gated_residual']['answer']['mean'])}; R35 fixed32: {pct(extension['r35_fixed32']['answer']['mean'])}; final exact path R38: {pct(extension['r38_gated_residual']['final_true_path_in_beam']['mean'])}."]
    L += ['','## Следующий шаг','', 'Если R38 подтверждён, следующий цикл может впервые вернуться к controlled sparse workspace scaling (100M/200M/300M) при frozen retrieval/reranking stack и одинаковом active compute. Если нет — масштабирование остаётся отложенным.']
    (out/'RESEARCH_REPORT_R38_RU.md').write_text('\n'.join(L)+'\n',encoding='utf-8')
    cfg={'train_seeds':train_seeds,'calibration_seeds':cal_seeds,'held_seeds':held_seeds,'noise':.30,'beam_width':32,'expand':8,'selected_beta':beta,'selected_gate_threshold':gate,'one_primary_hypothesis':True,'one_confirming_ablation':'ungated residual'}
    (out/'r38_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(json.dumps({'selected':payload['selected'],'held':{cn:{m:round(z['answer']['mean'],4) for m,z in md.items()} for cn,md in held.items()},'criterion':payload['success_criterion'],'extension256':None if extension is None else {m:round(z['answer']['mean'],4) for m,z in extension.items()},'elapsed_sec':payload['elapsed_sec']},indent=2))

if __name__=='__main__':
    import sys; main(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/flygraph_r38/output')