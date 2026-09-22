from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# FlyGraph R35: consistency verifier/reranker for noisy mutable memory.
# Self-contained continuation of R30/R33/R34 synthetic cognition benchmark.
R=4; S=16; PHYS=96; LOG=32; REP=3
CONDS={'id8':(8,48,80,.25),'ood32':(32,128,240,.35),'ood64':(64,192,320,.40),'ood128':(128,320,560,.45)}
FEATURE_NAMES=[
 'subject_similarity','relation_match','write_recency','local_logprob','retrieval_margin','retrieval_entropy',
 'is_latest_write_for_key','trajectory_repeated_key','next_relation_subject_compat','next_relation_compat_margin',
 'step_fraction','cum_mean_local_logprob','cum_mean_margin','cum_mean_future_compat'
]
LOCAL_ONLY=[0,1,2,3,4,5,10,11,12]

def flip(x,p,rng): return np.where(rng.random(x.shape)<p,-x,x).astype(np.float32)
def norm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-8)
def codes(n,rng): return np.where(rng.random((n,PHYS))<.5,-1.,1.).astype(np.float32)

def epgen(hops,nent,nedges,upd,p,seed,perms):
    rng=np.random.default_rng(seed); C=codes(nent,rng)
    path=rng.choice(nent,hops+1,replace=False).astype(int)
    rels=rng.integers(R,size=hops,dtype=np.int64); ops=rng.integers(S,size=hops,dtype=np.int64)
    pk={(int(path[j]),int(rels[j])) for j in range(hops)}; e=[]
    reserve=hops+int(round(hops*upd))+2
    while len(e)<max(0,nedges-reserve):
        e.append((int(rng.integers(nent)),int(rng.integers(R)),int(rng.integers(nent)),int(rng.integers(S))))
    for j in range(hops):
        s,r,o,op=int(path[j]),int(rels[j]),int(path[j+1]),int(ops[j])
        if rng.random()<upd and len(e)<nedges-1:
            e.append((s,r,int(rng.integers(nent)),int(rng.integers(S))))
        e.append((s,r,o,op))
    while len(e)<nedges:
        s=int(rng.integers(nent)); r=int(rng.integers(R))
        if (s,r) in pk: continue
        e.append((s,r,int(rng.integers(nent)),int(rng.integers(S))))
    if len(e)>nedges: e=e[len(e)-nedges:]
    latest={}; sub=[]; obj=[]; mr=[]; mo=[]; ts=[]
    for i,(s,r,o,op) in enumerate(e):
        latest[(s,r)]=i; sub.append(C[s]); obj.append(flip(C[o],p,rng)); mr.append(r); mo.append(op); ts.append(i/max(1,nedges-1))
    q=flip(C[path[0]],p,rng); st=int(rng.integers(S)); tar=st
    for op in ops: tar=int(perms[int(op),tar])
    return dict(sub=norm(np.stack(sub)),obj=norm(np.stack(obj)),rel=np.array(mr),op=np.array(mo),
                ts=np.array(ts,np.float32),q=norm(q),prog=rels,target=tar,start=st,
                idx=np.array([latest[(int(path[j]),int(rels[j]))] for j in range(hops)],dtype=np.int64))

def lsm(x):
    m=float(np.max(x)); z=np.exp(x-m); return x-m-math.log(float(z.sum())+1e-12)

def softstats(logits):
    m=float(np.max(logits)); z=np.exp(logits-m); p=z/(z.sum()+1e-12); order=np.argsort(logits)[::-1]
    margin=float(logits[order[0]]-logits[order[1]]) if len(order)>1 else 99.0
    ent=float(-(p*np.log(p+1e-12)).sum())
    return p,order,margin,ent

def prep(ep):
    bits=(ep['sub']>0).astype(np.uint8)
    keys=[np.packbits(row).tobytes() for row in bits]; mp={}; ids=[]
    for k in keys:
        if k not in mp: mp[k]=len(mp)
        ids.append(mp[k])
    sids=np.array(ids,dtype=np.int32); latest=np.zeros(len(sids),dtype=np.float32); last={}
    for i in range(len(sids)): last[(int(sids[i]),int(ep['rel'][i]))]=i
    for i in range(len(sids)): latest[i]=float(last[(int(sids[i]),int(ep['rel'][i]))]==i)
    # Precompute object -> latest-subject compatibility for every possible next relation.
    n=len(sids); fut=np.full((R,n),-1.0,dtype=np.float32); fm=np.zeros((R,n),dtype=np.float32)
    for rr in range(R):
        idx=np.where((ep['rel']==rr)&(latest>.5))[0]
        if len(idx)==0: continue
        M=ep['sub'][idx]@ep['obj'].T
        fut[rr]=M.max(axis=0)
        if len(idx)>=2:
            part=np.partition(M,-2,axis=0); fm[rr]=part[-1]-part[-2]
        else: fm[rr]=2.0
    return sids,latest,fut,fm

def feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf):
    r=int(ep['prog'][j]); sim=float(ep['sub'][ix]@q); relmatch=float(ep['rel'][ix]==r); lp=float(np.log(p[ix]+1e-12))
    key=(int(sids[ix]),int(ep['rel'][ix])); repeat=float(key in used)
    if j+1<len(ep['prog']):
        nr=int(ep['prog'][j+1]); fut=float(fut_tab[nr,ix]); fm=float(fm_tab[nr,ix])
    else: fut=0.0; fm=0.0
    return np.array([sim,relmatch,float(ep['ts'][ix]),lp,margin,ent,float(latest[ix]),repeat,fut,fm,
                     j/max(1,len(ep['prog'])-1),clp/max(1,j) if j else 0.0,cm/max(1,j) if j else 0.0,cf/max(1,j) if j else 0.0],dtype=np.float64)

def make_train(perms,seeds,n=48,k=8,noise=.30):
    X=[]; y=[]
    for cn in ('id8','ood32','ood64'):
      args=CONDS[cn]
      for sd in seeds:
       for i in range(n):
        ep=epgen(*args,noise,sd*100000+i,perms); sids,latest,fut_tab,fm_tab=prep(ep); q=ep['q']; used=set(); clp=cm=cf=0.0
        for j,r in enumerate(ep['prog']):
            logits=10*(ep['sub']@q)+3*(ep['rel']==r)+2*ep['ts']; p,order,margin,ent=softstats(logits)
            for ix in order[:k]:
                X.append(feat(ep,j,q,int(ix),p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)); y.append(int(ix==int(ep['idx'][j])))
            tx=int(ep['idx'][j]); tf=feat(ep,j,q,tx,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
            clp+=tf[3]; cm+=margin; cf+=tf[8]; used.add((int(sids[tx]),int(ep['rel'][tx]))); q=ep['obj'][tx]
    return np.stack(X),np.array(y,dtype=np.int64)

def fit_lr(X,y,cols):
    sc=StandardScaler().fit(X[:,cols]); z=sc.transform(X[:,cols])
    lr=LogisticRegression(max_iter=400,C=1.0,class_weight='balanced',solver='lbfgs').fit(z,y)
    return {'cols':np.array(cols,dtype=np.int64),'mean':sc.mean_.copy(),'scale':sc.scale_.copy(),'coef':lr.coef_[0].copy(),'intercept':float(lr.intercept_[0])}

def log_sigmoid(x): return -float(np.logaddexp(0.0,-x))
def verifier_score(model,f):
    z=(f[model['cols']]-model['mean'])/model['scale']; return log_sigmoid(model['intercept']+float(z@model['coef']))

def greedy(ep,perms):
    q=ep['q']; st=int(ep['start']); correct=0
    for j,r in enumerate(ep['prog']):
        logits=10*(ep['sub']@q)+3*(ep['rel']==r)+2*ep['ts']; ix=int(logits.argmax()); correct+=int(ix==int(ep['idx'][j])); st=int(perms[int(ep['op'][ix]),st]); q=ep['obj'][ix]
    return st,correct/len(ep['prog']),float(correct==len(ep['prog']))

def beam_local(ep,perms,width=8,expand=8):
    hyps=[(0.,ep['q'],int(ep['start']),True)]; surv=[]
    for j,r in enumerate(ep['prog']):
        cand=[]
        for lp,q,st,pok in hyps:
            logits=10*(ep['sub']@q)+3*(ep['rel']==r)+2*ep['ts']; lps=lsm(logits); order=np.argsort(lps)[::-1][:expand]
            for ix in order:
                ix=int(ix); cand.append((lp+float(lps[ix]),ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j])))
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:width]; surv.append(any(h[3] for h in hyps))
    m=max(h[0] for h in hyps); mass=np.zeros(S)
    for lp,q,st,pok in hyps: mass[st]+=math.exp(lp-m)
    return int(mass.argmax()),float(np.mean(surv)),float(surv[-1])

def adaptive(ep,perms,width=8,expand_low=4,margin_thr=.5):
    hyps=[(0.,ep['q'],int(ep['start']),True)]; surv=[]
    for j,r in enumerate(ep['prog']):
        cand=[]
        for lp,q,st,pok in hyps:
            logits=10*(ep['sub']@q)+3*(ep['rel']==r)+2*ep['ts']; order=np.argsort(logits)[::-1]
            mar=float(logits[order[0]]-logits[order[1]]); ex=1 if mar>=margin_thr else min(expand_low,len(order)); lps=lsm(logits)
            for ix in order[:ex]:
                ix=int(ix); cand.append((lp+float(lps[ix]),ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j])))
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:width]; surv.append(any(h[3] for h in hyps))
    m=max(h[0] for h in hyps); mass=np.zeros(S)
    for lp,q,st,pok in hyps: mass[st]+=math.exp(lp-m)
    return int(mass.argmax()),float(np.mean(surv)),float(surv[-1])

def beam_verifier(ep,perms,model,width=8,expand=8,local_alpha=0.0):
    sids,latest,fut_tab,fm_tab=prep(ep)
    # score,q,state,prefix_ok,used,cum_lp,cum_margin,cum_future
    hyps=[(0.,ep['q'],int(ep['start']),True,frozenset(),0.,0.,0.)]; surv=[]
    for j,r in enumerate(ep['prog']):
        cand=[]
        for score,q,st,pok,used,clp,cm,cf in hyps:
            logits=10*(ep['sub']@q)+3*(ep['rel']==r)+2*ep['ts']; p,order,margin,ent=softstats(logits)
            for ix in order[:expand]:
                ix=int(ix); f=feat(ep,j,q,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                ns=score+verifier_score(model,f)+local_alpha*f[3]
                key=(int(sids[ix]),int(ep['rel'][ix])); cand.append((ns,ep['obj'][ix],int(perms[int(ep['op'][ix]),st]),pok and ix==int(ep['idx'][j]),used|{key},clp+f[3],cm+margin,cf+f[8]))
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:width]; surv.append(any(h[3] for h in hyps))
    best=hyps[0]
    return int(best[2]),float(np.mean(surv)),float(surv[-1]),float(best[3])

def eval_method(method,args,seeds,n,perms,model=None,alpha=0.0):
    seed_rows=[]
    for sd in seeds:
        ans=surv=final=toptrue=addr=0.0
        for i in range(n):
            ep=epgen(*args,.30,sd*100000+i,perms)
            if method=='greedy':
                st,a,full=greedy(ep,perms); ans+=st==ep['target']; addr+=a; surv+=a; final+=full; toptrue+=full
            elif method=='beam8_local':
                st,s,f=beam_local(ep,perms); ans+=st==ep['target']; surv+=s; final+=f
            elif method=='r34_adaptive':
                st,s,f=adaptive(ep,perms); ans+=st==ep['target']; surv+=s; final+=f
            elif method=='beam8_verifier':
                st,s,f,t=beam_verifier(ep,perms,model,local_alpha=alpha); ans+=st==ep['target']; surv+=s; final+=f; toptrue+=t
            else: raise ValueError(method)
        seed_rows.append({'answer':ans/n,'path_survival':surv/n,'final_true_path_in_beam':final/n,'selected_true_path':toptrue/n,'address_step_acc':addr/n if method=='greedy' else None})
    def stat(k):
        vals=[r[k] for r in seed_rows if r[k] is not None]
        return {'mean':float(np.mean(vals)),'std':float(np.std(vals)),'seed_values':[float(v) for v in vals]} if vals else None
    return {k:stat(k) for k in ['answer','path_survival','final_true_path_in_beam','selected_true_path','address_step_acc']}|{'seeds':seed_rows}

def main(outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); t0=time.time()
    rng=np.random.default_rng(3000); perms=np.stack([rng.permutation(S) for _ in range(S)])
    train_seeds=[3501,3502,3503]; cal_seed=[3599]; held_seeds=[3601,3602,3603]; n_train=24; n_cal=8; n_held=16
    X,y=make_train(perms,train_seeds,n=n_train); full=fit_lr(X,y,list(range(len(FEATURE_NAMES)))); local=fit_lr(X,y,LOCAL_ONLY)
    # One calibration: choose how much old local log-likelihood to retain in the new verifier score.
    alpha_grid=[0.0,0.25,0.5]
    calibration=[]
    for a in alpha_grid:
        vals={}
        for cn in ('id8','ood32','ood64'):
            vals[cn]=eval_method('beam8_verifier',CONDS[cn],cal_seed,n_cal,perms,full,a)['answer']['mean']
        # Success-shaped objective: long robustness plus explicit short preservation.
        obj=2.0*vals['id8']+vals['ood32']+vals['ood64']
        calibration.append({'local_alpha':a,'objective':obj,'answer':vals})
    best_alpha=max(calibration,key=lambda x:x['objective'])['local_alpha']
    results={}
    for cn,args in CONDS.items():
        results[cn]={}
        for method in ('greedy','beam8_local','r34_adaptive','beam8_verifier'):
            results[cn][method]=eval_method(method,args,held_seeds,n_held,perms,full,best_alpha)
    # Single confirming ablation: same beam/reranking, verifier trained only on local retrieval features.
    ablation={}
    for cn in ('id8','ood32','ood64'):
        ablation[cn]=eval_method('beam8_verifier',CONDS[cn],held_seeds,n_held,perms,local,best_alpha)
    payload={
      'experiment':'FlyGraph R35 consistency verifier/reranker','status':'completed',
      'hypothesis':'A verifier using inference-available trajectory consistency features can rank retained memory hypotheses better than local retrieval likelihood, improving noisy 32/64-hop reasoning while keeping 8-hop regression within 1-2 pp of greedy.',
      'protocol':{'noise':0.30,'train_seeds':train_seeds,'calibration_seed':cal_seed,'held_seeds':held_seeds,'n_train_per_seed_per_condition':n_train,'n_calibration_per_condition':n_cal,'n_held_per_seed_per_condition':n_held,'beam_width':8,'beam_expand':8,'r34_margin_threshold':0.5,'feature_names':FEATURE_NAMES,'local_only_ablation_features':[FEATURE_NAMES[i] for i in LOCAL_ONLY]},
      'training':{'rows':int(len(y)),'positive_fraction':float(y.mean()),'full_model':{'coef':full['coef'].tolist(),'intercept':full['intercept'],'cols':full['cols'].tolist(),'mean':full['mean'].tolist(),'scale':full['scale'].tolist()},'local_only_model':{'coef':local['coef'].tolist(),'intercept':local['intercept'],'cols':local['cols'].tolist(),'mean':local['mean'].tolist(),'scale':local['scale'].tolist()}},
      'calibration':calibration,'selected_local_alpha':best_alpha,'held_results':results,'confirming_ablation_local_only_verifier':ablation,
      'elapsed_sec':time.time()-t0
    }
    (outdir/'r35_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (outdir/'r35_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['condition','method','answer_mean','answer_std','path_survival_mean','final_true_path_in_beam','selected_true_path'])
        for cn,md in results.items():
            for method,res in md.items():
                w.writerow([cn,method,res['answer']['mean'],res['answer']['std'],res['path_survival']['mean'],res['final_true_path_in_beam']['mean'],res['selected_true_path']['mean']])
        for cn,res in ablation.items():
            w.writerow([cn,'ablation_local_only_verifier',res['answer']['mean'],res['answer']['std'],res['path_survival']['mean'],res['final_true_path_in_beam']['mean'],res['selected_true_path']['mean']])
    # Lightweight model artifact for exact reranker reproduction.
    np.savez(outdir/'r35_verifier_weights.npz',feature_names=np.array(FEATURE_NAMES),cols=full['cols'],mean=full['mean'],scale=full['scale'],coef=full['coef'],intercept=np.array([full['intercept']]),local_alpha=np.array([best_alpha]))
    print(json.dumps({'selected_alpha':best_alpha,'calibration':calibration,'held':{cn:{m:round(v['answer']['mean'],4) for m,v in md.items()} for cn,md in results.items()},'ablation':{cn:round(v['answer']['mean'],4) for cn,v in ablation.items()},'elapsed_sec':payload['elapsed_sec']},indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/flygraph_r35/output')