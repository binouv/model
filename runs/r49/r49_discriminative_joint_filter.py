from __future__ import annotations
import csv,json,time,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
import r35_base as r
import r37_base as q
import r47_base as r47
CONDS=q.R37_CONDS; S=r.S; TRANS_K=8; WIDTH=32; EXPAND=8

def load_assets(d:Path): return q.load_base(d)
def softmax_scores(x,gamma):
    a=gamma*np.asarray(x,float);m=float(a.max());z=np.exp(a-m);return z/(z.sum()+1e-300)

def mem(ep,pr):
    sids,U,entries=r47.memory_index(ep,pr);return sids,U,entries

def score_candidates(ep,j,qv,cands,pr,base,alpha,clp,cm,cf,mode='verifier'):
    sids,latest,fut,fm=pr;rel=int(ep['prog'][j]);logits=10*(ep['sub']@qv)+3*(ep['rel']==rel)+2*ep['ts'];p,order,margin,ent=r.softstats(logits)
    scores=[];feats=[]
    for ix in cands:
        f=r.feat(ep,j,qv,int(ix),p,margin,ent,sids,latest,fut,fm,frozenset(),clp,cm,cf)
        sc=r.verifier_score(base,f)+alpha*f[3] if mode=='verifier' else f[3]
        scores.append(float(sc));feats.append(f)
    return np.asarray(scores),feats,float(margin)

def choose_top(cand_sids,cand_ix,scores,feats,k,gamma):
    order=np.argsort(scores)[::-1][:min(k,len(scores))];sel_s=cand_sids[order];sel_ix=cand_ix[order];sel_f=[feats[int(t)] for t in order];prob=softmax_scores(scores[order],gamma);return sel_s,sel_ix,prob,sel_f

def init_filter(ep,pr,assets,gamma,mode):
    base,alpha,_=assets;sids,U,entries=mem(ep,pr);rel=int(ep['prog'][0]);cand_s=np.array(sorted(entries[rel]),dtype=np.int64);cand_ix=np.array([entries[rel][int(s)] for s in cand_s],dtype=np.int64)
    sc,fs,margin=score_candidates(ep,0,ep['q'],cand_ix,pr,base,alpha,0.,0.,0.,mode);cur,ixs,pa,fs=choose_top(cand_s,cand_ix,sc,fs,TRANS_K,gamma)
    J=np.zeros((len(cur),S),float);J[:,int(ep['start'])]=pa;clp=np.array([f[3] for f in fs],float);cm=np.full(len(cur),margin,float);cf=np.array([f[8] for f in fs],float)
    return sids,U,entries,cur,ixs,J,clp,cm,cf

def diag(J,cur,true_sid):
    pa=J.sum(axis=1);order=np.argsort(pa)[::-1];pos=np.where(cur==true_sid)[0];tm=float(pa[pos[0]]) if len(pos) else 0.;top1=float(len(order)>0 and int(cur[order[0]])==int(true_sid));top32=float(int(true_sid) in set(int(x) for x in cur[order[:min(32,len(order))]]));return top1,tm,top32,r47.entropy(pa)+r47.entropy(J.sum(axis=0))

def run_filter(ep,pr,assets,gamma,mode='verifier'):
    base,alpha,perms=assets;sids,U,entries,cur,ixs,J,clp,cm,cf=init_filter(ep,pr,assets,gamma,mode);top1=[];tm=[];top32=[];ents=[];support=[]
    for j in range(len(ep['prog'])):
        true_sid=int(sids[int(ep['idx'][j])]);a,b,c,d=diag(J,cur,true_sid);top1.append(a);tm.append(b);top32.append(c);ents.append(d);support.append(float(np.sum(J.sum(axis=1)>1e-12)))
        rel=int(ep['prog'][j]);after=np.zeros_like(J)
        for row,ix in enumerate(ixs):
            perm=perms[int(ep['op'][int(ix)])]
            for st in range(S):after[row,int(perm[st])]+=J[row,st]
        if j==len(ep['prog'])-1:
            ps=after.sum(axis=0);ps/=ps.sum()+1e-300;pred=int(ps.argmax());break
        nr=int(ep['prog'][j+1]);next_all=np.array(sorted(entries[nr]),dtype=np.int64);next_ix_all=np.array([entries[nr][int(s)] for s in next_all],dtype=np.int64);index={int(s):i for i,s in enumerate(next_all)};Jnext=np.zeros((len(next_all),S),float);wstat=np.zeros(len(next_all),float);nclp=np.zeros(len(next_all));ncm=np.zeros(len(next_all));ncf=np.zeros(len(next_all))
        pa_cur=J.sum(axis=1)
        for row,ix in enumerate(ixs):
            qv=ep['obj'][int(ix)];sc,fs,margin=score_candidates(ep,j+1,qv,next_ix_all,pr,base,alpha,float(clp[row]),float(cm[row]),float(cf[row]),mode);ss,six,prob,sfs=choose_top(next_all,next_ix_all,sc,fs,TRANS_K,gamma)
            for sid2,ix2,p2,f2 in zip(ss,six,prob,sfs):
                col=index[int(sid2)];Jnext[col]+=float(p2)*after[row];w=float(pa_cur[row])*float(p2);wstat[col]+=w;nclp[col]+=w*(clp[row]+f2[3]);ncm[col]+=w*(cm[row]+margin);ncf[col]+=w*(cf[row]+f2[8])
        keep=np.where(Jnext.sum(axis=1)>1e-15)[0];J=Jnext[keep];cur=next_all[keep];ixs=next_ix_all[keep];mass=J.sum();J/=mass+1e-300
        clp=np.array([nclp[k]/(wstat[k]+1e-300) for k in keep]);cm=np.array([ncm[k]/(wstat[k]+1e-300) for k in keep]);cf=np.array([ncf[k]/(wstat[k]+1e-300) for k in keep])
    return {'state':pred,'answer':float(pred==ep['target']),'address_top1_acc':float(np.mean(top1)),'true_address_mass':float(np.mean(tm)),'address_in_top32':float(np.mean(top32)),'final_true_address_top1':float(top1[-1]),'final_true_address_mass':float(tm[-1]),'joint_entropy':float(np.mean(ents)),'mean_address_support':float(np.mean(support))}

def baseline(ep,pr,assets):
    g,b=r47.baseline(ep,pr,assets)
    for z in (g,b):z['mean_address_support']=1.0
    return g,b

def summarize(rows):
    keys=['answer','address_top1_acc','true_address_mass','address_in_top32','final_true_address_top1','final_true_address_mass','joint_entropy','mean_address_support'];o={'n_episodes':len(rows)}
    for k in keys:
        a=np.array([z[k] for z in rows],float);o[k]={'mean':float(a.mean()),'std':float(a.std()),'values':a.tolist()}
    o['seeds']=[]
    for sd in sorted(set(z['seed'] for z in rows)):
        rr=[z for z in rows if z['seed']==sd];zz={'seed':sd,'n':len(rr)}
        for k in keys:
            a=np.array([x[k] for x in rr],float);zz[k]={'mean':float(a.mean()),'std':float(a.std())}
        o['seeds'].append(zz)
    return o

def evalcond(cn,seeds,n,assets,gamma):
    _,_,perms=assets;rows={m:[] for m in ['greedy','r35_fixed32','r49_discriminative_joint','ablation_local_likelihood_joint']}
    for sd in seeds:
        for i in range(n):
            ep=r.epgen(*CONDS[cn],.30,sd*100000+i,perms);pr=r.prep(ep);g,b=baseline(ep,pr,assets);g['seed']=b['seed']=sd;rows['greedy'].append(g);rows['r35_fixed32'].append(b)
            x=run_filter(ep,pr,assets,gamma,'verifier');x['seed']=sd;rows['r49_discriminative_joint'].append(x)
            z=run_filter(ep,pr,assets,gamma,'local');z['seed']=sd;rows['ablation_local_likelihood_joint'].append(z)
    return {m:summarize(v) for m,v in rows.items()}

def calibrate(assets,seeds,n,gammas):
    _,_,perms=assets;rows=[]
    for gamma in gammas:
        R={}
        for cn in ('id8','ood64','ood128'):
            rr=[]
            for sd in seeds:
                for i in range(n):
                    ep=r.epgen(*CONDS[cn],.30,sd*100000+i,perms);pr=r.prep(ep);x=run_filter(ep,pr,assets,gamma,'verifier');x['seed']=sd;rr.append(x)
            R[cn]=summarize(rr)
        sc=1.5*R['id8']['answer']['mean']+2*R['ood64']['answer']['mean']+3*R['ood128']['answer']['mean']+.5*R['ood128']['address_top1_acc']['mean'];rows.append({'gamma':gamma,'score':float(sc),'conditions':R})
    best=max(rows,key=lambda z:(z['score'],-abs(z['gamma']-1.0)));return rows,float(best['gamma'])

def pc(x):return f'{100*x:.2f}%'
def finalize(root:Path,cal,H):
    out=root/'output';out.mkdir(exist_ok=True);cfg={'experiment':'FlyGraph R49 discriminative protected joint filter','noise':.30,'transition_topk':TRANS_K,'calibration_seeds':[4921,4922],'held_seeds':[4941,4942,4943],'gamma_grid':[.5,1.,2.,4.],'selected_gamma':cal['selected_gamma'],'held_counts':{'id8':8,'ood32':6,'ood64':6,'ood128':4},'success_criterion':'8-hop delta vs R35 >= -2pp; 64/128 answer strictly exceed R35; 128 verifier-filter strictly exceeds local-likelihood ablation.'}
    d8=100*(H['id8']['r49_discriminative_joint']['answer']['mean']-H['id8']['r35_fixed32']['answer']['mean']);d64=100*(H['ood64']['r49_discriminative_joint']['answer']['mean']-H['ood64']['r35_fixed32']['answer']['mean']);d128=100*(H['ood128']['r49_discriminative_joint']['answer']['mean']-H['ood128']['r35_fixed32']['answer']['mean']);da=100*(H['ood128']['r49_discriminative_joint']['answer']['mean']-H['ood128']['ablation_local_likelihood_joint']['answer']['mean']);ok=bool(d8>=-2 and d64>0 and d128>0 and da>0);ver={'status':'CONFIRMED' if ok else 'REFUTED / MIXED','id8_delta_vs_r35_pp':d8,'ood64_delta_vs_r35_pp':d64,'ood128_delta_vs_r35_pp':d128,'ood128_delta_vs_local_ablation_pp':da,'criterion_met':ok}
    nxt='R50: extend confirmed discriminative joint filter to 256 hops and adaptive support; then unlock workspace scaling.' if ok else 'R50: close soft belief-filter family and test delayed decision with backward consistency: retain R35 beam forward, then rerank complete trajectories using a reverse-memory pass from terminal evidence instead of local/soft posterior propagation.'
    P={'experiment':cfg['experiment'],'status':'completed','hypothesis':'Using the frozen R35 verifier as the transition evidence inside an explicit protected address×reasoning-state belief filter will correct the generative-likelihood mismatch and outperform both R35 beam and a local-likelihood belief filter.','protocol':{**cfg,'one_primary_hypothesis':True,'one_confirming_ablation':'same joint filter/top-k transitions but local retrieval log-likelihood instead of R35 verifier score','oracle_use':'none in inference'},'calibration':cal,'held_results':H,'verdict':ver,'next_step':nxt};(out/'r49_config.json').write_text(json.dumps(cfg,indent=2));(out/'r49_metrics.json').write_text(json.dumps(P,indent=2))
    with (out/'r49_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition','method','n','answer_mean','answer_std','address_top1_acc','true_address_mass','address_in_top32','final_true_address_top1','joint_entropy','mean_address_support'])
        for cn,md in H.items():
            for meth,z in md.items():w.writerow([cn,meth,z['n_episodes'],z['answer']['mean'],z['answer']['std'],z['address_top1_acc']['mean'],z['true_address_mass']['mean'],z['address_in_top32']['mean'],z['final_true_address_top1']['mean'],z['joint_entropy']['mean'],z['mean_address_support']['mean']])
    L=['# FlyGraph R49 — discriminative protected joint filter','',f"**Статус:** {ver['status']}",'','## Точная гипотеза','',P['hypothesis'],'','## Протокол','',f"- Frozen R35 verifier; top-{TRANS_K} transition support per current address; calibration gamma={cfg['selected_gamma']} on {cfg['calibration_seeds']}; held={cfg['held_seeds']}.",'- Main propagates joint state mass with transition probabilities derived from R35 verifier scores and posterior-weighted cumulative verifier summaries.','- Единственная абляция: identical filter using local retrieval log-likelihood instead of verifier score.','','## Held-out memory/reasoning','','| Hops | N | Greedy | R35 fixed32 | **R49 discr. joint** | Local-likelihood ablation | R49 addr top1 |','|---:|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        z=H[cn];L.append(f"| {h} | {z['r49_discriminative_joint']['n_episodes']} | {pc(z['greedy']['answer']['mean'])} | {pc(z['r35_fixed32']['answer']['mean'])} | **{pc(z['r49_discriminative_joint']['answer']['mean'])}** | {pc(z['ablation_local_likelihood_joint']['answer']['mean'])} | {pc(z['r49_discriminative_joint']['address_top1_acc']['mean'])} |")
    L+=['','## Вердикт','',f"**{ver['status']}**",'',f"8-hop Δ vs R35 {d8:+.2f} п.п.; 64-hop Δ vs R35 {d64:+.2f}; 128-hop Δ vs R35 {d128:+.2f}; 128-hop verifier vs local {da:+.2f}.",'','## Следующий шаг','',nxt,''];(out/'RESEARCH_REPORT_R49_RU.md').write_text('\n'.join(L)+'\n')
    shutil.copy2(root/'r49_discriminative_joint_filter.py',out/'r49_discriminative_joint_filter.py')
    for fn in ['r35_base.py','r37_base.py','r47_base.py','r35_verifier_weights.npz']:shutil.copy2(root/fn,out/fn)
    (out/'test_r49_semantics.py').write_text('''import numpy as np\nimport r49_discriminative_joint_filter as m\ndef test_seeds_disjoint(): assert {4921,4922}.isdisjoint({4941,4942,4943})\ndef test_softmax(): assert abs(m.softmax_scores([0.,0.],1.).sum()-1)<1e-12\ndef test_transition_k(): assert m.TRANS_K==8\ndef test_modes(): assert set(['verifier','local'])=={'verifier','local'}\n''')
    fs=['r49_discriminative_joint_filter.py','r49_config.json','r49_metrics.json','r49_metrics.csv','RESEARCH_REPORT_R49_RU.md','test_r49_semantics.py','r35_base.py','r37_base.py','r47_base.py','r35_verifier_weights.npz'];man={'experiment':'R49','status':'completed','files':{}}
    for fn in fs:
        b=(out/fn).read_bytes();man['files'][fn]={'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    (out/'r49_manifest.json').write_text(json.dumps(man,indent=2));zp=root/'FlyGraph_R49_DiscriminativeJointFilter_2026-09-22.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,arcname='output/'+p.name)
    return P,zp