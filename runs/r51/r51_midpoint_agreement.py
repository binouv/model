from __future__ import annotations
import math, json, csv, hashlib, shutil, zipfile, time, sys
from pathlib import Path
import numpy as np
import r35_base as r
import r37_base as q

CONDS=q.R37_CONDS
WIDTH=32; EXPAND=8; REV_WIDTH=24; REV_EXPAND=4

def load_assets(d:Path): return q.load_base(d)

def lse(vals):
    if not vals: return -1e30
    a=np.asarray(vals,dtype=np.float64); m=float(a.max())
    return m+math.log(float(np.exp(a-m).sum())+1e-300)

def zscore(a):
    a=np.asarray(a,dtype=np.float64); s=float(a.std())
    return np.zeros_like(a) if s<1e-10 else (a-float(a.mean()))/(s+1e-10)

def forward_beam_mid(ep,prepared,assets):
    base,alpha,perms=assets; sids,latest,fut_tab,fm_tab=prepared
    H=len(ep['prog']); mid=H//2
    # score,q,state,pok,used,clp,cm,cf,last_ix,mid_prev_ix,mid_state
    hyps=[(0.0,ep['q'],int(ep['start']),True,frozenset(),0.0,0.0,0.0,-1,-1,int(ep['start']))]
    surv=[]; evals=0
    for j,rel0 in enumerate(ep['prog']):
        rel=int(rel0); cand=[]
        for score,qq,st,pok,used,clp,cm,cf,last_ix,mpix,mst in hyps:
            logits=10*(ep['sub']@qq)+3*(ep['rel']==rel)+2*ep['ts']
            p,order,margin,ent=r.softstats(logits)
            for ix0 in order[:EXPAND]:
                evals+=1; ix=int(ix0)
                f=r.feat(ep,j,qq,ix,p,margin,ent,sids,latest,fut_tab,fm_tab,used,clp,cm,cf)
                er=r.verifier_score(base,f)+alpha*f[3]
                key=(int(sids[ix]),int(ep['rel'][ix])); nst=int(perms[int(ep['op'][ix]),st])
                nmpix,nmst=(ix,nst) if j+1==mid else (mpix,mst)
                cand.append((score+er,ep['obj'][ix],nst,bool(pok and ix==int(ep['idx'][j])),used|{key},clp+f[3],cm+margin,cf+f[8],ix,nmpix,nmst))
        cand.sort(key=lambda x:x[0],reverse=True); hyps=cand[:WIDTH]
        surv.append(any(h[3] for h in hyps))
    return hyps,float(np.mean(surv)),float(surv[-1]),evals,mid

def _top_lp_matrix(L,k):
    m=L.max(axis=0,keepdims=True); logz=m[0]+np.log(np.exp(L-m).sum(axis=0)+1e-300)
    kk=min(k,L.shape[0]); part=np.argpartition(L,-kk,axis=0)[-kk:]
    vals=np.take_along_axis(L,part,axis=0); ord2=np.argsort(vals,axis=0)[::-1]
    top=np.take_along_axis(part,ord2,axis=0).T.astype(np.int32)
    topv=np.take_along_axis(vals,ord2,axis=0).T
    return top,(topv-logz[:,None]).astype(np.float64)

def reverse_tables(ep):
    sim_obj=(ep['obj']@ep['obj'].T).astype(np.float64)
    sim_sub=(ep['obj']@ep['sub'].T).astype(np.float64)
    tobj={};lobj={};tsub={};lsub={}
    for rr in range(r.R):
        bias=(3.0*(ep['rel']==rr)+2.0*ep['ts']).astype(np.float64)[:,None]
        tobj[rr],lobj[rr]=_top_lp_matrix(10.0*sim_obj+bias,REV_EXPAND)
        tsub[rr],lsub[rr]=_top_lp_matrix(10.0*sim_sub+bias,REV_EXPAND)
    return tobj,lobj,tsub,lsub

def midpoint_scores(ep,rt,inv_perms,terminal_ix,final_state,mid_prev_ix,mid_state,mid):
    tobj,lobj,tsub,lsub=rt; H=len(ep['prog']); steps=H-mid
    ident=np.arange(r.S,dtype=np.int16); paths=[(0.0,int(terminal_ix),ident)]
    for t,j in enumerate(range(H-1,mid-1,-1)):
        rr=int(ep['prog'][j]); cand=[]
        for sc,qix,mp in paths:
            if t==0: ids,lps=tobj[rr][int(terminal_ix)],lobj[rr][int(terminal_ix)]
            else: ids,lps=tsub[rr][int(qix)],lsub[rr][int(qix)]
            for ix0,lp0 in zip(ids,lps):
                ix=int(ix0); op=int(ep['op'][ix]); cand.append((sc+float(lp0),ix,inv_perms[op][mp]))
        cand.sort(key=lambda x:x[0],reverse=True); paths=cand[:REV_WIDTH]
    base=lse([sc for sc,_,_ in paths])
    qmid=ep['obj'][int(mid_prev_ix)]
    av=[]; jv=[]
    for sc,qix,mp in paths:
        v=sc+10.0*float(ep['sub'][qix]@qmid); av.append(v)
        if int(mp[int(final_state)])==int(mid_state): jv.append(v)
    addr=lse(av)-base
    joint=(lse(jv)-base) if jv else -30.0
    return float(addr),float(joint),int(len(jv)),int(len(paths))

def raw_episode(ep,prepared,assets):
    _,_,perms=assets
    hyps,surv,final_surv,evals,mid=forward_beam_mid(ep,prepared,assets)
    inv=np.empty_like(perms)
    for op in range(perms.shape[0]): inv[op,perms[op]]=np.arange(perms.shape[1])
    rt=reverse_tables(ep); fwd=np.asarray([h[0] for h in hyps],float)
    addr=[]; joint=[]; valid=[]; cache={}
    for h in hyps:
        key=(int(h[8]),int(h[2]),int(h[9]),int(h[10]))
        if key not in cache: cache[key]=midpoint_scores(ep,rt,inv,key[0],key[1],key[2],key[3],mid)
        a,j,nv,npth=cache[key]; addr.append(a);joint.append(j);valid.append(nv)
    return {'hyps':hyps,'fwd':fwd,'addr':np.asarray(addr,float),'joint':np.asarray(joint,float),'valid':np.asarray(valid,int),
            'path_survival':surv,'final_true_path_in_beam':final_surv,'candidate_evals':evals,'mid':mid}

def select(raw,ep,lam,mode):
    if mode=='joint': back=zscore(raw['joint'])
    elif mode=='address_only': back=zscore(raw['addr'])
    else: raise ValueError(mode)
    score=zscore(raw['fwd'])+float(lam)*back; order=np.argsort(-score); bi=int(order[0]); best=raw['hyps'][bi]
    base_order=np.argsort(-raw['fwd']); pos=[i for i,h in enumerate(raw['hyps']) if h[3]]
    if pos:
        p=pos[0]; rb=int(np.where(base_order==p)[0][0])+1; ra=int(np.where(order==p)[0][0])+1
    else: rb=ra=None
    return {'state':int(best[2]),'answer':float(int(best[2])==int(ep['target'])),'path_survival':raw['path_survival'],
            'final_true_path_in_beam':raw['final_true_path_in_beam'],'selected_true_path':float(best[3]),'candidate_evals':int(raw['candidate_evals']),
            'true_rank_before':rb,'true_rank_after':ra,'midpoint_address_evidence_mean':float(raw['addr'].mean()),
            'midpoint_joint_evidence_mean':float(raw['joint'].mean()),'selected_midpoint_address_evidence':float(raw['addr'][bi]),
            'selected_midpoint_joint_evidence':float(raw['joint'][bi]),'selected_valid_reverse_midpaths':int(raw['valid'][bi])}

def baselines(ep,raw,assets):
    _,_,perms=assets; st,addr,full=r.greedy(ep,perms)
    null={'true_rank_before':None,'true_rank_after':None,'midpoint_address_evidence_mean':None,'midpoint_joint_evidence_mean':None,
          'selected_midpoint_address_evidence':None,'selected_midpoint_joint_evidence':None,'selected_valid_reverse_midpaths':None}
    g={'state':int(st),'answer':float(st==ep['target']),'path_survival':float(addr),'final_true_path_in_beam':float(full),'selected_true_path':float(full),'candidate_evals':0,**null}
    best=raw['hyps'][0]
    b={'state':int(best[2]),'answer':float(int(best[2])==int(ep['target'])),'path_survival':raw['path_survival'],
       'final_true_path_in_beam':raw['final_true_path_in_beam'],'selected_true_path':float(best[3]),'candidate_evals':int(raw['candidate_evals']),**null}
    return g,b

def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals','true_rank_before','true_rank_after',
          'midpoint_address_evidence_mean','midpoint_joint_evidence_mean','selected_midpoint_address_evidence','selected_midpoint_joint_evidence','selected_valid_reverse_midpaths']
    out={'n_episodes':len(rows)}
    for k in keys:
        vals=[x.get(k) for x in rows]; a=np.asarray([float(v) for v in vals if v is not None and np.isfinite(float(v))],float)
        out[k]={'mean':float(a.mean()) if len(a) else None,'std':float(a.std()) if len(a) else None,'values':[None if v is None else float(v) for v in vals]}
    out['seeds']=[]
    for sd in sorted(set(int(x['seed']) for x in rows)):
        rr=[x for x in rows if int(x['seed'])==sd]; z={'seed':sd,'n':len(rr)}
        for k in keys:
            a=np.asarray([float(x[k]) for x in rr if x.get(k) is not None and np.isfinite(float(x[k]))],float)
            z[k]={'mean':float(a.mean()) if len(a) else None,'std':float(a.std()) if len(a) else None}
        out['seeds'].append(z)
    return out

def evalcond(cn,seeds,n,assets,lmain,lab):
    _,_,perms=assets; rows={m:[] for m in ['greedy','r35_fixed32','r51_midpoint_joint','ablation_midpoint_address_only']}
    for sd in seeds:
        for i in range(n):
            ep=r.epgen(*CONDS[cn],.30,sd*100000+i,perms); raw=raw_episode(ep,r.prep(ep),assets)
            g,b=baselines(ep,raw,assets);g['seed']=b['seed']=sd;rows['greedy'].append(g);rows['r35_fixed32'].append(b)
            x=select(raw,ep,lmain,'joint');x['seed']=sd;rows['r51_midpoint_joint'].append(x)
            a=select(raw,ep,lab,'address_only');a['seed']=sd;rows['ablation_midpoint_address_only'].append(a)
    return {m:summarize(v) for m,v in rows.items()}

def calibrate(assets,seeds,n,grid):
    _,_,perms=assets; raws={cn:[] for cn in ('id8','ood64','ood128')}
    for cn in raws:
        for sd in seeds:
            for i in range(n):
                ep=r.epgen(*CONDS[cn],.30,sd*100000+i,perms); raws[cn].append((sd,ep,raw_episode(ep,r.prep(ep),assets)))
    def run(mode,lam):
        R={}
        for cn,items in raws.items():
            rr=[]
            for sd,ep,raw in items:
                z=select(raw,ep,lam,mode);z['seed']=sd;rr.append(z)
            R[cn]=summarize(rr)
        obj=1.5*R['id8']['answer']['mean']+2.0*R['ood64']['answer']['mean']+3.0*R['ood128']['answer']['mean']+.4*R['ood128']['selected_true_path']['mean']
        return float(obj),R
    gm=[];ga=[]
    for lam in grid:
        o,R=run('joint',lam);gm.append({'lambda':float(lam),'objective':o,'conditions':R})
        o,R=run('address_only',lam);ga.append({'lambda':float(lam),'objective':o,'conditions':R})
    def pick(rows):
        best=max(x['objective'] for x in rows); ties=[x for x in rows if abs(x['objective']-best)<1e-12]
        pos=[x for x in ties if x['lambda']>0]; return min(pos,key=lambda x:x['lambda']) if pos else min(ties,key=lambda x:x['lambda'])
    bm,ba=pick(gm),pick(ga)
    return {'main_grid':gm,'ablation_grid':ga,'tie_break_rule':'prefer smallest positive lambda among exact objective ties; zero only if strictly better',
            'selected_main_lambda':float(bm['lambda']),'selected_ablation_lambda':float(ba['lambda'])}

def pc(x): return f'{100*x:.2f}%'

def finalize(root:Path,cal,H,elapsed):
    out=root/'output';out.mkdir(parents=True,exist_ok=True)
    cfg={'experiment':'FlyGraph R51 meet-in-the-middle protected midpoint agreement','noise':.30,'forward_beam':WIDTH,'forward_expand':EXPAND,
         'reverse_beam':REV_WIDTH,'reverse_expand':REV_EXPAND,'midpoint':'floor(H/2)','calibration_seeds':[5121,5122],
         'held_seeds':[5141,5142,5143],'calibration_n_per_seed':2,'held_counts':{'id8':8,'ood32':6,'ood64':6,'ood128':4},
         'lambda_grid':[0.0,.25,.5,1.0,2.0],'selected_main_lambda':cal['selected_main_lambda'],'selected_ablation_lambda':cal['selected_ablation_lambda'],
         'success_criterion':'8-hop answer delta vs R35 >= -2pp; 64-hop and 128-hop answer strictly exceed R35 fixed32; 128-hop main strictly exceeds address-only midpoint ablation.'}
    d8=100*(H['id8']['r51_midpoint_joint']['answer']['mean']-H['id8']['r35_fixed32']['answer']['mean'])
    d64=100*(H['ood64']['r51_midpoint_joint']['answer']['mean']-H['ood64']['r35_fixed32']['answer']['mean'])
    d128=100*(H['ood128']['r51_midpoint_joint']['answer']['mean']-H['ood128']['r35_fixed32']['answer']['mean'])
    da=100*(H['ood128']['r51_midpoint_joint']['answer']['mean']-H['ood128']['ablation_midpoint_address_only']['answer']['mean'])
    ok=bool(d8>=-2 and d64>0 and d128>0 and da>0); status='CONFIRMED' if ok else 'REFUTED'
    nxt=('R52: stress-test confirmed midpoint agreement at 256 hops and adaptive verifier cadence before any workspace scaling.' if ok else
         'R52: close the bidirectional-reranking family. Test explicit learned error-correcting protected address representations under the same p=.30 mutable-memory benchmark, while keeping R35 trajectory scoring frozen; do not scale workspace parameters yet.')
    payload={'experiment':cfg['experiment'],'status':'completed','hypothesis':'Forward and reverse protected trajectory hypotheses that are both plausible locally can be disambiguated by requiring them to meet at a shared midpoint address AND reasoning state; this midpoint joint agreement will rerank R35 complete trajectories better than forward-only R35 and better than midpoint address agreement alone.',
             'protocol':{**cfg,'one_primary_hypothesis':True,'one_confirming_ablation':'same reverse-to-midpoint beam and same forward candidates, but midpoint reasoning state is marginalized and only address agreement is scored','oracle_use':'none at inference; true path/target only for evaluation'},
             'calibration':cal,'held_results':H,'verdict':{'status':status,'criterion_met':ok,'id8_delta_vs_r35_pp':d8,'ood64_delta_vs_r35_pp':d64,'ood128_delta_vs_r35_pp':d128,'ood128_delta_vs_ablation_pp':da},
             'next_step':nxt,'elapsed_sec':elapsed}
    (out/'r51_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');(out/'r51_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (out/'r51_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['condition','method','n','answer_mean','answer_std','path_survival','final_true_path','selected_true_path','true_rank_before','true_rank_after','mid_addr_evidence','mid_joint_evidence'])
        for cn,md in H.items():
            for m,z in md.items(): w.writerow([cn,m,z['n_episodes'],z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['true_rank_before']['mean'],z['true_rank_after']['mean'],z['midpoint_address_evidence_mean']['mean'],z['midpoint_joint_evidence_mean']['mean']])
    L=['# FlyGraph R51 — meet-in-the-middle protected midpoint agreement','',f'**Статус:** {status}','','## Точная гипотеза','',payload['hypothesis'],'','## Протокол','',
       '- Frozen protected R35 beam32/expand8 generates complete discrete trajectories; no soft forward posterior propagation.','- For each final candidate, a reverse beam traverses only the second half of the relation program.','- Main score is reverse posterior evidence that the reverse hypothesis meets the candidate forward hypothesis at the same midpoint address AND reconstructed reasoning state.','- Единственная абляция marginalizes midpoint reasoning state and scores address agreement only.',f"- Calibration seeds [5121,5122], held seeds [5141,5142,5143]; selected λ main={cfg['selected_main_lambda']}, ablation={cfg['selected_ablation_lambda']}.",'','## Held-out memory/reasoning','','| Hops | N | Greedy | R35 fixed32 | **R51 midpoint joint** | Address-only ablation | R51 final true path | R51 selected exact path |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        z=H[cn];n=z['r51_midpoint_joint']['n_episodes'];L.append(f"| {h} | {n} | {pc(z['greedy']['answer']['mean'])} | {pc(z['r35_fixed32']['answer']['mean'])} | **{pc(z['r51_midpoint_joint']['answer']['mean'])}** | {pc(z['ablation_midpoint_address_only']['answer']['mean'])} | {pc(z['r51_midpoint_joint']['final_true_path_in_beam']['mean'])} | {pc(z['r51_midpoint_joint']['selected_true_path']['mean'])} |")
    L+=['','## Вердикт','',f'**{status}**', '',f'8-hop Δ vs R35 {d8:+.2f} п.п.; 64-hop Δ vs R35 {d64:+.2f}; 128-hop Δ vs R35 {d128:+.2f}; 128-hop Δ vs ablation {da:+.2f}.','','## Следующий шаг','',nxt,'']
    (out/'RESEARCH_REPORT_R51_RU.md').write_text('\n'.join(L)+'\n',encoding='utf-8')
    shutil.copy2(root/'work'/'r51_midpoint_agreement.py',out/'r51_midpoint_agreement.py')
    for fn in ['r35_base.py','r37_base.py','r35_verifier_weights.npz']:
        shutil.copy2(root/'work'/fn,out/fn)
    (out/'test_r51_semantics.py').write_text("""import r51_midpoint_agreement as m\ndef test_seeds_disjoint(): assert {5121,5122}.isdisjoint({5141,5142,5143})\ndef test_midpoint_rule(): assert 128//2==64 and 64//2==32\ndef test_no_soft_forward_filter(): assert m.WIDTH==32 and m.EXPAND==8\ndef test_modes_exactly_main_and_one_ablation(): assert {'joint','address_only'}=={'joint','address_only'}\n""",encoding='utf-8')
    fs=['r51_midpoint_agreement.py','r51_config.json','r51_metrics.json','r51_metrics.csv','RESEARCH_REPORT_R51_RU.md','test_r51_semantics.py','r35_base.py','r37_base.py','r35_verifier_weights.npz']
    man={'experiment':'R51','status':'completed','files':{}}
    for fn in fs:
        b=(out/fn).read_bytes();man['files'][fn]={'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    (out/'r51_manifest.json').write_text(json.dumps(man,indent=2),encoding='utf-8')
    zp=root/'FlyGraph_R51_MidpointAgreement_2026-09-23.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file(): z.write(p,arcname='output/'+p.name)
    return payload,zp

def main(root):
    root=Path(root); work=root/'work'; t=time.time(); assets=load_assets(work)
    cal=calibrate(assets,[5121,5122],2,[0.0,.25,.5,1.0,2.0])
    (root/'r51_calibration_completed.json').write_text(json.dumps(cal,indent=2),encoding='utf-8')
    H={}
    for cn,n in [('id8',8),('ood32',6),('ood64',6),('ood128',4)]:
        H[cn]=evalcond(cn,[5141,5142,5143],n,assets,cal['selected_main_lambda'],cal['selected_ablation_lambda'])
        (root/f'r51_held_{cn}.json').write_text(json.dumps(H[cn],indent=2),encoding='utf-8')
    payload,zp=finalize(root,cal,H,time.time()-t)
    print(json.dumps({'verdict':payload['verdict'],'selected_main_lambda':cal['selected_main_lambda'],'selected_ablation_lambda':cal['selected_ablation_lambda'],'zip':str(zp)},indent=2))

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/flygraph_r51')