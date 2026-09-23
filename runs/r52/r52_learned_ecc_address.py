from __future__ import annotations
import csv, json, hashlib, shutil, zipfile, time, math
from pathlib import Path
import numpy as np
import r35_base as r
import r37_base as r37

K=9
PHYS=96
NCODE=1<<K
NOISE=.30
WIDTH=32
EXPAND=8
CONDS=r.CONDS
TRAIN_SEEDS=[5211,5212,5213]
ABLATION_SEED=5299
HELD_SEEDS=[5241,5242,5243]
HELD_COUNTS={'id8':12,'ood32':8,'ood64':8,'ood128':6}
HILL_STEPS=30000

MESSAGES=((np.arange(NCODE)[:,None] >> np.arange(K)[None,:]) & 1).astype(np.uint8)
NONZERO=MESSAGES[1:]

def load_assets(d:Path):
    return r37.load_base(d)

def gen_codebook(G:np.ndarray)->np.ndarray:
    bits=(MESSAGES @ G) & 1
    # 0 -> +1, 1 -> -1
    C=np.where(bits==0,1.0,-1.0).astype(np.float32)
    return C

def code_stats(G:np.ndarray)->dict:
    words=((NONZERO @ G)&1).astype(np.uint8)
    w=words.sum(axis=1).astype(np.int32)
    sv=np.linalg.svd(G.astype(np.float64),compute_uv=False)
    rank=int(np.linalg.matrix_rank(G.astype(np.float64)%2))  # diagnostic only; GF2 rank checked separately
    s=np.sort(w)
    return {
        'min_distance':int(s[0]),
        'bottom8_mean':float(s[:8].mean()),
        'bottom32_mean':float(s[:32].mean()),
        'bottom96_mean':float(s[:96].mean()),
        'p05_distance':float(np.quantile(w,.05)),
        'median_distance':float(np.median(w)),
        'mean_distance':float(w.mean()),
        'real_matrix_rank_diagnostic':rank,
    }

def gf2_rank(A:np.ndarray)->int:
    A=A.copy().astype(np.uint8)
    m,n=A.shape; row=0
    for col in range(n):
        piv=np.where(A[row:,col]==1)[0]
        if len(piv)==0: continue
        p=row+int(piv[0]); A[[row,p]]=A[[p,row]]
        for rr in range(m):
            if rr!=row and A[rr,col]: A[rr]^=A[row]
        row+=1
        if row==m: break
    return row

def objective_from_weights(w:np.ndarray):
    s=np.sort(w)
    return (int(s[0]),float(s[:8].mean()),float(s[:32].mean()),float(s[:96].mean()))

def train_generator(seed:int,steps:int=HILL_STEPS):
    rng=np.random.default_rng(seed)
    while True:
        G=rng.integers(0,2,size=(K,PHYS),dtype=np.uint8)
        if gf2_rank(G)==K: break
    C=(NONZERO@G)&1
    w=C.sum(axis=1).astype(np.int16)
    best=objective_from_weights(w); accepted=0
    for _ in range(steps):
        i=int(rng.integers(K)); j=int(rng.integers(PHYS))
        mask=NONZERO[:,i].astype(bool)
        nw=w.copy(); nw[mask]+=np.where(C[mask,j]==0,1,-1)
        no=objective_from_weights(nw)
        if no>best:
            C[mask,j]^=1; w=nw; G[i,j]^=1; best=no; accepted+=1
    st=code_stats(G); st['gf2_rank']=gf2_rank(G); st['seed']=seed; st['accepted_flips']=accepted; st['objective']=list(best)
    return G,st

def random_generator(seed:int):
    rng=np.random.default_rng(seed)
    while True:
        G=rng.integers(0,2,size=(K,PHYS),dtype=np.uint8)
        if gf2_rank(G)==K: break
    st=code_stats(G); st['gf2_rank']=gf2_rank(G); st['seed']=seed
    return G,st

def make_structure(hops,nent,nedges,upd,seed,perms):
    # Factor semantics from representation so all three representations share the exact same path,
    # relation program, updates, operations and channel-flip masks.
    rng=np.random.default_rng(seed)
    path=rng.choice(nent,hops+1,replace=False).astype(int)
    rels=rng.integers(r.R,size=hops,dtype=np.int64)
    ops=rng.integers(r.S,size=hops,dtype=np.int64)
    pk={(int(path[j]),int(rels[j])) for j in range(hops)}; e=[]
    reserve=hops+int(round(hops*upd))+2
    while len(e)<max(0,nedges-reserve):
        e.append((int(rng.integers(nent)),int(rng.integers(r.R)),int(rng.integers(nent)),int(rng.integers(r.S))))
    for j in range(hops):
        s,rr,o,op=int(path[j]),int(rels[j]),int(path[j+1]),int(ops[j])
        if rng.random()<upd and len(e)<nedges-1:
            e.append((s,rr,int(rng.integers(nent)),int(rng.integers(r.S))))
        e.append((s,rr,o,op))
    while len(e)<nedges:
        s=int(rng.integers(nent)); rr=int(rng.integers(r.R))
        if (s,rr) in pk: continue
        e.append((s,rr,int(rng.integers(nent)),int(rng.integers(r.S))))
    if len(e)>nedges: e=e[len(e)-nedges:]
    latest={}
    for i,(s,rr,o,op) in enumerate(e): latest[(s,rr)]=i
    idx=np.array([latest[(int(path[j]),int(rels[j]))] for j in range(hops)],dtype=np.int64)
    st=int(rng.integers(r.S)); tar=st
    for op in ops: tar=int(perms[int(op),tar])
    # representation-independent random choices
    code_ids=rng.choice(NCODE,nent,replace=False).astype(np.int64)
    raw_bits=rng.integers(0,2,size=(nent,PHYS),dtype=np.uint8)
    # one independent noisy mask per stored object, plus query mask; matched across reps
    obj_flip=(rng.random((len(e),PHYS))<NOISE)
    q_flip=(rng.random(PHYS)<NOISE)
    return dict(path=path,rels=rels,ops=ops,e=e,idx=idx,start=st,target=tar,code_ids=code_ids,
                raw_bits=raw_bits,obj_flip=obj_flip,q_flip=q_flip,nent=nent,nedges=nedges)

def episode_from_structure(st,rep:str,opt_codebook,rand_codebook):
    nent=st['nent']; e=st['e']
    if rep=='raw96':
        C=np.where(st['raw_bits']==0,1.0,-1.0).astype(np.float32)
    elif rep=='optimized_linear':
        C=opt_codebook[st['code_ids']].copy()
    elif rep=='random_linear':
        C=rand_codebook[st['code_ids']].copy()
    else: raise ValueError(rep)
    sub=[];obj=[];mr=[];mo=[];ts=[]
    for i,(s,rr,o,op) in enumerate(e):
        sub.append(C[s])
        v=C[o].copy(); v[st['obj_flip'][i]]*=-1.0; obj.append(v)
        mr.append(rr);mo.append(op);ts.append(i/max(1,len(e)-1))
    q=C[int(st['path'][0])].copy(); q[st['q_flip']]*=-1.0
    return dict(sub=r.norm(np.stack(sub)),obj=r.norm(np.stack(obj)),rel=np.array(mr),op=np.array(mo),
                ts=np.array(ts,np.float32),q=r.norm(q),prog=st['rels'],target=int(st['target']),start=int(st['start']),idx=st['idx'])

def semantic_validate(ep,perms):
    # Latest-write lookup and target trace must agree with the episode's stored index/program.
    prep=r.prep(ep); sids=prep[0]
    ok_latest=True
    for j,ix in enumerate(ep['idx']):
        ix=int(ix); sid=int(sids[ix]); rr=int(ep['prog'][j])
        later=np.where((sids==sid)&(ep['rel']==rr)&(np.arange(len(sids))>ix))[0]
        if len(later): ok_latest=False; break
    st=int(ep['start'])
    for ix in ep['idx']: st=int(perms[int(ep['op'][int(ix)]),st])
    return bool(ok_latest and st==int(ep['target']))

def eval_rep(ep,assets):
    base,alpha,perms=assets
    pr=r.prep(ep)
    z=r37.baseline_fixed(ep,pr,perms,base,alpha,WIDTH,EXPAND)
    z['answer']=float(z['state']==int(ep['target']))
    return z

def eval_greedy(ep,perms):
    st,addr,full=r.greedy(ep,perms)
    return {'state':int(st),'answer':float(st==int(ep['target'])),'path_survival':float(addr),
            'final_true_path_in_beam':float(full),'selected_true_path':float(full),'candidate_evals':0}

def summarize(rows):
    keys=['answer','path_survival','final_true_path_in_beam','selected_true_path','candidate_evals']
    out={'n_episodes':len(rows)}
    for k in keys:
        a=np.asarray([float(x[k]) for x in rows],dtype=float)
        out[k]={'mean':float(a.mean()),'std':float(a.std()),'values':a.tolist()}
    out['seeds']=[]
    for sd in sorted(set(int(x['seed']) for x in rows)):
        rr=[x for x in rows if int(x['seed'])==sd]; z={'seed':sd,'n':len(rr)}
        for k in keys:
            a=np.asarray([float(x[k]) for x in rr],float); z[k]={'mean':float(a.mean()),'std':float(a.std())}
        out['seeds'].append(z)
    return out

def evaluate_condition(cn,seeds,n,assets,opt_cb,rand_cb):
    base,alpha,perms=assets
    rows={m:[] for m in ['raw96_greedy','raw96_r35_fixed32','r52_optimized_ecc_r35','ablation_random_linear_r35']}
    sem_mismatch=0
    for sd in seeds:
        for i in range(n):
            seed=sd*100000+i
            st=make_structure(*CONDS[cn],seed,perms)
            eraw=episode_from_structure(st,'raw96',opt_cb,rand_cb)
            eopt=episode_from_structure(st,'optimized_linear',opt_cb,rand_cb)
            eran=episode_from_structure(st,'random_linear',opt_cb,rand_cb)
            sem_mismatch += int(not semantic_validate(eraw,perms) or not semantic_validate(eopt,perms) or not semantic_validate(eran,perms))
            z=eval_greedy(eraw,perms);z['seed']=sd;rows['raw96_greedy'].append(z)
            z=eval_rep(eraw,assets);z['seed']=sd;rows['raw96_r35_fixed32'].append(z)
            z=eval_rep(eopt,assets);z['seed']=sd;rows['r52_optimized_ecc_r35'].append(z)
            z=eval_rep(eran,assets);z['seed']=sd;rows['ablation_random_linear_r35'].append(z)
    return {m:summarize(v) for m,v in rows.items()},sem_mismatch

def pct(x): return f'{100*x:.2f}%'

def main(root):
    root=Path(root); out=root/'output'; out.mkdir(parents=True,exist_ok=True); t0=time.time()
    assets=load_assets(root)
    # Train exactly one protected-address model family: maximize minimum distance of a [96,9] binary linear code.
    trials=[]; trained=[]
    for sd in TRAIN_SEEDS:
        G,st=train_generator(sd); trials.append(st); trained.append((G,st))
    trained.sort(key=lambda x: tuple(x[1]['objective']),reverse=True)
    Gopt,opt_stats=trained[0]
    Grand,rand_stats=random_generator(ABLATION_SEED)
    opt_cb=gen_codebook(Gopt); rand_cb=gen_codebook(Grand)
    np.savez_compressed(out/'r52_ecc_model.npz',generator=Gopt,codebook=opt_cb,train_seeds=np.asarray(TRAIN_SEEDS),stats_json=json.dumps(opt_stats))
    np.savez_compressed(out/'r52_random_linear_ablation.npz',generator=Grand,codebook=rand_cb,seed=np.array([ABLATION_SEED]),stats_json=json.dumps(rand_stats))

    held={}; sem={}
    for cn in ['id8','ood32','ood64','ood128']:
        held[cn],sem[cn]=evaluate_condition(cn,HELD_SEEDS,HELD_COUNTS[cn],assets,opt_cb,rand_cb)

    def am(cn,m): return held[cn][m]['answer']['mean']
    d8=100*(am('id8','r52_optimized_ecc_r35')-am('id8','raw96_r35_fixed32'))
    d64=100*(am('ood64','r52_optimized_ecc_r35')-am('ood64','raw96_r35_fixed32'))
    d128=100*(am('ood128','r52_optimized_ecc_r35')-am('ood128','raw96_r35_fixed32'))
    a64=100*(am('ood64','r52_optimized_ecc_r35')-am('ood64','ablation_random_linear_r35'))
    a128=100*(am('ood128','r52_optimized_ecc_r35')-am('ood128','ablation_random_linear_r35'))
    ok=bool(d8>=-2 and d64>0 and d128>0 and a64>0 and a128>0)
    status='CONFIRMED' if ok else 'REFUTED / MIXED'
    cfg={
      'experiment':'FlyGraph R52 learned linear ECC protected address codebook','noise':NOISE,'physical_bits':PHYS,'logical_bits':K,
      'codewords':NCODE,'max_entities':max(v[1] for v in CONDS.values()),'beam_width':WIDTH,'expand':EXPAND,
      'train_seeds':TRAIN_SEEDS,'ablation_seed':ABLATION_SEED,'held_seeds':HELD_SEEDS,'held_counts_per_seed':HELD_COUNTS,
      'training_steps_per_seed':HILL_STEPS,
      'main':'optimized [96,9] binary linear code, randomly remapped to entities per episode; frozen R35 verifier/inference',
      'ablation':'same [96,9] linear-code representation and frozen R35 inference, but unoptimized random generator matrix',
      'success_criterion':'8-hop answer delta vs matched raw96 R35 >= -2pp; optimized ECC strictly improves 64 and 128 answer over raw96 R35 and over random-linear ablation.',
      'pairing':'same path/program/updates/ops/entity-to-codeword IDs and exact channel flip masks across raw/main/ablation within each episode'
    }
    payload={
      'experiment':cfg['experiment'],'status':'completed','hypothesis':
      'A learned high-minimum-distance protected address code, without increasing the 96-bit address width or changing the frozen reasoning/verifier stack, will reduce noisy memory ambiguity enough to improve long-horizon 64/128-hop reasoning. The improvement should exceed an equal-capacity random linear-code control, showing that distance optimization rather than merely using a fixed code family is responsible.',
      'protocol':cfg,'training':{'trials':trials,'selected':opt_stats,'ablation_random':rand_stats},
      'semantic_validation_mismatches':sem,'held_results':held,
      'verdict':{'status':status,'criterion_met':ok,'id8_delta_vs_raw_r35_pp':d8,'ood64_delta_vs_raw_r35_pp':d64,'ood128_delta_vs_raw_r35_pp':d128,
                 'ood64_delta_vs_random_linear_pp':a64,'ood128_delta_vs_random_linear_pp':a128},
      'next_step':('R53: stress the confirmed ECC representation at 256 hops and p=.35, then only if retrieval remains stable unlock sparse workspace scaling.' if ok else
                   'R53: keep the frozen R35 trajectory stack and test syndrome-aware multi-hypothesis decoding rather than a better static codebook. Preserve discrete trajectories; expose decoder uncertainty (top-k syndrome-consistent addresses) instead of hard nearest-code cleanup. Do not scale workspace parameters yet.'),
      'elapsed_sec':time.time()-t0
    }
    (out/'r52_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'r52_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (out/'r52_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['condition','method','n','answer_mean','answer_std','path_survival','final_true_path','selected_true_path','candidate_evals'])
        for cn,md in held.items():
            for m,z in md.items():
                w.writerow([cn,m,z['n_episodes'],z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['candidate_evals']['mean']])
    L=['# FlyGraph R52 — learned linear ECC protected address codebook','',f'**Статус:** {status}','','## Точная гипотеза','',payload['hypothesis'],'','## Протокол','',
       f'- Address width remains exactly **96 physical bits**; no workspace/reasoning parameter scaling.',
       f'- Main code: binary linear `[96,9]` code with 512 codewords; generator optimized only for code distance on train seeds {TRAIN_SEEDS}.',
       f"- Selected minimum Hamming distance: **{opt_stats['min_distance']}**; random-linear ablation: **{rand_stats['min_distance']}**.",
       '- Entity identities are randomly remapped to codewords every episode, so fixed semantic identity cannot be memorized.',
       '- Frozen R35 verifier, beam=32, expand=8. Exact path/program/update/noise masks are paired across conditions.',
       '- Единственная абляция: identical `[96,9]` linear-code address representation with an unoptimized random generator matrix.','','## Held-out memory/reasoning','','| Hops | N | Raw96 greedy | Raw96 R35 | **R52 optimized ECC** | Random-linear ablation | R52 final true path |','|---:|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        z=held[cn];n=z['r52_optimized_ecc_r35']['n_episodes']
        L.append(f"| {h} | {n} | {pct(z['raw96_greedy']['answer']['mean'])} | {pct(z['raw96_r35_fixed32']['answer']['mean'])} | **{pct(z['r52_optimized_ecc_r35']['answer']['mean'])}** | {pct(z['ablation_random_linear_r35']['answer']['mean'])} | {pct(z['r52_optimized_ecc_r35']['final_true_path_in_beam']['mean'])} |")
    L += ['','## Semantic validation','',f"Mismatches: `{sum(sem.values())}` total across all completed paired held-out episodes. Per condition: `{sem}`.",'','## Вердикт','',f'**{status}**','',
          f'- 8-hop Δ vs raw96 R35: **{d8:+.2f} п.п.**',f'- 64-hop Δ vs raw96 R35: **{d64:+.2f} п.п.**',f'- 128-hop Δ vs raw96 R35: **{d128:+.2f} п.п.**',f'- 64-hop Δ vs random-linear ablation: **{a64:+.2f} п.п.**',f'- 128-hop Δ vs random-linear ablation: **{a128:+.2f} п.п.**','',
          '## Архитектурный вывод','']
    if ok:
        L.append('Static protected-address geometry is a genuine cognition bottleneck at p=.30: improving minimum code distance while preserving full 96-bit protected state materially improves long-horizon trajectory survival/ranking without adding reasoning parameters.')
    else:
        L.append('A better static codebook alone is insufficient to close retrieval uncertainty. The next useful target is decoder uncertainty itself: keep multiple syndrome-consistent discrete addresses rather than forcing a single nearest-code correction. This preserves the established protected-state principle and avoids returning to soft workspace dilution.')
    L += ['','## Следующий шаг','',payload['next_step'],'']
    (out/'RESEARCH_REPORT_R52_RU.md').write_text('\n'.join(L)+'\n',encoding='utf-8')

    # Local reproducibility snapshot
    shutil.copy2(Path(__file__),out/'r52_learned_ecc_address.py')
    for fn in ['r35_base.py','r37_base.py','r35_verifier_weights.npz']:
        shutil.copy2(root/fn,out/fn)
    test='''import numpy as np\nimport r52_learned_ecc_address as m\ndef test_seeds_disjoint(): assert set(m.TRAIN_SEEDS).isdisjoint(m.HELD_SEEDS)\ndef test_capacity(): assert m.NCODE>=320 and m.PHYS==96\ndef test_code_rank_and_uniqueness():\n g,_=m.train_generator(m.TRAIN_SEEDS[0],2000); c=m.gen_codebook(g); assert m.gf2_rank(g)==m.K; assert np.unique(c,axis=0).shape[0]==m.NCODE\ndef test_one_ablation_only(): assert m.ABLATION_SEED==5299\n'''
    (out/'test_r52_semantics.py').write_text(test,encoding='utf-8')
    manifest={'experiment':'R52','status':'completed','files':{}}
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name!='r52_manifest.json':
            b=p.read_bytes();manifest['files'][p.name]={'size_bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    (out/'r52_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    zp=root/'FlyGraph_R52_LearnedECCProtectedAddress_2026-09-23.zip'
    with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file(): z.write(p,arcname='output/'+p.name)
    print(json.dumps({'status':status,'selected_code':opt_stats,'random_code':rand_stats,'verdict':payload['verdict'],
                      'answers':{cn:{m:round(z['answer']['mean'],4) for m,z in md.items()} for cn,md in held.items()},
                      'semantic_mismatches':sem,'elapsed_sec':payload['elapsed_sec'],'zip':str(zp)},indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parent))