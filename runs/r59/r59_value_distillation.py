from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import r35_base as r
import r52_base as r52
import r58_base as t

TRAIN_SEEDS=(5911,5912,5913)
CAL_SEEDS=(5941,5942)
HELD_SEEDS=(5951,5952,5953)
TRAIN_CONDS=('id8','ood32','ood64','ood128')
TRAIN_N=1
MODEL_SEED=5901
POOL=t.RERANK_POOL
WIDTH=t.WIDTH

def candidate_feature(ep,j,child,inc,f,pr,rank,ranked_scores):
    score,qsrc,st,_pok,used,clp,cm,cf=child
    src=int(qsrc)+1
    lp=np.clip(np.asarray(pr[5][src,:4],dtype=np.float64),-80.0,0.0)
    gap01=float(lp[0]-lp[1]); gap12=float(lp[1]-lp[2]); den=max(1,j+1)
    top=float(ranked_scores[0]); boundary=float(ranked_scores[min(WIDTH-1,len(ranked_scores)-1)])
    sd=float(np.std(ranked_scores[:min(POOL,len(ranked_scores))])) if ranked_scores else 0.0
    nr=int(ep['prog'][j+1]) if j+1<len(ep['prog']) else 0
    one=np.zeros(r.R,dtype=np.float64); one[nr]=1.0
    extra=np.asarray([float(inc),float(score/den),float(len(used)/den),float(clp/den),float(cm/den),float(cf/den),
                      *lp.tolist(),gap01,gap12,float(rank/max(1,POOL-1)),float(score-top),float(score-boundary),sd],dtype=np.float64)
    return np.concatenate([np.asarray(f,dtype=np.float64),extra,one])

def teacher_collect_episode(ep,assets):
    base,alpha,perms,cb,_=assets; pr=t.prep_episode(ep,cb)
    hyps=[(0.0,-1,int(ep['start']),True,frozenset(),0.0,0.0,0.0)]; X=[]; y=[]
    for j,rel0 in enumerate(ep['prog']):
        rec=[]
        for h in hyps:
            ch,_,_=t.expand_hyp(ep,j,int(rel0),h,perms,base,alpha,pr,return_feat=True); rec.extend(ch)
        rec.sort(key=lambda z:z[0][0],reverse=True); scores=[float(z[0][0]) for z in rec]
        if j+1<len(ep['prog']):
            pool=rec[:min(POOL,len(rec))]; ranked=[]
            for rk,(child,inc,f) in enumerate(pool):
                v,_=t.rollout_value(ep,j+1,child,1,perms,base,alpha,pr)
                X.append(candidate_feature(ep,j,child,inc,f,pr,rk,scores)); y.append(float(v))
                ranked.append((float(child[0]+t.LOOKAHEAD_WEIGHT*v),child))
            ranked.sort(key=lambda z:z[0],reverse=True); hyps=[z[1] for z in ranked[:WIDTH]]
        else: hyps=[z[0] for z in rec[:WIDTH]]
    return X,y

def train_student(root:Path,assets):
    X=[]; y=[]; _,_,perms,cb,_=assets
    for cn in TRAIN_CONDS:
        for sd in TRAIN_SEEDS:
            ep=t.build_episode(cn,sd*100000,perms,cb); xx,yy=teacher_collect_episode(ep,assets); X.extend(xx); y.extend(yy)
    X=np.asarray(X,np.float64); y=np.asarray(y,np.float64)
    sc=StandardScaler().fit(X); Z=sc.transform(X); ym=float(y.mean()); ys=float(y.std()+1e-8)
    mlp=MLPRegressor(hidden_layer_sizes=(48,24),activation='tanh',solver='adam',alpha=2e-4,learning_rate_init=1e-3,
                     max_iter=180,early_stopping=True,validation_fraction=.15,n_iter_no_change=12,random_state=MODEL_SEED,batch_size=512)
    mlp.fit(Z,(y-ym)/ys)
    np.savez_compressed(root/'r59_value_student.npz',xmean=sc.mean_,xscale=sc.scale_,ymean=np.asarray([ym]),yscale=np.asarray([ys]),
                        w0=mlp.coefs_[0],b0=mlp.intercepts_[0],w1=mlp.coefs_[1],b1=mlp.intercepts_[1],w2=mlp.coefs_[2],b2=mlp.intercepts_[2])

def load_student(path):
    z=np.load(path,allow_pickle=False)
    return {k:z[k] for k in z.files}

def predict_student(model,X):
    Z=(np.asarray(X,np.float64)-model['xmean'])/(model['xscale']+1e-12)
    h=np.tanh(Z@model['w0']+model['b0']); h=np.tanh(h@model['w1']+model['b1'])
    return (h@model['w2']+model['b2']).reshape(-1)*float(model['yscale'][0])+float(model['ymean'][0])
