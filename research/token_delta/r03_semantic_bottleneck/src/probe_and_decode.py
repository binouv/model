"""R03 -- supervised semantic probing of actual (cached) pretrained Qwen states.

No new Qwen network download. Inputs are SHA-verified R02 frozen-teacher
last-hidden-state features for 264 non-sensitive, fixed synthetic questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve()
TOKEN_DELTA=HERE.parents[2]
sys.path.insert(0,str(TOKEN_DELTA/"r02_qwen_teacher"/"src"))
from real_qwen_latent import (
    SEED, LATENT, MAX_LEN, BOS, PAD, EOS, IDX, VOCAB,
    SUBJECTS, ACTIONS, OBJECTS, question, response, split_facts,
    hashed_bow, encode_targets, Renderer, exact, matches_slots, show_text
)

R02_ROOT=TOKEN_DELTA/"r02_qwen_teacher"/"results"
SOURCE_SHA="e0add6315af0ac9f344c90d7339881d8711b49da28576eaae8e5e05055f8ff96"
# Source SHA pins the original independently-trained R02 CPU feature cache.

def read_features(path=R02_ROOT/"qwen_prompt_features.pt"):
    p=Path(path)
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != SOURCE_SHA:
        raise RuntimeError(f"Teacher cache SHA mismatch: {actual} (expected {SOURCE_SHA})")
    c=torch.load(p,map_location="cpu",weights_only=False)
    train,test=split_facts()
    prompts=[question(f) for f in train]+[question(f) for f in test]+[question(f,ood=True) for f in test]
    assert c["questions"]==prompts
    assert c["train_triples"]==train and c["test_triples"]==test
    assert c["embeddings"].shape==(264,1024)
    return c["embeddings"].float(),train,test,prompts

class FactProbe(nn.Module):
    def __init__(self,kind="linear"):
        super().__init__()
        if kind=="linear":
            self.net=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,18))
        elif kind=="mlp":
            self.net=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,128),
                  nn.SiLU(),nn.Linear(128,18))
        else:raise ValueError(kind)
        self.kind=kind
    def forward(self,x):return self.net(x).reshape(-1,3,6)

def probe_measure(net,x,facts):
    net.eval()
    with torch.no_grad():
        out=net(x).argmax(-1)
        target=torch.tensor(facts,dtype=torch.long)
        ok=(out==target).float()
        return {
          "subject":float(ok[:,0].mean()),
          "action":float(ok[:,1].mean()),
          "object":float(ok[:,2].mean()),
          "mean_slots":float(ok.mean()),
          "full_triple":float(ok.all(-1).float().mean()),
          "full_triple_count":int(ok.all(-1).sum().item()),
          "n":len(facts)
        }

def train_probe(name,kind,x_train,train,steps=700,lr=.005,shuffled_labels=False):
    torch.manual_seed(SEED)
    model=FactProbe(kind)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0001)
    labels=torch.tensor(train,dtype=torch.long)
    if shuffled_labels:
        labels=labels[torch.randperm(len(labels),generator=torch.Generator().manual_seed(SEED+33))]
    rng=torch.Generator().manual_seed(SEED+2)
    trace=[]
    t0=time.perf_counter()
    for i in range(steps):
        ids=torch.randint(0,len(train),(48,),generator=rng)
        scores=model(x_train[ids])
        gold=labels[ids]
        loss=F.cross_entropy(scores.reshape(-1,6),gold.reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward()
        opt.step()
        if (i+1)%max(1,steps//4)==0:trace.append([i+1,round(float(loss.detach()),6)])
    return model,opt,{
        "steps":steps,"train_seconds":time.perf_counter()-t0,
        "parameters":sum(p.numel() for p in model.parameters()),
        "loss_trace":trace,"shuffled_train_labels":shuffled_labels,
        "experiment":name
    }

class SemanticsFirstStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe=FactProbe("mlp")
        self.slot_basis=nn.Parameter(torch.randn(3,6,16)*.05)
        self.slot_to_latent=nn.Sequential(nn.Linear(48,LATENT),nn.Tanh())
        self.renderer=Renderer()
    def features(self,x):
        logits=self.probe(x)
        probabilities=logits.softmax(-1)
        # shape (batch, 3, 16), each vector is a weighted semantic slot
        slots=torch.einsum("bsc,scd->bsd",probabilities,self.slot_basis)
        z=self.slot_to_latent(slots.flatten(1))
        return logits,z
    def forward(self,x,inp,styles):
        logits,z=self.features(x)
        return logits,self.renderer(inp,z,styles)

def train_joint(name,x,train,steps=900):
    torch.manual_seed(SEED)
    net=SemanticsFirstStudent()
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.00001)
    labels=torch.tensor(train,dtype=torch.long)
    targets=encode_targets(train)
    rng=torch.Generator().manual_seed(SEED+71)
    trace=[]
    t0=time.perf_counter()
    for i in range(steps):
        pairs=torch.randint(0,len(train)*2,(48,),generator=rng)
        fidx=pairs//2
        styles=pairs%2
        target=targets[fidx,styles]
        inputs=torch.cat([torch.full((len(target),1),BOS,dtype=torch.long),
                          target[:,:-1]],dim=1)
        s_logits,word_logits=net(x[fidx],inputs,styles)
        slot_loss=F.cross_entropy(s_logits.reshape(-1,6),labels[fidx].reshape(-1))
        seq_loss=F.cross_entropy(word_logits.reshape(-1,len(VOCAB)),
                                 target.reshape(-1),ignore_index=PAD)
        loss=seq_loss+slot_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step()
        if (i+1)%max(1,steps//4)==0:
            trace.append({"step":i+1,"seq":float(seq_loss.detach()),
                          "slots":float(slot_loss.detach())})
    return net,opt,{
        "training_seconds":time.perf_counter()-t0,"loss_trace":trace,"steps":steps,
        "parameters":sum(p.numel() for p in net.parameters())
    }

@torch.no_grad()
def evaluate_joint(net,x,triples,intervention="none"):
    net.eval()
    logits,z=net.features(x)
    if intervention=="zero":
        z=torch.zeros_like(z)
    elif intervention=="shuffle":
        z=z[torch.randperm(len(z),generator=torch.Generator().manual_seed(97))]
    elif intervention!="none":raise ValueError(intervention)
    p=logits.argmax(-1)
    facts=torch.tensor(triples,dtype=torch.long)
    ok=(p==facts).float()
    results={
       "slots":{
           "subject":float(ok[:,0].mean()),
           "action":float(ok[:,1].mean()),
           "object":float(ok[:,2].mean()),
           "mean":float(ok.mean()),
           "exact_triple":float(ok.all(-1).float().mean()),
           "exact_triple_count":int(ok.all(-1).sum())
       }
    }
    generated={}
    for style,tag in enumerate(("short","detailed")):
        styles=torch.full((len(x),),style,dtype=torch.long)
        pred=net.renderer.generate(z,styles).tolist()
        checks=[int(exact(row,list(response(f,style)))) for row,f in zip(pred,triples)]
        slots=[matches_slots(row,f,style) for row,f in zip(pred,triples)]
        results[tag]={
          "exact":sum(checks)/len(checks),
          "exact_count":sum(checks),
          "semantic_slots":sum(sum(a) for a in slots)/(3*len(slots))
        }
        generated[tag]=pred
    both=[int(exact(generated["short"][i],list(response(f,0))) and
              exact(generated["detailed"][i],list(response(f,1))))
          for i,f in enumerate(triples)]
    results["paired_exact"]=sum(both)/len(both)
    return results,generated

def checkpoint(out,name,net,opt,meta):
    p=out/(name+".pt")
    torch.save({
      "student":net.state_dict(),"optimizer":opt.state_dict(),"torch_rng":torch.get_rng_state(),
      "numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
      "seed":SEED,"source_feature_sha256":SOURCE_SHA,"meta":meta
    },p)
    return {"path":p.name,"bytes":p.stat().st_size,
            "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}

def representation_geometry(x):
    y=F.normalize(x,dim=-1)
    norms=x.norm(dim=-1)
    centered=x-x.mean(0,keepdim=True)
    cosine=y@y.T
    n=len(x)
    off=(cosine.sum()-cosine.diag().sum())/(n*(n-1))
    return {
       "mean_pairwise_cosine":float(off),
       "mean_l2_norm":float(norms.mean()),
       "mean_centered_l2":float(centered.norm(dim=-1).mean()),
       "centered_to_full_ratio":float(centered.norm(dim=-1).mean()/norms.mean()),
       "mean_channel_std":float(x.std(0).mean()),
       "full_std":float(x.std()),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results")
    ap.add_argument("--probe-steps",type=int,default=700)
    ap.add_argument("--joint-steps",type=int,default=900)
    args=ap.parse_args()
    torch.set_num_threads(2)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    all_x,train,test,texts=read_features()
    bows=hashed_bow(texts)
    index=len(train)
    result={
      "status":"completed",
      "experiment":"R03 semantic bottleneck pretrained Qwen vs hashed lexical",
      "source_cache_sha256":SOURCE_SHA,
      "train":len(train),"test":len(test),"seed":SEED,
      "representations":{
         "qwen":representation_geometry(all_x),
         "hash_bow":representation_geometry(bows)
      },"probes":{},"joint":{}}
    for name,features in (("qwen",all_x),("hash_bow",bows)):
        training=features[:index]
        holdout=features[index:index+len(test)]
        ood=features[index+len(test):]
        for kind in ("linear","mlp"):
            probe,optim,trace=train_probe(name,kind,training,train,steps=args.probe_steps)
            measurements={
                "train":probe_measure(probe,training,train),
                "heldout":probe_measure(probe,holdout,test),
                "ood_synonyms":probe_measure(probe,ood,test),
                "trace":trace
            }
            measurements["checkpoint"]=checkpoint(out,f"{name}_{kind}_probe",probe,optim,trace)
            result["probes"][name+"_"+kind]=measurements
            print("PROBE",name,kind,measurements["heldout"],measurements["ood_synonyms"],flush=True)
        if name=="qwen":
            shuffled_probe,shuffled_opt,trace=train_probe(name,"mlp",training,train,
                steps=args.probe_steps,shuffled_labels=True)
            result["probes"]["qwen_shuffled_train_labels"]={
              "heldout":probe_measure(shuffled_probe,holdout,test),
              "ood_synonyms":probe_measure(shuffled_probe,ood,test),
              "trace":trace,
              "checkpoint":checkpoint(out,"qwen_shuffled_probe",shuffled_probe,shuffled_opt,trace)
            }
        joint,optim,trace=train_joint(name,training,train,steps=args.joint_steps)
        iid,pred_iid=evaluate_joint(joint,holdout,test)
        sem,pred_ood=evaluate_joint(joint,ood,test)
        zero,_=evaluate_joint(joint,holdout,test,"zero")
        shuffle,_=evaluate_joint(joint,holdout,test,"shuffle")
        cp=checkpoint(out,f"{name}_joint_student",joint,optim,trace)
        copy=SemanticsFirstStudent()
        copy.load_state_dict(torch.load(out/cp["path"],map_location="cpu",weights_only=False)["student"],strict=True)
        iid_reloaded,_=evaluate_joint(copy,holdout,test)
        assert iid_reloaded["paired_exact"]==iid["paired_exact"]
        result["joint"][name]={
           "heldout":iid,"ood_synonyms":sem,
           "zero_latent":zero,"shuffled_latent":shuffle,
           "training":trace,"checkpoint":cp
        }
        result.setdefault("predictions",{})[name]=[
            {"triple":[SUBJECTS[f[0]],ACTIONS[f[1]],OBJECTS[f[2]]],
             "expected_short":show_text(response(f,0)),
             "iid_short":show_text(pred_iid["short"][i]),
             "ood_short":show_text(pred_ood["short"][i]),
             "expected_detailed":show_text(response(f,1)),
             "iid_detailed":show_text(pred_iid["detailed"][i]),
             "ood_detailed":show_text(pred_ood["detailed"][i])}
            for i,f in enumerate(test)]
        print("JOINT",name,"IID",iid,"OOD",sem,flush=True)
    preds=result.pop("predictions")
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    (out/"predictions.json").write_text(json.dumps(preds,indent=2)+"\n")
    print("R03 COMPLETE",flush=True)
if __name__=="__main__":main()
