"""R04-H1c: preregistered frozen-feature recurrent vs no-recurrence readouts.

This TRAINs a tiny task-specific solver. Positive performance is evidence about
additional compute in the adapter, NOT evidence that Qwen precomputed an answer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from safetensors import safe_open

HERE=Path(__file__).resolve()
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"h1"/"src"))
from h1_probes import lexical_features,structured_features
from dataset import load,file_sha
from latent_reasoner import LatentReasoner,two_digit_loss

SEEDS=(11,22,33)
INPUTS=("qwen_last","hash","structured")
DEPTHS=(0,2,8)
CONFIDENCE_THRESHOLDS=(.55,.65,.75,.85,.95)


def evaluate(model:LatentReasoner,x:torch.Tensor,y:torch.Tensor,steps:int,indices:list[int],
             *,device:str="cpu",full_predictions:bool=False):
    model.eval()
    p=[]
    with torch.no_grad():
        for i in range(0,len(indices),128):
            block=indices[i:i+128]
            q=model(x[block].to(device),steps).argmax(-1).cpu()
            p.append(q)
    pred=torch.cat(p)
    gold=y[indices]
    correct=(pred==gold)
    exact=correct.all(-1)
    metrics={"exact":float(exact.float().mean()),"exact_count":int(exact.sum()),
             "n":len(indices),"digit0":float(correct[:,0].float().mean()),
             "digit1":float(correct[:,1].float().mean())}
    return (metrics,pred.tolist()) if full_predictions else metrics


def fixed_seed(seed:int)->None:
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def train_one(name:str,depth:int,seed:int,x:torch.Tensor,y:torch.Tensor,split:dict,
              output_root:Path,steps:int,batch:int,lr:float,device:str,shuffle_labels:bool=False):
    fixed_seed(seed)
    model=LatentReasoner(x.shape[1],hidden=96).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    indices=torch.tensor(split["train"],dtype=torch.long)
    xdev=x.to(device)
    labels=y.clone()
    if shuffle_labels:
        # fixed, disjoint permutation of train labels ONLY
        perm=torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed+943))
        labels[indices]=labels[indices[perm]]
    ydev=labels.to(device)
    ridx=torch.Generator().manual_seed(seed+67)
    t=time.perf_counter()
    traces=[]
    for step in range(steps):
        positions=torch.randint(0,len(indices),(batch,),generator=ridx)
        selected=indices[positions].to(device)
        logits=model(xdev[selected],depth)
        loss=two_digit_loss(logits,ydev[selected])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        if (step+1)%max(1,steps//4)==0:
            traces.append({"step":step+1,"loss":float(loss.detach())})
    if device=="cuda":torch.cuda.synchronize()
    duration=time.perf_counter()-t
    model=model.to("cpu").eval()
    val=evaluate(model,x,y,depth,split["val"])
    slug=f"{name}_T{depth}_s{seed}"+("_shuffled" if shuffle_labels else "")
    cp=output_root/"checkpoints"/f"{slug}.pt"
    cp.parent.mkdir(parents=True,exist_ok=True)
    # Native complete optimizer state, random generators, and model tensor bytes.
    native_opt=optimizer.state_dict()
    for st in native_opt["state"].values():
        for k,v in st.items():
            if isinstance(v,torch.Tensor):st[k]=v.detach().cpu()
    torch.save({"model":model.state_dict(),"optimizer":native_opt,
        "rng_torch":torch.get_rng_state(),"rng_numpy":np.random.get_state(),
        "rng_python":random.getstate(),
        "rng_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "source":name,"depth":depth,"seed":seed,"shuffled_labels":shuffle_labels,
        "hidden":96,"features_dim":x.shape[1],"train_steps":steps,"batch":batch,"lr":lr},cp)
    if cp.stat().st_size>=90_000_000:raise RuntimeError("Checkpoint exceeds Git per-file ceiling")
    info={"source":name,"depth":depth,"seed":seed,"shuffled_labels":shuffle_labels,
          "val":val,"train_seconds":duration,"loss_trace":traces,
          "parameters":sum(p.numel() for p in model.parameters()),
          "checkpoint":{"path":cp.relative_to(output_root).as_posix(),
                        "bytes":cp.stat().st_size,"sha256":file_sha(cp)}}
    del xdev,ydev,optimizer
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return model,info


def adaptive_measure(model,x,y,indices,threshold:float):
    model.eval()
    all_pred=[];all_steps=[]
    with torch.inference_mode():
        for i in range(0,len(indices),128):
            ids=indices[i:i+128]
            pred,depth=model.adaptive(x[ids],threshold)
            all_pred.append(pred.argmax(-1).cpu())
            all_steps.append(depth.cpu())
    predictions=torch.cat(all_pred)
    steps=torch.cat(all_steps)
    matching=predictions.eq(y[indices]).all(-1)
    counts={str(t):int((steps==t).sum()) for t in (0,2,4,8)}
    return {"exact":float(matching.float().mean()),"exact_count":int(matching.sum()),
            "mean_steps":float(steps.float().mean()),"n":len(indices),"stop_counts":counts},predictions.tolist()


def restore_model(output_root:Path,record:dict):
    p=output_root/record["checkpoint"]["path"]
    if file_sha(p)!=record["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint SHA mismatch: "+str(p))
    raw=torch.load(p,map_location="cpu",weights_only=False)
    net=LatentReasoner(raw["features_dim"],raw["hidden"])
    net.load_state_dict(raw["model"],strict=True)
    if "optimizer" not in raw or "rng_torch" not in raw:
        raise RuntimeError("Incomplete native training checkpoint")
    return net.eval()


def load_teacher_features(path:Path,manifest:Path,expected_dataset_sha:str):
    receipt=json.loads(manifest.read_text(encoding="utf-8"))
    if receipt["dataset_sha256"]!=expected_dataset_sha or receipt["features_sha256"]!=file_sha(path):
        raise RuntimeError("Frozen teacher cache does not match dataset or SHA")
    with safe_open(str(path),framework="pt",device="cpu") as fd:
        metadata=fd.metadata()
        if metadata["dataset_sha256"]!=expected_dataset_sha:
            raise RuntimeError("Safetensors metadata mismatch")
        z=fd.get_tensor("last")
    if z.shape!=(1700,2560) or z.dtype!=torch.bfloat16 or not torch.isfinite(z).all():
        raise RuntimeError("Unexpected pretrained teacher feature cache")
    return z.float().contiguous(),receipt


def main():
    p=argparse.ArgumentParser()
    root=HERE.parents[1]
    p.add_argument("--data",type=Path,default=root/"data")
    p.add_argument("--cache",type=Path,default=root/"cache")
    p.add_argument("--out",type=Path,default=root/"results")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--batch",type=int,default=96)
    p.add_argument("--lr",type=float,default=.0015)
    args=p.parse_args()
    if (args.steps,args.batch,args.lr)!=(600,96,.0015):
        raise RuntimeError("H1c is preregistered at 600 steps/batch96/lr.0015; use a separate experiment for alternatives")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(min(8,torch.get_num_threads()))
    rows,ds=load(args.data)
    qwen,teacher_receipt=load_teacher_features(args.cache/"h1c_last_features.safetensors",
        args.cache/"feature_manifest.json",ds["dataset_sha256_canonical"])
    features={"qwen_last":qwen,"hash":lexical_features(rows),
              "structured":structured_features(rows)}
    target=torch.tensor([r["digits"] for r in rows],dtype=torch.long)
    split={s:[i for i,r in enumerate(rows) if r["split"]==s] for s in ("train","val","test")}
    if [len(split[x]) for x in ("train","val","test")] != [1200,200,300]:
        raise RuntimeError("H1c data splits invalid")
    output=args.out;output.mkdir(parents=True,exist_ok=True)
    records=[];models={}
    for name in INPUTS:
        for depth in DEPTHS:
            for seed in SEEDS:
                net,rec=train_one(name,depth,seed,features[name],target,split,
                                  output,args.steps,args.batch,args.lr,args.device)
                records.append(rec)
                models[(name,depth,seed)]=net
                print("completed train",name,depth,seed,"val exact",rec["val"]["exact"],flush=True)
    # Exactly preregistered negative label-shuffle control, no posthoc selection.
    for seed in SEEDS:
        net,rec=train_one("qwen_last",8,seed,features["qwen_last"],target,split,
                          output,args.steps,args.batch,args.lr,args.device,shuffle_labels=True)
        records.append(rec)
        models[("qwen_last_shuffle",8,seed)]=net

    lookup={(r["source"]+("_shuffle" if r["shuffled_labels"] else ""),r["depth"],r["seed"]):r for r in records}
    predictions=[]
    for rec in records:
        net=restore_model(output,rec)
        input_name=rec["source"]
        m,pred=evaluate(net,features[input_name],target,rec["depth"],split["test"],full_predictions=True)
        rec["test"]=m
        for k,(i,guess) in enumerate(zip(split["test"],pred)):
            predictions.append({"id":rows[i]["id"],"source":input_name,"depth":rec["depth"],
                 "seed":rec["seed"],"shuffle":rec["shuffled_labels"],
                 "gold":target[i].tolist(),"prediction":guess,"exact":guess==target[i].tolist()})
    gates={}
    for seed in SEEDS:
        q0=lookup[("qwen_last",0,seed)]["test"]["exact"]
        q8=lookup[("qwen_last",8,seed)]["test"]["exact"]
        input0=max(lookup[(x,0,seed)]["test"]["exact"] for x in ("hash","structured"))
        gates[str(seed)]={"qwen_T0":q0,"qwen_T8":q8,"best_input_T0":input0,
            "C0_readout":bool(q0>=.4 and q0-input0>=.15),
            "C1_recurrence":bool(q8>=.30 and q8-q0>=.10),
            "label_shuffle_T8":lookup[("qwen_last_shuffle",8,seed)]["test"]["exact"]}
    conclusion={"C0":sum(x["C0_readout"] for x in gates.values())>=2,
                "C1":sum(x["C1_recurrence"] for x in gates.values())>=2}

    readiness={}
    for seed in SEEDS:
        net=models[("qwen_last",8,seed)].cpu().eval()
        fixed_val=lookup[("qwen_last",8,seed)]["val"]["exact"]
        calibrations=[]
        for threshold in CONFIDENCE_THRESHOLDS:
            v,_=adaptive_measure(net,features["qwen_last"],target,split["val"],threshold)
            calibrations.append({"threshold":threshold,**v})
        eligible=[c for c in calibrations if c["exact"]>=fixed_val-.03]
        selected=sorted(eligible,key=lambda a:(a["mean_steps"],-a["threshold"]))[0] if eligible else calibrations[-1]
        fresh,preds=adaptive_measure(net,features["qwen_last"],target,split["test"],selected["threshold"])
        full8=lookup[("qwen_last",8,seed)]["test"]["exact"]
        readiness[str(seed)]={"threshold_selected_using_val_only":selected["threshold"],
            "validation_calibration":calibrations,"test":fresh,"test_fixed8":full8,
            "C2_simulated_computation_steps":bool(fresh["mean_steps"]<=4 and fresh["exact"]>=full8-.03),
            "note":"SIMULATED branching: trajectory computed to 8 internally; no runtime savings measured"}
        for idx,guess in zip(split["test"],preds):
            predictions.append({"id":rows[idx]["id"],"source":"qwen_adaptive",
               "seed":seed,"threshold":selected["threshold"],"gold":target[idx].tolist(),
               "prediction":guess,"exact":guess==target[idx].tolist()})
    conclusion["C2_simulated"] = sum(v["C2_simulated_computation_steps"] for v in readiness.values())>=2
    with (output/"predictions.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for row in predictions:f.write(json.dumps(row,sort_keys=True)+"\n")
    report={"status":"completed_diagnostic_only_not_full_llm_acceleration",
        "dataset_sha256":ds["dataset_sha256_canonical"],
        "features_sha256":teacher_receipt["features_sha256"],
        "locked_teacher_revision":"c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97",
        "train_count":1200,"validation_count":200,"test_count":300,
        "methods":list(INPUTS),"depths":list(DEPTHS),"seeds":list(SEEDS),
        "records":records,"gate_by_seed":gates,"readiness":readiness,"gates_passed":conclusion,
        "crucial_limitation":"Adapter is a supervised NEW solver; improvement does not prove Qwen had calculated answers during prefill",
        "no_frozen_writer_run":True,"no_total_speed_claim":True,
        "checkpoint_bytes_verified":True,"test_predictions_sha256":file_sha(output/"predictions.jsonl")}
    (output/"metrics.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("H1c conclusion (separate from original H1):",conclusion,flush=True)

if __name__=="__main__":main()