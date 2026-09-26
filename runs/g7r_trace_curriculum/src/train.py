"""Finite preregistered G7R paired trace-curriculum training."""
from __future__ import annotations
import collections,copy,gzip,json,math,random,time
import torch
from safetensors.torch import load_file
from common import ROOT,Model,Config,ByteTokenizer,EOS,ex,rows,mat,evaluate,save
from train_setup import prepare

ARMS=("trace_then_final","final_only")
SEEDS=(7801,7802,7803)
SPLITS=("test_iid","test_surface","test_extrapolation","test_counterfactual")

def lr_for(step:int)->float:
    warm=min(1.0,step/100.0)
    cosine=.15+.85*.5*(1+math.cos(math.pi*min(step,2000)/2000))
    return .001*warm*cosine

def family_counts(result_rows):
    out=collections.defaultdict(lambda:{"n":0,"correct":0})
    for z in result_rows:
        q=out[z["family"]];q["n"]+=1;q["correct"]+=int(z["exact"])
    return {k:{**v,"accuracy":v["correct"]/v["n"]} for k,v in sorted(out.items())}

@torch.inference_mode()
def trace_diagnostic(m,source):
    tok=ByteTokenizer(); groups=collections.defaultdict(list); out=[]
    for z in source:
        q=mat(z,True); groups[q["prompt_tokens"]].append(q)
    start=time.perf_counter()
    for g in groups.values():
        for j in range(0,len(g),16):
            b=g[j:j+16]; ids=[tok.encode(z["prompt"],bos=True) for z in b]
            pred=m.generate(torch.tensor(ids),torch.full((len(ids),),len(ids[0])),max_new=128)
            for z,x in zip(b,pred.tolist()):
                eos=EOS in x;x=x[:x.index(EOS)] if eos else x;s=tok.decode(x)
                tr,a=ex.trace(z["case"]); target=f"T={tr};F={a}"
                out.append({"id":z["id"],"family":z["family"],"generated":s,"eos":eos,"exact_trace":eos and s==target})
    return {"n":len(out),"exact_trace":sum(z["exact_trace"] for z in out),
            "accuracy":sum(z["exact_trace"] for z in out)/len(out),
            "seconds":time.perf_counter()-start,"rows":out}

def profile_supported_flops(m,batch_rows):
    x,y,p=ex.batch(batch_rows)
    mm=Model(Config());mm.load_state_dict(m.state_dict());mm.train()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
        loss=mm(x,y,p);loss.backward()
    flops=sum(int(e.flops or 0) for e in prof.key_averages())
    return {"supported_forward_backward_flops":flops,"batch":len(batch_rows),
            "note":"torch.profiler supported ops only; triangular-solve and other unsupported FLOPs may be omitted"}

def replay_next_update(native,pools,keys):
    hashes=[]
    for _ in range(2):
        m=Model(Config());m.load_state_dict(native["model"])
        opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01)
        opt.load_state_dict(copy.deepcopy(native["optimizer"]))
        rng=random.Random();rng.setstate(native["sampler_rng"])
        random.setstate(native["python_rng"]);torch.set_rng_state(native["torch_rng"])
        key=keys[rng.randrange(len(keys))];bb=rng.choices(pools[False][key],k=16)
        x,y,p=ex.batch(bb);m.train();opt.zero_grad(set_to_none=True)
        loss=m(x,y,p);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        for g in opt.param_groups:g["lr"]=lr_for(2000)
        opt.step();hashes.append(ex.state_hash(m))
    return hashes[0]==hashes[1],hashes[0]

def train_one(arm,seed,checkpoint_hook=None):
    assert arm in ARMS and seed in SEEDS
    out,m,pools,keys,rng,opt,logs,bh,initial_hash=prepare(arm,seed)
    started=time.perf_counter();last_cp=None
    log_path=ROOT/"metrics"/f"{arm}_s{seed}.train.jsonl";log_path.parent.mkdir(parents=True,exist_ok=True)
    for step in range(1,2001):
        trace_phase=(arm=="trace_then_final" and step<=1000)
        key=keys[rng.randrange(len(keys))];bb=rng.choices(pools[trace_phase][key],k=16)
        x,y,p=ex.batch(bb);t0=time.perf_counter();m.train();opt.zero_grad(set_to_none=True)
        loss=m(x,y,p);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):raise RuntimeError("non-finite training")
        lr=lr_for(step)
        for g in opt.param_groups:g["lr"]=lr
        opt.step()
        row={"step":step,"trace_phase":trace_phase,"loss":float(loss.detach()),"lr":lr,"grad_norm":gn,
             "seconds":time.perf_counter()-t0,"case_ids":[z["id"] for z in bb],
             "input_tokens":sum(len(z["tokens"])-1 for z in bb),
             "supervised_tokens":int(y.ne(-100).sum()),"padded_tokens":x.numel()}
        logs.append(row)
        for z in bb:bh.update((z["id"]+"\n").encode())
        with log_path.open("a") as f:f.write(json.dumps(row,separators=(",",":"))+"\n")
        if step%100==0:print("TRAIN",arm,seed,step,round(row["loss"],4),round(row["seconds"],4),flush=True)
        if step%500==0:
            opt.zero_grad(set_to_none=True);last_cp=save(m,opt,rng,arm,seed,step,logs)
            if checkpoint_hook:checkpoint_hook(last_cp,arm,seed,step)
    assert last_cp is not None and last_cp.name=="step2000"
    tests={sp:evaluate(m,rows(sp)) for sp in SPLITS}
    trace_eval=trace_diagnostic(m,rows("test_iid"))
    reloaded=Model(Config());reloaded.load_state_dict(load_file(str(last_cp/"model.safetensors")))
    a=evaluate(m,rows("test_iid")[:16])["rows"];b=evaluate(reloaded,rows("test_iid")[:16])["rows"]
    stable=("id","family","pair_id","generated","eos","exact")
    mismatches=sum(any(x[k]!=y[k] for k in stable) for x,y in zip(a,b))
    native=torch.load(last_cp/"resume.pt",weights_only=False,map_location="cpu")
    replay_ok,replay_hash=replay_next_update(native,pools,keys)
    prof_rows=pools[arm=="trace_then_final"][keys[0]][:16]
    profiler=profile_supported_flops(m,prof_rows)
    result={
        "status":"completed","run":"G7R","arm":arm,"seed":seed,"parameters":sum(p.numel() for p in m.parameters()),
        "steps":2000,"trace_phase_steps":1000 if arm=="trace_then_final" else 0,
        "initial_state_sha256":initial_hash,"final_state_sha256":ex.state_hash(m),
        "batch_sequence_sha256":bh.hexdigest(),"train_seconds":sum(z["seconds"] for z in logs),
        "elapsed_seconds":time.perf_counter()-started,
        "input_tokens":sum(z["input_tokens"] for z in logs),
        "supervised_tokens":sum(z["supervised_tokens"] for z in logs),
        "padded_tokens":sum(z["padded_tokens"] for z in logs),
        "checkpoint":str(last_cp),"checkpoint_manifest":json.loads((last_cp/"MANIFEST.json").read_text()),
        "reload_cases":16,"reload_mismatches":mismatches,
        "resume_next_update_verified":replay_ok,"resume_replay_state_sha256":replay_hash,
        "profiler":profiler,
        "tests":tests,"trace_diagnostic_iid":trace_eval,
        "family_iid":family_counts(tests["test_iid"]["rows"])
    }
    ex.put(out,result)
    raw=ROOT/"metrics"/f"{arm}_s{seed}.RAW.json.gz"
    raw.write_bytes(gzip.compress(json.dumps(result,separators=(",",":")).encode(),mtime=0))
    print("COMPLETE",arm,seed,{s:tests[s]["correct"] for s in SPLITS},"trace",trace_eval["exact_trace"],flush=True)
    return result

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=ARMS,required=True);ap.add_argument("--seed",type=int,choices=SEEDS,required=True)
    a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);train_one(a.arm,a.seed)
