import collections,json,os,re,sys,torch
from pathlib import Path
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]
G5=ROOT.parent/"g5s_trace"/"src"
sys.path.insert(0,str(G5))
import experiment as ex
from model import Model,Config,ByteTokenizer,EOS

def rows(sp):
    return list(map(json.loads,(ROOT/"data"/f"{sp}.jsonl").read_text().splitlines()))

def mat(z,use_trace=False):
    tok=ByteTokenizer(); tr,a=ex.trace(z["case"])
    assert a==ex.independent_answer(z["case"])==int(z["answer"])
    prompt=z["prompt"]+"\nFinish with F=<integer>.\n"
    target=("T="+tr+";" if use_trace else "")+"F="+str(a)
    pt=tok.encode(prompt,bos=True)
    return {**z,"prompt":prompt,"tokens":pt+tok.encode(target,eos=True),"prompt_tokens":len(pt)}

@torch.inference_mode()
def evaluate(m,source):
    tok=ByteTokenizer(); groups=collections.defaultdict(list); out=[]
    for z in source:
        q=mat(z,False); groups[q["prompt_tokens"]].append(q)
    for g in groups.values():
        for j in range(0,len(g),16):
            b=g[j:j+16]; ids=[tok.encode(z["prompt"],bos=True) for z in b]
            pred=m.generate(torch.tensor(ids),torch.full((len(ids),),len(ids[0])),max_new=32)
            for z,x in zip(b,pred.tolist()):
                eos=EOS in x; x=x[:x.index(EOS)] if eos else x; s=tok.decode(x)
                mm=re.fullmatch(r"F=(-?\d+)",s); p=int(mm.group(1)) if mm else None
                out.append({"id":z["id"],"family":z["family"],"pair_id":z["pair_id"],"generated":s,"eos":eos,"exact":p==int(z["answer"]) if p is not None else False})
    pairs=collections.defaultdict(list)
    for x in out:
        if x["pair_id"]: pairs[x["pair_id"]].append(x)
    return {"n":len(out),"correct":sum(x["exact"] for x in out),"accuracy":sum(x["exact"] for x in out)/len(out),"both_correct_pairs":sum(all(y["exact"] for y in p) for p in pairs.values()),"pair_count":len(pairs),"rows":out}

def save(m,opt,rng,arm,seed,step,logs):
    d=ROOT/"checkpoints"/f"{arm}_s{seed}"/f"step{step:04d}"; d.mkdir(parents=True)
    torch.save({"schema":"G7R_v1","arm":arm,"seed":seed,"step":step,"model":m.state_dict(),"optimizer":opt.state_dict(),"sampler_rng":rng.getstate(),"python_rng":__import__("random").getstate(),"torch_rng":torch.get_rng_state(),"log":logs},d/"resume.pt")
    save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},str(d/"model.safetensors"))
    ex.put(d/"config.json",m.config.__dict__)
    ex.put(d/"MANIFEST.json",{"step":step,"optimizer_rng":True,"files":{n:{"bytes":(d/n).stat().st_size,"sha256":ex.sha(d/n)} for n in ["resume.pt","model.safetensors","config.json"]}})
    return d
