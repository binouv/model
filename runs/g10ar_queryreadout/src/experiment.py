from __future__ import annotations
import collections, copy, csv, hashlib, json, math, os, random, re, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"base"))
from model import Model,Config,ByteTokenizer,EOS

ARMS=("query_readout_attention","token_mlp_control")
SEEDS=(7601,7602,7603)
KEYS=tuple("wxyz")
SPLITS=("iid","surface","extrapolation")
PARENT_TAG="flygraph-g8c-consolidation-36238890991"
PARENT_SHA={
 7601:"8dc94305108b1dd97f68f4d4b3d54c95e34e380847a2b04752b90eb8bcd1164e",
 7602:"f4d5239f1e59c94f64666d7528f655efa1a1fdf7829b0826604297fb481fe69a",
 7603:"195ec50d85347cd124d09bfb65ef4f848075da7755d069c4d177f3057c4daa4c",
}
DATA_SPEC={
 "train":(2048,92301,0,0),
 "validation":(128,92302,0,0),
 "iid":(200,92303,0,0),
 "surface":(100,92304,0,1),
 "extrapolation":(100,92305,1,1),
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def put(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp")
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def read_jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def state_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def trainable_flat_hash(m):
 h=hashlib.sha256()
 for p in m.adapter.parameters():h.update(p.detach().cpu().contiguous().view(-1).numpy().tobytes())
 return h.hexdigest()
def current_values(writes):
 d={}
 for k,v in writes:d[k]=v
 return d
def canonical_history(writes):
 histories={k:[] for k in KEYS}
 for k,v in writes:histories[k].append(v)
 return stable({"keys":KEYS,"histories":histories})
def make_group(rng,ood=False):
 lo,hi=(128,255) if ood else (0,127)
 writes=[[k,rng.randint(lo,hi)] for k in KEYS]
 for _ in range(8 if ood else 4):writes.append([rng.choice(KEYS),rng.randint(lo,hi)])
 rng.shuffle(writes);mem=current_values(writes);qs=list(KEYS);rng.shuffle(qs);pair=None
 for i in range(len(qs)):
  for j in range(i+1,len(qs)):
   if mem[qs[i]]!=mem[qs[j]]:pair=[qs[i],qs[j]];break
  if pair:break
 if pair is None:return None
 return {"writes":writes,"queries":pair,"answers":[mem[pair[0]],mem[pair[1]]],"canonical":canonical_history(writes)}
def render(group,query,variant=0):
 s="; ".join(f"{k}={v}" for k,v in group["writes"])
 if variant==0:return f"Writes in time order: {s}. Latest value of {query}?\nAnswer:"
 return f"Apply these updates chronologically: {s}. Read current {query}.\nAnswer:"
def group_rows(group,variant=0,split=""):
 tok=ByteTokenizer();pid=stable({"split":split,"canonical":group["canonical"]})[:20];out=[]
 for idx,(q,a) in enumerate(zip(group["queries"],group["answers"])):
  prompt=render(group,q,variant)+"\nFinish with F=<integer>.\n";pt=tok.encode(prompt,bos=True);target=tok.encode("F="+str(a),eos=True)
  out.append({"id":stable({"pair":pid,"query":q}),"pair_id":pid,"query":q,"gold":int(a),"other_gold":int(group["answers"][1-idx]),
   "prompt":prompt,"prompt_tokens":len(pt),"tokens":pt+target,"canonical":group["canonical"],"family":"memory_update"})
 return out
def make_mixed(rng,n=200):
 out=[];kinds=("arithmetic","conditional","code_trace","list_reasoning")
 for i in range(n):
  fam=kinds[i%4]
  if fam=="arithmetic":
   a,b=rng.randint(256,511),rng.randint(256,511);op=rng.choice("+-*");ans=a+b if op=="+" else a-b if op=="-" else a*b
   prompt=f"What integer is {a} {op} {b}?\nAnswer:";sem={"family":fam,"values":sorted([a,b]) if op in "+*" else [a,b],"op":op}
  elif fam=="conditional":
   x,k,a,b=[rng.randint(256,511) for _ in range(4)];ans=x+a if x<k else x-b
   prompt=f"Start at {x}. Below {k}: +{a}. Otherwise: -{b}.\nAnswer:";sem={"family":fam,"values":[x,k,a,b]}
  elif fam=="code_trace":
   start=rng.randint(256,511);program=[[rng.choice("+-"),rng.randint(16,63)] for _ in range(3)];val=start
   for op,a in program:val=val+a if op=="+" else val-a
   ans=val;body="x = "+str(start)+"\n"+"".join(f"x {op}= {a}\n" for op,a in program)+"print(x)"
   prompt="Trace this program and return only the printed integer.\n"+body+"\nAnswer:";sem={"family":fam,"start":start,"program":program}
  else:
   vals=[rng.randint(256,511) for _ in range(6)];kind=rng.choice(("sum","min","max"));ans=sum(vals) if kind=="sum" else min(vals) if kind=="min" else max(vals)
   lab="minimum" if kind=="min" else "maximum" if kind=="max" else "sum";prompt=f"Numbers: {vals}. Return their {lab}.\nAnswer:";sem={"family":fam,"values":sorted(vals),"kind":kind}
  prompt+="\nFinish with F=<integer>.\n";tok=ByteTokenizer();pt=tok.encode(prompt,bos=True);target=tok.encode("F="+str(ans),eos=True)
  out.append({"id":stable({"mixed":sem}),"pair_id":None,"query":None,"gold":int(ans),"other_gold":None,"prompt":prompt,"prompt_tokens":len(pt),"tokens":pt+target,"canonical":stable(sem),"family":fam})
 assert len({z["canonical"] for z in out})==len(out);return out
def prepare_data():
 d=ROOT/"data";d.mkdir(parents=True,exist_ok=True);occupied=set();manifest={}
 for split,(n,seed,ood,variant) in DATA_SPEC.items():
  rng=random.Random(seed);groups=[];attempts=0
  while len(groups)<n:
   attempts+=1
   if attempts>n*1000:raise RuntimeError("unique data exhausted")
   g=make_group(rng,bool(ood))
   if g is None or g["canonical"] in occupied:continue
   rr=group_rows(g,variant,split)
   if max(len(z["tokens"]) for z in rr)>256:continue
   assert rr[0]["gold"]!=rr[1]["gold"];groups.append(g);occupied.add(g["canonical"])
  p=d/f"{split}.jsonl";p.write_text("".join(json.dumps(g,separators=(",",":"))+"\n" for g in groups))
  manifest[split]={"groups":n,"cases":2*n,"attempts":attempts,"sha256":sha(p)}
 mixed=make_mixed(random.Random(92306),200);p=d/"mixed.jsonl";p.write_text("".join(json.dumps(z,separators=(",",":"))+"\n" for z in mixed));manifest["mixed"]={"cases":200,"sha256":sha(p)}
 allg=[g for split in DATA_SPEC for g in read_jsonl(d/f"{split}.jsonl")]
 assert len({g["canonical"] for g in allg})==len(allg)
 audit={"status":"completed_pretraining_data_audit","memory_key_space":list(KEYS),"prior_memory_key_space":list("abcd"),
  "structurally_disjoint_from_G5S_G8C_G9QR_memory":True,"query_invariant_groups":len(allg),"cross_split_overlap":0,
  "per_key_chronology_preserved":True,"cross_key_permutations_canonicalized":True,"mixed_value_range":[256,511],"g8c_known_nonmemory_max_value":255,"manifests":manifest}
 put(ROOT/"reports/DATA_AUDIT.json",audit);return audit

class QueryReadout(nn.Module):
 def __init__(self,dim=96,heads=4):
  super().__init__();self.dim=dim;self.heads=heads;self.hd=dim//heads
  self.q=nn.Linear(dim,dim,bias=False);self.k=nn.Linear(dim,dim,bias=False);self.v=nn.Linear(dim,dim,bias=False);self.o=nn.Linear(dim,dim,bias=False);self.gate=nn.Parameter(torch.zeros(()))
 def forward(self,x,cache=None):
  B,T,D=x.shape;H=self.heads;hd=self.hd
  q=self.q(x).view(B,T,H,hd).transpose(1,2);k=self.k(x).view(B,T,H,hd).transpose(1,2);v=self.v(x).view(B,T,H,hd).transpose(1,2)
  if cache is not None:k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2);y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
  else:y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  y=self.o(y.transpose(1,2).reshape(B,T,D));return x+torch.tanh(self.gate)*y,(k,v)
class TokenMLP(nn.Module):
 def __init__(self,dim=96):
  super().__init__();self.up=nn.Linear(dim,192,bias=False);self.down=nn.Linear(192,dim,bias=False);self.gate=nn.Parameter(torch.zeros(()))
 def forward(self,x,cache=None):return x+torch.tanh(self.gate)*self.down(F.silu(self.up(x))),None
def init_adapter(adapter,seed):
 gen=torch.Generator(device="cpu");gen.manual_seed(101000+seed)
 with torch.no_grad():
  for name,p in adapter.named_parameters():
   if name=="gate":p.zero_()
   else:p.copy_(torch.randn(p.shape,generator=gen,dtype=p.dtype)*.02)
class AdapterLM(nn.Module):
 def __init__(self,base,arm,seed):
  super().__init__();self.base=base;self.arm=arm
  for p in self.base.parameters():p.requires_grad_(False)
  self.adapter=QueryReadout(base.config.dim,base.config.heads) if arm==ARMS[0] else TokenMLP(base.config.dim);init_adapter(self.adapter,seed)
 def base_hidden(self,ids,cache=None):
  with torch.no_grad():
   x=self.base.embed(ids);new=[]
   for i,b in enumerate(self.base.blocks):x,s=b(x,None if cache is None else cache[i]);new.append(s)
   x=self.base.norm(x)
  return x,new
 def forward_logits(self,ids,base_cache=None,adapter_cache=None):
  x,bc=self.base_hidden(ids,base_cache);y,ac=self.adapter(x,adapter_cache);return F.linear(y,self.base.embed.weight.detach()),bc,ac
 def loss(self,x,y):
  logits,_,_=self.forward_logits(x);keep=y.ne(-100);return F.cross_entropy(logits[keep],y[keep])
 @torch.inference_mode()
 def generate(self,ids,max_new=32):
  self.eval();logits,bc,ac=self.forward_logits(ids);out=[];done=torch.zeros(ids.shape[0],dtype=torch.bool)
  for _ in range(max_new):
   z=logits[:,-1].argmax(-1);z=torch.where(done,torch.full_like(z,EOS),z);out.append(z);done|=z.eq(EOS)
   if bool(done.all()):break
   logits,bc,ac=self.forward_logits(z[:,None],bc,ac)
  return torch.stack(out,1)
def report_model(m):
 return {"total_parameters":sum(p.numel() for p in m.parameters()),"active_trainable_parameters":sum(p.numel() for p in m.parameters() if p.requires_grad),
  "base_parameters":sum(p.numel() for p in m.base.parameters()),"adapter_parameters":sum(p.numel() for p in m.adapter.parameters()),"arm":m.arm}
def rows_for_split(split):
 if split=="mixed":return read_jsonl(ROOT/"data/mixed.jsonl")
 variant=1 if split in ("surface","extrapolation") else 0
 return [z for g in read_jsonl(ROOT/f"data/{split}.jsonl") for z in group_rows(g,variant,split)]
def train_groups():return read_jsonl(ROOT/"data/train.jsonl")
def make_batch(rows):
 n=max(len(z["tokens"]) for z in rows)-1;x=torch.zeros((len(rows),n),dtype=torch.long);y=torch.full_like(x,-100)
 for i,z in enumerate(rows):
  t=torch.tensor(z["tokens"]);end=len(t)-1;x[i,:end]=t[:-1];y[i,z["prompt_tokens"]-1:end]=t[z["prompt_tokens"]:]
 return x,y
def strict_parse(text):
 m=re.fullmatch(r"F=(-?\d+)",text.strip());return int(m.group(1)) if m else None
@torch.inference_mode()
def evaluate(m,rows,paired=False):
 tok=ByteTokenizer();groups=collections.defaultdict(list)
 for z in rows:groups[z["pair_id"] or z["id"]].append(z)
 outs=[];seconds=0.
 for _,gg in sorted(groups.items()):
  for start in range(0,len(gg),16):
   bb=gg[start:start+16];seq=[tok.encode(z["prompt"],bos=True) for z in bb]
   batches=[bb] if len({len(s) for s in seq})==1 else [[z] for z in bb]
   for batch in batches:
    ss=[tok.encode(z["prompt"],bos=True) for z in batch];t0=time.perf_counter();pred=m.generate(torch.tensor(ss),32).tolist();seconds+=time.perf_counter()-t0
    for z,ids in zip(batch,pred):
     eos=EOS in ids;ids=ids[:ids.index(EOS)] if eos else ids;text=tok.decode(ids);val=strict_parse(text)
     outs.append({**{k:z[k] for k in ("id","pair_id","query","gold","other_gold","family")},"text":text,"ids":ids,"eos":eos,"parsed":val,"exact":eos and val==z["gold"],
      "other_confusion":bool(z["other_gold"] is not None and eos and val==z["other_gold"])})
 result={"n":len(outs),"correct":sum(z["exact"] for z in outs),"accuracy":sum(z["exact"] for z in outs)/len(outs),"truncated":sum(not z["eos"] for z in outs),
  "seconds":seconds,"generated_bytes":sum(len(z["ids"]) for z in outs),"rows":sorted(outs,key=lambda z:z["id"])}
 if paired:
  pp=collections.defaultdict(list)
  for z in outs:pp[z["pair_id"]].append(z)
  assert all(len(v)==2 for v in pp.values())
  result.update({"pair_n":len(pp),"both_correct_pairs":sum(all(x["exact"] for x in v) for v in pp.values()),
   "query_sensitive_pairs":sum(v[0]["parsed"] is not None and v[1]["parsed"] is not None and v[0]["parsed"]!=v[1]["parsed"] for v in pp.values()),
   "other_key_confusions":sum(x["other_confusion"] for v in pp.values() for x in v)})
 return result
@torch.inference_mode()
def teacher_forced_nll(m,rows):
 total=0.;count=0
 for i in range(0,len(rows),16):
  x,y=make_batch(rows[i:i+16]);logits,_,_=m.forward_logits(x);keep=y.ne(-100);total+=float(F.cross_entropy(logits[keep],y[keep],reduction="sum"));count+=int(keep.sum())
 return {"loss_sum":total,"tokens":count,"nll_per_token":total/count}
def update(m,opt,rows,lr):
 x,y=make_batch(rows);m.train();opt.zero_grad(set_to_none=True)
 for g in opt.param_groups:g["lr"]=lr
 t=time.perf_counter();loss=m.loss(x,y);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.adapter.parameters(),1.))
 if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):raise ValueError("nonfinite")
 opt.step();return {"loss":float(loss.detach()),"grad_norm":gn,"lr":lr,"seconds":time.perf_counter()-t,"input_tokens":int(x.ne(0).sum()),
  "supervised_tokens":int(y.ne(-100).sum()),"padded_tokens":x.numel(),"ids":[z["id"] for z in rows]}
def make_model_from_parent(parent_file,arm,seed):
 if sha(parent_file)!=PARENT_SHA[seed]:raise ValueError("parent SHA256 mismatch")
 st=torch.load(parent_file,weights_only=False,map_location="cpu")
 if st.get("schema")!="G8C_v1" or st.get("seed")!=seed or st.get("step")!=2000 or st.get("arm")!="verified_trace":raise ValueError("wrong parent")
 base=Model(Config(**st["config"]));base.load_state_dict(st["model"]);m=AdapterLM(base,arm,seed);r=report_model(m)
 assert (r["base_parameters"],r["adapter_parameters"],r["total_parameters"],r["active_trainable_parameters"])==(469648,36865,506513,36865)
 return m
def save_checkpoint(m,opt,rng,arm,seed,step,logs,parent_file):
 dest=ROOT/"checkpoints"/f"{arm}_s{seed}"/f"step{step:04d}";dest.mkdir(parents=True,exist_ok=False)
 native={"schema":"G10AR_v1","arm":arm,"seed":seed,"step":step,"parent_release_tag":PARENT_TAG,"parent_checkpoint_sha256":sha(parent_file),
  "base_config":m.base.config.__dict__,"model":m.state_dict(),"optimizer":opt.state_dict(),"sampler_rng":rng.getstate(),"python_rng":random.getstate(),
  "torch_rng":torch.get_rng_state(),"log":logs,"preregistered_sha256":sha(ROOT/"configs/preregistered.json")}
 torch.save(native,dest/"resume.pt");save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},str(dest/"model.safetensors"))
 put(dest/"config.json",{"arm":arm,"seed":seed,"base_config":m.base.config.__dict__,"schema":"G10AR_v1"})
 files={n:{"bytes":(dest/n).stat().st_size,"sha256":sha(dest/n)} for n in ("resume.pt","model.safetensors","config.json")}
 put(dest/"MANIFEST.json",{"step":step,**report_model(m),"files":files,"optimizer_rng":True});(dest/"COMPLETE").write_text("complete\n");return dest
def restore_native(path):
 st=torch.load(path,weights_only=False,map_location="cpu");base=Model(Config(**st["base_config"]));m=AdapterLM(base,st["arm"],st["seed"]);m.load_state_dict(st["model"])
 opt=torch.optim.AdamW(m.adapter.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);opt.load_state_dict(copy.deepcopy(st["optimizer"]))
 rng=random.Random();rng.setstate(st["sampler_rng"]);random.setstate(st["python_rng"]);torch.set_rng_state(st["torch_rng"]);return m,opt,rng,st
def profiler_flops(m,rows):
 x,y=make_batch(rows);m.zero_grad(set_to_none=True)
 try:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
   loss=m.loss(x,y);loss.backward()
  flops=sum(int(getattr(e,"flops",0) or 0) for e in prof.key_averages());m.zero_grad(set_to_none=True)
  return {"supported_ops_train_batch_flops":flops,"batch_prompts":len(rows),"caveat":"torch.profiler supports only a subset of operations"}
 except Exception as e:
  m.zero_grad(set_to_none=True);return {"supported_ops_train_batch_flops":None,"error":type(e).__name__}
def train_one(arm,seed,parent_file,callback=None):
 out=ROOT/"metrics"/f"{arm}_s{seed}.json"
 if out.exists():raise FileExistsError("refuse duplicate completed model")
 torch.manual_seed(seed);random.seed(seed);m=make_model_from_parent(parent_file,arm,seed);init_flat=trainable_flat_hash(m)
 opt=torch.optim.AdamW(m.adapter.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);rng=random.Random(92311);groups=train_groups();logs=[];seqhash=hashlib.sha256()
 validation=[z for g in read_jsonl(ROOT/"data/validation.jsonl") for z in group_rows(g,0,"validation")];before={"validation_nll":teacher_forced_nll(m,validation)}
 for step in range(1,201):
  chosen=rng.choices(groups,k=8);rows=[z for g in chosen for z in group_rows(g,0,"train")];lr=.001*(.1+.9*.5*(1+math.cos(math.pi*step/200)))
  rec=update(m,opt,rows,lr);rec["step"]=step;logs.append(rec)
  for z in chosen:seqhash.update((z["canonical"]+"\n").encode())
  with (ROOT/"metrics"/f"{arm}_s{seed}.train.jsonl").open("a") as f:f.write(json.dumps(rec)+"\n")
  if step in (100,200):
   cp=save_checkpoint(m,opt,rng,arm,seed,step,logs,parent_file)
   if callback:callback("checkpoint",cp,arm,seed,step,None)
 held={sp:evaluate(m,rows_for_split(sp),True) for sp in SPLITS};mixed=evaluate(m,rows_for_split("mixed"),False)
 diagnostics={"validation_nll":teacher_forced_nll(m,validation),"iid_nll":teacher_forced_nll(m,rows_for_split("iid")),
  "profiler":profiler_flops(m,[z for g in groups[:8] for z in group_rows(g,0,"train")])}
 cfg=json.loads((cp/"config.json").read_text());base=Model(Config(**cfg["base_config"]));reload_m=AdapterLM(base,arm,seed);reload_m.load_state_dict(load_file(str(cp/"model.safetensors")))
 a=evaluate(m,rows_for_split("iid")[:16],True)["rows"];b=evaluate(reload_m,rows_for_split("iid")[:16],True)["rows"]
 assert [{k:z[k] for k in ("id","text","ids","eos","exact")} for z in a]==[{k:z[k] for k in ("id","text","ids","eos","exact")} for z in b]
 rh=[]
 for _ in range(2):
  mm,oo,rr,_=restore_native(cp/"resume.pt");chosen=rr.choices(groups,k=8);rows=[z for g in chosen for z in group_rows(g,0,"train")];update(mm,oo,rows,.0001);rh.append(state_hash(mm))
 assert rh[0]==rh[1] and all(bool(torch.isfinite(p).all()) for p in m.parameters())
 result={"status":"completed","arm":arm,"seed":seed,**report_model(m),"updates":200,"parent_checkpoint_sha256":sha(parent_file),"adapter_initial_flat_sha256":init_flat,
  "final_state_sha256":state_hash(m),"batch_group_sequence_sha256":seqhash.hexdigest(),"train_seconds":sum(x["seconds"] for x in logs),
  "input_tokens":sum(x["input_tokens"] for x in logs),"supervised_tokens":sum(x["supervised_tokens"] for x in logs),"padded_tokens":sum(x["padded_tokens"] for x in logs),
  "reload_generation_cases":16,"reload_generation_mismatches":0,"resume_next_update_verified":True,"checkpoint_manifest":json.loads((cp/"MANIFEST.json").read_text()),
  "before":before,"held":held,"mixed":mixed,"diagnostics":diagnostics}
 put(out,result);raw=ROOT/"metrics"/f"{arm}_s{seed}.raw.jsonl"
 with raw.open("w") as f:
  for sp,t in held.items():
   for z in t["rows"]:f.write(json.dumps({"split":sp,**z},separators=(",",":"))+"\n")
  for z in mixed["rows"]:f.write(json.dumps({"split":"mixed",**z},separators=(",",":"))+"\n")
 if callback:callback("completed",cp,arm,seed,200,result)
 return result
def pair_stats(a,b):
 aa={z["id"]:z for z in a};bb={z["id"]:z for z in b};assert set(aa)==set(bb);w=sum(aa[k]["exact"] and not bb[k]["exact"] for k in aa);l=sum(bb[k]["exact"] and not aa[k]["exact"] for k in aa);n=w+l
 p=min(1.,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1.;return {"wins":w,"losses":l,"delta_pp":100*(w-l)/len(aa),"exact_two_sided_p":p}
def aggregate():
 runs={(a,s):json.loads((ROOT/"metrics"/f"{a}_s{s}.json").read_text()) for a in ARMS for s in SEEDS}
 assert all(r["status"]=="completed" for r in runs.values()) and len({r["batch_group_sequence_sha256"] for r in runs.values()})==1
 out={}
 for arm in ARMS:
  rs=[runs[arm,s] for s in SEEDS];out[arm]={}
  for sp in SPLITS:
   out[arm][sp]={"mean_individual_accuracy":sum(r["held"][sp]["accuracy"] for r in rs)/3,"individual_correct_by_seed":[r["held"][sp]["correct"] for r in rs],
    "pair_n":rs[0]["held"][sp]["pair_n"],"both_correct_by_seed":[r["held"][sp]["both_correct_pairs"] for r in rs],"query_sensitive_by_seed":[r["held"][sp]["query_sensitive_pairs"] for r in rs],
    "other_key_confusions_by_seed":[r["held"][sp]["other_key_confusions"] for r in rs],"truncated_total":sum(r["held"][sp]["truncated"] for r in rs),"mean_seconds":sum(r["held"][sp]["seconds"] for r in rs)/3}
  out[arm]["mixed"]={"mean_accuracy":sum(r["mixed"]["accuracy"] for r in rs)/3,"correct_by_seed":[r["mixed"]["correct"] for r in rs],"n":200}
 qa,qm=ARMS
 def pm(arm,key):return sum(runs[arm,s]["held"]["iid"][key]/runs[arm,s]["held"]["iid"]["pair_n"] for s in SEEDS)/3
 iid=100*(out[qa]["iid"]["mean_individual_accuracy"]-out[qm]["iid"]["mean_individual_accuracy"]);pair=100*(pm(qa,"both_correct_pairs")-pm(qm,"both_correct_pairs"))
 sens=100*(pm(qa,"query_sensitive_pairs")-pm(qm,"query_sensitive_pairs"));conf=100*sum((runs[qa,s]["held"]["iid"]["other_key_confusions"]-runs[qm,s]["held"]["iid"]["other_key_confusions"])/(2*runs[qa,s]["held"]["iid"]["pair_n"]) for s in SEEDS)/3
 mixed=100*(out[qm]["mixed"]["mean_accuracy"]-out[qa]["mixed"]["mean_accuracy"]);pos=sum(runs[qa,s]["held"]["iid"]["both_correct_pairs"]>runs[qm,s]["held"]["iid"]["both_correct_pairs"] for s in SEEDS)
 paired={str(s):{sp:pair_stats(runs[qa,s]["held"][sp]["rows"],runs[qm,s]["held"][sp]["rows"]) for sp in SPLITS} for s in SEEDS}
 gates={"both_correct_gain_ge3pp":pair>=3,"individual_gain_ge3pp":iid>=3,"positive_both_correct_direction_2seeds":pos>=2,"query_sensitivity_gain_ge3pp":sens>=3,"negative_confusion_change_le0pp":conf<=0,"mixed_regression_drop_le2pp":mixed<=2}
 result={"run":"G10AR","status":"completed","training_models":6,"ancestral_parent_seeds":list(SEEDS),"base_parameters":469648,"adapter_parameters":36865,"total_parameters":506513,
  "active_trainable_parameters":36865,"updates_each":200,"results":out,"paired_exact":paired,"gains_pp":{"iid_individual":iid,"iid_both_correct_pairs":pair,"iid_query_sensitivity":sens,
  "iid_other_key_confusion_change":conf,"mixed_regression_drop":mixed},"gates":gates,"verdict":"CONFIRMED_IN_REGISTERED_SMALL_SCREEN" if all(gates.values()) else "NOT_CONFIRMED_IN_REGISTERED_PROTOCOL",
  "same_held_cases_across_seeds":True,"same_training_group_sequence_across_all_models":True,"scope":"506,513-parameter narrow synthetic memory-addressing screen; not broad reasoning/code, not300-800M, not Qwen parity"}
 put(ROOT/"metrics/FINAL.json",result)
 with (ROOT/"metrics/FINAL.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["arm","seed","split","n","correct","accuracy","both_correct_pairs","pair_n","query_sensitive_pairs","other_key_confusions","truncated","seconds"])
  for arm in ARMS:
   for seed in SEEDS:
    r=runs[arm,seed]
    for sp in SPLITS:
     t=r["held"][sp];w.writerow([arm,seed,sp,t["n"],t["correct"],t["accuracy"],t["both_correct_pairs"],t["pair_n"],t["query_sensitive_pairs"],t["other_key_confusions"],t["truncated"],t["seconds"]])
    t=r["mixed"];w.writerow([arm,seed,"mixed",t["n"],t["correct"],t["accuracy"],"","","","",t["truncated"],t["seconds"]])
 put(ROOT/"reports/NEXT_STATE.json",{"status":"completed","run":"G10AR","verdict":result["verdict"],"goal_300_800m_qwen_thinking_not_achieved":True,"do_not_repeat":["G8C","G10AR"],
  "next_if_positive":"port readout to broad generative reasoning/code curriculum before scale","next_if_negative":"change memory write/address representation; do not merely extend updates"})
 return result
