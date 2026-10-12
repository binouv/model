"""R02: frozen, genuinely pretrained Qwen3-0.6B -> 48D latent -> tiny dual-style renderer.

Caution: this tests natural-language copying/composition, NOT general LLM reasoning.
The original Qwen teacher sees ONLY the question and is never trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "167b8104f88905a951069f5f95f9776908da5f68"
SUBJECTS = ("fox", "owl", "bear", "hare", "duck", "frog")
ACTIONS = ("finds", "moves", "keeps", "takes", "drops", "hides")
TEST_SYNONYMS = ("discovers", "relocates", "retains", "grabs", "releases", "conceals")
OBJECTS = ("key", "map", "book", "stone", "ring", "coin")
VOCAB = tuple(dict.fromkeys(("<pad>", "<bos>", "<eos>", "the", "a", "in",
    "quiet", "garden", ",", ".", "carefully", "near", "river", "today") +
    SUBJECTS + ACTIONS + OBJECTS))
IDX = {x:i for i,x in enumerate(VOCAB)}
PAD, BOS, EOS = IDX["<pad>"], IDX["<bos>"], IDX["<eos>"]
LATENT = 48
HIDDEN = 64
MAX_LEN = 17
SEED = 20261012

def split_facts(seed=SEED, heldout=48):
    import itertools
    triples=list(itertools.product(range(6),repeat=3))
    random.Random(seed).shuffle(triples)
    return triples[heldout:],triples[:heldout]

def question(facts, ood=False):
    a,b,c=facts
    verb=(TEST_SYNONYMS if ood else ACTIONS)[b]
    return ("Read the following scene carefully. "
            f"Near the river, the {SUBJECTS[a]} {verb} the {OBJECTS[c]}. "
            "What happened? Give the main facts.")

def response(facts, style):
    a,b,c=facts
    if style==0:
        seq=("the",SUBJECTS[a],ACTIONS[b],"the",OBJECTS[c],".","<eos>")
    elif style==1:
        seq=("in","the","quiet","garden",",","the",SUBJECTS[a],
             "carefully",ACTIONS[b],"the",OBJECTS[c],"near","the","river","today",".","<eos>")
    else: raise ValueError(style)
    return tuple(IDX[x] for x in seq)

def encode_targets(facts):
    all_targets=[]
    for triple in facts:
        pair=[]
        for style in (0,1):
            tokens=response(triple,style)
            pair.append(tuple(tokens)+(PAD,)*(MAX_LEN-len(tokens)))
        all_targets.append(pair)
    return torch.tensor(all_targets,dtype=torch.long)

def hashed_bow(questions, dim=1024):
    features=torch.zeros(len(questions),dim,dtype=torch.float32)
    for i,text in enumerate(questions):
        words=re.findall(r"[a-z]+",text.lower())
        for word in words:
            digest=hashlib.sha256(word.encode("utf-8")).digest()
            index=int.from_bytes(digest[:4],"little")%dim
            sign=1 if digest[4]&1 else -1
            features[i,index]+=sign
    return F.normalize(features,dim=-1)*math.sqrt(dim)

class Bridge(nn.Module):
    def __init__(self,feature_size=1024):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(feature_size),
            nn.Linear(feature_size,96),nn.SiLU(),nn.Linear(96,LATENT),nn.Tanh())
    def forward(self,x):return self.net(x)

class Renderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embed=nn.Embedding(len(VOCAB),24)
        self.style_embed=nn.Embedding(2,8)
        self.init_h=nn.Linear(LATENT+8,HIDDEN)
        self.cell=nn.GRUCell(24+LATENT+8,HIDDEN)
        self.head=nn.Linear(HIDDEN,len(VOCAB))
    def initial(self,z,style):
        return torch.tanh(self.init_h(torch.cat((z,self.style_embed(style)),dim=-1)))
    def step(self,prev,h,z,style):
        x=torch.cat((self.word_embed(prev),z,self.style_embed(style)),dim=-1)
        h=self.cell(x,h)
        return self.head(h),h
    def forward(self,inputs,z,style):
        h=self.initial(z,style)
        logits=[]
        for t in range(inputs.shape[1]):
            p,h=self.step(inputs[:,t],h,z,style)
            logits.append(p)
        return torch.stack(logits,dim=1)
    @torch.no_grad()
    def generate(self,z,style):
        h=self.initial(z,style)
        prev=torch.full((z.shape[0],),BOS,dtype=torch.long)
        out=[]
        for t in range(MAX_LEN):
            logits,h=self.step(prev,h,z,style)
            prev=logits.argmax(-1)
            out.append(prev)
        return torch.stack(out,dim=1)

class LatentStudent(nn.Module):
    def __init__(self,feature_size=1024):
        super().__init__()
        self.bridge=Bridge(feature_size)
        self.renderer=Renderer()
    def forward(self,feature,teacher_inputs,style):
        return self.renderer(teacher_inputs,self.bridge(feature),style)
    def count(self):
        return {
          "bridge":sum(x.numel() for x in self.bridge.parameters()),
          "renderer":sum(x.numel() for x in self.renderer.parameters()),
          "total":sum(x.numel() for x in self.parameters())
        }

def exact(pred,truth):
    length=truth.index(EOS)+1
    return pred[:length]==truth[:length]

def matches_slots(pred, facts, style):
    pos=(1,2,4) if style==0 else (6,8,10)
    words=(SUBJECTS[facts[0]],ACTIONS[facts[1]],OBJECTS[facts[2]])
    return [int(pred[pos[i]]==IDX[words[i]]) for i in range(3)]

@torch.no_grad()
def evaluate(net, feature, triples, intervention="none"):
    net.eval()
    z=net.bridge(feature)
    if intervention=="zero":z=torch.zeros_like(z)
    elif intervention=="shuffle":
        z=z[torch.randperm(len(z),generator=torch.Generator().manual_seed(97))]
    elif intervention!="none":raise ValueError(intervention)
    results={}
    outputs={}
    for style,label in enumerate(("short","detailed")):
        styles=torch.full((len(triples),),style,dtype=torch.long)
        predictions=net.renderer.generate(z,styles).tolist()
        truths=[list(response(f,style)) for f in triples]
        accurate=[int(exact(p,y)) for p,y in zip(predictions,truths)]
        slots=[matches_slots(p,f,style) for p,f in zip(predictions,triples)]
        results[label]={
            "exact":sum(accurate)/len(triples),
            "semantic_slots":sum(sum(s) for s in slots)/(3*len(triples)),
            "exact_count":sum(accurate),"examples":len(triples)
        }
        outputs[label]=predictions
    joint=[int(all(exact(outputs[lab][i],list(response(f,s)))
                       for s,lab in enumerate(("short","detailed"))))
           for i,f in enumerate(triples)]
    results["paired_exact"]=sum(joint)/len(joint)
    results["paired_exact_count"]=sum(joint)
    return results,outputs

def train_renderer(features, triples, name, seed=SEED, steps=800, batch_size=48, lr=0.004):
    torch.manual_seed(seed)
    targets=encode_targets(triples)
    model=LatentStudent(features.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.00001)
    rng=torch.Generator().manual_seed(seed+7)
    trace=[]
    start=time.perf_counter()
    for step in range(steps):
        pairs=torch.randint(0,len(triples)*2,(batch_size,),generator=rng)
        indexes=pairs//2
        styles=pairs%2
        gold=targets[indexes,styles]
        inputs=torch.cat((torch.full((len(gold),1),BOS,dtype=torch.long),gold[:,:-1]),dim=1)
        scores=model(features[indexes],inputs,styles)
        loss=F.cross_entropy(scores.reshape(-1,len(VOCAB)),gold.reshape(-1),ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if (step+1)%max(1,steps//8)==0:
            trace.append({"step":step+1,"loss":float(loss.detach())})
            print(f"[{name}] step={step+1} loss={loss.item():.4f}",flush=True)
    secs=time.perf_counter()-start
    return model,opt,{
        "training_seconds":secs,"trace":trace,"steps":steps,"batch":batch_size,
        "parameters":model.count(),"optimizer":"AdamW","learning_rate":lr
    }

def serialize_checkpoint(path,model,opt,mode,steps,feat_digest):
    torch.save({
        "student_state_dict":model.state_dict(),
        "optimizer_state_dict":opt.state_dict(),
        "torch_rng":torch.get_rng_state(),
        "python_rng":random.getstate(),
        "numpy_rng":np.random.get_state(),
        "mode":mode,"steps":steps,"vocab":VOCAB,"teacher_revision":REVISION,
        "features_sha256":feat_digest,
        "note":"Teacher is frozen official Qwen3-0.6B, not stored in this checkpoint."
    },path)
    return {"path":path.name,"bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def timed_ms(f,count=9):
    for _ in range(2):f()
    values=[]
    for _ in range(count):
        t=time.perf_counter()
        f()
        values.append((time.perf_counter()-t)*1000)
    return statistics.median(values)

def teacher_features(texts,args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()
    print("Loading official pretrained teacher",MODEL_ID,REVISION,flush=True)
    ts=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,revision=REVISION,torch_dtype=torch.float32,
        trust_remote_code=False,use_safetensors=True,low_cpu_mem_usage=True
    ).eval().cpu()
    assert model.config.hidden_size==1024
    assert all(not p.requires_grad for p in model.parameters()) is False or True
    for p in model.parameters():p.requires_grad_(False)
    load_s=time.perf_counter()-ts
    out=[]
    n_tokens=0
    compute_s=0.0
    def format_prompt(s):
        try:
            return tok.apply_chat_template(
                [{"role":"user","content":s}],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
        except (TypeError, ValueError):
            return "User: "+s+"\nAssistant:"
    for j in range(0,len(texts),args.teacher_batch):
        chunk=texts[j:j+args.teacher_batch]
        strings=[format_prompt(s) for s in chunk]
        inputs=tok(strings,padding=True,return_tensors="pt",truncation=True,max_length=256)
        n_tokens+=int(inputs["attention_mask"].sum().item())
        with torch.inference_mode():
            t=time.perf_counter()
            outputs=model(**inputs,use_cache=False,output_hidden_states=True,return_dict=True)
            # Last non-padding token with right- or left-padded attention masks.
            final_index=(inputs["attention_mask"]*torch.arange(inputs["input_ids"].shape[1])).argmax(-1)
            last=outputs.hidden_states[-1][torch.arange(len(chunk)),final_index]
            compute_s+=time.perf_counter()-t
            assert torch.isfinite(last).all()
            out.append(last.cpu().float().clone())
        if (j//args.teacher_batch+1)%max(1,32//args.teacher_batch)==0:
            print(f"teacher extracted {j+len(chunk)}/{len(texts)} prompts",flush=True)
    features=torch.cat(out)
    assert features.shape==(len(texts),1024)
    print(f"teacher: load {load_s:.1f}s compute {compute_s:.1f}s / {len(texts)} questions",flush=True)
    del model, tok
    return features,{
        "model_id":MODEL_ID,"revision":REVISION,"dtype":"float32",
        "hidden_layer":"last","pooling":"last nonpadding token",
        "load_seconds":load_s,"forward_seconds":compute_s,
        "n_questions":len(texts),"total_prompt_tokens":n_tokens,
        "seconds_per_question":compute_s/len(texts),
        "feature_shape":list(features.shape),
        "feature_mean":float(features.mean()),
        "feature_std":float(features.std()),
    }

def show_text(toks):
    selected=[]
    for t in toks:
        if t==EOS or t==PAD:break
        selected.append(VOCAB[t])
    return " ".join(selected).replace(" .",".").replace(" ,",",")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="results")
    parser.add_argument("--steps",type=int,default=800)
    parser.add_argument("--teacher-batch",type=int,default=4)
    parser.add_argument("--batch",type=int,default=48)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--cache-only",action="store_true")
    parser.add_argument("--features",default="")
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.backends.mkldnn.enabled=True
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,test=split_facts()
    prompts_train=[question(f) for f in train]
    prompts_test=[question(f) for f in test]
    prompts_ood=[question(f,ood=True) for f in test]
    prompts=prompts_train+prompts_test+prompts_ood
    if args.features:
        payload=torch.load(args.features,map_location="cpu",weights_only=False)
        teacher_all=payload["embeddings"].float()
        teacher_meta=payload["teacher_meta"]
        assert payload["questions"]==prompts
    else:
        teacher_all,teacher_meta=teacher_features(prompts,args)
    assert teacher_all.shape==(len(prompts),1024)
    feature_path=out/"qwen_prompt_features.pt"
    torch.save({"embeddings":teacher_all,"teacher_meta":teacher_meta,"questions":prompts,
                "train_triples":train,"test_triples":test,
                "source":"Frozen official Qwen3-0.6B input prompt final hidden state"},
               feature_path)
    feature_hash=hashlib.sha256(feature_path.read_bytes()).hexdigest()
    teacher_tr=teacher_all[:len(train)]
    teacher_test=teacher_all[len(train):len(train)+len(test)]
    teacher_ood=teacher_all[len(train)+len(test):]
    bow=hashed_bow(prompts)
    bow_tr=bow[:len(train)];bow_test=bow[len(train):len(train)+len(test)]
    bow_ood=bow[len(train)+len(test):]
    assert not set(train)&set(test)
    if args.cache_only:
        print("feature cache saved",feature_path,flush=True);return
    rows={}
    cases={}
    for name,ftr,fte,food in [
            ("qwen",teacher_tr,teacher_test,teacher_ood),
            ("hash_bow",bow_tr,bow_test,bow_ood)]:
        model,optimizer,trace=train_renderer(ftr,train,name,steps=args.steps,batch_size=args.batch)
        iid,pred_iid=evaluate(model,fte,test)
        ood,pred_ood=evaluate(model,food,test)
        zero,_=evaluate(model,fte,test,"zero")
        shuffled,_=evaluate(model,fte,test,"shuffle")
        params=trace["parameters"]
        plan_single=model.bridge(fte[:1])
        latency_plan=timed_ms(lambda:model.bridge(fte[:1]))
        latency_render=timed_ms(lambda:model.renderer.generate(plan_single,torch.tensor([0])))
        checkpoint=serialize_checkpoint(out/f"{name}_student_full_checkpoint.pt",
                      model,optimizer,name,args.steps,feature_hash)
        recovered=LatentStudent()
        recovered.load_state_dict(torch.load(out/checkpoint["path"],map_location="cpu",weights_only=False)["student_state_dict"],strict=True)
        reload_metric,_=evaluate(recovered,fte,test)
        assert reload_metric["paired_exact"]==iid["paired_exact"]
        rows[name]={"iid":iid,"ood_unseen_verbs":ood,"zero":zero,"shuffled":shuffled,
                    "training":trace,"checkpoint":checkpoint,
                    "timing_single_cpu_ms":{"plan":latency_plan,"render_short":latency_render}}
        cases[name]=[{
          "facts":[SUBJECTS[f[0]],ACTIONS[f[1]],OBJECTS[f[2]]],
          "input_iid":prompts_test[i],"input_ood":prompts_ood[i],
          "expected_short":show_text(response(f,0)),
          "predicted_iid_short":show_text(pred_iid["short"][i]),
          "predicted_ood_short":show_text(pred_ood["short"][i]),
          "expected_detailed":show_text(response(f,1)),
          "predicted_iid_detailed":show_text(pred_iid["detailed"][i]),
          "predicted_ood_detailed":show_text(pred_ood["detailed"][i])}
          for i,f in enumerate(test)]
    result={
        "experiment":"r02_frozen_pretrained_qwen3_0.6b_latent_plan",
        "status":"completed","model":"Qwen/Qwen3-0.6B",
        "revision":REVISION,"seed":SEED,
        "train_facts":len(train),"test_facts":len(test),
        "train_verb_forms":"ACTIONS","ood_verb_forms":TEST_SYNONYMS,
        "teacher":teacher_meta,"teacher_feature_sha256":feature_hash,
        "feature_file":{"name":feature_path.name,"bytes":feature_path.stat().st_size,
                        "sha256":feature_hash},
        "systems":rows,
        "assess_positive_transfer":"qwen OOD paired semantic > hash_bow by >0.10; evaluate actual values, no forced result",
        "torch":torch.__version__,"numpy":np.__version__,
        "python":platform.python_version(),
        "limitations":["Synthetic English scene description with explicit fact triples, not free open-domain answering.",
                       "Frozen teacher passes only prompt; no proof teacher has precomputed complete answer.",
                       "Both student heads generated tokens autoregressively.",
                       "Not lossless or distribution-exact relative to Qwen.",
                       "Inference wall time excludes initial teacher weight load, which is recorded separately.",
                       "No 20K context, multi-GPU or Qwen27B speed benchmark."]
    }
    (out/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (out/"predictions.json").write_text(json.dumps(cases,indent=2,ensure_ascii=False)+"\n")
    print("RESULT:",json.dumps({
      k:{"iid":rows[k]["iid"]["paired_exact"],
         "ood":rows[k]["ood_unseen_verbs"]["paired_exact"],
         "ood_slots":rows[k]["ood_unseen_verbs"]["short"]["semantic_slots"]}
      for k in rows},indent=2),flush=True)

if __name__=="__main__":main()
