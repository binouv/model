"""Synthetic latent-plan-to-text proof of concept. No natural-language understanding claimed."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AGENTS = ('fox','owl','wolf','bear','hare','duck','frog','lynx')
VERBS = ('finds','moves','keeps','takes','drops','hides','shows','holds')
OBJECTS = ('key','map','book','stone','ring','coin','lamp','shell')
VOCAB = ['<pad>','<bos>','<eos>', 'the', 'a', 'in','quiet', 'garden',',', '.',
         'carefully','near','river','today','at','dawn','and'] + list(AGENTS) + list(VERBS) + list(OBJECTS)
TOKEN = {word: i for i,word in enumerate(VOCAB)}
PAD, BOS, EOS = 0,1,2
MAX_LEN = 19

def grammar(facts: tuple[int,int,int], style: int) -> list[int]:
    a,b,c = facts
    if style == 0:
        words = ['the',AGENTS[a],VERBS[b],'a',OBJECTS[c],'.','<eos>']
    elif style == 1:
        words = ['in','the','quiet','garden',',','the',AGENTS[a], 'carefully',VERBS[b],
                 'a',OBJECTS[c],'near','the','river','today','.','<eos>']
    else: raise ValueError(style)
    return [TOKEN[w] for w in words]

def split(seed: int, fraction: float=0.2):
    facts = [(a,b,c) for a in range(8) for b in range(8) for c in range(8)]
    random.Random(seed).shuffle(facts)
    n = int(len(facts)*fraction)
    return facts[n:], facts[:n]

class Planner(nn.Module):
    def __init__(self, emb=12, latent=32):
        super().__init__()
        self.lookup = nn.ModuleList(nn.Embedding(8,emb) for _ in range(3))
        self.network = nn.Sequential(nn.Linear(emb*3,64), nn.Tanh(),nn.Linear(64,latent),nn.Tanh())
    def forward(self, facts):
        return self.network(torch.cat([layer(facts[:,i]) for i,layer in enumerate(self.lookup)],dim=-1))

class Renderer(nn.Module):
    def __init__(self, vocab=len(VOCAB), embed=24,latent=32,hidden=64):
        super().__init__()
        self.token_embed=nn.Embedding(vocab, embed)
        self.style_embed=nn.Embedding(2,8)
        self.start_state=nn.Linear(latent+8,hidden)
        self.cell=nn.GRUCell(embed+latent+8,hidden)
        self.output=nn.Linear(hidden,vocab)
    def step(self, prev, state, latent, style):
        x=torch.cat([self.token_embed(prev),latent,self.style_embed(style)],dim=-1)
        state=self.cell(x,state)
        return self.output(state),state
    def init(self,latent,style):
        return torch.tanh(self.start_state(torch.cat([latent,self.style_embed(style)],dim=-1)))
    def forward(self, inputs,latent,style):
        h=self.init(latent,style)
        logits=[]
        for i in range(inputs.size(1)):
            pred,h=self.step(inputs[:,i],h,latent,style)
            logits.append(pred)
        return torch.stack(logits,dim=1)
    @torch.no_grad()
    def generate(self,latent,style,max_len=MAX_LEN):
        state=self.init(latent,style)
        prev=torch.full((len(style),),BOS,dtype=torch.long,device=style.device)
        generated=[]
        for _ in range(max_len):
            pred,state=self.step(prev,state,latent,style)
            prev=pred.argmax(-1)
            generated.append(prev)
        return torch.stack(generated,dim=1)

class LatentSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.planner=Planner()
        self.renderer=Renderer()
    def forward(self,facts,inputs,style):
        return self.renderer(inputs,self.planner(facts),style)

def make_batch(facts,device='cpu'):
    x=torch.tensor([f for f,_ in facts], dtype=torch.long,device=device)
    style=torch.tensor([s for _,s in facts],dtype=torch.long,device=device)
    target=torch.full((len(facts),MAX_LEN),PAD,dtype=torch.long,device=device)
    for i,(f,s) in enumerate(facts):
        seq=grammar(f,s)
        target[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
    inputs=torch.cat([torch.full((len(facts),1),BOS,dtype=torch.long,device=device),target[:,:-1]],dim=1)
    return x,style,inputs,target

def sequences(pred,tar):
    vals=[]
    for p,t in zip(pred.tolist(),tar.tolist()):
        k=t.index(EOS)+1
        vals.append(p[:k]==t[:k])
    return vals

def semantic_correct(pred,facts,style):
    slots=(1,2,4) if style==0 else (6,8,10)
    return all(pred[slots[i]]==TOKEN[(AGENTS,VERBS,OBJECTS)[i][facts[i]]] for i in range(3))

@torch.no_grad()
def evaluate(system,facts,ablation='none',seed=500):
    system.eval()
    xs=torch.tensor(facts,dtype=torch.long)
    latent=system.planner(xs)
    if ablation=='zero': latent=torch.zeros_like(latent)
    elif ablation=='shuffle': latent=latent[torch.randperm(len(facts),generator=torch.Generator().manual_seed(seed))]
    metrics={}
    outputs={}
    for style in [0,1]:
        ss=torch.full((len(facts),),style,dtype=torch.long)
        pred=system.renderer.generate(latent,ss)
        target=make_batch([(f,style) for f in facts])[3]
        exact=torch.tensor(sequences(pred,target),dtype=torch.float32)
        sem=[semantic_correct(row.tolist(),f,style) for row,f in zip(pred,facts)]
        metrics[('short','detailed')[style]]={'exact_accuracy':float(exact.mean()),'semantic_slots_accuracy':sum(sem)/len(sem),'mean_target_tokens':float((target!=PAD).sum()/len(facts))}
        outputs[style]=pred
    metrics['paired_exact'] = float(torch.tensor([all(sequences(outputs[s][i:i+1],make_batch([(f,s)])[3])[0] for s in [0,1]) for i,f in enumerate(facts)],dtype=torch.float32).mean())
    return metrics,outputs

def readable(tokens):
    r=[]
    for t in tokens:
        if t==PAD or t==EOS:break
        r.append(VOCAB[int(t)])
    return ' '.join(r).replace(' .','.').replace(' ,',',')

def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train,test=split(args.seed)
    all_train=[(f,s) for f in train for s in [0,1]]
    model=LatentSystem()
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.00001)
    rng=random.Random(args.seed+1)
    t0=time.perf_counter()
    losses=[]
    for step in range(args.steps):
        batch=[all_train[rng.randrange(len(all_train))] for _ in range(args.batch)]
        facts,style,x,y=make_batch(batch)
        pred=model(facts,x,style)
        loss=F.cross_entropy(pred.reshape(-1,len(VOCAB)),y.reshape(-1),ignore_index=PAD)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()
        if (step+1)%max(1,args.steps//5)==0:
            losses.append({'step':step+1,'loss':round(float(loss.detach()),5)})
            print('step',step+1,'loss',float(loss.detach()),flush=True)
    train_time=time.perf_counter()-t0
    metrics,outputs=evaluate(model,test)
    metrics_zero,_=evaluate(model,test,'zero')
    metrics_shuffle,_=evaluate(model,test,'shuffle')
    torch.set_grad_enabled(False)
    lat=model.planner(torch.tensor(test,dtype=torch.long))
    def timeit(fn,repeats=7):
        samples=[]
        for _ in range(repeats):
            t1=time.perf_counter();fn();samples.append((time.perf_counter()-t1)*1e3)
        return float(np.median(samples))
    planner_ms=timeit(lambda:model.planner(torch.tensor(test,dtype=torch.long)))
    pair_ms=timeit(lambda:(model.renderer.generate(lat,torch.zeros(len(test),dtype=torch.long)),model.renderer.generate(lat,torch.ones(len(test),dtype=torch.long))))
    elapsed=time.perf_counter()-t0
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'latent_answer_training_checkpoint.pt'
    torch.save({'planner':model.planner.state_dict(),'renderer':model.renderer.state_dict(),
                'optimizer':optim.state_dict(),'torch_rng':torch.get_rng_state(),
                'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'train_facts':train,'test_facts':test,
                'config':vars(args),'steps':args.steps,'vocab':VOCAB},checkpoint)
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    results={'status':'completed','task':'synthetic agent-verb-object to short/detailed',
             'seed':args.seed,'parameters':sum(p.numel() for p in model.parameters()),
             'planner_parameters':sum(p.numel() for p in model.planner.parameters()),
             'renderer_parameters':sum(p.numel() for p in model.renderer.parameters()),
             'train_combinations':len(train),'heldout_combinations':len(test),'train_steps':args.steps,
             'loss_trace':losses,'heldout':metrics,'ablations':{'zero_latent':metrics_zero,'shuffled_latent':metrics_shuffle},
             'timings_cpu_ms':{'planner_batch_102':planner_ms,'two_styles_renderer_batch_102':pair_ms},
             'training_seconds':train_time,'total_seconds':elapsed,
             'checkpoint':{'name':checkpoint.name,'bytes':checkpoint.stat().st_size,'sha256':sha},
             'limitations':'Synthetic deterministic grammar; no pretrained LLM, no natural language reasoning; no verified speedup on Qwen.'}
    (out/'latent_answer_metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    sample=[]
    for i in [0,1,2,3,4,5,6,7,8,9]:
        sample.append({'facts':test[i], 'expected_short':readable(grammar(test[i],0)), 'predicted_short':readable(outputs[0][i].tolist()),
                       'expected_detailed':readable(grammar(test[i],1)),'predicted_detailed':readable(outputs[1][i].tolist())})
    (out/'examples.json').write_text(json.dumps(sample,indent=2))
    print(json.dumps(results,indent=2),flush=True)
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=800)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--seed',type=int,default=20261012)
    ap.add_argument('--lr',type=float,default=.006)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--out',default='results')
    args=ap.parse_args();run(args)
if __name__=='__main__':main()
