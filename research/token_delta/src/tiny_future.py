"""Deterministic toy test: can frozen AR states encode several future tokens?"""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

VOCAB=16
SEQ=28
CONTEXT=20
K=8

class TinyGRU(nn.Module):
    def __init__(self, hidden=96, embed=48):
        super().__init__()
        self.embed=nn.Embedding(VOCAB,embed)
        self.gru=nn.GRU(embed,hidden,batch_first=True)
        self.head=nn.Linear(hidden,VOCAB)
        self.hidden_size=hidden

    def forward(self, x, hidden=None):
        states,h=self.gru(self.embed(x), hidden)
        return self.head(states), states,h

class FutureProbe(nn.Module):
    def __init__(self, hidden=96, k=K):
        super().__init__()
        self.head=nn.Linear(hidden,k*VOCAB)
        self.k=k
    def forward(self, state):
        return self.head(state).reshape(-1,self.k,VOCAB)

def batch(task, size, rng, device='cpu'):
    if task == 'motif':
        motif=torch.from_numpy(rng.integers(0,VOCAB,size=(size,4),dtype=np.int64))
        tokens=motif.repeat(1,SEQ//4)
    elif task == 'random':
        tokens=torch.from_numpy(rng.integers(0,VOCAB,size=(size,SEQ),dtype=np.int64))
    else:
        raise ValueError(task)
    return tokens.to(device)

def train_task(task, args, seed=24):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=TinyGRU(hidden=args.hidden)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    t0=time.perf_counter()
    losses=[]
    for j in range(args.train_steps):
        x=batch(task,args.batch,rng)
        logits,_,_=model(x[:,:-1])
        loss=nn.functional.cross_entropy(logits.flatten(0,1),x[:,1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (j+1)%max(1,args.train_steps//4)==0:losses.append([j+1,round(loss.item(),5)])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    probe=FutureProbe(hidden=args.hidden)
    optim=torch.optim.AdamW(probe.parameters(),lr=args.lr)
    probe_losses=[]
    for j in range(args.probe_steps):
        x=batch(task,args.batch,rng)
        with torch.no_grad():_,states,_=model(x[:,:CONTEXT])
        pred=probe(states[:,-1,:])
        loss=nn.functional.cross_entropy(pred.flatten(0,1),x[:,CONTEXT:CONTEXT+K].flatten())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (j+1)%max(1,args.probe_steps//4)==0:probe_losses.append([j+1,round(loss.item(),5)])
    probe.eval()
    x=batch(task,args.eval_samples,np.random.default_rng(seed+9439))
    with torch.no_grad():
        logits,states,h=model(x[:,:CONTEXT]); p=probe(states[:,-1,:]); future=x[:,CONTEXT:CONTEXT+K]
        top=p.argmax(-1)
        horizon_acc=(top==future).float().mean(0).tolist()
        autoreg_next_accuracy=(logits[:,-1,:].argmax(-1)==future[:,0]).float().mean().item()
        draft_match=(top[:,0] == logits[:,-1,:].argmax(-1)).float().mean().item()
        accepted=((top==future).int().cumprod(-1)).sum(-1).float().mean().item()
    elapsed=time.perf_counter()-t0
    metrics={
        'task':task,'seed':seed,'samples':args.eval_samples,'train_steps':args.train_steps,
        'probe_steps':args.probe_steps,'train_loss_trace':losses,'probe_loss_trace':probe_losses,
        'autoreg_next_accuracy':autoreg_next_accuracy,'probe_horizon_accuracy':horizon_acc,
        'probe_next_matches_model_next':draft_match,
        'mean_prefix_correct_against_ground_truth':accepted,
        'chance_accuracy':1.0/VOCAB,'elapsed_wall_s':round(elapsed,3),
        'toy_caveat':'synthetic motif or iid-random tokens; not an LLM and not tok/s'
    }
    return model,probe,opt,optim,metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-steps',type=int,default=600)
    parser.add_argument('--probe-steps',type=int,default=400)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--hidden',type=int,default=96)
    parser.add_argument('--lr',type=float,default=.003)
    parser.add_argument('--eval-samples',type=int,default=1024)
    parser.add_argument('--out',default='')
    args=parser.parse_args()
    torch.set_num_threads(2)
    results={}
    for task in ['motif','random']:
        model,probe,opt,optim,metrics=train_task(task,args)
        results[task]=metrics
        print(task,metrics,flush=True)
        if args.out:
            out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'probe':probe.state_dict(),'model_optimizer':opt.state_dict(),
                        'probe_optimizer':optim.state_dict(),'torch_rng':torch.get_rng_state(),
                        'numpy_generator_note':'Samples deterministic from seed 24; see source',
                        'config':vars(args),'task':task,'steps':{'model':args.train_steps,'probe':args.probe_steps}},out/f'{task}_training_checkpoint.pt')
    if args.out:(Path(args.out)/'metrics.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
