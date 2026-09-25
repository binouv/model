"""Finite, resumable full-parameter LM training, no hidden background service."""
from __future__ import annotations
import argparse,csv,dataclasses,gc,json,math,os,random,resource,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from model import Config,FlyGraphLM,parameter_report
from checkpoint_io import save_checkpoint,load_checkpoint
ROOT=Path(__file__).resolve().parents[1]

def atomic_json(p,z):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(z,indent=2,ensure_ascii=False));os.replace(t,p)

def sample_fingerprints(model):
    out={}
    for n,p in model.named_parameters():
        flat=p.detach().view(-1);stride=max(1,flat.numel()//64)
        out[n]=flat[::stride][:64].float().cpu().tolist()
    return out

class Sampler:
    def __init__(self,path,seed,seq,batch):
        self.rng=np.random.default_rng(seed);self.seq=seq;self.batch=batch
        # Balance domains, not the raw source-file size. Exact sources in data manifest.
        self.groups=[('python_source',.30),('english_docstring',.15),('memory_update',.12),('memory_chain',.12),('arithmetic',.06),('multistep',.06),('code_trace',.06),('sorting',.04),('list_sum',.04),('synthetic_python',.03),('ru_prose',.02)]
        self.arrays={k:np.memmap(path/f'train_{k}.bin',dtype=np.uint16,mode='r') for k,_ in self.groups}
        self.weights=np.array([w for _,w in self.groups]);self.weights/=self.weights.sum()
        self.counts={k:0 for k,_ in self.groups}
    def get(self):
        rows=[]
        for _ in range(self.batch):
            k=self.groups[int(self.rng.choice(len(self.groups),p=self.weights))][0];a=self.arrays[k]
            start=int(self.rng.integers(0,max(1,len(a)-self.seq-1)))
            if len(a)<self.seq+1:row=np.array(a[np.arange(self.seq+1)%len(a)],dtype=np.int64)
            else:row=np.array(a[start:start+self.seq+1],dtype=np.int64)
            rows.append(row);self.counts[k]+=self.seq
        tt=torch.from_numpy(np.stack(rows));return tt[:,:-1].contiguous(),tt[:,1:].contiguous()

@torch.no_grad()
def evaluate(model,data,seq=128,n=8,disable_graph_edges=False):
    model.eval();out={}
    for split in ['validation']:
        for kind in ['python_source','english_docstring','memory_update','memory_chain','arithmetic','multistep','code_trace','sorting','ru_prose']:
            p=data/f'{split}_{kind}.bin'
            if not p.exists():continue
            a=np.memmap(p,dtype=np.uint16,mode='r');L=min(seq,len(a)-1)
            rng=np.random.default_rng(7190);loss=0.;correct=0;tokens=0
            for i in range(min(n,max(1,len(a)//L))):
                start=int(rng.integers(0,max(1,len(a)-L-1)))
                t=torch.tensor(np.array(a[start:start+L+1],dtype=np.int64),device=next(model.parameters()).device)[None]
                ll,lg=model(t[:,:-1],t[:,1:],disable_graph_edges=disable_graph_edges)
                loss+=float(ll)*L;correct+=int(lg.argmax(-1).eq(t[:,1:]).sum());tokens+=L
            out[kind]={'nll':loss/tokens,'token_accuracy':correct/tokens,'tokens':tokens}
    out['macro_nll']=sum(z['nll'] for z in out.values())/len(out)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=384);ap.add_argument('--seq',type=int,default=128);ap.add_argument('--batch',type=int,default=2);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--lr',type=float,default=.02);ap.add_argument('--seed',type=int,default=7101);ap.add_argument('--resume');ap.add_argument('--eval-every',type=int,default=64);ap.add_argument('--save-every',type=int,default=128);ap.add_argument('--probe',action='store_true');ap.add_argument('--no-checkpoint',action='store_true');ap.add_argument('--device',default='cpu');ap.add_argument('--amp',choices=['none','bf16'],default='none');args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.manual_seed(args.seed);random.seed(args.seed)
    config=Config(init_seed=args.seed,checkpoint_blocks=not args.no_checkpoint)
    if args.resume:
        model,idx=load_checkpoint(args.resume);step0=idx['step']
    else:model=FlyGraphLM(config);step0=0
    model=model.to(args.device)
    model.config.checkpoint_blocks=not args.no_checkpoint
    prior_tokens=idx.get('training',{}).get('total_tokens',0) if args.resume else 0
    optimizer=torch.optim.Adafactor(model.parameters(),lr=args.lr,weight_decay=0.,foreach=False)
    sampler=Sampler(ROOT/'data',7111,args.seq,args.batch)
    if args.resume:
        state=torch.load(Path(args.resume)/'training_state.pt',weights_only=False)
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
        sampler.rng.bit_generator.state=state['sampler_rng']
        if args.device.startswith('cuda') and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    report=parameter_report(model);print(json.dumps(report),flush=True)
    config_out={'model':dataclasses.asdict(model.config),'run_args':vars(args),'parameter_report':report,'optimizer':'PyTorch Adafactor, FP32 full parameters, factored second moments, no weight decay','hypothesis':'A true causal 310M generative model with a recurrent graph workspace can learn local RU/EN/code distributions; graph-message ablation measures whether its recurrent edges help held token prediction. This is a bootstrap feasibility check, not a claim of broad cognition superiority.','ablation':'same checkpoint, two workspace GRU rounds but zero inter-node messages; all other computation and data unchanged','stage':'bootstrap from scratch; NOT completed foundation pretraining','seed':args.seed,'git_used':False}
    atomic_json(ROOT/'configs/train_config.json',config_out)
    initial=sample_fingerprints(model) if step0==0 else json.loads((ROOT/'metrics/initial_parameter_samples.json').read_text())
    if step0==0:atomic_json(ROOT/'metrics/initial_parameter_samples.json',initial)
    start_time=time.perf_counter();curves=[]
    if not args.probe and step0==0:
        val=evaluate(model,ROOT/'data',args.seq,n=2);atomic_json(ROOT/'metrics/initial_validation.json',val);print('INITIAL',json.dumps(val),flush=True)
    gradient_seen={n:False for n,p in model.named_parameters()}
    for step in range(step0+1,args.steps+1):
        model.train();t0=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        x,y=sampler.get();x=x.to(args.device);y=y.to(args.device)
        ctx=torch.autocast(device_type=args.device.split(':')[0],dtype=torch.bfloat16) if args.amp=='bf16' else nullcontext()
        with ctx:loss,logits=model(x,y)
        loss.backward();del logits
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss; incomplete run not aggregatable')
        if step==step0+1 or step%32==0:
            for name,p in model.named_parameters():
                if p.grad is not None:
                    gg=p.grad.view(-1);ss=gg[::max(1,gg.numel()//64)]
                    if not bool(torch.isfinite(ss).all()):raise RuntimeError('nonfinite gradient')
                    gradient_seen[name]|=bool(ss.abs().max()>0)
        gnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=False))
        # Warmup then constant for the finite bootstrap; resume keeps global step count.
        lr=args.lr*min(1.0,step/24)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();ll=float(loss.detach());del loss
        tok=args.batch*args.seq;dt=time.perf_counter()-t0
        z={'step':step,'loss':ll,'lr':lr,'grad_norm':gnorm,'tokens_this_step':tok,'total_tokens':prior_tokens+(step-step0)*tok,'seconds':dt,'tokens_per_sec':tok/dt,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        curves.append(z)
        with (ROOT/'metrics/train_steps.jsonl').open('a') as f:f.write(json.dumps(z)+'\n')
        if step==step0+1 or step%8==0:print('TRAIN',json.dumps(z),flush=True)
        if step%8==0 or args.probe:atomic_json(ROOT/'metrics/progress.json',{'status':'running','last_completed_step':step,'config':config_out,'last_step':z,'domain_token_counts_this_invocation':sampler.counts})
        if args.probe:
            if step>=3:break
            continue
        if step%args.eval_every==0 or step==args.steps:
            optimizer.zero_grad(set_to_none=True);gc.collect()
            val=evaluate(model,ROOT/'data',args.seq,n=2);atomic_json(ROOT/f'metrics/validation_step_{step:04d}.json',val);print('VALIDATION',step,json.dumps(val),flush=True)
        if step%args.save_every==0 or step==args.steps:
            optimizer.zero_grad(set_to_none=True);gc.collect()
            dst=ROOT/f'checkpoints/step_{step:04d}'
            save_checkpoint(model,dst,optimizer,step,sampler.rng.bit_generator.state,{'total_tokens':prior_tokens+(step-step0)*tok,'full_parameter_training':True,'corpus':'local CPython + synthetic RU/EN/code only','stage':'BOOTSTRAP_NOT_PRETRAINED'})
            atomic_json(ROOT/'checkpoints/latest.json',{'path':str(dst),'step':step});print('CHECKPOINT',str(dst),flush=True)
    if args.probe:
        atomic_json(ROOT/'metrics/performance_probe.json',{'completed_steps':len(curves),'rows':curves,'status':'engineering_probe_only','checkpoint_saved':False});return
    optimizer.zero_grad(set_to_none=True)
    final=sample_fingerprints(model)
    changed={name:any(a!=b for a,b in zip(initial[name],v)) for name,v in final.items()}
    audit={'parameter_tensor_count':len(changed),'changed_tensor_count':sum(changed.values()),'unchanged_names':[n for n,v in changed.items() if not v],'nonzero_gradient_seen_tensor_count':sum(gradient_seen.values()),'gradient_not_seen_names':[n for n,v in gradient_seen.items() if not v],'details_changed':changed,'gradient_sampling':'64 samples per parameter tensor on first/every32 step; not an elementwise proof'}
    atomic_json(ROOT/'metrics/parameter_audit.json',audit)
    atomic_json(ROOT/'metrics/progress.json',{'status':'completed','last_completed_step':args.steps,'total_tokens':prior_tokens+(args.steps-step0)*args.seq*args.batch,'elapsed_sec_this_invocation':time.perf_counter()-start_time,'config':config_out,'parameter_audit':audit,'domain_token_counts_this_invocation':sampler.counts})
    print('FINISHED',args.steps,flush=True)
if __name__=='__main__':main()
