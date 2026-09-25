"""Finite G3L training continuation, optimizer/RNG included, no test selection."""
from __future__ import annotations
import argparse,collections,gc,hashlib,json,math,os,random,time
from pathlib import Path
import torch
from model import Model,Config,report
from train import rows,batch,val,evaluate,atomic,save,SPLITS
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['full','shared','hybrid']);p.add_argument('--seed',type=int,required=True);p.add_argument('--parent',required=True);p.add_argument('--resume');p.add_argument('--eval-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    name=f'{a.variant}_s{a.seed}';out=ROOT/'metrics'/name;out.mkdir(parents=True,exist_ok=True)
    path=Path(a.resume or a.parent);st=torch.load(path,weights_only=False)
    if st['config']['variant']!=a.variant:raise ValueError('variant/checkpoint mismatch')
    c=Config(**st['config']);m=Model(c);m.load_state_dict(st['model']);m.eval()
    if a.eval_only:
        if st['step']!=8000:raise ValueError('only fixedfinal8000 evaluated')
        tok=Tokenizer.load(ROOT/'data/tokenizer.json')
        for split in SPLITS:atomic(out/f'{split}.json',evaluate(m,tok,rows(split)))
        if a.variant=='hybrid':
            for split in SPLITS:atomic(out/f'reset_{split}.json',evaluate(m,tok,rows(split),True))
        atomic(out/'EVAL_COMPLETE.json',{'status':'completed','checkpoint':str(path),'sha256':digest(path)});return
    if st['step']<2000 or st['step']>=8000:raise ValueError('invalid continuation boundary')
    datahash=json.loads((ROOT/'data/manifest.json').read_text())['hashes']
    if st['data_hashes']!=datahash:raise ValueError('data changed')
    for n,h in datahash.items():
        if digest(ROOT/'data'/n)!=h:raise ValueError('data file hash mismatch')
    pools=collections.defaultdict(list)
    for z in rows('train'):pools[(z['family'],z['lang'])].append(z)
    keys=sorted(pools);rng=random.Random();rng.setstate(st['sampler_rng'])
    opt=torch.optim.AdamW(m.parameters(),lr=.0005,betas=(.9,.95),weight_decay=.01);opt.load_state_dict(st['optimizer'])
    torch.set_rng_state(st['torch_rng']);initial={k:v.detach().flatten()[::max(1,v.numel()//32)][:32].clone() for k,v in m.named_parameters()}
    logs=st['logs'];step0=st['step'];parent_hash=digest(a.parent);seen={k for z in logs for k in z['batch_ids']}
    if [z['step'] for z in logs]!=list(range(1,step0+1)):raise ValueError('invalid parent log')
    target=out/'train.jsonl'
    if target.exists() and not a.resume:raise FileExistsError('refuse overwrite started continuation')
    if a.resume and target.exists():
        existing=[json.loads(s) for s in target.read_text().splitlines()]
        if any(z['step']>step0 for z in existing):raise ValueError('unsaved tail exists; quarantine it before exact resume')
    elif a.resume and not target.exists():
        target.write_text(''.join(json.dumps(z)+'\n' for z in logs if z['step']>2000))
    validation=rows('validation');v=val(m,validation);atomic(out/f'initial_step{step0}.json',{'validation_nll':v,'step':step0,'parent_sha256':parent_hash,'optimizer_and_rng_restored':True})
    start=time.perf_counter()
    for step in range(step0+1,8001):
        m.train();opt.zero_grad(set_to_none=True);t=time.perf_counter();key=keys[rng.randrange(len(keys))];rr=rng.choices(pools[key],k=32)
        ids=[z['id'] for z in rr];seen.update(ids);xx,yy=batch(rr);loss=m(xx,yy);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(grad):raise RuntimeError('nonfinite')
        phase=step-2000;lr=.0005*min(1.,phase/100)*(.15+.85*.5*(1+math.cos(math.pi*phase/6000)))
        for g in opt.param_groups:g['lr']=lr
        opt.step();row={'step':step,'loss':float(loss.detach()),'grad_norm':grad,'lr':lr,'seconds':time.perf_counter()-t,'input_tokens':sum(len(z['tokens'])-1 for z in rr),'supervised_tokens':int(yy.ne(-100).sum()),'padded_tokens':xx.numel(),'batch_ids':ids,'family':key[0],'lang':key[1]};logs.append(row)
        with target.open('a') as f:f.write(json.dumps(row)+'\n')
        if step%250==0:print(name,'STEP',step,'LOSS',round(row['loss'],4),'SEC',round(row['seconds'],3),flush=True)
        if phase%2000==0:
            opt.zero_grad(set_to_none=True);v=val(m,validation);atomic(out/f'val{step:04d}.json',{'step':step,'nll':v})
            cp=ROOT/'checkpoints'/name/f'step{step:04d}.pt';ss=save(cp,m,opt,step,rng,logs,vars(c));atomic(out/'progress.json',{'durable_step':step,'sha256':ss,'path':str(cp),'status':'completed_training' if step==8000 else 'running'});print(name,'CHECKPOINT',step,'VAL',v,flush=True)
    changed=sum(not torch.equal(initial[k],v.detach().flatten()[::max(1,v.numel()//32)][:32]) for k,v in m.named_parameters());hh=hashlib.sha256()
    for z in logs:hh.update(('\n'.join(z['batch_ids'])+'\n').encode())
    summary={'status':'completed','variant':a.variant,'seed':a.seed,'steps':8000,'new_steps':8000-step0,'parent_steps':step0,'parent_sha256':parent_hash,'parent_checkpoint':str(a.parent),'parameters':report(m),'input_tokens':sum(z['input_tokens'] for z in logs),'new_input_tokens':sum(z['input_tokens'] for z in logs if z['step']>2000),'supervised_tokens':sum(z['supervised_tokens'] for z in logs),'padded_tokens':sum(z['padded_tokens'] for z in logs),'examples':8000*32,'unique_examples':len(seen),'batch_sequence_sha256':hh.hexdigest(),'seconds':time.perf_counter()-start,'validation_nll':v,'checkpoint_sha256':ss,'changed_parameter_tensors':changed,'total_parameter_tensors':len(initial),'training_seed_not_new_relative_to_G3':True}
    atomic(out/'TRAIN_COMPLETE.json',summary);print('TRAIN_DONE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
