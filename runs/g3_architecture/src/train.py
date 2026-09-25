"""Finite paired architecture screen. Checkpoints carry optimizer/RNG and data hashes.
Each model receives exactly the same ordered batches, shuffled by a data-only seed.
"""
from __future__ import annotations
import argparse,collections,csv,gc,hashlib,json,math,os,random,sys,time
from pathlib import Path
import torch
from model import Model,Config,report
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
SPLITS=['test_iid','test_surface','test_extrapolation','test_counterfactual']

def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def rows(split):return [json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
def batch(rr):
    T=max(len(z['tokens']) for z in rr)-1;x=torch.zeros((len(rr),T),dtype=torch.long);y=torch.full_like(x,-100)
    for i,z in enumerate(rr):
        t=torch.tensor(z['tokens']);n=len(t)-1;x[i,:n]=t[:-1];y[i,z['prompt_tokens']-1:n]=t[z['prompt_tokens']:]
    return x,y
@torch.inference_mode()
def val(m,rr):
    m.eval();s=0;n=0
    for j in range(0,len(rr),32):
        x,y=batch(rr[j:j+32]);nn=int(y.ne(-100).sum());s+=float(m(x,y))*nn;n+=nn
    return s/n
@torch.inference_mode()
def evaluate(m,tok,rr,reset=False):
    m.eval();groups=collections.defaultdict(list);out=[]
    for z in rr:groups[z['prompt_tokens']].append(z)
    for plen,gg in sorted(groups.items()):
        for j in range(0,len(gg),32):
            zz=gg[j:j+32];prompt_ids=[tok.encode(z['prompt'],bos=True) for z in zz]
            preds=m.generate(torch.tensor(prompt_ids),max_new=16,reset_memory=reset)
            for z,pr in zip(zz,preds.tolist()):
                eos=2 in pr
                if eos:pr=pr[:pr.index(2)]
                text=tok.decode(pr)
                out.append({'id':z['id'],'family':z['family'],'lang':z['lang'],'pair_id':z.get('pair_id'),'prompt':z['prompt'],'gold':z['answer'],'generated':text,'token_ids':pr,'eos':eos,'exact':text.strip()==z['answer']})
    out.sort(key=lambda z:z['id']);pairs=collections.defaultdict(list)
    for z in out:
        if z['pair_id']:pairs[z['pair_id']].append(z)
    by={f:{'n':sum(z['family']==f for z in out),'correct':sum(z['exact'] and z['family']==f for z in out)} for f in sorted({z['family'] for z in out})}
    return {'status':'completed','n':len(out),'correct':sum(z['exact'] for z in out),'eos_count':sum(z['eos'] for z in out),'by_family':by,'pair_count':len(pairs),'both_correct':sum(all(z['exact'] for z in rr) for rr in pairs.values()),'rows':out}
def save(path,m,opt,step,rng,logs,config):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    state={'schema':'G3-v1','config':config,'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'sampler_rng':rng.getstate(),'logs':logs,'data_hashes':json.loads((ROOT/'data/manifest.json').read_text())['hashes']}
    tmp=p.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,p)
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--variant',choices=['full','shared','hybrid'],required=True);a.add_argument('--seed',type=int,default=7301);a.add_argument('--steps',type=int,default=2000);a.add_argument('--batch',type=int,default=32);a.add_argument('--threads',type=int,default=4);a.add_argument('--probe',action='store_true');a.add_argument('--eval-only');x=a.parse_args()
    torch.set_num_threads(x.threads);torch.set_num_interop_threads(1);torch.manual_seed(x.seed)
    cfg=Config(variant=x.variant);m=Model(cfg);name=f'{x.variant}_s{x.seed}';out=ROOT/'metrics'/name;out.mkdir(parents=True,exist_ok=True)
    if x.eval_only:
        st=torch.load(x.eval_only,weights_only=False);m.load_state_dict(st['model']);tok=Tokenizer.load(ROOT/'data/tokenizer.json')
        for split in SPLITS:atomic(out/f'{split}.json',evaluate(m,tok,rows(split)))
        if x.variant=='hybrid':
            for split in SPLITS:atomic(out/f'reset_{split}.json',evaluate(m,tok,rows(split),True))
        atomic(out/'EVAL_COMPLETE.json',{'status':'completed','checkpoint':x.eval_only});return
    train=rows('train');validation=rows('validation');pools=collections.defaultdict(list)
    for z in train:pools[(z['family'],z['lang'])].append(z)
    keys=sorted(pools);rng=random.Random(73311);opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01)
    initial={k:v.detach().flatten()[::max(1,v.numel()//32)][:32].clone() for k,v in m.named_parameters()}
    logs=[];seen=set();digest=hashlib.sha256();input_tokens=0;supervised=0;padded=0;start=time.perf_counter()
    if not x.probe:
        if (out/'train.jsonl').exists():raise RuntimeError('refuse overwrite existing run')
        atomic(out/'initial.json',{'parameter_report':report(m),'validation_nll':val(m,validation)})
    for step in range(1,x.steps+1):
        m.train();opt.zero_grad(set_to_none=True);t=time.perf_counter()
        key=keys[rng.randrange(len(keys))];rr=rng.choices(pools[key],k=x.batch)
        ids=[z['id'] for z in rr];digest.update(('\n'.join(ids)+'\n').encode());seen.update(ids)
        xx,yy=batch(rr);inp=sum(len(z['tokens'])-1 for z in rr);sup=int(yy.ne(-100).sum());input_tokens+=inp;supervised+=sup;padded+=xx.numel()
        loss=m(xx,yy);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(grad):raise RuntimeError('nonfinite training')
        lr=.001*min(1.,step/100)*(.15+.85*.5*(1+math.cos(math.pi*step/x.steps)))
        for g in opt.param_groups:g['lr']=lr
        opt.step();row={'step':step,'loss':float(loss.detach()),'grad_norm':grad,'lr':lr,'seconds':time.perf_counter()-t,'input_tokens':inp,'supervised_tokens':sup,'padded_tokens':xx.numel(),'batch_ids':ids,'family':key[0],'lang':key[1]};logs.append(row)
        if not x.probe:
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step%100==0 or step==1:print(name,'STEP',step,'LOSS',round(row['loss'],4),'SEC',round(row['seconds'],3),flush=True)
        if x.probe and step>=10:
            print('PROBE',json.dumps({'parameters':report(m),'median_seconds':sorted(z['seconds'] for z in logs)[len(logs)//2],'peak_first':logs[0]['seconds']}),flush=True);return
        if step%500==0 or step==x.steps:
            opt.zero_grad(set_to_none=True);v=val(m,validation);atomic(out/f'val{step:04d}.json',{'step':step,'nll':v});print(name,'VAL',step,v,flush=True)
            cp=ROOT/'checkpoints'/name/f'step{step:04d}.pt';sha=save(cp,m,opt,step,rng,logs,vars(cfg))
            atomic(out/'progress.json',{'status':'completed_training' if step==x.steps else 'running','durable_step':step,'checkpoint':str(cp),'sha256':sha})
    changed=[k for k,v in m.named_parameters() if not torch.equal(initial[k],v.detach().flatten()[::max(1,v.numel()//32)][:32])]
    summary={'status':'completed','variant':x.variant,'seed':x.seed,'steps':x.steps,'batch':x.batch,'examples':x.steps*x.batch,'unique_examples':len(seen),'input_tokens':input_tokens,'supervised_tokens':supervised,'padded_tokens':padded,'batch_sequence_sha256':digest.hexdigest(),'seconds':time.perf_counter()-start,'parameters':report(m),'changed_parameter_tensors':len(changed),'total_parameter_tensors':len(list(m.parameters())),'checkpoint_sha256':sha,'validation_nll':v}
    atomic(out/'TRAIN_COMPLETE.json',summary);print('TRAIN_DONE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
