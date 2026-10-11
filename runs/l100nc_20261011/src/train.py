"""Finite local full-parameter continuation; checkpoints before evaluation.
No process survives as an unreported background training promise.
"""
from __future__ import annotations
import argparse,collections,gc,hashlib,json,math,random,resource,time
import torch
from common import *

def new_optimizer(m):
    return torch.optim.Adafactor(m.parameters(),lr=.006,weight_decay=0.,foreach=False)

def update(m,opt,rr,lr):
    m.train();opt.zero_grad(set_to_none=True);x,y=batch(rr)
    for g in opt.param_groups:g['lr']=lr
    start=time.perf_counter()
    with torch.autocast('cpu',dtype=torch.bfloat16):loss=m(x,y)
    loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,foreach=False))
    if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):raise RuntimeError('nonfinite training')
    opt.step()
    return {'loss':float(loss.detach()),'grad_norm':gn,'lr':lr,'seconds':time.perf_counter()-start,
      'input_tokens':sum(len(r['ids'])-1 for r in rr),'supervised_tokens':int(y.ne(-100).sum()),
      'padded_tokens':x.numel(),'batch_ids':[r['id'] for r in rr],
      'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

@torch.inference_mode()
def val(m,rr):
    m.eval();total=nt=0
    for i in range(0,len(rr),2):
        x,y=batch(rr[i:i+2]);n=int(y.ne(-100).sum())
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=m(x,y)
        total+=float(loss)*n;nt+=n
    return {'completion_nll':total/nt,'tokens':nt,'cases':len(rr),'teacher_forced_not_solving':True}

def samples(m):
    return {n:p.detach().flatten()[::max(1,p.numel()//32)][:32].tolist() for n,p in m.named_parameters()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume');ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(111061);random.seed(111061)
    cfg=json.loads((ROOT/'configs/preregistered.json').read_text());steps=cfg['updates'];rng=random.Random(111062)
    m=load_model(a.resume);m.config.checkpoint_blocks=True;opt=new_optimizer(m);step0=0;records=[]
    if a.resume:
        st=torch.load(Path(a.resume)/'training_state.pt',weights_only=False,map_location='cpu')
        if st['schema']!='L100NC-v1' or st['protocol_sha256']!=sha(ROOT/'configs/preregistered.json'):raise ValueError('resume protocol mismatch')
        opt.load_state_dict(st['optimizer']);rng.setstate(st['sampler_rng']);random.setstate(st['python_rng']);torch.set_rng_state(st['torch_rng']);step0=st['step'];records=st['records']
    elif (ROOT/'metrics/TRAIN.jsonl').exists() and not a.probe:raise FileExistsError('existing run: resume rather than restart')
    before=samples(m);rr=tokenize(rows('train'));pool=collections.defaultdict(list)
    for r in rr:pool[r['family'],r['lang']].append(r)
    keys=sorted(pool);vr=tokenize(rows('validation'))
    print('LOADED',parameter_report(m),flush=True)
    if not a.probe and step0==0:put(ROOT/'metrics/INITIAL_VALIDATION.json',val(m,vr[:32]))
    if a.probe:
        result=[]
        for j in range(3):
            # Engineering only; short English samples, discarded updates.
            bb=pool['pipeline','en'][:cfg['batch']]
            result.append(update(m,opt,bb,.0001))
        put(ROOT/'metrics/ENGINEERING_THROUGHPUT.json',{'discarded':True,'rows':result})
        print(json.dumps(result),flush=True);return
    begin=time.perf_counter()
    for step in range(step0+1,steps+1):
        key=keys[rng.randrange(len(keys))];bb=rng.choices(pool[key],k=cfg['batch'])
        lr=.006*min(1.,step/32)*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
        rec=update(m,opt,bb,lr);rec.update(step=step,family=key[0],lang=key[1]);records.append(rec)
        with (ROOT/'metrics/TRAIN.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        if step%16==0:print('TRAIN',step,round(rec['loss'],4),round(rec['seconds'],2),int(rec['peak_rss_mb']),flush=True)
        if step%cfg['save_every']==0 or step==steps:
            opt.zero_grad(set_to_none=True);gc.collect();cp=save(m,opt,rng,step,records)
            put(ROOT/'reports/NEXT_STATE.json',{'status':'running' if step<steps else 'training_completed_evaluation_pending',
                'run':'L100NC','last_durable_step':step,'checkpoint':str(cp),'do_not_duplicate':True,'goal_achieved':False})
            put(ROOT/f'metrics/VALIDATION_{step:04d}.json',val(m,vr[:32]));print('CHECKPOINT',step,flush=True)
    final=samples(m);audit={k:final[k]!=v for k,v in before.items()}
    put(ROOT/'metrics/TRAINING_COMPLETED.json',{'status':'completed','updates':steps,'parameters':100028328,
        'input_tokens':sum(r['input_tokens'] for r in records),'supervised_tokens':sum(r['supervised_tokens'] for r in records),
        'padded_tokens':sum(r['padded_tokens'] for r in records),'presentations':steps*cfg['batch'],
        'unique_training_cases_presented':len({i for r in records for i in r['batch_ids']}),
        'changed_sampled_tensors':sum(audit.values()),'tensor_count':len(audit),'all_parameters_trainable':all(p.requires_grad for p in m.parameters()),
        'all_values_finite':all(bool(torch.isfinite(p).all()) for p in m.parameters()),
        'training_step_seconds':sum(r['seconds'] for r in records),'elapsed_this_invocation':time.perf_counter()-begin,
        'new_optimizer':True,'exact_ancestor_resume':False})
    save(m,None,rng,steps,records,dtype=torch.bfloat16)
    print('TRAINING_DONE',flush=True)
if __name__=='__main__':main()
