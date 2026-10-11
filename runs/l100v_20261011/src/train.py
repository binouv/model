"""Finite full-parameter continuation; exact resume only from this run's native state."""
from __future__ import annotations
import argparse,collections,gc,math,random,resource,time
import torch
from common import *
from data import pack
from engine import batch,loss_forward

def validation(m,tok,rr,device):
    m.eval();losses=[]
    with torch.no_grad():
        for row in rr:
            x,y,w=batch([pack(row,tok)],device)
            with torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):loss,ce=loss_forward(m,x,y,w)
            losses.append([float(loss),float(ce)])
    return {'weighted_nll':sum(x[0] for x in losses)/len(losses),'nll':sum(x[1] for x in losses)/len(losses),'n':len(rr)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--parent-config',type=Path,required=True)
    ap.add_argument('--resume',type=Path);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    cfg=json.loads((ROOT/'configs/preregistered.json').read_text());ph=sha(ROOT/'configs/preregistered.json');device=args.device
    torch.set_num_threads(4);torch.set_num_interop_threads(1);random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if (ROOT/'metrics/TRAINING_COMPLETE.json').exists():raise FileExistsError('Run already completed')
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');pool={}
    for stage in ['elementary','composed']:
        pool[stage]=collections.defaultdict(list)
        for r in rows(ROOT/'data'/f'{stage}.jsonl'):
            p=pack(r,tok);pool[stage][(r['family'],r['lang'])].append(p)
    val=rows(ROOT/'data/validation.jsonl')[::10]
    if args.resume:
        mf=verify_checkpoint(args.resume)
        if mf['protocol_sha256']!=ph:raise ValueError('Protocol differs from saved run')
        m=load_model(args.resume,device=device)
    else:
        if sha(args.parent)!=cfg['parent_sha256']:raise ValueError('Wrong parent weights')
        m=load_model(args.parent,args.parent_config,device);m.config.checkpoint_blocks=True
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=.01,foreach=False)
    rng=random.Random(cfg['sampler_seed']);start=0;counts={'input_tokens':0,'supervised_tokens':0,'padded_tokens':0};began=time.perf_counter()
    if args.resume:
        state=torch.load(args.resume/'training_state.pt',map_location=device,weights_only=False)
        opt.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu());rng.setstate(state['sampler_rng'])
        if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
        start=state['step'];counts=state['counts'];del state;gc.collect()
        # Preserve non-durable suffix separately instead of counting repeated steps twice.
        logfile=ROOT/'metrics/train.jsonl'
        if logfile.exists():
            old=rows(logfile);suffix=[z for z in old if z['step']>start]
            if suffix:write_rows(ROOT/'metrics'/f'non_durable_suffix_after_{start}.jsonl',suffix)
            write_rows(logfile,[z for z in old if z['step']<=start])
    else:
        initial=tensor_hashes(m);put(ROOT/'metrics/INITIAL_TENSORS.json',initial)
        print('INITIAL',validation(m,tok,val,device),flush=True)
    for step in range(start+1,cfg['updates']+1):
        stage='elementary' if step<=cfg['elementary_updates'] else 'composed'
        keys=sorted(pool[stage]);key=keys[(step-1)%len(keys)];rr=rng.choices(pool[stage][key],k=cfg['batch'])
        x,y,w=batch(rr,device);m.train();opt.zero_grad(set_to_none=True)
        lr=cfg['lr']*min(step/32,1.)
        for g in opt.param_groups:g['lr']=lr
        tick=time.perf_counter()
        with torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):loss,ce=loss_forward(m,x,y,w)
        loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,foreach=False))
        if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):raise RuntimeError('Nonfinite loss/gradient')
        opt.step();elapsed=time.perf_counter()-tick
        inp=sum(len(r['tokens'])-1 for r in rr);sup=int(y.ne(-100).sum());pad=x.numel()
        for k,v in zip(counts,(inp,sup,pad)):counts[k]+=v
        z={'step':step,'stage':stage,'family':key[0],'lang':key[1],'weighted_loss':float(loss.detach()),'nll':float(ce),
           'grad_norm':gn,'lr':lr,'seconds':elapsed,'input_tokens':inp,'supervised_tokens':sup,'padded_tokens':pad,
           'batch_ids':[r['id'] for r in rr],'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        with (ROOT/'metrics/train.jsonl').open('a') as f:f.write(json.dumps(z)+'\n');f.flush()
        del loss,ce,x,y,w
        if step%32==0:print('TRAIN',step,stage,key,round(z['weighted_loss'],4),round(z['nll'],4),round(elapsed,2),'RSS',round(z['peak_rss_mib']),flush=True)
        if step%cfg['save_every']==0 or step==cfg['updates']:
            opt.zero_grad(set_to_none=True);gc.collect();path=save_checkpoint(m,opt,step,rng,counts,ph,device)
            vv=validation(m,tok,val,device);put(ROOT/'metrics'/f'validation_{step:04d}.json',vv)
            print('CHECKPOINT',step,vv,flush=True)
    opt.zero_grad(set_to_none=True);del opt;gc.collect()
    before=json.loads((ROOT/'metrics/INITIAL_TENSORS.json').read_text());after=tensor_hashes(m)
    put(ROOT/'metrics/TRAINING_COMPLETE.json',{'status':'completed_training_not_yet_evaluated','updates':cfg['updates'],
      'parameters':sum(p.numel() for p in m.parameters()),'changed_tensors':sum(before[k]!=v for k,v in after.items()),
      'total_tensors':len(after),'all_parameters_trainable':all(p.requires_grad for p in m.parameters()),
      'all_values_finite':all(bool(torch.isfinite(p).all()) for p in m.parameters()),'counts':counts,'device':device,
      'wall_seconds_this_process':time.perf_counter()-began,'parent_warmstart_new_optimizer':True,'Qwen_superiority':False})
    print('TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
