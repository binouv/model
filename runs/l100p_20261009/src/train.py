"""Finite local full-backbone continuation: one pointer arm and one no-copy arm."""
from __future__ import annotations
import argparse,collections,copy,json,math,random,re,resource,sys,time,gc,hashlib
from pathlib import Path
import torch
from pointer_model import (ROOT,PointerModel,Tokenizer,init_parent,load_checkpoint,
                          save_checkpoint,atomic_json,sha,state_hash)
from data import read,training_rows,batch,pack
SPLITS=['iid','surface','extrapolation','counterfactual','parent_regression']
def parse(text):
    if text.count('F=')!=1:return None
    m=re.fullmatch(r'(?:T=[^;\n]+;)?F=(-?\d+)',text.strip())
    return int(m.group(1)) if m else None
@torch.inference_mode()
def predict(m,tok,prompts):
    # Inference accepts no cases, answers, candidate list, or executable code.
    ids=[tok.encode(p,bos=True) for p in prompts]
    if len({len(z) for z in ids})!=1:raise ValueError('prompt length mismatch')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=m.generate(torch.tensor(ids),max_new_tokens=64)
    result=[]
    for seq in out.tolist():
        eos=tok.EOS in seq;seq=seq[:seq.index(tok.EOS)] if eos else seq
        result.append({'text':tok.decode(seq),'token_ids':seq,'eos':eos})
    return result
@torch.inference_mode()
def evaluate(m,tok,rows):
    m.eval();groups=collections.defaultdict(list);out=[];start=time.perf_counter()
    for z in rows:groups[len(tok.encode(z['prompt'],bos=True))].append(z)
    for _,rr in sorted(groups.items()):
        for j in range(0,len(rr),8):
            bb=rr[j:j+8];pred=predict(m,tok,[z['prompt'] for z in bb])
            for z,p in zip(bb,pred):
                val=parse(p['text']);exact=p['eos'] and val==int(z['answer']) if val is not None else False
                trace=p['eos'] and p['text'].strip()=='T='+z['canonical_trace']+';F='+z['answer']
                out.append({**p,'id':z['id'],'family':z['family'],'lang':z['lang'],
                 'prompt':z['prompt'],'pair_id':z.get('pair_id'),'gold':z['answer'],
                 'parsed_answer':val,'exact':exact,'exact_legacy_trace':trace})
    pairs=collections.defaultdict(list)
    for z in out:
        if z['pair_id'] is not None:pairs[z['pair_id']].append(z)
    if not all(len(z)==2 for z in pairs.values()):raise ValueError('incomplete counterfactual pairs')
    return {'status':'completed','n':len(out),'correct':sum(z['exact'] for z in out),
      'accuracy':sum(z['exact'] for z in out)/len(out),
      'truncated':sum(not z['eos'] for z in out),
      'by_family':{f:{'n':len(rr:= [z for z in out if z['family']==f]),'correct':sum(z['exact'] for z in rr)} for f in sorted({z['family'] for z in out})},
      'pairs':len(pairs),'both_correct_pairs':sum(all(z['exact'] for z in p) for p in pairs.values()),
      'changed_answer_pairs':sum(p[0]['parsed_answer']!=p[1]['parsed_answer'] for p in pairs.values()),
      'seconds':time.perf_counter()-start,'generated_tokens':sum(len(z['token_ids']) for z in out),
      'rows':sorted(out,key=lambda z:z['id'])}
@torch.inference_mode()
def validation(m,tok):
    rr=[pack(z,tok) for z in read(ROOT/'data/validation.jsonl')[:40]]
    total=0.;n=0;m.eval()
    for i in range(0,len(rr),4):
        x,y,p=batch(rr[i:i+4]);nn=int(y.ne(-100).sum())
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=m(x,y,prefix_lens=p)
        total+=float(loss)*nn;n+=nn
    return {'completion_nll':total/n,'tokens':n,'n':len(rr)}
def pools():
    pp=collections.defaultdict(list)
    for z in training_rows():pp[z['family'],z['prompt_tokens']].append(z)
    keys=[k for k in sorted(pp) if len(pp[k])>=4]
    return keys,pp
def one_update(m,opt,rr,lr):
    x,y,p=batch(rr);m.train();opt.zero_grad(set_to_none=True)
    for g in opt.param_groups:g['lr']=lr
    start=time.perf_counter()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss,gate=m(x,y,prefix_lens=p,return_gate=True)
    loss.backward()
    gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,foreach=False))
    if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):raise RuntimeError('nonfinite training')
    opt.step()
    return {'loss':float(loss.detach()),'gate_mean':float(gate),'lr':lr,'grad_norm':gn,
     'seconds':time.perf_counter()-start,'input_tokens':sum(len(z['tokens'])-1 for z in rr),
     'supervised_tokens':int(y.ne(-100).sum()),'padded_tokens':x.numel(),'batch_ids':[z['id'] for z in rr],
     'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
def samples(m):
    return {n:p.detach().flatten()[::max(1,p.numel()//32)][:32].float().tolist() for n,p in m.named_parameters()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True)
    ap.add_argument('--arm',choices=['pointer','no_copy','baseline'],required=True)
    ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    random.seed(9201);torch.manual_seed(9201)
    tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    if (ROOT/'metrics'/f'{a.arm}_DONE.json').exists():raise FileExistsError('arm already done; do not duplicate')
    m=init_parent(a.parent,enabled=a.arm=='pointer')
    print('LOADED',a.arm,sum(p.numel() for p in m.parameters()),flush=True)
    if a.arm!='baseline':
        cfg=json.loads((ROOT/'configs/preregistered.json').read_text());steps=cfg['updates_per_arm']
        opt=torch.optim.Adafactor(m.parameters(),lr=.006,foreach=False,weight_decay=0)
        rng=random.Random(92021);keys,pp=pools();logs=[];before=samples(m);initial_hash=state_hash(m)
        vv=validation(m,tok);atomic_json(ROOT/'metrics'/f'{a.arm}_initial_validation.json',vv)
        print('VALIDATION_INITIAL',a.arm,vv,flush=True)
        started=time.perf_counter()
        for step in range(1,steps+1):
            k=keys[rng.randrange(len(keys))];bb=rng.choices(pp[k],k=4)
            lr=.006*min(1,step/32)*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
            row=one_update(m,opt,bb,lr);row['step']=step;logs.append(row)
            with (ROOT/'metrics'/f'{a.arm}_TRAIN.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if step%32==0:
                print('TRAIN',a.arm,step,round(row['loss'],4),round(row['gate_mean'],4),round(row['seconds'],2),flush=True)
                atomic_json(ROOT/'reports/PROGRESS.json',{'status':'running','arm':a.arm,'step':step,'target':steps})
            if step%256==0 or step==steps:
                opt.zero_grad(set_to_none=True);gc.collect()
                counts={key:sum(z[key] for z in logs) for key in ['input_tokens','supervised_tokens','padded_tokens']}
                cp=save_checkpoint(m,ROOT/f'checkpoints/{a.arm}/step_{step:04d}',step,opt,rng,counts)
                atomic_json(ROOT/'reports/NEXT_STATE.json',{'status':'partial_paired_experiment','arm':a.arm,'last_durable_step':step,'target':steps,'checkpoint':str(cp),'do_not_duplicate':True})
                vv=validation(m,tok);atomic_json(ROOT/'metrics'/f'{a.arm}_validation_{step:04d}.json',vv)
                print('CHECKPOINT',a.arm,step,vv,flush=True)
        opt.zero_grad(set_to_none=True)
        after=samples(m)
        trained={'status':'completed_training','arm':a.arm,'steps':steps,
          'parameters':sum(p.numel() for p in m.parameters()),'initial_hash':initial_hash,'final_hash':state_hash(m),
          'sampled_changed_tensors':sum(after[k]!=v for k,v in before.items()),
          'unchanged_sampled_tensors':[k for k,v in before.items() if after[k]==v],
          'tensor_count':len(before),'counts':counts,'elapsed_seconds':time.perf_counter()-started,
          'training_seconds':sum(z['seconds'] for z in logs),'batch_sequence_sha256':hashlib.sha256(json.dumps([z['batch_ids'] for z in logs]).encode()).hexdigest(),
          'all_values_finite':all(bool(torch.isfinite(p).all()) for p in m.parameters()),
          'copier_parameters_in_control_have_zero_gradient':a.arm=='no_copy',
          'parent_fp32_optimizer_was_not_available':True,'warmstart_new_optimizer':True}
        atomic_json(ROOT/'metrics'/f'{a.arm}_TRAINING.json',trained)
        save_checkpoint(m,ROOT/f'checkpoints/{a.arm}/export_bf16',steps,counts=counts,dtype=torch.bfloat16)
        del opt;gc.collect()
    tests={}
    for split in SPLITS:
        rr=read(ROOT/'data'/f'{split}.jsonl')
        tests[split]=evaluate(m,tok,rr)
        atomic_json(ROOT/'metrics'/f'{a.arm}_{split}.json',tests[split])
        print('EVAL',a.arm,split,tests[split]['correct'],tests[split]['n'],'cf',tests[split]['both_correct_pairs'],flush=True)
    atomic_json(ROOT/'metrics'/f'{a.arm}_DONE.json',{'status':'completed','arm':a.arm,'evaluations':SPLITS})
    print('DONE',a.arm,flush=True)
if __name__=='__main__':main()
