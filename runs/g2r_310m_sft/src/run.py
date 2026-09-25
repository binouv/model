"""G2R: finite matched full-parameter response-loss experiment from durable G1.
No exact G2-step64 resume: only G1 BF16 weight bytes are mounted.
"""
from __future__ import annotations
import argparse,collections,gc,hashlib,json,math,os,random,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base_src'))
from model import parameter_report
from checkpoint_io import load_checkpoint,save_checkpoint
from tokenizer import Tokenizer
SPLITS=('test_iid','test_surface','test_extrapolation','test_counterfactual')

def atomic_json(p,z):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)

def load_rows(split):return [json.loads(x) for x in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]

def batch(rows,arm):
    n=max(len(z['tokens']) for z in rows)-1
    x=torch.zeros((len(rows),n),dtype=torch.long);y=torch.full_like(x,-100)
    for i,z in enumerate(rows):
        t=torch.tensor(z['tokens']);nn=len(t)-1;x[i,:nn]=t[:-1];y[i,:nn]=t[1:]
        if arm=='answer_only':y[i,:z['prompt_tokens']-1]=-100
    return x,y

@torch.inference_mode()
def validation(m,rows,n=40):
    m.eval();loss=0.;nt=0;acc=0
    for i in range(0,min(n,len(rows)),4):
        x,y=batch(rows[i:i+4],'answer_only');ll,lg=m(x,y);nn=int(y.ne(-100).sum())
        nt+=nn;loss+=float(ll)*nn;acc+=int((lg.argmax(-1).eq(y)&y.ne(-100)).sum())
    return {'response_nll':loss/nt,'teacher_forced_token_accuracy':acc/nt,'response_tokens':nt,'examples':min(n,len(rows))}

@torch.inference_mode()
def generate_texts(m,tok,prompts):
    """The model interface receives prompts only, not cases/labels/completions."""
    plen=len(tok.encode(prompts[0],bos=True))
    ids=torch.tensor([tok.encode(p,bos=True) for p in prompts]);assert ids.shape[1]==plen
    pred=m.generate(ids,max_new_tokens=8,eos_id=tok.EOS)
    out=[]
    for pp in pred:
        new=pp[plen:].tolist();eos=tok.EOS in new
        if eos:new=new[:new.index(tok.EOS)]
        out.append((tok.decode(new),new,eos))
    return out

@torch.inference_mode()
def evaluate(m,tok,arm,split):
    m.eval();rows=load_rows(split);groups=collections.defaultdict(list)
    for z in rows:groups[z['prompt_tokens']].append(z)
    out=[];elapsed=time.perf_counter()
    for plen,rr in sorted(groups.items()):
        for start in range(0,len(rr),4):
            bb=rr[start:start+4];pred=generate_texts(m,tok,[z['prompt'] for z in bb])
            for z,(text,idsnew,eos) in zip(bb,pred):
                out.append({'id':z['id'],'pair_id':z.get('pair_id'),'family':z['family'],'lang':z['lang'],
                  'prompt':z['prompt'],'gold':z['answer'],'generated':text,'token_ids':idsnew,
                  'exact':text.strip()==z['answer'],'emitted_eos':eos})
    out.sort(key=lambda z:z['id']);by={}
    for f in sorted({z['family'] for z in out}):
        zz=[z for z in out if z['family']==f];by[f]={'n':len(zz),'correct':sum(z['exact'] for z in zz),'accuracy':sum(z['exact'] for z in zz)/len(zz)}
    result={'status':'completed','arm':arm,'split':split,'n':len(out),'correct':sum(z['exact'] for z in out),
            'accuracy':sum(z['exact'] for z in out)/len(out),'by_family':by,'rows':out,
            'seconds':time.perf_counter()-elapsed,'sampling':'greedy_full_vocab_8192','max_new_tokens':8,'tools_used':False}
    if split=='test_counterfactual':
        pairs=collections.defaultdict(list)
        for z in out:pairs[z['pair_id']].append(z)
        assert all(len(zz)==2 for zz in pairs.values())
        result['both_correct']=sum(all(x['exact'] for x in zz) for zz in pairs.values())
        result['pair_count']=len(pairs);result['pair_accuracy']=result['both_correct']/len(pairs)
    atomic_json(ROOT/'metrics'/f'{arm}_{split}.json',result)
    print('EVAL',arm,split,result['accuracy'],flush=True)
    return result

def fp_samples(m):
    return {n:p.detach().flatten()[::max(1,p.numel()//32)][:32].float().tolist() for n,p in m.named_parameters()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True)
    ap.add_argument('--arm',choices=['answer_only','full_sequence','baseline'],required=True)
    ap.add_argument('--steps',type=int,default=256);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--resume');ap.add_argument('--eval-only',action='store_true');a=ap.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(7201);random.seed(7201)
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');m,base_idx=load_checkpoint(a.resume or a.base,dtype=torch.float32)
    m.config.checkpoint_blocks=True
    assert sum(p.numel() for p in m.parameters())==309993253
    print('LOAD',a.arm,parameter_report(m),flush=True)
    outdir=ROOT/'checkpoints'/a.arm;outdir.mkdir(parents=True,exist_ok=True)
    if a.arm!='baseline' and not a.eval_only:
        train=load_rows('train');val=load_rows('validation');pools=collections.defaultdict(list)
        for z in train:pools[(z['family'],z['lang'])].append(z)
        pool_keys=sorted(pools);rng=random.Random(72011)
        opt=torch.optim.Adafactor(m.parameters(),lr=.006,weight_decay=0,foreach=False)
        step0=0;tokens=0;loss_tokens=0;start_samples=fp_samples(m)
        if a.resume:
            st=torch.load(Path(a.resume)/'training_state.pt',weights_only=False)
            opt.load_state_dict(st['optimizer']);torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng'])
            rng.setstate(st['sampler_rng']);step0=st['step'];tokens=base_idx['training']['input_tokens'];loss_tokens=base_idx['training']['supervised_tokens']
        logs=ROOT/'metrics'/f'{a.arm}_steps.jsonl'
        if logs.exists() and not a.resume:raise RuntimeError('Refuse silently append fresh training to existing log')
        start=time.perf_counter();vi=validation(m,val)
        atomic_json(ROOT/'metrics'/f'{a.arm}_initial_validation.json',vi);print('VAL_INITIAL',a.arm,vi,flush=True)
        for step in range(step0+1,a.steps+1):
            t0=time.perf_counter();m.train();opt.zero_grad(set_to_none=True)
            key=rng.choice(pool_keys);rr=rng.choices(pools[key],k=4);x,y=batch(rr,a.arm)
            loss,lg=m(x,y);loss.backward();del lg
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss')
            grad=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,foreach=False))
            if not math.isfinite(grad):raise RuntimeError('nonfinite grad norm')
            lr=.006*min(1.,step/16)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
            for group in opt.param_groups:group['lr']=lr
            opt.step();sup=int(y.ne(-100).sum());inp=sum(len(z['tokens'])-1 for z in rr)
            tokens+=inp;loss_tokens+=sup
            row={'step':step,'train_loss':float(loss.detach()),'lr':lr,'grad_norm':grad,'input_tokens':inp,
                 'supervised_tokens':sup,'cumulative_input_tokens':tokens,'cumulative_supervised_tokens':loss_tokens,
                 'padded_tokens':x.numel(),'family':key[0],'lang':key[1],'batch_ids':[z['id'] for z in rr],
                 'seconds':time.perf_counter()-t0,'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
            with logs.open('a') as f:f.write(json.dumps(row)+'\n')
            if step%8==0:print('TRAIN',a.arm,step,round(row['train_loss'],4),round(row['seconds'],2),flush=True)
            atomic_json(ROOT/'metrics'/f'{a.arm}_progress.json',{'status':'running','last_optimizer_step':step,'input_tokens':tokens,'supervised_tokens':loss_tokens})
            del loss,x,y
            if step%64==0 or step==a.steps:
                opt.zero_grad(set_to_none=True);gc.collect();vv=validation(m,val)
                atomic_json(ROOT/'metrics'/f'{a.arm}_validation_{step:04d}.json',vv);print('VAL',a.arm,step,vv,flush=True)
                cp=outdir/f'step_{step:04d}'
                if cp.exists():raise RuntimeError('refuse overwrite checkpoint')
                save_checkpoint(m,cp,opt,step,rng.getstate(),{'arm':a.arm,'input_tokens':tokens,'supervised_tokens':loss_tokens,
                       'base_training_step':192,'base_precision':'BF16 rounded -> FP32','all_parameters_trainable':True,'new_optimizer':True})
                atomic_json(ROOT/'metrics'/f'{a.arm}_checkpoint_status.json',{'last_durable_step':step,'path':str(cp),'status':'completed_checkpoint_not_final_experiment'})
                print('CHECKPOINT',str(cp),flush=True)
        opt.zero_grad(set_to_none=True);gc.collect();after=fp_samples(m)
        changed={k:after[k]!=v for k,v in start_samples.items()}
        atomic_json(ROOT/'metrics'/f'{a.arm}_training.json',{'status':'completed','steps':a.steps,'input_tokens':tokens,
           'supervised_tokens':loss_tokens,'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
           'changed_sampled_tensors':sum(changed.values()),'total_tensors':len(changed),
           'unchanged_sampled_tensors':[k for k,v in changed.items() if not v],
           'elapsed':time.perf_counter()-start,'base_index_sha256':hashlib.sha256((Path(a.base)/'index.json').read_bytes()).hexdigest(),
           'warm_start_not_exact_G1_or_G2_resume':True})
        save_checkpoint(m,outdir/'release_bf16',step=a.steps,training={'stage':'G2R_experimental_SFT','arm':a.arm,
                     'input_tokens':tokens,'supervised_tokens':loss_tokens},dtype=torch.bfloat16)
        del opt;gc.collect()
    for split in SPLITS:evaluate(m,tok,a.arm,split)
    atomic_json(ROOT/'metrics'/f'{a.arm}_DONE.json',{'status':'completed','arm':a.arm,'all_splits_completed':True})
    print('DONE',a.arm,flush=True)
if __name__=='__main__':main()
