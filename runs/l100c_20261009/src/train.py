"""Local paired 100M continuation. Fixed endpoints, complete-only evaluations.
Control computes the same positive/negative forward graph with rank weight zero.
Native checkpoint contains the new optimizer and RNG for exact stage resume.
"""
from __future__ import annotations
import argparse,collections,copy,gc,json,math,random,resource,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from common import *
from checkpoint_io import load,save,state_hash
ARMS=('contrastive','nll_control')

def pair_list(split='train'):
 rr=read(ROOT/'data'/f'{split}.jsonl');gg=collections.OrderedDict()
 for r in rr:gg.setdefault(r['pair_id'],[]).append(r)
 assert all(len(v)==2 for v in gg.values())
 return list(gg.values())

class Sampler:
 def __init__(self,cfg):
  self.cfg=cfg;self.rng=random.Random(cfg['sampler_seed']);self.pools=collections.defaultdict(list)
  for pair in pair_list():self.pools[(pair[0]['family'],pair[0]['lang'])].append(pair)
  self.keys=sorted(self.pools);self.consumed=0
 def next(self):
  k=self.rng.choice(self.keys);pairs=self.rng.choices(self.pools[k],k=self.cfg['pairs_per_batch']);self.consumed+=len(pairs);return pairs
 def state(self):return {'rng':self.rng.getstate(),'consumed_pairs':self.consumed}
 def restore(self,s):self.rng.setstate(s['rng']);self.consumed=s['consumed_pairs']

def make_batch(pairs,tok):
 """For each query: correct then swapped target. Targets are offline supervision."""
 rows=[];ids=[]
 for pp in pairs:
  assert len(pp)==2 and pp[0]['answer']!=pp[1]['answer']
  for i,q in enumerate(pp):
   prompt=tok.encode(q['prompt'],bos=True)
   for target in [q['answer'],pp[1-i]['answer']]:
    tt=prompt+tok.encode('F='+target,eos=True)
    rows.append((tt,len(prompt)))
   ids.append(q['id'])
 T=max(len(t)-1 for t,p in rows);x=torch.zeros((len(rows),T),dtype=torch.long);y=torch.full_like(x,-100)
 for i,(tt,p) in enumerate(rows):
  x[i,:len(tt)-1]=torch.tensor(tt[:-1]);y[i,p-1:len(tt)-1]=torch.tensor(tt[p:])
 return x,y,ids

def loss_components(m,x,y,lam,margin):
 losses=m(x,y,loss_reduction='none');valid=y.ne(-100)
 rowidx=torch.arange(y.shape[0],device=y.device)[:,None].expand_as(y)[valid]
 sums=torch.zeros(y.shape[0],device=losses.device,dtype=losses.dtype).scatter_add(0,rowidx,losses)
 lens=valid.sum(1);scores=-sums/lens
 assert torch.equal(lens[::2],lens[1::2])
 nll=sums[::2].sum()/lens[::2].sum()
 rank=F.softplus(margin+scores[1::2]-scores[::2]).mean()
 return nll+lam*rank,nll,rank,scores,lens

def sample_parameters(m):
 return {n:p.detach().flatten()[::max(1,p.numel()//32)][:32].float().tolist() for n,p in m.named_parameters()}

def update(m,opt,pairs,tok,lr,lam,margin):
 m.train();opt.zero_grad(set_to_none=True);x,y,ids=make_batch(pairs,tok)
 for g in opt.param_groups:g['lr']=lr
 started=time.perf_counter()
 with torch.autocast('cpu',dtype=torch.bfloat16):loss,nll,rank,scores,lens=loss_components(m,x,y,lam,margin)
 loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.,foreach=False))
 if not math.isfinite(gn) or not torch.isfinite(loss):raise ValueError('nonfinite update')
 opt.step()
 return {'loss':float(loss.detach()),'positive_nll':float(nll.detach()),'rank_loss':float(rank.detach()),'correct_over_wrong_fraction':float((scores[::2]>scores[1::2]).float().mean().detach()),'grad_norm':gn,'lr':lr,'seconds':time.perf_counter()-started,'scored_input_tokens':sum(len(tok.encode(z['prompt'],bos=True))+len(tok.encode('F='+z['answer'],eos=True))-1 for pp in pairs for z in pp)*2,'positive_supervised_tokens':int(lens[::2].sum()),'scored_supervised_tokens':int(lens.sum()),'padded_tokens':x.numel(),'batch_ids':ids,'rss_max_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}

@torch.inference_mode()
def validate(m,tok):
 m.eval();rr=pair_list('validation');stats=[]
 for pp in rr:
  x,y,_=make_batch([pp],tok)
  with torch.autocast('cpu',dtype=torch.bfloat16):_,nll,rank,score,lens=loss_components(m,x,y,.5,.2)
  stats.append({'pair_id':pp[0]['pair_id'],'positive_nll':float(nll),'rank_loss':float(rank),'rank_correct':int((score[::2]>score[1::2]).sum()),'queries':2})
 return {'n_pairs':len(stats),'n_queries':2*len(stats),'positive_nll':sum(x['positive_nll'] for x in stats)/len(stats),'ranking_accuracy':sum(x['rank_correct'] for x in stats)/(2*len(stats)),'not_free_generation':True,'rows':stats}

def train(arm,parent,resume=None):
 cfg=json.loads((ROOT/'configs/preregistered.json').read_text());cfg={**cfg,'arm':arm};done=ROOT/'metrics'/f'{arm}_TRAINING_COMPLETE.json'
 if done.exists():raise FileExistsError('already complete; no duplicate training')
 logpath=ROOT/'metrics'/f'{arm}_steps.jsonl';sam=Sampler(cfg);tok=Tokenizer.load(ROOT/'data/tokenizer.json')
 if not resume:
  if logpath.exists():raise FileExistsError('fresh run would mix existing logs; explicit resume required')
  if sha(Path(parent)/'model.safetensors')!=cfg['starting_weights_sha256']:raise ValueError('wrong parent')
  m,mf=load(parent);m=m.float();m.config.checkpoint_blocks=cfg['activation_checkpointing'];opt=torch.optim.Adafactor(m.parameters(),lr=cfg['learning_rate'],foreach=False,weight_decay=0.)
  torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);start=0;counts={'scored_input_tokens':0,'positive_supervised_tokens':0,'scored_supervised_tokens':0,'padded_tokens':0,'positive_presentations':0}
 else:
  m,mf=load(resume);st=torch.load(Path(resume)/'training_state.pt',weights_only=False,map_location='cpu')
  assert st['run_config']['arm']==arm and st['run_config']['run']==cfg['run']
  opt=torch.optim.Adafactor(m.parameters(),lr=cfg['learning_rate'],foreach=False,weight_decay=0.);opt.load_state_dict(st['optimizer']);sam.restore(st['sampler_state']);torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);start=st['step']-1024;counts=st['counts']
  logs=read(logpath);canonical=[x for x in logs if x['local_step']<=start]
  if len(canonical)!=len(logs):
   put(ROOT/'metrics'/f'{arm}_uncheckpointed_excluded.json',[x for x in logs if x['local_step']>start]);logpath.write_text(''.join(json.dumps(z)+'\n' for z in canonical))
 assert sum(p.numel() for p in m.parameters())==100028328 and all(p.requires_grad for p in m.parameters())
 init=sample_parameters(m);put(ROOT/'metrics'/f'{arm}_initial_parameter_samples.json',init)
 if start==0:
  put(ROOT/'metrics'/f'{arm}_initial_validation.json',validate(m,tok));put(ROOT/'metrics'/f'{arm}_initial_state.json',{'sha256':state_hash(m),'exact_ancestor_resume':False})
 print('TRAIN_START',arm,start,flush=True);lam=cfg['contrastive_lambda'] if arm=='contrastive' else 0.;tstart=time.perf_counter()
 for step in range(start+1,cfg['updates_per_arm']+1):
  pairs=sam.next();lr=cfg['learning_rate']*min(1,step/cfg['warmup'])*(cfg['cosine_floor']+(1-cfg['cosine_floor'])*.5*(1+math.cos(math.pi*step/cfg['updates_per_arm'])))
  rec=update(m,opt,pairs,tok,lr,lam,cfg['ranking_margin']);rec['local_step']=step;rec['global_step']=step+1024
  for k in counts:
   counts[k]+=rec[k] if k!='positive_presentations' else 2*len(pairs)
  with logpath.open('a') as f:f.write(json.dumps(rec)+'\n')
  if step%16==0:put(ROOT/'metrics'/f'{arm}_PROGRESS.json',{'status':'running','last_optimizer_step':step,'counts':counts,'last':rec})
  if step%64==0:print('TRAIN',arm,step,round(rec['positive_nll'],4),round(rec['rank_loss'],4),round(rec['seconds'],3),flush=True)
  if step%cfg['save_interval']==0 or step==cfg['updates_per_arm']:
   opt.zero_grad(set_to_none=True);gc.collect();cp=ROOT/'checkpoints'/arm/f'step_{1024+step:04d}'
   save(m,cp,step=1024+step,optimizer=opt,sampler=sam,run_config=cfg,counts=counts)
   put(ROOT/'reports/NEXT_STATE.json',{'status':'running','active_arm':arm,'last_durable_global_step':1024+step,'last_durable_stage_step':step,'checkpoint_relative':str(cp.relative_to(ROOT)),'native_optimizer_rng':True,'completed_models_not_whole_pair':False,'do_not_duplicate':True,'goal_achieved':False})
   put(ROOT/'metrics'/f'{arm}_validation_{step:04d}.json',validate(m,tok));print('CHECKPOINT',arm,step,flush=True)
 opt.zero_grad(set_to_none=True);final=sample_parameters(m);changed={k:v!=final[k] for k,v in init.items()}
 outcome={'status':'completed','arm':arm,'additional_updates':cfg['updates_per_arm'],'global_step':1024+cfg['updates_per_arm'],'parameters':sum(p.numel() for p in m.parameters()),'counts':counts,'elapsed_training_validation_save_seconds_this_invocation':time.perf_counter()-tstart,'sampled_changed_tensors':sum(changed.values()),'parameter_tensors':len(changed),'all_parameter_values_finite':all(torch.isfinite(p).all().item() for p in m.parameters()),'unchanged_sampled': [k for k,v in changed.items() if not v],'final_state_sha256':state_hash(m),'final_checkpoint':str(cp.relative_to(ROOT)),'new_optimizer_warm_start':True,'same_ancestor_seed_not_new_independent_seeds':True}
 put(done,outcome);print('TRAIN_COMPLETE',arm,json.dumps(outcome),flush=True)
 save(m,ROOT/'exports'/arm,step=1024+cfg['updates_per_arm'],counts=counts,dtype=torch.bfloat16)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--parent',required=True);ap.add_argument('--resume');args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);train(args.arm,args.parent,args.resume)
