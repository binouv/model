"""One bounded continuation: balanced choice loss with a language anchor."""
import datetime,json,math,os,random,resource,shutil,time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file,load_file
from common import ROOT,setup,load,rows,encode,sha,atomic_json
from objective import configure,permute,label_logits,balanced_dev

def lr_at(step,cfg):
 if step<=cfg['warmup_steps']:return cfg['learning_rate']*step/cfg['warmup_steps']
 x=(step-cfg['warmup_steps'])/(cfg['max_steps']-cfg['warmup_steps'])
 return cfg['learning_rate']*(.1+.9*(1+math.cos(math.pi*x))/2)
def train_update(m,opt,batch,step,cfg,tok):
 m.train();opt.zero_grad(set_to_none=True);total=0.;tokens=0;losses={}
 for r in batch:
  if r['task']=='alpaca':
   x,y,n=encode(tok,r)
   with torch.autocast('cpu',dtype=torch.bfloat16):loss=m(x,targets=y)
   (loss*.2).backward();v=float(loss.detach());losses['alpaca_nll']=v;total+=.2*v;tokens+=x.numel()
  else:
   n=len(r['choices']);vals=[]
   for shift in range(n):
    q=permute(r,shift)
    with torch.autocast('cpu',dtype=torch.bfloat16):lg=label_logits(m,tok,q);loss=F.cross_entropy(lg[None],torch.tensor([q['gold_index']]))
    assert torch.isfinite(loss);(loss*(.4/n)).backward();vals.append(float(loss.detach()));tokens+=len(tok.encode(q['prompt'],bos=True))
   losses[r['task']+'_choice_nll']=sum(vals)/n;total+=.4*sum(vals)/n
 grad=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
 lr=lr_at(step,cfg)
 for group in opt.param_groups:group['lr']=lr
 opt.step()
 return dict(loss=total,**losses,input_tokens=tokens,lr=lr,grad_norm=float(grad))
def checkpoint(m,opt,step,cfg,best):
 dest=ROOT/'checkpoints'/f'step_{step:04d}';tmp=dest.with_name(dest.name+'.tmp');tmp.mkdir()
 wp=tmp/'model.safetensors';save_file(m.state_dict(),str(wp))
 with wp.open('rb') as f:os.fsync(f.fileno())
 nr=np.random.get_state();state=dict(optimizer=opt.state_dict(),step=step,sampler_cursor=3*step,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=dict(kind=nr[0],keys=nr[1].tolist(),position=nr[2],has_gauss=nr[3],cached_gaussian=nr[4]),scaler=None,cuda_rng=None,schedule_config=cfg,best_dev_step=best,data_manifest_sha256=sha(ROOT/'data/manifest.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),kind='native exact continuation within EXT3; new AdamW warm start from EXT2')
 sp=tmp/'training_state.pt'
 with sp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
 restored=load_file(str(wp));assert all(torch.equal(v,restored[k]) for k,v in m.state_dict().items());del restored
 s=torch.load(sp,weights_only=True);assert s['step']==step;del s
 atomic_json(tmp/'manifest.json',dict(step=step,status='saved_local',scientific_status='in_progress',model_sha256=sha(wp),state_sha256=sha(sp),model_bytes=wp.stat().st_size,state_bytes=sp.stat().st_size,strict_reload_equal=True))
 tmp.rename(dest)
 if step==128:os.link(dest/'model.safetensors',ROOT/'verification/step_0128_model.safetensors')
 for p in (ROOT/'checkpoints').glob('step_[0-9][0-9][0-9][0-9]'):
  old=int(p.name[5:])
  if old not in {64,best,step}:
   atomic_json(ROOT/f'metrics/checkpoint_{old:04d}_receipt.json',json.loads((p/'manifest.json').read_text()));shutil.rmtree(p)
 print('CHECKPOINT',step,flush=True)
def main():
 setup();cfg=json.loads((ROOT/'config.json').read_text());parent=(ROOT/cfg['parent_path']).resolve();assert sha(parent)==cfg['parent_sha256'];assert not (ROOT/'logs/train.jsonl').exists()
 m,tok=load(parent);params=configure(m);assert sum(p.numel() for p in params)==cfg['trainable_parameters'];data=rows('train.jsonl');assert len(data)==3*cfg['max_steps']
 opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False)
 base=json.loads((ROOT/'metrics/parent_EXT2_balanced_dev_summary.json').read_text());best=0;score=base['selection_objective'];atomic_json(ROOT/'metrics/selection.json',dict(best_step=0,best_objective=score,criterion='mean normalized choice NLL over balanced cyclic permutations of new OBQA/PIQA dev'))
 start=time.perf_counter();reason='fixed_step_count';last=0
 with (ROOT/'logs/train.jsonl').open('x') as f:
  for step in range(1,cfg['max_steps']+1):
   t=time.perf_counter();batch=data[3*(step-1):3*step];v=train_update(m,opt,batch,step,cfg,tok);last=step
   v.update(step=step,seconds=time.perf_counter()-t,elapsed_seconds=time.perf_counter()-start,sample_ids=[r['id'] for r in batch]);f.write(json.dumps(v)+'\n');f.flush()
   if step==1 or step%16==0:print(json.dumps(v),flush=True)
   if step%128==0 or step==cfg['max_steps']:
    dev=balanced_dev(m,tok,f'step_{step:04d}');print('DEV',json.dumps(dev),flush=True)
    if dev['selection_objective']<score:best=step;score=dev['selection_objective'];atomic_json(ROOT/'metrics/selection.json',dict(best_step=best,best_objective=score,criterion='mean normalized choice NLL over balanced cyclic permutations of new OBQA/PIQA dev'))
   if step%64==0 or step==cfg['max_steps']:checkpoint(m,opt,step,cfg,best)
   if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(cfg['training_stop_utc']) and step%64==0:reason='registered_time_budget_stop';break
 atomic_json(ROOT/'metrics/training_summary.json',dict(steps=last,completed_fixed_step_count=last==cfg['max_steps'],stop_reason=reason,unique_source_examples=3*last,qa_permutation_presentations=6*last,alpaca_presentations=last,wall_seconds=time.perf_counter()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_step=best,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,test_used=False))
 print('TRAINING_FINISHED',last,reason,flush=True)
if __name__=='__main__':main()
