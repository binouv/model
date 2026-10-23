"""Bounded, fully dense continuation; selection uses frozen dev only."""
import datetime,json,math,os,random,resource,shutil,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from common import ROOT,setup,load,rows,encode,sha,atomic_json,evaluate
def lr_at(step,cfg):
    if step<=cfg['warmup_steps']:return cfg['learning_rate']*step/cfg['warmup_steps']
    progress=(step-cfg['warmup_steps'])/(cfg['max_steps']-cfg['warmup_steps'])
    return cfg['learning_rate']*(.1+.9*(1+math.cos(math.pi*progress))/2)
def checkpoint(m,opt,step,cfg,best):
    target=ROOT/'checkpoints'/f'step_{step:04d}';stage=target.with_name(target.name+'.tmp')
    stage.mkdir(exist_ok=False);wp=stage/'model.safetensors';save_file(m.state_dict(),str(wp))
    with wp.open('rb') as f:os.fsync(f.fileno())
    nr=np.random.get_state()
    state={'optimizer':opt.state_dict(),'step':step,'sampler_cursor':4*step,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':{'kind':nr[0],'keys':nr[1].tolist(),'position':nr[2],'has_gauss':nr[3],'cached_gaussian':nr[4]},'cuda_rng':None,'scaler':None,'schedule_config':cfg,'model_config':asdict(m.config),'best_dev_step':best,'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'parent_sha256':cfg['parent_sha256'],'kind':'new-run exact-resume checkpoint after a weight-only warm start'}
    tp=stage/'training_state.pt'
    with tp.open('wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    restored=load_file(str(wp));assert all(torch.equal(t,restored[k]) for k,t in m.state_dict().items());del restored
    st=torch.load(tp,weights_only=True);assert st['step']==step and st['sampler_cursor']==step*4;del st
    atomic_json(stage/'manifest.json',{'status':'saved_local','scientific_status':'in_progress','step':step,'parameters':100028328,'native_model_dtype':'float32','optimizer_dtype':'float32','parent_sha256':cfg['parent_sha256'],'model_sha256':sha(wp),'state_sha256':sha(tp),'model_bytes':wp.stat().st_size,'state_bytes':tp.stat().st_size,'strict_reload_equal':True,'data_manifest_sha256':sha(ROOT/'data/manifest.json')})
    stage.rename(target)
    # Keep the first replay checkpoint, best selected checkpoint, and two most
    # recent validated recoveries. Prune only this run's redundant older bytes.
    keep={64,best,step,step-64}
    for p in (ROOT/'checkpoints').glob('step_[0-9][0-9][0-9][0-9]'):
        old=int(p.name[5:])
        if old not in keep and old<step-64:
            atomic_json(ROOT/'metrics'/f'checkpoint_{old:04d}_receipt.json',json.loads((p/'manifest.json').read_text()))
            shutil.rmtree(p)
    print('CHECKPOINT',step,flush=True)
def train_update(m,opt,batch,step,cfg):
    m.train();opt.zero_grad(set_to_none=True);loss_sum=0.;input_tokens=0;target_tokens=0
    for x,y,n in batch:
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=m(x,targets=y)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
        (loss/len(batch)).backward();loss_sum+=float(loss.detach());input_tokens+=x.numel();target_tokens+=n
    grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    lr=lr_at(step,cfg)
    for g in opt.param_groups:g['lr']=lr
    opt.step()
    return {'loss':loss_sum/len(batch),'input_tokens':input_tokens,'supervised_tokens':target_tokens,'lr':lr,'grad_norm':float(grad)}
def main():
    setup();cfg=json.loads((ROOT/'config.json').read_text());parent=(ROOT/cfg['parent_path']).resolve();assert sha(parent)==cfg['parent_sha256']
    assert not (ROOT/'logs/train.jsonl').exists(),'Refuse to overwrite existing training; use checkpoint recovery.'
    m,tok=load(parent);data=rows('train.jsonl');encoded=[encode(tok,r) for r in data]
    initial,_,_=evaluate(m,tok,'dev','step_0000');best_score=initial['selection_objective'];best_step=0
    atomic_json(ROOT/'metrics/selection.json',{'best_step':best_step,'best_objective':best_score,'selected_weight_path':str(parent),'criterion':'equal-task dev NLL: Alpaca, OpenBookQA, PIQA; ARC excluded'})
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False)
    start=time.perf_counter();deadline=datetime.datetime.fromisoformat(cfg['training_stop_utc']).timestamp();last_step=0;reason='fixed_step_count'
    with (ROOT/'logs/train.jsonl').open('x') as log:
        for step in range(1,cfg['max_steps']+1):
            before=time.perf_counter();batch=encoded[4*(step-1):4*step];assert len(batch)==4
            metrics=train_update(m,opt,batch,step,cfg);last_step=step
            metrics.update({'step':step,'seconds':time.perf_counter()-before,'elapsed_seconds':time.perf_counter()-start,'sample_ids':[r['id'] for r in data[4*(step-1):4*step]]})
            log.write(json.dumps(metrics)+'\n');log.flush()
            if step==1 or step%16==0:print(json.dumps(metrics),flush=True)
            is_dev=step%cfg['dev_interval']==0 or step==cfg['max_steps']
            if is_dev:
                summary,_,_=evaluate(m,tok,'dev',f'step_{step:04d}')
                print('DEV',json.dumps(summary),flush=True)
                if summary['selection_objective']<best_score:
                    best_score=summary['selection_objective'];best_step=step
                    atomic_json(ROOT/'metrics/selection.json',{'best_step':best_step,'best_objective':best_score,'selected_weight_path':str(ROOT/'checkpoints'/f'step_{step:04d}'/'model.safetensors'),'criterion':'equal-task dev NLL: Alpaca, OpenBookQA, PIQA; ARC excluded'})
            if step%cfg['checkpoint_interval']==0 or step==cfg['max_steps']:checkpoint(m,opt,step,cfg,best_step)
            if time.time()>=deadline and step%cfg['checkpoint_interval']==0:
                reason='registered_time_budget_stop';break
    atomic_json(ROOT/'metrics/training_summary.json',{'completed_fixed_step_count':last_step==cfg['max_steps'],'steps':last_step,'unique_records':4*last_step,'wall_seconds':time.perf_counter()-start,'stop_reason':reason,'best_dev_step':best_step,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoint_retention':'first step64, best dev, latest two; redundant earlier recovery bytes pruned only after verified successor','test_evaluated_during_training':False})
    print('TRAINING_FINISHED',last_step,reason,flush=True)
if __name__=='__main__':main()
