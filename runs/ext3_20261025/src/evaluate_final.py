"""One final, frozen, three-model comparison; no test-driven model selection."""
import collections,datetime,json,math,time,csv
import numpy as np
import torch
from common import ROOT,setup,load,rows,sha,atomic_json,evaluate

def wilson(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def compare(before,after):
 rng=np.random.default_rng(20261026);out={};boot=[];diffs=[]
 for task in ['openbookqa','piqa','arc_challenge']:
  b=[p for p in before if p['task']==task];a=[p for p in after if p['task']==task];assert [p['id'] for p in b]==[p['id'] for p in a];n=len(a);idx=rng.integers(n,size=(10000,n));out[task]={}
  for metric,key in [('label','predicted_label'),('generation','parsed_generation')]:
   bb=np.array([p[key]==p['gold'] for p in b],dtype=int);aa=np.array([p[key]==p['gold'] for p in a],dtype=int);dd=aa-bb;boots=dd[idx].mean(1)
   out[task][metric]=dict(before_correct=int(bb.sum()),after_correct=int(aa.sum()),n=n,difference=float(dd.mean()),paired_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),after_wilson95=wilson(int(aa.sum()),n))
   if metric=='label':boot.append(boots);diffs.append(float(dd.mean()))
 return dict(by_task=out,mean_label_difference=float(np.mean(diffs)),stratified_paired_bootstrap95=np.quantile(np.stack(boot).mean(0),[.025,.975]).tolist(),positive_tasks=sum(x>0 for x in diffs))
def main():
 setup();cfg=json.loads((ROOT/'config.json').read_text());training=json.loads((ROOT/'metrics/training_summary.json').read_text());sel=json.loads((ROOT/'metrics/selection.json').read_text());assert not (ROOT/'metrics/final_results.json').exists();parent=(ROOT/cfg['parent_path']).resolve();chosen=parent if sel['best_step']==0 else ROOT/f"checkpoints/step_{sel['best_step']:04d}/model.safetensors"
 lock=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_step=sel['best_step'],selection_sha256=sha(ROOT/'metrics/selection.json'),test_mcq_sha256=sha(ROOT/'data/test_mcq.jsonl'),test_alpaca_sha256=sha(ROOT/'data/test_alpaca.jsonl'))
 atomic_json(ROOT/'metrics/final_evaluation_lock.json',lock);summaries={};raw={};examples=[]
 for arm,weights in [('original_EXT256',(ROOT/cfg['original_EXT256_path']).resolve()),('parent_EXT2',parent),('selected_EXT3',chosen)]:
  m,tok=load(weights);s,rr,_=evaluate(m,tok,'test',arm);s['weight_sha256']=sha(weights);summaries[arm]=s;raw[arm]=rr
  with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
   for r in rows('test_alpaca.jsonl')[:4]:
    seq=m.generate(torch.tensor([tok.encode(r['prompt'],bos=True)]),max_new_tokens=32)[0].tolist();body=seq[:seq.index(2)] if 2 in seq else seq;examples.append(dict(arm=arm,id=r['id'],prompt=r['prompt'],gold=r['completion'],generation=tok.decode(body),token_ids=seq))
  print('TEST',json.dumps(s),flush=True);del m
 comparisons={k:compare(raw[k],raw['selected_EXT3']) for k in ['original_EXT256','parent_EXT2']};primary=comparisons['parent_EXT2'];no_worse=summaries['selected_EXT3']['alpaca_nll']<=summaries['parent_EXT2']['alpaca_nll'];success=primary['stratified_paired_bootstrap95'][0]>0 and primary['positive_tasks']>=2 and no_worse
 freq={arm:{task:dict(collections.Counter(p['predicted_label'] for p in rr if p['task']==task)) for task in ['openbookqa','piqa','arc_challenge']} for arm,rr in raw.items()}
 result=dict(completed=True,training=training,selection=sel,summaries=summaries,comparisons=comparisons,alpaca_nll_not_worse_than_EXT2=no_worse,registered_improvement_criterion_met=bool(success),prediction_counts=freq,bootstrap_seed=20261026,bootstrap_repeats=10000,general_superiority_established=False,references=json.loads((ROOT/'metrics/reference_status.json').read_text()))
 atomic_json(ROOT/'metrics/final_results.json',result)
 with (ROOT/'metrics/test_alpaca_generations.jsonl').open('x') as f:
  for x in examples:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 with (ROOT/'metrics/final_results.csv').open('w',newline='') as f:
  fields=['model','dataset','status','n','label_correct','label_accuracy','generation_correct','generation_accuracy','valid_generation','alpaca_nll'];w=csv.DictWriter(f,fields);w.writeheader()
  for arm,s in summaries.items():
   for task,t in s['by_task'].items():w.writerow(dict(model=arm,dataset=task,status='measured',alpaca_nll=s['alpaca_nll'],**{k:t[k] for k in fields if k in t}))
  for mid in cfg['reference_models']:w.writerow(dict(model=mid,status='blocked_proxy403_not_measured'))
 print('FINAL',json.dumps(primary),'criterion',success,flush=True)
if __name__=='__main__':main()
