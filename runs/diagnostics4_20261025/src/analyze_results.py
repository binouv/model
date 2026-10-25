"""Export every fixed scoring result and exploratory uncertainty intervals."""
import csv,datetime,json,math,platform
from pathlib import Path
import numpy as np
import torch
from common import ROOT,rows,sha,atomic_json

def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-r,c+r]
def main():
 r=json.loads((ROOT/'metrics/results.json').read_text());assert r['completed'];records=[];unc={};rng=np.random.default_rng(2026102504)
 for arm,splits in r['scoring'].items():
  unc[arm]={}
  for split,tasks in splits.items():
   raw=[json.loads(s) for s in (ROOT/f'metrics/{arm}_{split}_scoring.jsonl').read_text().splitlines()];unc[arm][split]={}
   for task,summary in tasks.items():
    rr=[x for x in raw if x['task']==task];sample=rng.integers(0,len(rr),size=(10000,len(rr)));base=np.array([x['predictions']['label_bf16']==x['gold'] for x in rr],dtype=int);unc[arm][split][task]={}
    for method,m in summary['methods'].items():
     pred=np.array([(x['parsed_generation'] if method=='generation16' else x['predictions'][method])==x['gold'] for x in rr],dtype=int);ci=wilson(m['correct'],m['n']);paired=np.quantile((pred-base)[sample].mean(1),[.025,.975]).tolist();unc[arm][split][task][method]=dict(wilson95=ci,delta_vs_same_model_label=float((pred-base).mean()),paired_bootstrap95=paired)
     records.append(dict(arm=arm,split=split,task=task,method=method,correct=m['correct'],n=m['n'],accuracy=m['correct']/m['n'],wilson95_low=ci[0],wilson95_high=ci[1],delta_vs_same_model_label=float((pred-base).mean()),paired95_low=paired[0],paired95_high=paired[1],valid=m.get('valid','')))
 with (ROOT/'metrics/scoring.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 pp=[]
 for task,v in r['probe'].items():
  for split in ['train','tiny_train','dev_mcq','test_mcq']:
   if split+'_correct' in v:k,n=v[split+'_correct'],v[split+'_n']
   elif split in v:k,n=v[split]['correct'],v[split]['n']
   else:continue
   pp.append(dict(task=task,split=split,correct=k,n=n,accuracy=k/n))
 with (ROOT/'metrics/probe.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(pp[0]));w.writeheader();w.writerows(pp)
 atomic_json(ROOT/'metrics/uncertainty.json',dict(bootstrap_seed=2026102504,replicates=10000,scope='Exploratory per-task intervals; not multiplicity-adjusted, no test-driven scoring-rule selection',results=unc))
 atomic_json(ROOT/'metrics/environment.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version(),torch=torch.__version__,cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),gpu_available=torch.cuda.is_available(),data_sampling_seed=2026102504,runtime_setup_seed=20261025,probe_fit='deterministic closed form; no random initialization or optimizer updates',generation='greedy max16; identical per model',candidate_scoring='Same complete external answer strings for both models, no EOS, sum/mean; variable length per candidate is retained and reported',scope='Reference access not retried; no paid resources or APIs used'))
 print(json.dumps(dict(scoring=r['scoring'],probe=r['probe'],wall_seconds=r['wall_seconds']),indent=2))
if __name__=='__main__':main()
