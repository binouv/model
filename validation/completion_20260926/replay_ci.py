"""Reproduce completed local G5S on public CPU to publish real weight bytes.
Not new independent seeds; do not pool these scores with the original run.
Local-to-connector binary upload is unavailable. Every asset is re-downloaded
and hashed. Only six completed endpoints permit a published prerelease.
"""
from __future__ import annotations
import gzip,hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile,urllib.request
from pathlib import Path
import torch
from safetensors.torch import load_file
TOP=Path(__file__).resolve().parents[2];SRC=TOP/'runs/g5s_trace';ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC/'src'))
import experiment as ex
import data
REPO='binouv/model';BRANCH='flygraph/completion-replay-20260926'
TAG='flygraph-g5s-cpu-replay-'+os.environ['GITHUB_RUN_ID']
LOCAL_WEIGHTS={'verified_trace_s7601':'83afc4b6962fcee9bbe0105ca186b9354146f94d20d6d4ee4d9768265080dfa9','final_only_s7601':'952a6ed40060ecec551bce8fab7dc46cb6dd1d60a21ed1965f72fbc0e47e4916','verified_trace_s7602':'5b077f158092495f8a4f7e52842e50cedb728ceeb8c27fe3b3dbb3fab8333a78','final_only_s7602':'8e79e18e39379b5958581cdef44b45a0c560aa82cde3e684718232967230ab35','verified_trace_s7603':'f46e2c6cd72ddef970f4cd0cfc1c97b6b0d974c6650aff73ce3bb4c169d178cd','final_only_s7603':'3ed64121a00b08f75fe7aceedac79d206f444d761bc1e47379904d69c24949e3'}
LOCAL_IID={'verified_trace':[1,4,7],'final_only':[1,1,3]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gh(*a):return subprocess.check_output(['gh',*a],text=True)
def api(method,path,obj=None):
 headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'}
 body=None if obj is None else json.dumps(obj).encode()
 with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=body,headers=headers,method=method),timeout=90) as f:return json.load(f)
def push(files):
 head=api('GET','/git/ref/heads/'+BRANCH)['object']['sha'];tree=api('GET','/git/commits/'+head)['tree']['sha']
 entries=[dict(path=str(p.relative_to(TOP)),mode='100644',type='blob',content=p.read_text()) for p in files]
 tr=api('POST','/git/trees',dict(base_tree=tree,tree=entries))['sha'];co=api('POST','/git/commits',dict(tree=tr,parents=[head],message='FlyGraph G5S reproduction: actual verified checkpoint, not independent seed'))['sha'];api('PATCH','/git/refs/heads/'+BRANCH,dict(sha=co,force=False));print('FG_PUSH',co,flush=True)
def upload(paths):
 gh('release','upload',TAG,'--repo',REPO,*map(str,paths));rows=[]
 with tempfile.TemporaryDirectory() as td:
  for p in paths:
   gh('release','download',TAG,'--repo',REPO,'--pattern',p.name,'--dir',td);q=Path(td)/p.name;assert sha(q)==sha(p) and q.stat().st_size==p.stat().st_size
   rows.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),remote_bytes_redownload_verified=True))
 return rows

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 work=ROOT/'work';assets=ROOT/'assets'
 for p in [work/'configs',work/'data',work/'reports',work/'metrics',assets]:p.mkdir(parents=True,exist_ok=True)
 # A separate runtime root avoids overwriting completed local-study files.
 cfg=json.loads((SRC/'configs/preregistered.json').read_text());shutil.copy2(SRC/'configs/preregistered.json',work/'configs/preregistered.json')
 data.build(work/'data')
 for split,h in cfg['data_sha256'].items():assert sha(work/'data'/f'{split}.jsonl')==h
 ex.ROOT=work
 for split in ['train','validation',*ex.SPLITS]:ex.load(split,'verified_trace')
 gh('release','create',TAG,'--repo',REPO,'--target',os.environ['GITHUB_SHA'],'--title','FlyGraph G5S CPU reproduction: six small research models','--notes','Draft until all six fixed1000-step reproductions complete. Repeated original seeds, not new independent seeds or300M training.','--draft','--prerelease')
 receipts=[];runs=[];old_save=ex.save
 def saved(m,opt,rng,arm,seed,step,logs):
  cp=old_save(m,opt,rng,arm,seed,step,logs)
  name=f'{arm}_s{seed}_step{step:04d}.resume.pt';p=assets/name;shutil.copy2(cp/'resume.pt',p);rr=upload([p]);receipts.extend(rr)
  progress=ROOT/'PROGRESS.json';progress.write_text(json.dumps(dict(status='completed_reproduction_checkpoint_only',arm=arm,seed=seed,step=step,asset=rr[0],not_independent_seed=True),indent=2));push([progress]);return cp
 ex.save=saved
 for seed in [7601,7602,7603]:
  for arm in ['verified_trace','final_only']:
   result=ex.train(arm,seed);assert result['status']=='completed' and result['resume_next_update_verified'] and result['reload_mismatches']==0
   cp=Path(result['checkpoint']);stem=f'{arm}_s{seed}'
   sf=assets/(stem+'.safetensors');shutil.copy2(cp/'model.safetensors',sf)
   conf=assets/(stem+'.config.json');shutil.copy2(cp/'config.json',conf)
   raw=assets/(stem+'.RAW.json.gz');raw.write_bytes(gzip.compress(json.dumps(result).encode(),mtime=0));receipts.extend(upload([sf,conf,raw]))
   summary={k:v for k,v in result.items() if k!='tests'};summary['tests']={s:{k:v for k,v in t.items() if k!='rows'} for s,t in result['tests'].items()}
   summary['local_original_safetensors_sha256']=LOCAL_WEIGHTS[stem];summary['safetensors_byte_identical_to_original_local']=sha(sf)==LOCAL_WEIGHTS[stem]
   summary['IID_matches_original_local']=summary['tests']['test_iid']['correct']==LOCAL_IID[arm][seed-7601]
   runs.append(summary);p=ROOT/(stem+'.COMPLETED.json');p.write_text(json.dumps(summary,indent=2));push([p])
 assert len(runs)==6 and len({r['batch_sequence_sha256'] for r in runs})==1
 assert all(len({r['initial_state_sha256'] for r in runs if r['seed']==s})==1 for s in [7601,7602,7603])
 out=dict(status='completed_CPU_reproduction',models=6,independent_new_seeds=0,parameters_each=469648,steps_each=1000,canonical_local_study='runs/g5s_trace',local_bytes_are_not_assumed_identical=True,not_broad_Qwen_comparison=True,runs=runs)
 final=ROOT/'FINAL_REPLAY.json';final.write_text(json.dumps(out,indent=2))
 package=assets/'SOURCE_DATA_RAW_RESULTS.tar.gz'
 with tarfile.open(package,'w:gz') as t:
  for folder in [SRC/'src',SRC/'configs',ROOT]:
   for p in sorted(folder.rglob('*')):
    if not p.is_file() or any(x in p.parts for x in ['assets','checkpoints','__pycache__']):continue
    t.add(p,arcname=str(p.relative_to(TOP)))
 receipts.extend(upload([package]));mf=assets/'MANIFEST.json';mf.write_text(json.dumps(dict(status='completed',source_commit=os.environ['GITHUB_SHA'],workflow_run=os.environ['GITHUB_RUN_ID'],release_tag=TAG,models=6,independent_new_seeds=0,native_optimizer_rng_present=True,all_assets_redownload_verified=True,assets=receipts),indent=2));upload([mf])
 gh('release','edit',TAG,'--repo',REPO,'--draft=false','--prerelease','--notes','Six completed CPU reproductions of G5S original seeds7601/7602/7603; NOT new independent seeds. Actual model safetensors plus native optimizer/torch-python-sampler RNG every200steps and full source/data/raw outputs. Every uploaded asset re-downloaded and SHA256-verified. Each model records equality/difference from original local safetensors. Small469648-parameter English synthetic probes, not300-800M or Qwen parity.')
 receipt=ROOT/'RELEASE_RECEIPT.json';receipt.write_bytes(mf.read_bytes());push([final,receipt]);print('FG_FINAL',json.dumps(dict(tag=TAG,models=6,IID={a:[r['tests']['test_iid']['correct'] for r in runs if r['arm']==a] for a in ['verified_trace','final_only']},byte_matches=[r['safetensors_byte_identical_to_original_local'] for r in runs])),flush=True)
if __name__=='__main__':main()
