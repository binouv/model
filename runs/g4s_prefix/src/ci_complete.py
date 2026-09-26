"""Finish existing G4S without changing model/data/loss/800-step training.
Each completed arm is uploaded to a DRAFT prerelease. Only six complete arms
permit final publication. Writes are non-force and restricted to this branch.
"""
import csv,gzip,hashlib,json,os,shutil,subprocess,tarfile,tempfile,urllib.request
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import run
from model import Model,Config
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'results'
REPO='binouv/model';BRANCH='flygraph/g4s-finish-20260926'
TAG='flygraph-g4s-completed-'+os.environ.get('GITHUB_RUN_ID','local')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gh(*a):return subprocess.check_output(['gh',*a],text=True)
def api(method,path,data=None):
 h={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'}
 b=None if data is None else json.dumps(data).encode()
 with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=b,headers=h,method=method),timeout=90) as r:return json.load(r)
def push(files):
 head=api('GET','/git/ref/heads/'+BRANCH)['object']['sha'];base=api('GET','/git/commits/'+head)['tree']['sha']
 tree=[{'path':'runs/g4s_prefix/'+str(p.relative_to(ROOT)),'mode':'100644','type':'blob','content':p.read_text()} for p in files]
 tr=api('POST','/git/trees',{'base_tree':base,'tree':tree})['sha'];co=api('POST','/git/commits',{'tree':tr,'parents':[head],'message':'FlyGraph G4S: persist completed checkpoint/metrics, no partial-result claim'})['sha']
 api('PATCH','/git/refs/heads/'+BRANCH,{'sha':co,'force':False});print('FG_PUSH',co,flush=True);return co

def upload_verified(paths):
 gh('release','upload',TAG,'--repo',REPO,*map(str,paths))
 receipts=[]
 with tempfile.TemporaryDirectory() as td:
  for p in paths:
   gh('release','download',TAG,'--repo',REPO,'--pattern',p.name,'--dir',td)
   q=Path(td)/p.name;assert q.stat().st_size==p.stat().st_size and sha(q)==sha(p)
   receipts.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'remote_bytes_download_verified':True})
 return receipts

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 WORK.mkdir(exist_ok=True);assets=WORK/'release_assets';assets.mkdir(exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
 gh('release','create',TAG,'--repo',REPO,'--target',os.environ['GITHUB_SHA'],'--title','FlyGraph G4S: six small paired architecture probes','--notes','Draft: no completed campaign claim until all six endpoints and integrity checks finish.','--draft','--prerelease')
 receipts=[];completed=[];original=run.train
 def wrapped(root,variant,seed,steps=800):
  met=original(root,variant,seed,steps);assert met['status']=='completed' and met['steps']==800 and met['all_losses_finite']
  cp=root/'checkpoints'/f'{variant}_s{seed}'/'checkpoint.pt';z=torch.load(cp,weights_only=False)
  assert all(k in z for k in ('model','optimizer','torch_rng','python_rng','sampler_rng'))
  assert all(bool(torch.isfinite(t).all()) for t in z['model'].values() if t.is_floating_point())
  prefix=f'{variant}_s{seed}';native=assets/(prefix+'.checkpoint.pt');shutil.copy2(cp,native)
  sf=assets/(prefix+'.safetensors');save_file({k:v.contiguous() for k,v in z['model'].items()},str(sf));m=Model(Config(**z['config']));m.load_state_dict(load_file(str(sf)))
  replay=run.evaluate(m,run.rows(root/'data','test_iid')[:16])['rows'];orig=json.loads((root/'metrics'/(prefix+'.json')).read_text())['raw_tests']['test_iid']['rows'];orig={r['id']:r for r in orig}
  assert all(r['generated']==orig[r['id']]['generated'] and r['token_ids']==orig[r['id']]['token_ids'] for r in replay)
  assert all(torch.equal(t,load_file(str(sf))[k]) for k,t in z['model'].items())
  raw=assets/(prefix+'.metrics.json.gz');raw.write_bytes(gzip.compress((root/'metrics'/(prefix+'.json')).read_bytes(),mtime=0))
  cfg=assets/(prefix+'.config.json');cfg.write_text(json.dumps(z['config'],indent=2))
  rr=upload_verified([native,sf,cfg,raw]);receipts.extend(rr);completed.append(met)
  progress=ROOT/'reports/PROGRESS.json';progress.write_text(json.dumps({'status':'completed_arms_only','completed_models':len(completed),'expected_models':6,'models':completed,'assets':receipts},indent=2));push([progress]);print('FG_ARM_COMPLETE',variant,seed,json.dumps(met['tests']),flush=True)
  return met
 run.train=wrapped
 result=run.campaign(WORK,[7501,7502,7503],800)
 assert result['status']=='completed' and len(result['runs'])==6
 assert len({r['batch_sequence_sha256'] for r in result['runs']})==1
 result['scope']='469648-parameter English synthetic mechanism probes, not 300-800M or Qwen parity';result['no_real_code_generation_benchmark']=True
 (ROOT/'metrics').mkdir(exist_ok=True);summary=ROOT/'metrics/FINAL_RESULT.json';summary.write_text(json.dumps(result,indent=2))
 csvp=ROOT/'metrics/METRICS.csv'
 with csvp.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['variant','seed','split','n','correct','accuracy','both_correct_pairs'])
  for r in result['runs']:
   for split,m in r['tests'].items():w.writerow([r['variant'],r['seed'],split,m['n'],m['correct'],m['accuracy'],m['both_correct_pairs']])
 nextp=ROOT/'reports/NEXT_STATE.json';nextp.write_text(json.dumps({'status':'G4S_completed','source_commit':os.environ['GITHUB_SHA'],'run_id':os.environ['GITHUB_RUN_ID'],'release_tag':TAG,'six_native_optimizer_rng_checkpoints':True,'goal_achieved':False,'next':'G5 trace supervision needs exact new data registration; G3 long requires optimizer bytes or explicitly fresh optimizer warm-start. Do not repeat G4S.'},indent=2))
 package=assets/'SOURCE_RESULTS.tar.gz'
 with tarfile.open(package,'w:gz') as tar:
  for p in sorted(ROOT.rglob('*')):
   if not p.is_file() or any(x in p.parts for x in ['release_assets','checkpoints','__pycache__','.pytest_cache']):continue
   tar.add(p,arcname='g4s_prefix/'+str(p.relative_to(ROOT)))
 receipts.extend(upload_verified([package]));manifest=assets/'MANIFEST.json';manifest.write_text(json.dumps({'status':'completed','source_commit':os.environ['GITHUB_SHA'],'workflow_run':os.environ['GITHUB_RUN_ID'],'models':6,'parameters_each':469648,'native_optimizer_rng':True,'all_remote_bytes_verified':True,'assets':receipts},indent=2));upload_verified([manifest])
 gh('release','edit',TAG,'--repo',REPO,'--draft=false','--prerelease','--notes','Six completed fixed-step800 G4S models, three paired seeds, actual FP32 safetensors + native optimizer/RNG + raw metrics/source/data. All bytes verified by remote re-download. Synthetic small probes, not general intelligence or 300M pretraining.')
 receipt=ROOT/'reports/RELEASE_RECEIPT.json';receipt.write_text(manifest.read_text());push([summary,csvp,nextp,receipt])
 print('FG_FINAL',json.dumps({'comparison':result['comparison'],'causal_hybrid':result['causal_hybrid'],'prefix_hybrid':result['prefix_hybrid'],'tag':TAG,'assets':len(receipts)+1,'complete_models':6}),flush=True)
if __name__=='__main__':main()
