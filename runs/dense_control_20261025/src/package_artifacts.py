"""Package the completed scaffold and clearly UNTRAINED tiny initialization."""
import csv,datetime,hashlib,json,zipfile
from pathlib import Path
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 smoke=json.loads((ROOT/'metrics/smoke.json').read_text());assert smoke['status']=='passed' and smoke['optimizer_updates']==0;profile=json.loads((ROOT/'metrics/tiny_profile.json').read_text());assert profile['status']=='passed';counts=json.loads((ROOT/'metrics/parameter_counts.json').read_text())
 with (ROOT/'metrics/checks.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['check','passed','details'])
  for name,passed in smoke['checks'].items():w.writerow([name,passed,json.dumps(smoke['details'].get(name,{}))])
 with (ROOT/'metrics/parameters.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['configuration','parameters','tensor_count','trained'])
  for name,v in counts.items():w.writerow([name,v['parameters'],v['tensor_count'],False])
 weights=ROOT/'artifacts/tiny_UNTRAINED_initial.safetensors'
 with safe_open(str(weights),framework='pt') as sf:
  inventory={k:dict(shape=list(sf.get_tensor(k).shape),elements=sf.get_tensor(k).numel(),dtype=str(sf.get_tensor(k).dtype)) for k in sf.keys()};metadata=sf.metadata()
 assert metadata['optimizer_updates']=='0' and metadata['status']=='UNTRAINED_INITIALIZATION_ONLY';assert sha(weights)==smoke['untrained_weight_sha256']
 manifest=dict(schema='flygraph.architecture-scaffold.v1',status='completed_engineering_scaffold',trained=False,quality_evaluated=False,full_size_weights_created=False,native_component=dict(path=str(weights.relative_to(ROOT)),bytes=weights.stat().st_size,sha256=sha(weights),tensor_count=len(inventory),parameters=sum(v['elements'] for v in inventory.values()),tensor_inventory=inventory,metadata=metadata),optimizer_state_applicable=False,parent_trained_checkpoint=None,initialization_seed=smoke['seed'],publication_status_source='../PUBLICATION.json');dump(ROOT/'artifacts/MANIFEST.json',manifest)
 dump(ROOT/'source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and (p.parent==ROOT/'src' and p.suffix=='.py' or p.parent in [ROOT/'configs',ROOT/'data'])})
 dump(ROOT/'metrics/summary.json',dict(completed=True,scope='Engineering checks only; no language training or quality test',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=smoke['check_count'],optimizer_updates=0,canonical_smoke_seconds=smoke['seconds'],tiny_profile=profile,full_size_counts={k:v['parameters'] for k,v in counts.items() if k!='tiny_smoke'},environment=dict(cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),gpu_used=False),remaining_work=['A resumable training/data runner is not implemented in this minimal scaffold','32k tokenizer is absent; template is deliberately rejected by profile tool with8k tokenizer','Full-size throughput, GPU behavior, BF16 and trained quality are unmeasured']))
 bundle=ROOT/'artifacts/dense_control_UNTRAINED_scaffold.zip';files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p not in [bundle,ROOT/'artifacts/BUNDLE.json'] and p.name not in ['PUBLICATION.json','PUSH_COMPLETE.json','CHECKSUMS.sha256'])
 with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:assert p.stat().st_size<90_000_000;z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
 assert bundle.stat().st_size<90_000_000;dump(ROOT/'artifacts/BUNDLE.json',dict(path=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),files=len(files),weights='Untrained tiny init only, not EXT3 or full-size dense weights'));print('PACKAGED',bundle.stat().st_size,sha(bundle))
if __name__=='__main__':main()
