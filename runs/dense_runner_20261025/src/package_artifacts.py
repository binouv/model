import csv,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((ROOT/'metrics/resume_verification.json').read_text());assert r['status']=='passed' and r['entire_native_state_equal']
 with (ROOT/'metrics/train_trace.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(r['trace'][0]));w.writeheader();w.writerows(r['trace'])
 manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*.py'))};(ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 checkpoints=[]
 for p in sorted((ROOT/'artifacts/local_smoke').glob('*/step_*/MANIFEST.json')):
  m=json.loads(p.read_text());checkpoints.append(dict(path=str(p.parent.relative_to(ROOT)),step=m['step'],input_tokens=m['input_tokens'],model_sha256=m['model_sha256'],training_state_sha256=m['training_state_sha256'],status=m['status'],scope=m['scope']))
 (ROOT/'artifacts/MANIFEST.json').write_text(json.dumps(dict(status='completed_tiny_resume_engineering_proof',full_size_training_executed=False,checkpoints=checkpoints,source_model='dense_control_20261025/src/model.py unchanged',source_model_sha256=sha(ROOT/'src/model.py')),indent=2)+'\n')
 bundle=ROOT/'artifacts/dense_runner_verified_bundle.zip';chosen=[]
 for p in sorted(ROOT.rglob('*')):
  rel=p.relative_to(ROOT)
  if not p.is_file() or '__pycache__' in rel.parts or p in [bundle,ROOT/'artifacts/BUNDLE.json'] or p.name in ['PUBLICATION.json','PUSH_COMPLETE.json','CHECKSUMS.sha256']:continue
  if rel.parts[:2]==('artifacts','local_smoke'):
   if len(rel.parts)>3 and rel.parts[2] in ['continuous','split','budget']:
    if rel.parts[3].startswith('step_') and str(rel.parts[3]) not in {'continuous':'step_000006','split':'step_000003','budget':'step_000002'}.get(rel.parts[2],''):continue
   elif len(rel.parts)>3:continue
  chosen.append(p)
 with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in chosen:z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
 assert bundle.stat().st_size<90_000_000
 (ROOT/'artifacts/BUNDLE.json').write_text(json.dumps(dict(path=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),files=len(chosen),contains='Source/config/data/metrics and canonical step3, step6, budget70 native checkpoints and parts; duplicate replay checkpoint bytes remain separately in repository'),indent=2)+'\n');print('BUNDLE',bundle.stat().st_size,sha(bundle))
if __name__=='__main__':main()
