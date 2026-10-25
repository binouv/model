"""Preserve real native readout/feature bytes; all files below90,000,000B."""
import collections,datetime,json,zipfile
from pathlib import Path
from safetensors import safe_open
from common import ROOT,sha,atomic_json

def main():
 assert json.loads((ROOT/'metrics/results.json').read_text())['completed']
 for n in ['artifact_verification','scoring_protocol_verification','chunked_delta_verification']:assert json.loads((ROOT/f'metrics/{n}.json').read_text())['status']=='passed'
 components=[]
 for p in sorted((ROOT/'artifacts').glob('*.safetensors')):
  with safe_open(str(p),framework='pt') as f:
   inventory={k:dict(shape=list(f.get_tensor(k).shape),dtype=str(f.get_tensor(k).dtype),elements=f.get_tensor(k).numel()) for k in f.keys()}
  assert p.stat().st_size<90_000_000;components.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),tensor_count=len(inventory),total_elements=sum(t['elements'] for t in inventory.values()),inventory=inventory,kind='native closed-form readout coefficients and sufficient standardization state' if 'weights' in p.name else 'frozen EXT3 last-normalized-hidden features, not model weights',splitting_required=False))
 atomic_json(ROOT/'artifacts/MANIFEST.json',dict(schema='flygraph.frozen-diagnostics.v1',run_id='diagnostics4_20261025',completed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),neural_optimizer_updates=0,optimizer_state_applicable=False,readout_training='closed-form float64 ridge; deterministic, no neural resume claim',parent_model_parameter_count=100028328,parent_model_sha256=json.loads((ROOT/'config.json').read_text())['EXT3_sha256'],components=components,publication_status_source='../PUBLICATION.json'))
 # Freeze postflight helper sources separately from the unchanged preregistration manifest.
 atomic_json(ROOT/'postflight_source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*.py'))})
 path=ROOT/'artifacts/diagnostics4_complete_bundle.zip'
 files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp' and p not in [path,ROOT/'artifacts/BUNDLE.json'] and p.name not in ['PUBLICATION.json','PUSH_COMPLETE.json','CHECKSUMS.sha256'])
 with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,str(p.relative_to(ROOT)))
 assert path.stat().st_size<90_000_000
 with zipfile.ZipFile(path) as z:assert z.testzip() is None;assert len(z.namelist())==len(files)
 atomic_json(ROOT/'artifacts/BUNDLE.json',dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path),files=len(files),contains_actual_native_probe_and_feature_bytes=True,full_backbone_reference='PARENT_CHECKPOINTS.json; already published native EXT2/EXT3 parts, intentionally not duplicated',publication_receipt='External PUBLICATION.json created after clean remote readback'))
 print(json.dumps(dict(components=[{k:v for k,v in c.items() if k!='inventory'} for c in components],bundle_bytes=path.stat().st_size,bundle_sha256=sha(path)),indent=2))
if __name__=='__main__':main()
