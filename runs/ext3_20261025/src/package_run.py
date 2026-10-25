import json,hashlib,zipfile,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file
from common import ROOT,sha,atomic_json
from checkpoint_parts import split,verify
VOLATILE={'STATUS.json','PIPELINE_STATE.json','PIPELINE_LOCK','CHECKSUMS.sha256','PUBLICATION.json','PUSH_COMPLETE.json'}
def main():
 r=json.loads((ROOT/'metrics/final_results.json').read_text());assert r['completed'];assert json.loads((ROOT/'metrics/resume_verification.json').read_text())['status']=='passed';assert json.loads((ROOT/'metrics/protocol_verification.json').read_text())['status']=='passed'
 cfg=json.loads((ROOT/'config.json').read_text());step=r['training']['steps'];chosen=r['selection']['best_step'];parent=(ROOT/cfg['parent_path']).resolve();final=ROOT/f'checkpoints/step_{step:04d}'
 runtime=Path(cfg.get('runtime_directory','/workspace/ext3_runtime_20261025'))
 for name in ['verify_protocol','verify_resume','evaluate_final','make_report']:
  p=runtime/(name+'.log')
  if p.exists():shutil.copyfile(p,ROOT/'logs'/p.name)
 components=[('final_model',final/'model.safetensors',step),('final_state',final/'training_state.pt',step)]
 if chosen not in [0,step]:
  d=ROOT/f'checkpoints/step_{chosen:04d}';components.extend([('selected_model',d/'model.safetensors',chosen),('selected_state',d/'training_state.pt',chosen)])
 init=load_file(str(parent));sd=load_file(str(final/'model.safetensors'));assert len(sd)==135 and sum(t.numel() for t in sd.values())==100028328
 prefixes=tuple(cfg['trainable_prefixes']);changed=[k for k,v in sd.items() if not torch.equal(v,init[k])];assert changed and all(k.startswith(prefixes) for k in changed);assert all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in sd.values());del init,sd
 inv=[]
 for role,path,n in components:
  if role.endswith('state'):
   state=torch.load(path,weights_only=True);assert state['step']==n and state['sampler_cursor']==3*n
   for v in state['optimizer']['state'].values():
    for k in ['exp_avg','exp_avg_sq']:assert v[k].dtype==torch.float32 and torch.isfinite(v[k]).all()
   del state
  folder=ROOT/'artifacts'/(role+'_parts');m=verify(folder) if folder.exists() else split(path,folder,identity='EXT3_20261025_'+role,expected_sha256=sha(path));assert m['sha256']==sha(path);m.update(run_id='ext3_20261025',role=role,completed_new_optimizer_step=n,parameters=100028328,trainable_parameters=13391640,native_precision='float32',parent_checkpoint_sha256=cfg['parent_sha256'],data_manifest_sha256=sha(ROOT/'data/manifest.json'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),status='saved_local');atomic_json(folder/'MANIFEST.json',m);verify(folder);inv.append(dict(role=role,directory=str(folder.relative_to(ROOT)),filename=m['filename'],size_bytes=m['size_bytes'],sha256=m['sha256'],parts=len(m['parts']),step=n))
 inventory=dict(status='saved_local_pending_repository_push',components=inv,final_step=step,selected_step=chosen,selected_model_role='published_EXT2_selected_model' if chosen==0 else 'final_model' if chosen==step else 'selected_model',parent_checkpoint_sha256=cfg['parent_sha256'],parent_published_commit='7b712f2d557a85248639037348dd7b27f2705790',parent_parts='runs/ext2_20261022/artifacts/selected_model_parts',original_EXT256_parts='runs/ext2_20261022/artifacts/parent_parts',native_model_dtype='float32',native_optimizer_dtype='float32',changed_tensors=changed,all_frozen_parameters_bitwise_unchanged=True,registered_improvement_criterion_met=r['registered_improvement_criterion_met'],general_superiority_established=False,all_parts_strictly_under_bytes=90000000)
 atomic_json(ROOT/'artifacts/MANIFEST.json',inventory)
 from enrich_metadata import main as enrich
 inventory=enrich()
 atomic_json(ROOT/'final_source_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')})
 included=[]
 for p in ROOT.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.name in VOLATILE or p.name.endswith('.tmp'):continue
  rel=p.relative_to(ROOT)
  if rel.parts[0] in ['checkpoints','verification']:continue
  if rel.parts[0]=='artifacts' and p.name!='MANIFEST.json':continue
  if p.suffix in ['.pt','.safetensors','.zip']:continue
  included.append(p)
 included.append(ROOT/'checkpoints/config.json');archive=ROOT/'artifacts/FlyGraph_EXT3_20261025_Report_Code_Data.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in included:z.write(p,str(p.relative_to(ROOT)))
  z.writestr('CHECKSUMS.sha256',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(included)))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for line in z.read('CHECKSUMS.sha256').decode().splitlines():h,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==h
 print(json.dumps(inventory,indent=2),flush=True)
if __name__=='__main__':main()
