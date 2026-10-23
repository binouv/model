"""Preserve exact native bytes and small reproducibility artifacts locally."""
import json,zipfile,hashlib,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file
from common import ROOT,sha,atomic_json
from checkpoint_parts import split,verify
VOLATILE={'STATUS.json','PIPELINE_STATE.json','PIPELINE_LOCK','CHECKSUMS.sha256'}
def main():
    # Copy only closed verification/evaluation logs; the package/controller
    # logs are still active and must not enter their own archive/checksums.
    runtime=Path('/workspace/ext2_runtime_20261022')
    for name in ['verify_protocol','verify_resume','evaluate_final','evaluate_math','make_report']:
        p=runtime/(name+'.log')
        if p.exists():shutil.copyfile(p,ROOT/'logs'/(name+'.log'))
    result=json.loads((ROOT/'metrics/final_results.json').read_text());assert result['evaluation_completed']
    assert json.loads((ROOT/'metrics/resume_verification.json').read_text())['all_135_native_tensors_bitwise_equal']
    assert json.loads((ROOT/'metrics/protocol_verification.json').read_text())['status']=='passed'
    cfg=json.loads((ROOT/'config.json').read_text());step=result['training']['steps'];chosen=result['selected_step'];parent=(ROOT/cfg['parent_path']).resolve();final=ROOT/'checkpoints'/f'step_{step:04d}'
    components=[('parent',parent,0,'float32 model input from prior completed EXT run'),('final_model',final/'model.safetensors',step,'float32 native model'),('final_state',final/'training_state.pt',step,'float32 native AdamW; RNG, cursor, schedule and config')]
    if chosen not in [0,step]:
        selected=ROOT/'checkpoints'/f'step_{chosen:04d}';components.extend([('selected_model',selected/'model.safetensors',chosen,'float32 native model'),('selected_state',selected/'training_state.pt',chosen,'float32 native AdamW; RNG, cursor, schedule and config')])
    model=load_file(str(final/'model.safetensors'));initial=load_file(str(parent));assert len(model)==len(initial)==135
    assert sum(v.numel() for v in model.values())==100028328
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in model.values())
    changed=sum(not torch.equal(v,initial[k]) for k,v in model.items());assert changed>0;del model,initial
    state=torch.load(final/'training_state.pt',weights_only=True);assert state['step']==step and state['sampler_cursor']==step*4
    assert state['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
    for s in state['optimizer']['state'].values():
        for name in ['exp_avg','exp_avg_sq']:assert s[name].dtype==torch.float32 and torch.isfinite(s[name]).all()
    del state
    inventory=[]
    for role,path,completed,dtype in components:
        folder=ROOT/'artifacts'/(role+'_parts');expected=sha(path)
        if folder.exists():
            m=verify(folder);assert m['sha256']==expected and m['size_bytes']==path.stat().st_size
        else:m=split(path,folder,identity='EXT2_20261022_'+role,expected_sha256=expected)
        m.update({'run_id':'ext2_20261022','role':role,'completed_new_optimizer_step':completed,'parameters':100028328,'native_contents':dtype,'parent_checkpoint_sha256':cfg['parent_sha256'],'data_manifest_sha256':sha(ROOT/'data/manifest.json'),'tokenizer_sha256':sha(ROOT/'data/tokenizer.json'),'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'status':'saved_local','model_tensor_count':135 if role!='final_state' and role!='selected_state' else None})
        atomic_json(folder/'MANIFEST.json',m);verify(folder)
        inventory.append({'role':role,'directory':str(folder.relative_to(ROOT)),'filename':m['filename'],'size_bytes':m['size_bytes'],'sha256':m['sha256'],'parts':len(m['parts']),'step':completed})
    inventory_obj={'status':'saved_local_pending_repository_push','run_id':'ext2_20261022','components':inventory,'selected_model_role':'parent' if chosen==0 else 'final_model' if chosen==step else 'selected_model','selected_state_role':None if chosen==0 else 'final_state' if chosen==step else 'selected_state','final_step':step,'selected_step':chosen,'changed_parameter_tensors':changed,'general_superiority_established':False,'all_parts_strictly_under_bytes':90000000}
    inventory_obj.update(training_completed_fixed_step_count=result['training']['completed_fixed_step_count'],measured_comparison='parent_EXT256 versus selected_EXT2 on frozen external holdouts',registered_improvement_criterion_met=result['registered_improvement_criterion_met'],reference_models_measured=False)
    atomic_json(ROOT/'artifacts/MANIFEST.json',inventory_obj)
    src={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').glob('*.py')}
    atomic_json(ROOT/'final_source_manifest.json',src)
    archive=ROOT/'artifacts/FlyGraph_EXT2_20261022_Report_Code_Data.zip'
    included=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.name in VOLATILE or p.name.endswith('.tmp'):continue
        rel=p.relative_to(ROOT)
        if rel.parts[0] in ['checkpoints','verification']:continue
        if rel.parts[0]=='artifacts' and p.name!='MANIFEST.json':continue
        if p.suffix in ['.safetensors','.pt','.zip']:continue
        included.append(p)
    included.append(ROOT/'checkpoints/config.json')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in included:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('CHECKSUMS.sha256',''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(included)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in z.read('CHECKSUMS.sha256').decode().splitlines():
            h,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==h
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in VOLATILE and not p.name.endswith('.tmp')]
    (ROOT/'CHECKSUMS.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(files)))
    print(json.dumps({'inventory':inventory_obj,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive)},indent=2),flush=True)
if __name__=='__main__':main()
