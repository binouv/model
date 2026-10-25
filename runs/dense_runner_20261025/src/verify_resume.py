"""Fresh-process uninterrupted vs interrupted/resumed tiny training proof."""
import argparse,datetime,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import torch
from safetensors.torch import load_file
from checkpoint_parts import verify as verify_parts,restore as restore_parts
ROOT=Path(__file__).resolve().parents[1]
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--saved-root',type=Path);a=p.parse_args();work=a.workspace
 if work.exists():raise FileExistsError('new workspace required')
 work.mkdir(parents=True);base=[sys.executable,str(ROOT/'src/train.py'),'--model-config',str(ROOT/'configs/tiny_smoke.json'),'--train-config',str(ROOT/'configs/tiny_train.json'),'--data',str(ROOT/'data/external_dev_fixtures.jsonl'),'--tokenizer',str(ROOT/'data/tokenizer.json')];logs=[]
 def run(name,extra=(),cmd=None,expected_error=None):
  args=(cmd or base)+['--output',str(work/name)]+list(extra);result=subprocess.run(args,text=True,capture_output=True);(work/(name+'.log')).write_text(result.stdout+result.stderr);logs.append(dict(name=name,exit_code=result.returncode))
  if expected_error is None:assert result.returncode==0,result.stderr
  else:assert result.returncode!=0 and expected_error in result.stderr,result.stderr
 if a.saved_root is None:
  run('continuous');run('split',extra=['--stop-after','3']);saved=work
 else:saved=a.saved_root
 before=saved/'split/step_000003';reference=saved/'continuous/step_000006';run('resumed',extra=['--resume',str(before)]);after=work/'resumed/step_000006'
 A=load_file(str(reference/'model.safetensors'));B=load_file(str(after/'model.safetensors'));assert equal(A,B)
 SA=torch.load(reference/'training_state.pt',map_location='cpu',weights_only=True);SB=torch.load(after/'training_state.pt',map_location='cpu',weights_only=True);assert equal(SA,SB)
 assert len(SA['trace'])==6 and SA['step']==6 and SA['sampler']['epoch']>=1
 part_checks=[]
 for folder in [before,reference,after]:
  for role in ['model','training_state']:
   m=verify_parts(folder/(role+'_parts'));part_checks.append(dict(checkpoint=str(folder),role=role,sha256=m['sha256'],parts=len(m['parts'])))
 # Reconstruct actual checkpoint components from parts; compare native tensors.
 restored=work/'restored';restored.mkdir();restore_parts(before/'model_parts',restored/'model.safetensors');restore_parts(before/'training_state_parts',restored/'training_state.pt');shutil.copyfile(before/'MANIFEST.json',restored/'MANIFEST.json')
 run('resumed_from_parts',extra=['--resume',str(restored)]);assert equal(load_file(str(work/'resumed_from_parts/step_000006/model.safetensors')),A);assert equal(torch.load(work/'resumed_from_parts/step_000006/training_state.pt',map_location='cpu',weights_only=True),SA)
 budget=base.copy();budget[budget.index('--train-config')+1]=str(ROOT/'configs/tiny_budget.json');run('budget',cmd=budget);bs=json.loads((work/'budget/SUMMARY.json').read_text());assert bs['complete'] and bs['input_tokens']==70 and bs['step']==2 and bs['reason']=='token_budget'
 altered=work/'altered_source.jsonl';altered.write_text((ROOT/'data/external_dev_fixtures.jsonl').read_text().replace('\n',' \n',1));bad=base.copy();bad[bad.index('--data')+1]=str(altered);run('reject_data_change',cmd=bad,extra=['--resume',str(before)],expected_error='exact resume refused')
 broken=work/'corrupt_checkpoint';broken.mkdir()
 for name in ['MANIFEST.json','model.safetensors','training_state.pt']:shutil.copyfile(before/name,broken/name)
 with (broken/'model.safetensors').open('r+b') as f:f.seek(-1,2);b=f.read(1);f.seek(-1,2);f.write(bytes([b[0]^1]))
 run('reject_corruption',extra=['--resume',str(broken)],expected_error='checkpoint hash mismatch')
 # Failed/corrupt fixtures are temporary engineering data, not published models.
 shutil.rmtree(broken);altered.unlink()
 result=dict(status='passed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Tiny optimizer/resume engineering smoke only; not pretrained quality',canonical_steps=6,split_after=3,resumed_steps=3,parameters=598336,input_tokens=SA['tokens'],epochs_crossed=SA['sampler']['epoch'],model_tensors_bitwise_equal=len(A),entire_native_state_equal=True,optimizer_state_equal=True,python_numpy_torch_rng_equal=True,sampler_state_equal=True,all_losses_and_gradient_norms_equal=True,trace=SA['trace'],parts_reconstruction_resume_equal=True,token_budget_test=bs,changed_data_rejected=True,corrupted_bytes_rejected=True,part_checks=part_checks,subprocess_results=logs,full_size_training_executed=False)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['trace','part_checks','subprocess_results']},indent=2))
if __name__=='__main__':main()
