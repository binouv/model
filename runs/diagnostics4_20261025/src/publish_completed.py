"""Publish only this completed diagnostic into authorized binouv/model.

Ordinary non-force push; uncertain outcomes are inspected once, never retried.
Only this new run directory enters the index; prior artifacts are untouched.
"""
import datetime,json,os,shutil,subprocess,sys
from pathlib import Path
from common import ROOT,atomic_json,sha
REPO=ROOT.parents[1];URL='https://github.com/binouv/model.git';BRANCH='flygraph/diagnostics4-20261025';REL=Path('runs/diagnostics4_20261025');STAGE=Path('/workspace/diagnostics4_publish_20261025');READBACK=Path('/workspace/diagnostics4_readback_20261025');RUNTIME=Path('/workspace/diagnostics4_runtime_20261025');BASE='44aaa833f23e663f21415142bdd23369549fd707';ENV=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1')
def git(*args,cwd=REPO,timeout=600):
 p=subprocess.run(['git',*args],cwd=cwd,env=ENV,text=True,capture_output=True,timeout=timeout)
 if p.returncode:
  if args and args[0]=='push':
   q=subprocess.run(['git','ls-remote',URL,'refs/heads/'+BRANCH],env=ENV,text=True,capture_output=True,timeout=120);atomic_json(ROOT/'PUBLICATION.json',dict(status='push_failed_or_uncertain_no_retry',error=p.stderr[-6000:],remote_readback_after_error=q.stdout.strip(),remote_readback_exit=q.returncode))
  raise RuntimeError('git '+args[0]+' failed: '+p.stderr[-6000:])
 if p.stderr.strip():print(p.stderr[-3000:],flush=True)
 return p.stdout.strip()
def checksums(dest):
 files=sorted(p for p in dest.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256' and '__pycache__' not in p.parts)
 (dest/'CHECKSUMS.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(dest))+'\n' for p in files))
def commit(message):
 changed=git('diff','--cached','--name-only',cwd=STAGE).splitlines();assert changed and all(p.startswith(str(REL)+'/') for p in changed),changed
 git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',message,cwd=STAGE);return git('rev-parse','HEAD',cwd=STAGE)
def push_once(expected):
 git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=STAGE);assert git('ls-remote','origin','refs/heads/'+BRANCH,cwd=STAGE).split()[0]==expected

def main():
 RUNTIME.mkdir(exist_ok=True)
 assert json.loads((ROOT/'metrics/results.json').read_text())['completed']
 for name in ['artifact_verification','scoring_protocol_verification','chunked_delta_verification']:assert json.loads((ROOT/f'metrics/{name}.json').read_text())['status']=='passed'
 assert git('remote','get-url','origin')==URL;assert not git('ls-remote','origin','refs/heads/'+BRANCH),'Existing branch requires inspection, no overwrite';assert not STAGE.exists() and not READBACK.exists()
 git('fetch','origin','flygraph/ext3-20261025');base=git('rev-parse','FETCH_HEAD');assert base==BASE,'Upstream changed; inspect before publication'
 assert not git('ls-tree',base,'--',str(REL))
 cfg=json.loads((ROOT/'config.json').read_text())
 for arm,folder in [('EXT2','ext2_20261022'),('EXT3','ext3_20261025')]:
  path=f'runs/{folder}/artifacts/selected_model_parts/MANIFEST.json';mm=json.loads(git('show',base+':'+path));assert mm['sha256']==cfg[arm+'_sha256']
 git('worktree','add','--no-checkout','-b',BRANCH,str(STAGE),base);git('reset','--mixed','--quiet','HEAD',cwd=STAGE);dest=STAGE/REL
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.name in ['PUBLICATION.json','PUSH_COMPLETE.json','CHECKSUMS.sha256'] or p.name.endswith('.tmp'):continue
  assert p.stat().st_size<90_000_000,(p,p.stat().st_size);target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 (dest/'.gitattributes').write_text('*.safetensors -text -filter -diff -merge\n*.zip -text -filter -diff -merge\n')
 atomic_json(dest/'PUBLICATION.json',dict(status='uploaded_unverified_pending_readback',repo=URL,branch=BRANCH,base_commit=base));checksums(dest)
 files=sorted(p for p in dest.rglob('*') if p.is_file())
 for i in range(0,len(files),100):git('add','-f','--',*[str(p.relative_to(STAGE)) for p in files[i:i+100]],cwd=STAGE)
 artifact=commit('FlyGraph: complete frozen diagnostics and architecture audit');push_once(artifact)
 atomic_json(ROOT/'PUBLICATION.json',dict(status='uploaded_unverified',artifact_commit=artifact,branch=BRANCH))
 git('clone','--filter=blob:none','--no-checkout','--depth','1','--single-branch','--branch',BRANCH,URL,str(READBACK),cwd=Path('/workspace'));git('sparse-checkout','init','--cone',cwd=READBACK);git('sparse-checkout','set',str(REL),cwd=READBACK);git('checkout',BRANCH,cwd=READBACK);assert git('rev-parse','HEAD',cwd=READBACK)==artifact;remote=READBACK/REL
 n=0
 for line in (remote/'CHECKSUMS.sha256').read_text().splitlines():
  h,name=line.split('  ',1);assert sha(remote/name)==h,name;n+=1
 output=RUNTIME/'remote_probe_verification.json';subprocess.run([sys.executable,str(remote/'src/verify_artifacts.py'),'--output',str(output)],cwd=remote,env=ENV,check=True,stdout=subprocess.DEVNULL)
 # Use bytes already independently restored from the immutable EXT3 remote commit.
 weights=Path('/workspace/ext3_readback_20261025/restored/selected_model_model.safetensors');assert sha(weights)==cfg['EXT3_sha256'];kernel=RUNTIME/'remote_chunked_verification.json';subprocess.run([sys.executable,str(remote/'src/verify_chunked_delta.py'),'--weights',str(weights),'--output',str(kernel)],cwd=remote,env=ENV,check=True,stdout=subprocess.DEVNULL)
 receipt=dict(status='published_verified',repo=URL,branch=BRANCH,base_commit=base,artifact_commit=artifact,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_published_file_checksums_verified=n,remote_probe_refit=json.loads(output.read_text()),remote_kernel_check=json.loads(kernel.read_text()),no_force_push=True,new_backbone_weights=False,backbone_reference='Existing immutable EXT2/EXT3 native checkpoints; no mutation or repack',general_superiority_established=False)
 atomic_json(ROOT/'PUBLICATION.json',receipt);atomic_json(dest/'PUBLICATION.json',receipt);checksums(dest);git('add','--',str(REL/'PUBLICATION.json'),str(REL/'CHECKSUMS.sha256'),cwd=STAGE);receipt_commit=commit('FlyGraph diagnostics: verify remote bytes, probe refit and kernel checks');push_once(receipt_commit)
 # Read the final receipt commit through a fresh fetch in the independent clone.
 git('fetch','--depth','1','origin',BRANCH,cwd=READBACK);assert git('rev-parse','FETCH_HEAD',cwd=READBACK)==receipt_commit
 for name in ['PUBLICATION.json','CHECKSUMS.sha256']:assert git('show',receipt_commit+':'+str(REL/name),cwd=READBACK)==(dest/name).read_text().strip()
 atomic_json(ROOT/'PUSH_COMPLETE.json',dict(status='published_verified',branch=BRANCH,artifact_commit=artifact,receipt_commit=receipt_commit,report_url=f'https://github.com/binouv/model/blob/{receipt_commit}/{REL}/REPORT_RU.md',architecture_url=f'https://github.com/binouv/model/blob/{receipt_commit}/{REL}/ARCHITECTURE_PLAN_RU.md',remote_receipt_readback=True));print('PUSH_COMPLETE',receipt_commit,flush=True)
if __name__=='__main__':main()
