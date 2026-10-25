"""Publish completed untrained architecture scaffold, then clean-readback checks.

Authorized target is only binouv/model. No force or automatic write retries.
"""
import datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];URL='https://github.com/binouv/model.git';BRANCH='flygraph/dense-control-20261025';BASE='767a4d5468cc2bd175b8c9b63d15e3147db321ff';REL=Path('runs/dense_control_20261025');STAGE=Path('/workspace/dense_control_publish_20261025');READBACK=Path('/workspace/dense_control_readback_20261025');RUNTIME=Path('/workspace/dense_control_runtime_20261025');ENV=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def git(*args,cwd=REPO,timeout=600):
 p=subprocess.run(['git',*args],cwd=cwd,env=ENV,text=True,capture_output=True,timeout=timeout)
 if p.returncode:
  if args and args[0]=='push':
   check=subprocess.run(['git','ls-remote',URL,'refs/heads/'+BRANCH],env=ENV,text=True,capture_output=True,timeout=120);dump(ROOT/'PUBLICATION.json',dict(status='push_failed_or_uncertain_no_retry',error=p.stderr[-6000:],remote_readback_after_error=check.stdout.strip(),remote_readback_exit=check.returncode))
  raise RuntimeError('git '+args[0]+' failed: '+p.stderr[-6000:])
 if p.stderr.strip():print(p.stderr[-3000:],flush=True)
 return p.stdout.strip()
def checksums(root):
 files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256' and '__pycache__' not in p.parts);(root/'CHECKSUMS.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in files))
def commit(message):
 changed=git('diff','--cached','--name-only',cwd=STAGE).splitlines();assert changed and all(p.startswith(str(REL)+'/') for p in changed);git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',message,cwd=STAGE);return git('rev-parse','HEAD',cwd=STAGE)
def push_once(head):
 git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=STAGE);assert git('ls-remote','origin','refs/heads/'+BRANCH,cwd=STAGE).split()[0]==head

def main():
 RUNTIME.mkdir(exist_ok=True);smoke=json.loads((ROOT/'metrics/smoke.json').read_text());assert smoke['status']=='passed' and smoke['optimizer_updates']==0;assert json.loads((ROOT/'metrics/tiny_profile.json').read_text())['status']=='passed'
 assert git('remote','get-url','origin')==URL;assert not git('ls-remote','origin','refs/heads/'+BRANCH),'Existing branch requires inspection';assert not STAGE.exists() and not READBACK.exists()
 git('fetch','origin','flygraph/diagnostics4-20261025');base=git('rev-parse','FETCH_HEAD');assert base==BASE,'Upstream changed; inspect before write';assert not git('ls-tree',base,'--',str(REL))
 git('worktree','add','--no-checkout','-b',BRANCH,str(STAGE),base);git('reset','--mixed','--quiet','HEAD',cwd=STAGE);dest=STAGE/REL
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.name in ['PUBLICATION.json','PUSH_COMPLETE.json','CHECKSUMS.sha256'] or p.suffix=='.tmp':continue
  assert p.stat().st_size<90_000_000;target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 (dest/'.gitattributes').write_text('*.safetensors -text -filter -diff -merge\n*.zip -text -filter -diff -merge\n');dump(dest/'PUBLICATION.json',dict(status='uploaded_unverified_pending_readback',repo=URL,branch=BRANCH,base_commit=base,trained=False));checksums(dest)
 files=sorted(p for p in dest.rglob('*') if p.is_file())
 for i in range(0,len(files),100):git('add','-f','--',*[str(p.relative_to(STAGE)) for p in files[i:i+100]],cwd=STAGE)
 artifact=commit('FlyGraph: add tested untrained dense deep-thin control');push_once(artifact);dump(ROOT/'PUBLICATION.json',dict(status='uploaded_unverified',artifact_commit=artifact,branch=BRANCH))
 git('clone','--filter=blob:none','--no-checkout','--depth','1','--single-branch','--branch',BRANCH,URL,str(READBACK),cwd=Path('/workspace'));git('sparse-checkout','init','--cone',cwd=READBACK);git('sparse-checkout','set',str(REL),cwd=READBACK);git('checkout',BRANCH,cwd=READBACK);assert git('rev-parse','HEAD',cwd=READBACK)==artifact;remote=READBACK/REL;n=0
 for line in (remote/'CHECKSUMS.sha256').read_text().splitlines():h,name=line.split('  ',1);assert sha(remote/name)==h,name;n+=1
 remote_smoke=RUNTIME/'remote_smoke.json';subprocess.run([sys.executable,str(remote/'src/smoke.py'),'--readback','--output',str(remote_smoke)],cwd=remote,env=ENV,check=True,stdout=subprocess.DEVNULL)
 remote_counts=RUNTIME/'remote_counts.json';subprocess.run([sys.executable,str(remote/'src/count_parameters.py'),'--output',str(remote_counts)],cwd=remote,env=ENV,check=True,stdout=subprocess.DEVNULL);assert json.loads(remote_counts.read_text())==json.loads((ROOT/'metrics/parameter_counts.json').read_text())
 receipt=dict(status='published_verified',repo=URL,branch=BRANCH,base_commit=base,artifact_commit=artifact,verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_published_file_checksums_verified=n,remote_smoke=json.loads(remote_smoke.read_text()),remote_parameter_counts_equal=True,no_force_push=True,scope='Untrained architecture scaffold; tiny init weights only; no quality benchmark or full-size training',original_EXT3_unchanged=True)
 dump(ROOT/'PUBLICATION.json',receipt);dump(dest/'PUBLICATION.json',receipt);checksums(dest);git('add','--',str(REL/'PUBLICATION.json'),str(REL/'CHECKSUMS.sha256'),cwd=STAGE);head=commit('FlyGraph dense control: verify remote source and native tiny replay');push_once(head)
 git('fetch','--depth','1','origin',BRANCH,cwd=READBACK);assert git('rev-parse','FETCH_HEAD',cwd=READBACK)==head
 for name in ['PUBLICATION.json','CHECKSUMS.sha256']:assert git('show',head+':'+str(REL/name),cwd=READBACK)==(dest/name).read_text().strip()
 git('merge-base','--is-ancestor',base,head,cwd=STAGE);changed=git('diff','--name-only',base,head,cwd=STAGE).splitlines();assert all(p.startswith(str(REL)+'/') for p in changed)
 for name in ['CHECKSUMS.sha256','.gitattributes']:shutil.copyfile(dest/name,ROOT/name)
 dump(ROOT/'PUSH_COMPLETE.json',dict(status='published_verified',branch=BRANCH,artifact_commit=artifact,receipt_commit=head,remote_receipt_readback=True,only_new_directory_changed=True,report_url=f'https://github.com/binouv/model/blob/{head}/{REL}/README_RU.md'));print('PUSH_COMPLETE',head,flush=True)
if __name__=='__main__':main()
