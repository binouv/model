"""Publish this completed run, then read back real binary bytes from GitHub.

No force pushes, alternate hosts, credentials in logs, or automatic push retries.
The caller must be authorized to publish to binouv/model.
"""
import datetime,json,os,shutil,subprocess,sys
from pathlib import Path
from common import ROOT,atomic_json,sha

REPO=ROOT.parents[1]
BRANCH='flygraph/ext-20261022'
URL='https://github.com/binouv/model.git'
REL=Path('runs/ext2_20261022')
STAGE=Path('/workspace/ext2_publish_20261022')
READBACK=Path('/workspace/ext2_readback_20261022')
RUNTIME=Path('/workspace/ext2_runtime_20261022')
VOLATILE={'STATUS.json','PIPELINE_STATE.json','PIPELINE_LOCK','CHECKSUMS.sha256','PUBLICATION.json'}
ENV=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1')

def git(*args,cwd=REPO,timeout=1800):
    # Native binary checkpoint chunks do not benefit from an expensive broad
    # delta search. Keep packing bounded on this four-core/16-GiB machine.
    options=['-c','pack.window=0','-c','pack.threads=2','-c','core.compression=1'] if args and args[0]=='push' else []
    p=subprocess.run(['git',*options,*args],cwd=cwd,env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if p.returncode:
        # A failed push might have reached the server: inspect before stopping.
        if args and args[0]=='push':
            check=subprocess.run(['git','ls-remote',URL,'refs/heads/'+BRANCH],env=ENV,text=True,capture_output=True,timeout=120)
            atomic_json(ROOT/'PUBLICATION.json',{'status':'push_failed_or_uncertain_no_retry','failed_action':list(args),'remote_readback_after_error':check.stdout.strip(),'remote_readback_exit':check.returncode,'error':p.stderr[-6000:]})
        raise RuntimeError('git '+args[0]+' failed: '+p.stderr[-6000:])
    if p.stderr.strip():print(p.stderr[-3000:],flush=True)
    return p.stdout.strip()

def eligible(p):
    rel=p.relative_to(ROOT)
    if not p.is_file() or p.is_symlink() or '__pycache__' in rel.parts or p.name in VOLATILE or p.name.endswith('.tmp'):return False
    if rel.parts[0]=='verification':return False
    if rel.parts[0]=='checkpoints' and rel!=Path('checkpoints/config.json'):return False
    return True

def checksums(dest):
    files=sorted(p for p in dest.rglob('*') if p.is_file() and p.name!='CHECKSUMS.sha256' and '__pycache__' not in p.parts)
    (dest/'CHECKSUMS.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(dest))+'\n' for p in files))

def commit(message):
    changed=git('diff','--cached','--name-only',cwd=STAGE).splitlines()
    assert changed and all(p.startswith(str(REL)+'/') for p in changed),changed
    git('-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m',message,cwd=STAGE)
    return git('rev-parse','HEAD',cwd=STAGE)

def push_once(expected):
    git('push','origin','HEAD:refs/heads/'+BRANCH,cwd=STAGE)
    remote=git('ls-remote','origin','refs/heads/'+BRANCH,cwd=STAGE).split()[0]
    assert remote==expected,(remote,expected)

def main():
    RUNTIME.mkdir(exist_ok=True)
    final=json.loads((ROOT/'metrics/final_results.json').read_text());assert final['evaluation_completed']
    assert json.loads((ROOT/'metrics/resume_verification.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'metrics/protocol_verification.json').read_text())['status']=='passed'
    assert json.loads((ROOT/'metrics/math_results.json').read_text())['completed']
    inventory=json.loads((ROOT/'artifacts/MANIFEST.json').read_text())
    assert git('remote','get-url','origin')==URL
    assert not git('ls-remote','origin','refs/heads/'+BRANCH),'Target branch exists; inspect instead of overwriting'
    assert not STAGE.exists() and not READBACK.exists(),'Existing publication workspaces require inspection'
    git('fetch','origin','main');base=git('rev-parse','origin/main')
    assert not git('ls-tree','origin/main','--',str(REL)),'Run path already exists upstream'
    git('worktree','add','-b',BRANCH,str(STAGE),'origin/main')
    dest=STAGE/REL
    for src in sorted(ROOT.rglob('*')):
        if not eligible(src):continue
        target=dest/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        assert src.stat().st_size<90_000_000,(src,src.stat().st_size)
        if '.part' in src.name:os.link(src,target)
        else:shutil.copyfile(src,target)
    (dest/'.gitattributes').write_text('*.part* -text -filter -diff -merge\n')
    atomic_json(dest/'PUBLICATION.json',{'status':'uploaded_unverified_pending_readback','repo':URL,'branch':BRANCH,'base_commit':base,'general_superiority_established':False})
    checksums(dest)
    # At most ~1.6 GB raw bytes in the first push; selected earlier checkpoint,
    # if present, forms a second <=1.2 GB push. Each is one normal Git push.
    allfiles=sorted(p for p in dest.rglob('*') if p.is_file())
    second=[p for p in allfiles if any(s in {'selected_model_parts','selected_state_parts'} for s in p.relative_to(dest).parts)]
    first=[p for p in allfiles if p not in second]
    assert sum(p.stat().st_size for p in first)<1_900_000_000
    assert sum(p.stat().st_size for p in second)<1_900_000_000
    commits=[]
    for files,label in [(first,'completed experiment and native final checkpoint'),(second,'dev-selected native checkpoint')]:
        if not files:continue
        # Chunk argument count while preserving literal file paths.
        for i in range(0,len(files),100):git('add','-f','--',*[str(p.relative_to(STAGE)) for p in files[i:i+100]],cwd=STAGE)
        head=commit('FlyGraph EXT2: '+label);push_once(head);commits.append(head)
    artifact_commit=commits[-1]
    atomic_json(ROOT/'PUBLICATION.json',{'status':'uploaded_unverified','repo':URL,'branch':BRANCH,'artifact_commit':artifact_commit,'commits':commits})
    git('clone','--filter=blob:none','--no-checkout','--depth','1','--single-branch','--branch',BRANCH,URL,str(READBACK),cwd=Path('/workspace'))
    git('sparse-checkout','init','--cone',cwd=READBACK)
    git('sparse-checkout','set',str(REL),cwd=READBACK)
    git('checkout',BRANCH,cwd=READBACK)
    assert git('rev-parse','HEAD',cwd=READBACK)==artifact_commit
    remote=READBACK/REL
    for line in (remote/'CHECKSUMS.sha256').read_text().splitlines():
        h,name=line.split('  ',1);assert sha(remote/name)==h,name
    # Import the remote byte verifier, not any local/working-tree binaries.
    import importlib.util
    spec=importlib.util.spec_from_file_location('remote_parts',remote/'src/checkpoint_parts.py');parts=importlib.util.module_from_spec(spec);spec.loader.exec_module(parts)
    receipts=[];restored=READBACK/'restored';restored.mkdir()
    for component in inventory['components']:
        folder=remote/component['directory'];target=restored/(component['role']+'_'+component['filename'])
        m=parts.verify(folder);assert m['sha256']==component['sha256']
        r=parts.restore(folder,target);receipts.append(dict(role=component['role'],**r))
    replay_receipts=[]
    for model_role,state_role in [('final_model','final_state'),('selected_model','selected_state')]:
        byrole={c['role']:c for c in inventory['components']}
        if model_role not in byrole:continue
        weights=restored/(model_role+'_'+byrole[model_role]['filename']);state=restored/(state_role+'_'+byrole[state_role]['filename'])
        output=RUNTIME/(model_role+'_remote_resume.json')
        subprocess.run([sys.executable,str(remote/'src/verify_remote_checkpoint.py'),'--weights',str(weights),'--state',str(state),'--output',str(output)],cwd=remote,check=True,env=ENV)
        replay_receipts.append(json.loads(output.read_text()))
    # Fixed external dev prompt inference with reconstructed final bytes.
    byrole={c['role']:c for c in inventory['components']}
    prompt=json.loads((remote/'data/dev_mcq.jsonl').read_text().splitlines()[0])['prompt']
    weights=restored/('final_model_'+byrole['final_model']['filename'])
    p=subprocess.run([sys.executable,str(remote/'src/infer.py'),'--weights',str(weights),'--prompt',prompt,'--max-new-tokens','16'],cwd=remote,env=ENV,text=True,capture_output=True,check=True)
    receipt={'status':'published_verified','repo':URL,'branch':BRANCH,'artifact_commit':artifact_commit,'base_commit':base,'artifact_commits':commits,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_published_file_checksums_verified':True,'all_remote_binary_components_reconstructed':receipts,'native_resume_receipts':replay_receipts,'fixed_external_dev_prompt':prompt,'fixed_prompt_inference_stdout':p.stdout,'no_force_push':True,'general_superiority_established':False}
    atomic_json(ROOT/'PUBLICATION.json',receipt);atomic_json(dest/'PUBLICATION.json',receipt)
    checksums(dest);git('add','--',str(REL/'PUBLICATION.json'),str(REL/'CHECKSUMS.sha256'),cwd=STAGE)
    receipt_commit=commit('FlyGraph EXT2: verify remote binary readback and native resume');push_once(receipt_commit)
    atomic_json(ROOT/'PUSH_COMPLETE.json',{'status':'published_verified','branch':BRANCH,'artifact_commit':artifact_commit,'receipt_commit':receipt_commit,'report_url':f'https://github.com/binouv/model/blob/{receipt_commit}/{REL}/REPORT_RU.md','receipt_url':f'https://github.com/binouv/model/blob/{receipt_commit}/{REL}/PUBLICATION.json'})
    print('PUSH_COMPLETE',receipt_commit,flush=True)

if __name__=='__main__':main()
