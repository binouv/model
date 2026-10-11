"""Publish exact, locally trained artifacts via encrypted one-shot handoff; no training."""
from __future__ import annotations
import base64,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
RUN='runs/l100da_20261012';PREFIX=RUN+'/transport'
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'runs/artifact_recovery_20261012'))
import checkpoint_parts as cp
spec=importlib.util.spec_from_file_location('handoff',ROOT/'runs/artifact_transfer_20261012/transport.py')
handoff=importlib.util.module_from_spec(spec);spec.loader.exec_module(handoff)

def checked_relative(value):
    if not isinstance(value,str) or '\\' in value:raise ValueError('Invalid path')
    p=PurePosixPath(value)
    if p.is_absolute() or any(x in ('..','.git','') for x in p.parts):raise ValueError('Unsafe path')
    return p

def extract_checked(archive:Path,destination:Path,max_total:int,allowed_roots:set[str]|None=None):
    paths=[];total=0
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            rel=checked_relative(entry.filename)
            if entry.is_dir() or not rel.parts:continue
            if allowed_roots is not None and rel.parts[0] not in allowed_roots:raise ValueError('Unexpected source member')
            if (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Archive symlink not allowed')
            total+=entry.file_size
            if total>max_total:raise ValueError('Archive expansion bound exceeded')
            target=destination.joinpath(*rel.parts)
            if target.is_symlink() or not target.resolve().is_relative_to(destination.resolve()):raise ValueError('Destination escapes root')
            if target.exists():
                h=hashlib.sha256()
                with z.open(entry) as f:
                    for block in iter(lambda:f.read(1048576),b''):h.update(block)
                if cp.digest(target)!=h.hexdigest():raise FileExistsError('Different existing source file')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(entry) as f,target.open('xb') as out:shutil.copyfileobj(f,out,1048576)
            paths.append(str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(rel))
    return paths

def commit(paths,message):
    cp.git('add','-f','--',*paths)
    if cp.git('diff','--cached','--name-only'):
        cp.git('commit','-m',message+' [skip ci]')
        for attempt in range(3):
            try:cp.git('push','origin','HEAD:refs/heads/main');break
            except RuntimeError:
                if attempt==2:raise
                cp.git('pull','--rebase','origin','main')
    return cp.git('rev-parse','HEAD')

def run():
    if os.environ.get('GITHUB_REPOSITORY')!='binouv/model':raise ValueError('Wrong repository')
    os.chdir(ROOT)
    s=json.loads((ROOT/PREFIX/'TRANSFER_SPEC.json').read_text())
    if s.get('run_id')!='l100da_20261012' or s.get('candidate_training_steps')!=256:raise ValueError('Wrong run binding')
    if (ROOT/PREFIX/'PUBLISHED_VERIFIED.json').exists():raise FileExistsError('Already published; no repeat')
    token=os.environ.get('GH_TOKEN')
    if not token:raise ValueError('Missing workflow token')
    os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode())
    cp.git('config','user.name','github-actions[bot]');cp.git('config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
    handoff.MODEL_SHA=s['native_model_sha256']
    with tempfile.TemporaryDirectory(prefix='flygraph-da-publish-') as temporary:
        tmp=Path(temporary);key=tmp/'private.pem';public=handoff.make_key(key)
        public.update(request_commit=os.environ.get('GITHUB_SHA'),workflow_run_id=os.environ.get('GITHUB_RUN_ID'))
        cp.git('pull','--ff-only','origin','main')
        (ROOT/PREFIX/'PUBLIC_KEY.json').write_text(json.dumps(public,indent=2)+'\n')
        commit([PREFIX+'/PUBLIC_KEY.json'],'L100DA publish one-shot transfer public key')
        print('Public key ready for exact model-artifact handoff.',flush=True)
        expected={'source.zip':s['source']};staged={};outputs={'source.zip':tmp/'source.zip'}
        for name,artifact in s['artifacts'].items():
            if name not in ('native','bf16'):raise ValueError('Unknown artifact type')
            if artifact['destination']!='models/l100da_20261012_step0256/'+name:raise ValueError('Unexpected model path')
            folder=tmp/name;folder.mkdir();staged[name]=folder
            (folder/'MANIFEST.json').write_text(json.dumps(artifact['manifest'],indent=2)+'\n')
            manifest=cp.read_manifest(folder)
            for part in manifest['parts']:
                label=name+'/'+part['name'];expected[label]=part;outputs[label]=folder/part['name']
        names=list(expected);deadline=time.monotonic()+1200
        for batch,start in enumerate(range(0,len(names),5)):
            batch_names=names[start:start+5];envelope=None
            while time.monotonic()<deadline:
                cp.git('fetch','origin','main')
                try:
                    raw=cp.git('show','FETCH_HEAD:'+PREFIX+f'/ENCRYPTED_URLS_{batch:02d}.json')
                    if len(raw)>50000:raise ValueError('Oversized envelope')
                    value=json.loads(raw)
                    if value.get('transfer_id')==public['transfer_id']:envelope=value;break
                except RuntimeError:pass
                time.sleep(5)
            if envelope is None:raise RuntimeError('Finite handoff timeout')
            payload=handoff.decrypt(key,public,envelope)
            if payload.get('checkpoint_sha256')!=s['native_model_sha256'] or payload.get('batch')!=batch:raise ValueError('Wrong checkpoint or batch identity')
            entries=payload.get('files')
            if not isinstance(entries,list) or len(entries)!=len(batch_names):raise ValueError('Wrong artifact count')
            by_name={e['name']:e for e in entries}
            if set(by_name)!=set(batch_names):raise ValueError('Artifact name mismatch')
            for name in batch_names:
                wanted=expected[name];actual=by_name[name]
                if actual['size_bytes']!=wanted['size_bytes'] or actual['sha256']!=wanted['sha256']:raise ValueError('Artifact metadata mismatch')
                handoff.download(actual['url'],outputs[name],wanted)
                print('Downloaded and verified '+name,flush=True)
            del payload,entries,by_name,actual
        cp.git('pull','--ff-only','origin','main')
        changed=extract_checked(tmp/'source.zip',ROOT/RUN,50_000_000,{'base','src','data','configs','tests','metrics','reports','checkpoints','requirements.txt','README.md','MODEL_CARD.md'})
        for name,artifact in s['artifacts'].items():
            cp.verify(staged[name]);target=ROOT/artifact['destination']
            if target.exists():raise FileExistsError('Model destination exists')
            shutil.copytree(staged[name],target)
            (target/'.gitattributes').write_text('*.part* -text -filter -diff -merge\n')
            changed.extend([artifact['destination']])
        binary_commit=commit(changed,'L100DA store real step256 native/optimizer and BF16 bytes in <90MB parts with source/data/results')
        fresh=tmp/'fresh-clone';cp.git('clone','--depth','1','--branch','main','https://github.com/binouv/model.git',str(fresh))
        restored={}
        for name,artifact in s['artifacts'].items():
            folder=fresh/artifact['destination'];cp.verify(folder)
            output=tmp/('remote-'+artifact['manifest']['filename']);restored[name]=cp.restore(folder,output)
        check=tmp/'native-restored';check.mkdir()
        extract_checked(Path(restored['native']['output']),check,2_000_000_000)
        test=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=fresh/RUN,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        if test.returncode:raise RuntimeError('Readback engineering tests failed')
        verify=subprocess.run([sys.executable,'src/verify_export.py','--native',str(check),'--bf16',restored['bf16']['output']],cwd=fresh/RUN,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=300)
        if verify.returncode:raise RuntimeError('Native/BF16 load or inference verification failed')
        receipt={'status':'published_verified','binary_commit':binary_commit,'workflow_run_id':os.environ.get('GITHUB_RUN_ID'),'candidate_training_steps':256,'remote_candidate_training_steps':0,'fresh_clone_verified':True,'artifacts':restored,'verification':json.loads(verify.stdout),'engineering_tests':test.stdout,'overall_qwen_superiority_goal_achieved':False}
        cp.git('pull','--ff-only','origin','main')
        (ROOT/PREFIX/'PUBLISHED_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
        result=commit([PREFIX+'/PUBLISHED_VERIFIED.json'],'L100DA verify real published model/optimizer by independent clone and strict load')
        print(json.dumps({'status':'published_verified','commit':result,'binary_commit':binary_commit}),flush=True)
if __name__=='__main__':
    try:run()
    except Exception as error:
        print('Artifact publication failed: '+type(error).__name__+'. No unverified success claimed.',file=sys.stderr)
        raise SystemExit(1) from None
