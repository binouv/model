"""Resume verified binary publication; save each part batch before final verification."""
from __future__ import annotations
import base64, hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile, time, zlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('publisher',Path(__file__).with_name('publish.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def unpack(value):
    if set(value)!={'transfer_id','compressed'}:raise ValueError('Unexpected compact envelope')
    encoded=base64.b64decode(value['compressed'],validate=True)
    if len(encoded)>16000:raise ValueError('Compressed payload too large')
    d=zlib.decompressobj();raw=d.decompress(encoded,16001)
    if len(raw)>16000 or not d.eof or d.unused_data:raise ValueError('Invalid bounded compression')
    result=json.loads(raw)
    if not isinstance(result,dict) or result.get('transfer_id')!=value['transfer_id']:raise ValueError('Inner transfer identity mismatch')
    return result

def run():
    os.chdir(p.ROOT)
    if os.environ.get('GITHUB_REPOSITORY')!='binouv/model':raise ValueError('Wrong repository')
    s=json.loads((p.ROOT/p.PREFIX/'TRANSFER_SPEC.json').read_text())
    if s.get('run_id')!='l100da_20261012' or s.get('candidate_training_steps')!=256:raise ValueError('Wrong run')
    token=os.environ.get('GH_TOKEN')
    if not token:raise ValueError('Missing workflow token')
    os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode())
    p.cp.git('config','user.name','github-actions[bot]');p.cp.git('config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
    p.cp.git('pull','--ff-only','origin','main')
    expected={'source.zip':s['source']};outputs={'source.zip':p.ROOT/p.PREFIX/'SOURCE_ARCHIVE.zip'};initial=[]
    for name,a in s['artifacts'].items():
        if name not in ('bf16','native') or a['destination']!='models/l100da_20261012_step0256/'+name:raise ValueError('Unexpected destination')
        folder=p.ROOT/a['destination'];folder.mkdir(parents=True,exist_ok=True)
        manifest=folder/'MANIFEST.json'
        if manifest.exists() and json.loads(manifest.read_text())!=a['manifest']:raise ValueError('Conflicting manifest')
        manifest.write_text(json.dumps(a['manifest'],indent=2)+'\n');p.cp.read_manifest(folder)
        (folder/'.gitattributes').write_text('*.part* -text -filter -diff -merge\n')
        initial.extend([str(manifest.relative_to(p.ROOT)),str((folder/'.gitattributes').relative_to(p.ROOT))])
        for part in a['manifest']['parts']:
            label=name+'/'+part['name'];expected[label]=part;outputs[label]=folder/part['name']
    def valid(name):
        f=outputs[name];e=expected[name]
        if f.is_symlink():raise ValueError('Unsafe existing part')
        if not f.exists():return False
        if f.stat().st_size!=e['size_bytes'] or p.cp.digest(f)!=e['sha256']:raise ValueError('Conflicting existing bytes')
        return True
    with tempfile.TemporaryDirectory(prefix='flygraph-resumable-') as temporary:
        tmp=Path(temporary);key=tmp/'private.pem';p.handoff.MODEL_SHA=s['native_model_sha256'];public=p.handoff.make_key(key)
        public.update(request_commit=os.environ.get('GITHUB_SHA'),workflow_run_id=os.environ.get('GITHUB_RUN_ID'))
        (p.ROOT/p.PREFIX/'PUBLIC_KEY.json').write_text(json.dumps(public,indent=2)+'\n')
        p.commit(initial+[p.PREFIX+'/PUBLIC_KEY.json'],'L100DA initialize resumable binary publication; no completion claim')
        names=list(expected);deadline=time.monotonic()+1200
        for batch,start in enumerate(range(0,len(names),5)):
            bn=names[start:start+5];seen=set()
            while not all(valid(n) for n in bn):
                if time.monotonic()>deadline:raise TimeoutError('Finite handoff deadline')
                p.cp.git('fetch','origin','main')
                remote=p.PREFIX+f'/BATCH_{public["transfer_id"]}_{batch:02d}.json'
                try:raw=p.cp.git('show','FETCH_HEAD:'+remote)
                except RuntimeError:time.sleep(5);continue
                fingerprint=hashlib.sha256(raw.encode()).hexdigest()
                if fingerprint in seen:time.sleep(5);continue
                seen.add(fingerprint)
                if len(raw)>50000:raise ValueError('Envelope too large')
                value=unpack(p.handoff.decrypt(key,public,json.loads(raw)))
                if value.get('batch')!=batch or value.get('checkpoint_sha256')!=s['native_model_sha256']:raise ValueError('Wrong batch identity')
                urls=value.get('urls')
                if not isinstance(urls,list) or len(urls)!=len(bn):raise ValueError('Wrong URL count')
                changed=[];failed=False
                for name,url in zip(bn,urls):
                    if valid(name):continue
                    try:p.handoff.download(url,outputs[name],expected[name])
                    except RuntimeError:
                        print('Fresh handoff required for '+name,flush=True);failed=True;break
                    changed.append(str(outputs[name].relative_to(p.ROOT)));print('Verified '+name,flush=True)
                del value,urls,url
                if changed:p.commit(changed,'L100DA preserve verified artifact batch parts; publication in_progress')
                if failed:time.sleep(5)
            print('Batch '+str(batch)+' preserved in Git.',flush=True)
        for a in s['artifacts'].values():p.cp.verify(p.ROOT/a['destination'])
        p.cp.git('pull','--ff-only','origin','main')
        files=p.extract_checked(outputs['source.zip'],p.ROOT/p.RUN,50_000_000,{'base','src','data','configs','tests','metrics','reports','checkpoints','requirements.txt','README.md','MODEL_CARD.md'})
        binary=p.commit(files,'L100DA publish all original source/data/672 predictions after complete native and BF16 byte transfer')
        fresh=tmp/'fresh';p.cp.git('clone','--depth','1','--branch','main','https://github.com/binouv/model.git',str(fresh))
        restored={}
        for name,a in s['artifacts'].items():
            restored[name]=p.cp.restore(fresh/a['destination'],tmp/('remote-'+a['manifest']['filename']))
        native=tmp/'native';native.mkdir();p.extract_checked(Path(restored['native']['output']),native,2_000_000_000)
        test=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=fresh/p.RUN,capture_output=True,text=True,timeout=180)
        if test.returncode:raise RuntimeError('Readback engineering test failed')
        check=subprocess.run([sys.executable,'src/verify_export.py','--native',str(native),'--bf16',restored['bf16']['output']],cwd=fresh/p.RUN,capture_output=True,text=True,timeout=300)
        if check.returncode:raise RuntimeError('Readback model validation failed')
        receipt={'status':'published_verified','binary_commit':binary,'workflow_run_id':os.environ.get('GITHUB_RUN_ID'),'candidate_training_steps':256,'remote_candidate_training_steps':0,'fresh_clone_verified':True,'artifacts':restored,'verification':json.loads(check.stdout),'engineering_tests':test.stdout,'overall_qwen_superiority_goal_achieved':False}
        p.cp.git('pull','--ff-only','origin','main')
        (p.ROOT/p.PREFIX/'PUBLISHED_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(p.commit([p.PREFIX+'/PUBLISHED_VERIFIED.json'],'L100DA verify all binary weights and optimizer by fresh clone, SHA256, strict loads and tests'),flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:
        print('Resumable publication stopped: '+type(exc).__name__+'. Previously committed parts remain recoverable.',file=sys.stderr)
        raise SystemExit(1) from None
