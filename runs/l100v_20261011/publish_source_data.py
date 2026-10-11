"""Exact-source/data publication only. No torch import, model creation or training."""
from __future__ import annotations
import argparse,contextlib,hashlib,importlib.util,io,json,lzma,os,subprocess,sys,tempfile,time,types,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO='binouv/model'
SOURCE_SHA='83bc17c576a1d647245664f810930a5b3cb9d7c9eb03e51b1eedf0c26bc1998d'
MANIFEST_SHA='3dfdc2ecdc802595039b137d33ba2514736b4fe9dc44591d077b2cfce4bd887a'
def digest(b):return hashlib.sha256(b).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def prepare():
    raw=b''.join((ROOT/'source_parts'/f'source.{i:03d}.xzpart').read_bytes() for i in range(3))
    if digest(raw)!=SOURCE_SHA:raise ValueError('source SHA256')
    source=json.loads(lzma.decompress(raw))
    for name,text in source.items():
        if not (ROOT/name).resolve().is_relative_to(ROOT):raise ValueError('unsafe source name')
        if not name.startswith('.pytest_cache/'):
            p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists() and p.read_bytes()!=text.encode():raise ValueError('conflicting source '+name)
            p.write_bytes(text.encode())
    original=(ROOT/'configs/original_data_manifest.json').read_bytes()
    if digest(original)!=MANIFEST_SHA:raise ValueError('original manifest SHA256')
    mf=json.loads(original);tokenizer=ROOT/'data/tokenizer.json'
    if not tokenizer.exists():
        subprocess.run([sys.executable,str(ROOT.parent/'l100c_20261009/src/setup_tokenizer.py'),'--output',str(tokenizer)],check=True)
    if digest(tokenizer.read_bytes())!=mf['tokenizer_sha256']:raise ValueError('tokenizer SHA256')
    tok=module('publication_tokenizer',ROOT/'base/tokenizer.py')
    payload={n:t for n,t in source.items() if not n.startswith('.pytest_cache/')}
    with tempfile.TemporaryDirectory() as td:
        tr=Path(td);(tr/'data').mkdir();(tr/'empty').mkdir();(tr/'data/tokenizer.json').write_bytes(tokenizer.read_bytes())
        stub=types.ModuleType('common');stub.ROOT=tr;stub.Tokenizer=tok.Tokenizer
        stub.sha=lambda p:digest(Path(p).read_bytes())
        stub.rows=lambda p:[json.loads(z) for z in Path(p).read_text().splitlines() if z.strip()]
        def put(p,z):
            p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        def write(p,rr):
            p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in rr))
        stub.put=put;stub.write_rows=write;sys.modules['common']=stub
        data=module('publication_data',ROOT/'src/data.py');data.ROOT=tr
        argv=sys.argv;sys.argv=['data.py','--ancestor-data',str(tr/'empty')]
        try:
            with contextlib.redirect_stdout(io.StringIO()):data.main()
        finally:sys.argv=argv
        expected={'data/'+k:v for k,v in mf['files'].items()}
        expected.update({'bench/prompts.jsonl':'1144d1b3eb23b7f33d223782d76e4fcc3dad839ea880a681fdf323d47dbb0a90','bench/reference_cases.jsonl':'fcda6352f682ca22ba32c664b4b365552e55f8aa1fc0fd86124915ea6bc854b3'})
        for name,h in expected.items():
            b=(tr/name).read_bytes()
            if digest(b)!=h:raise ValueError('data mismatch '+name)
            payload[name]=b.decode()
    payload['data/tokenizer.json']=tokenizer.read_text();payload['data/MANIFEST.json']=original.decode()
    for name,text in payload.items():
        if name.endswith('.py'):compile(text,name,'exec')
    receipt={'status':'completed_source_and_exact_data_publication_preparation','source_sha256':SOURCE_SHA,'original_manifest_sha256':MANIFEST_SHA,
      'row_files_verified':expected,'all_row_hashes_match':True,'torch_imported':'torch' in sys.modules,'model_training_performed':False,
      'replay_mode':'fixed-seed empty-ancestor reproduction, empirically identical to original local dataset bytes; not a decontamination proof',
      'legacy_overlap_caveat':'Schema-converted local audit found one old arithmetic item in validation and one in OOD; IID/surface/memory-CF have none among compatible available L100M cases. EC train is unavailable.',
      'original_manifest_provenance':'Preserved from actual local initial build, not a claim this publication worker inspected the old corpus.',
      'weights_in_commit':False}
    payload['reports/SOURCE_DATA_PUBLICATION.json']=json.dumps(receipt,indent=2)+'\n'
    return payload,receipt

def api(method,path,obj=None):
    assert os.environ['GITHUB_REPOSITORY']==REPO
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,method=method,
      data=None if obj is None else json.dumps(obj).encode(),headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
def publish(payload):
    prefix='runs/l100v_20261011/'
    for attempt in range(10):
        head=api('GET','/git/ref/heads/main')['object']['sha'];base=api('GET','/git/commits/'+head)['tree']['sha']
        tree=api('POST','/git/trees',{'base_tree':base,'tree':[{'path':prefix+n,'mode':'100644','type':'blob','content':t} for n,t in payload.items()]})['sha']
        commit=api('POST','/git/commits',{'tree':tree,'parents':[head],'message':'L100V: publish readable executed sources and all exact dataset/tokenizer bytes; no remote model training'})['sha']
        try:api('PATCH','/git/refs/heads/main',{'sha':commit,'force':False});return commit
        except urllib.error.HTTPError as e:
            if e.code not in (409,422) or attempt==9:raise
            time.sleep(attempt+1)
    raise RuntimeError('publication failed')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    payload,receipt=prepare();print(json.dumps(receipt,indent=2),flush=True)
    if not a.prepare_only:print('PUBLISHED_MAIN',publish(payload),flush=True)
