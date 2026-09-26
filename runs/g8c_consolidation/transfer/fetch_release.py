"""Finite byte transfer, not training. Verify committed receipt and every asset.
Run in the user's authorized public CI. No secrets leave GitHub or enter files.
"""
import base64,hashlib,json,os,subprocess
from pathlib import Path
REPO='binouv/model'
TAG='flygraph-g8c-consolidation-36238890991'

def gh(*args):return subprocess.check_output(['gh',*args],text=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ['GITHUB_REPOSITORY']==REPO
    receipt_blob=os.environ['EXPECTED_RECEIPT_BLOB']
    b=json.loads(gh('api',f'repos/{REPO}/git/blobs/{receipt_blob}'))
    raw=base64.b64decode(b['content'])
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==receipt_blob
    rec=json.loads(raw);assert rec['status']=='six_continuations_completed' and rec['tag']==TAG
    meta=json.loads(gh('api',f'repos/{REPO}/releases/tags/{TAG}'));assert not meta['draft']
    out=Path('downloaded');out.mkdir()
    gh('release','download',TAG,'--repo',REPO,'--pattern','MANIFEST.json','--dir',str(out))
    assert (out/'MANIFEST.json').read_bytes()==raw
    wanted=[x for x in rec['assets'] if x['name'].endswith(('.safetensors','.config.json','.RAW.json.gz','stage2_1000.resume.pt')) or x['name']=='SOURCE_DATA_RESULTS.tar.gz']
    assert len(wanted)==25
    for x in wanted:
        assert Path(x['name']).name==x['name']
        gh('release','download',TAG,'--repo',REPO,'--pattern',x['name'],'--dir',str(out))
        p=out/x['name'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
    result={'status':'completed_byte_transfer_not_training','tag':TAG,'receipt_blob':receipt_blob,'verified_assets':len(wanted),'model_count':6,'native_optimizer_rng_files':6,'all_sha256_verified':True,'no_model_retraining':True,'files':[{**x,'locally_downloaded_sha256':sha(out/x['name'])} for x in wanted]}
    (out/'TRANSFER_RECEIPT.json').write_text(json.dumps(result,indent=2))
    print('TRANSFER_COMPLETE',json.dumps({'tag':TAG,'verified_assets':len(wanted),'total_bytes':sum(x['bytes'] for x in wanted)}),flush=True)
if __name__=='__main__':main()
