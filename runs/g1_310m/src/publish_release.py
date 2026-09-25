"""User-side local-file GitHub release uploader. Dry-run unless --publish.

Uses an already authenticated GitHub CLI. Never accepts plaintext tokens.
This script was syntax/unit checked, NOT executed for a real upload here.
Creates a DRAFT prerelease, verifies asset names/sizes, then publishes it.
It refuses to overwrite an existing tag/release. Failed upload leaves a draft.
"""
from __future__ import annotations
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def verify(manifest_path:Path):
    m=json.loads(manifest_path.read_text()); assets=[]
    for row in m['archives']:
        name=row['name']
        if Path(name).name!=name:raise ValueError('archive names must be basenames')
        p=manifest_path.parent/name
        if not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            raise ValueError('missing file or checksum mismatch: '+name)
        if p.stat().st_size>=2_000_000_000:raise ValueError('asset is too large for this uploader')
        assets.append(p)
    return m,assets

def gh(args,check=True):
    return subprocess.run(['gh',*args],check=check,text=True,capture_output=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--repo',default='binouv/model')
    ap.add_argument('--tag',default='flygraph-g1-310m-bootstrap-step192')
    ap.add_argument('--target',required=True,help='Existing full commit SHA containing the source and metrics')
    ap.add_argument('--notes',type=Path,required=True)
    ap.add_argument('--publish',action='store_true')
    a=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',a.repo):raise ValueError('invalid repo')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]+',a.tag):raise ValueError('invalid tag')
    if not re.fullmatch(r'[0-9a-fA-F]{40}',a.target):raise ValueError('target must be a full commit SHA')
    if not a.notes.is_file():raise ValueError('notes file missing')
    m,assets=verify(a.manifest)
    plan={'repo':a.repo,'tag':a.tag,'target':a.target,'assets':[p.name for p in assets],
          'total_bytes':sum(p.stat().st_size for p in assets),'prerelease':True,'publish_requested':a.publish,
          'status':'locally_verified_dry_run' if not a.publish else 'upload_requested'}
    print(json.dumps(plan,indent=2),flush=True)
    if not a.publish:return
    if shutil.which('gh') is None:raise RuntimeError('GitHub CLI is required; use its normal authentication flow')
    gh(['auth','status','--hostname','github.com'])
    gh(['api',f'repos/{a.repo}/commits/{a.target}'])
    releases=json.loads(gh(['release','list','--repo',a.repo,'--limit','1000','--json','tagName']).stdout)
    if any(x['tagName']==a.tag for x in releases):raise RuntimeError('release already exists; refusing overwrite')
    # A pre-existing tag can target another commit; refuse rather than move it.
    tags=json.loads(gh(['api',f'repos/{a.repo}/git/matching-refs/tags/{a.tag}']).stdout)
    if any(x.get('ref')=='refs/tags/'+a.tag for x in tags):raise RuntimeError('tag already exists; refusing overwrite')
    gh(['release','create',a.tag,'--repo',a.repo,'--target',a.target,'--title',
        'FlyGraph G1-310M — research bootstrap, not a pretrained assistant','--notes-file',str(a.notes),
        '--draft','--prerelease',*[str(p) for p in assets],str(a.manifest)])
    remote=json.loads(gh(['release','view',a.tag,'--repo',a.repo,'--json','assets,url']).stdout)
    got={x['name']:x['size'] for x in remote['assets']}
    expected={p.name:p.stat().st_size for p in [*assets,a.manifest]}
    if got!=expected:raise RuntimeError('remote name/size verification failed; draft was left unpublished')
    # GitHub asset content hashes are recorded locally; this is NOT a byte-download verification.
    gh(['release','edit',a.tag,'--repo',a.repo,'--draft=false','--prerelease'])
    receipt={'status':'published_prerelease','url':remote['url'],'local_sha256_verified':True,
             'remote_asset_sizes_verified':True,'remote_bytewise_download_verified':False,**plan}
    receipt['status']='published_prerelease'
    (a.manifest.parent/'GITHUB_RELEASE_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
