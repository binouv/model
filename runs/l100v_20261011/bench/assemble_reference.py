"""Archive existing completed Qwen response files. Performs no inference or training."""
from __future__ import annotations
import base64,hashlib,json,lzma,os,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO='binouv/model';BRANCH='flygraph/l100v-reference-20261011'
FIXTURE='1144d1b3eb23b7f33d223782d76e4fcc3dad839ea880a681fdf323d47dbb0a90'
MODEL_SHA='9a7bed4041b7975e0f71fa34670d1e9025213bc92905ac0db75d36c4fa3fa623'
def sha(b):return hashlib.sha256(b).hexdigest()
def api(method,path,obj=None):
    if os.environ['GITHUB_REPOSITORY']!=REPO:raise ValueError('Wrong repository')
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,method=method,
      data=None if obj is None else json.dumps(obj).encode(),headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as f:return json.load(f)
def prepare():
    pb=(ROOT/'PROMPTS.jsonl').read_bytes()
    if sha(pb)!=FIXTURE:raise ValueError('Prompt hash mismatch')
    prompts=[json.loads(x) for x in pb.splitlines()];allrows=[];sources={}
    for s in range(4):
        p=ROOT/'outputs'/f'shard_{s}'/'FINAL.json';raw=p.read_bytes();result=json.loads(raw)
        sources[str(p.relative_to(ROOT))]=sha(raw)
        if result['status']!='completed_attempts' or result['metadata']['model_sha256']!=MODEL_SHA:raise ValueError('Incomplete or wrong shard')
        if result['metadata']['max_new_tokens']!=32768 or not result['metadata']['thinking']:raise ValueError('Wrong thinking configuration')
        if len(result['rows'])!=len(prompts[s::4]):raise ValueError('Wrong shard size')
        for r in result['rows']:
            i=r['index']
            if {k:r[k] for k in ('id','family','lang','prompt')}!=prompts[i]:raise ValueError('Wrong prompt association')
            if i%4!=s or r['seed']!=110362+i:raise ValueError('Wrong index/seed')
            allrows.append(r)
    if sorted(r['index'] for r in allrows)!=list(range(22)):raise ValueError('Missing or duplicated requests')
    allrows.sort(key=lambda r:r['index'])
    obj={'status':'all22_attempts_preserved','source_commit':os.environ['GITHUB_SHA'],'inference_workflow_run':38108269209,
      'fixture_sha256':FIXTURE,'model_sha256':MODEL_SHA,'source_file_hashes':sources,'rows':allrows,'no_new_requests':True,'no_training':True}
    raw=json.dumps(obj,ensure_ascii=False,separators=(',',':')).encode();packed=lzma.compress(raw,preset=6);encoded=base64.b64encode(packed).decode()
    files={};parts=[]
    for i,start in enumerate(range(0,len(encoded),6000)):
        name=f'outputs/assembled/part{i:03d}.b64';text=encoded[start:start+6000];files[name]=text
        parts.append({'name':name,'bytes':len(text),'sha256':sha(text.encode())})
    manifest={'status':'completed_lossless_archive','archive_sha256':sha(packed),'archive_bytes':len(packed),'uncompressed_sha256':sha(raw),'uncompressed_bytes':len(raw),'parts':parts,'requests':22,'source_commit':os.environ['GITHUB_SHA']}
    files['outputs/assembled/MANIFEST.json']=json.dumps(manifest,indent=2)+'\n'
    summary={'status':'completed_all_attempts','n':22,'uncensored':sum(bool(r.get('uncensored')) for r in allrows),'censored_ids':[r['id'] for r in allrows if not r.get('uncensored')],
      'rows':[{k:r.get(k) for k in ('id','index','family','lang','status','final_text','thinking_closed','nonempty_thinking','stop_type','tokens_predicted','seconds','uncensored','error')} for r in allrows],
      'not_scored_here':True,'not_a_broad_benchmark':True,'archive_sha256':sha(packed),'source_commit':os.environ['GITHUB_SHA']}
    files['outputs/assembled/SUMMARY.json']=json.dumps(summary,ensure_ascii=False,indent=2)+'\n'
    return files,manifest

def main():
    if os.environ['GITHUB_REF_NAME']!=BRANCH:raise ValueError('Wrong branch')
    files,manifest=prepare();prefix='runs/l100v_20261011/bench/'
    for attempt in range(10):
        head=api('GET','/git/ref/heads/'+BRANCH)['object']['sha'];base=api('GET','/git/commits/'+head)['tree']['sha']
        tree=api('POST','/git/trees',{'base_tree':base,'tree':[{'path':prefix+n,'mode':'100644','type':'blob','content':t} for n,t in files.items()]})['sha']
        commit=api('POST','/git/commits',{'tree':tree,'parents':[head],'message':'L100V: assemble lossless22-case Qwen response archive; no new inference or training'})['sha']
        try:api('PATCH','/git/refs/heads/'+BRANCH,{'sha':commit,'force':False});print('ASSEMBLED',commit,json.dumps(manifest),flush=True);return
        except urllib.error.HTTPError as e:
            if e.code not in (409,422) or attempt==9:raise
            time.sleep(attempt+1)
if __name__=='__main__':main()
