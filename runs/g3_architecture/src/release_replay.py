"""Publish completed independent CPU replay; not a mirror of local-container weights."""
from __future__ import annotations
import hashlib,json,os,sys,urllib.request,zipfile
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from model import Model,Config,report
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def request(url,payload=None,method='GET',binary=False):
    headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'}
    data=None if payload is None else payload if binary else json.dumps(payload).encode()
    if data is not None:headers['Content-Type']='application/octet-stream' if binary else 'application/json'
    req=urllib.request.Request(url,data=data,method=method,headers=headers)
    with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
def main():
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    assets=ROOT/'release';assets.mkdir(exist_ok=True);models=[]
    for kind in ['full','shared','hybrid']:
        cp=ROOT/f'checkpoints/{kind}_s7301/step2000.pt';s=torch.load(cp,weights_only=False)
        met=ROOT/f'metrics/{kind}_s7301';tr=json.loads((met/'TRAIN_COMPLETE.json').read_text());ev=json.loads((met/'EVAL_COMPLETE.json').read_text())
        assert s['step']==2000 and tr['status']==ev['status']=='completed'
        model=Model(Config(**s['config']));model.load_state_dict(s['model']);model.eval()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        f=assets/f'{kind}_seed7301.safetensors';save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(f))
        reloaded=Model(Config(**s['config']));reloaded.load_state_dict(load_file(str(f)));reloaded.eval()
        torch.manual_seed(7431);x=torch.randint(3,8192,(1,21));a=model(x)[0];b=reloaded(x)[0];assert torch.equal(a,b)
        (assets/f'{kind}_seed7301.config.json').write_text(json.dumps(s['config'],indent=2))
        models.append({'variant':kind,'seed':7301,'parameters':report(model),'training':tr,'weights':f.name,'weight_sha256':sha(f),'weight_bytes':f.stat().st_size,'exact_reload_logits_equal':True,'scope':'independent CPU replay, not claimed identical to local-container weights'})
    from shutil import copy2
    copy2(ROOT/'data/tokenizer.json',assets/'tokenizer.json')
    source=assets/'SourceDataResults.zip'
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ['src','tests','configs','metrics','data']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':z.write(p,str(p.relative_to(ROOT)))
    manifest={'run_id':os.environ['GITHUB_RUN_ID'],'source_commit':os.environ['GITHUB_SHA'],'status':'completed_CPU_replay','models':models,'not_broad_pretrained_models':True,'not_all_historical_assets':True,'Qwen_parity_claim':False,'assets':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(assets.iterdir())]}
    mf=assets/'MANIFEST.json';mf.write_text(json.dumps(manifest,indent=2))
    repo=os.environ['GITHUB_REPOSITORY'];tag='flygraph-g3-cpu-replay-'+os.environ['GITHUB_RUN_ID'];api='https://api.github.com/repos/'+repo
    body='Three actually trained compact generative architecture probes (seed7301), each2000updates. Independent CPU replay of the G3 protocol, not necessarily byte-identical to local checkpoints. Not a300Mmodel, not a broad pretrained assistant, not Qwen parity. Includes real FP32safetensors, exact tokenizer, source/data/results. See MANIFEST for hashes and scope.'
    rel=request(api+'/releases',{'tag_name':tag,'target_commitish':os.environ['GITHUB_SHA'],'name':'FlyGraph G3 — three trained architecture probes','body':body,'draft':True,'prerelease':True},'POST')
    remote=[]
    for p in sorted(assets.iterdir()):
        u=rel['upload_url'].split('{',1)[0]+'?name='+p.name;r=request(u,p.read_bytes(),'POST',True)
        assert r['size']==p.stat().st_size
        if r.get('digest'):assert r['digest']=='sha256:'+sha(p)
        remote.append({'name':p.name,'asset_id':r['id'],'bytes':r['size'],'github_digest':r.get('digest'),'local_sha256':sha(p)})
    rel=request(api+'/releases/'+str(rel['id']),{'draft':False,'prerelease':True},'PATCH')
    receipt={'status':'published_prerelease','url':rel['html_url'],'tag':tag,'release_id':rel['id'],'assets':remote,'local_all_hashes_verified':True,'all_remote_digests_verified':all(x['github_digest'] for x in remote),'bytewise_remote_download_repeated':False,'not_the_local_container_model_bytes':True}
    print('FG_RELEASE_RECEIPT',json.dumps(receipt),flush=True)
    path='runs/g3_architecture/reports/release_'+os.environ['GITHUB_RUN_ID']+'.json'
    import base64
    request(api+'/contents/'+path,{'message':'FlyGraph G3: record actual model-release asset hashes','branch':os.environ['GITHUB_REF_NAME'],'content':base64.b64encode(json.dumps(receipt,indent=2).encode()).decode()},'PUT')
if __name__=='__main__':main()
