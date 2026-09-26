"""Finite warm-start of actual G3 release bytes. Fresh optimizer is explicit.
No architecture/data/held changes. Completed checkpoints have actual RNG state.
"""
import collections,gzip,hashlib,json,math,os,random,subprocess,sys,tarfile,tempfile,time,urllib.request
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1];G3=ROOT.parent/'g3_architecture';sys.path.insert(0,str(G3/'src'))
import train as base
from model import Model,Config,report
from tokenizer import Tokenizer
REPO='binouv/model';BRANCH='flygraph/g3w-warmstart-20260926';SOURCE_TAG='flygraph-g3-cpu-replay-36198964333'
TAG='flygraph-g3w-completed-'+os.environ.get('GITHUB_RUN_ID','local')
BASE_SHA={'full':'2d48baddb9172049cc06c4728afffa4bd66f83646a968da88bdbfbf709470bd9','hybrid':'6c007e6b0f337a375e2517ad173c6cd888a9b971e6bef03a13ae7d94746317c6'}
STEPS=6000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):base.atomic(p,x)
def gh(*a):return subprocess.check_output(['gh',*a],text=True)
def api(method,path,obj=None):
    h={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'}
    body=None if obj is None else json.dumps(obj).encode()
    with urllib.request.urlopen(urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=body,headers=h,method=method),timeout=90) as r:return json.load(r)
def push(paths):
    hd=api('GET','/git/ref/heads/'+BRANCH)['object']['sha'];tree=api('GET','/git/commits/'+hd)['tree']['sha']
    entries=[dict(path='runs/g3w_warmstart/'+str(p.relative_to(ROOT)),mode='100644',type='blob',content=p.read_text()) for p in paths]
    tr=api('POST','/git/trees',dict(base_tree=tree,tree=entries))['sha'];co=api('POST','/git/commits',dict(tree=tr,parents=[hd],message='FlyGraph G3W: completed checkpoint/metrics, actual bytes verified'))['sha']
    api('PATCH','/git/refs/heads/'+BRANCH,dict(sha=co,force=False));print('FG_PUSH',co,flush=True)
def upload(paths):
    gh('release','upload',TAG,'--repo',REPO,*map(str,paths));out=[]
    with tempfile.TemporaryDirectory() as td:
        for p in paths:
            gh('release','download',TAG,'--repo',REPO,'--pattern',p.name,'--dir',td);q=Path(td)/p.name
            assert sha(q)==sha(p) and q.stat().st_size==p.stat().st_size
            out.append(dict(name=p.name,sha256=sha(p),bytes=p.stat().st_size,remote_redownload_sha256_verified=True))
    return out

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    for d in ('checkpoints','metrics','reports','parents','assets'): (ROOT/d).mkdir(exist_ok=True,parents=True)
    subprocess.run([sys.executable,str(G3/'src/setup_data.py')],check=True)
    tr=base.rows('train');vr=base.rows('validation');tok=Tokenizer.load(G3/'data/tokenizer.json')
    pools=collections.defaultdict(list)
    for z in tr:pools[(z['family'],z['lang'])].append(z)
    keys=sorted(pools);assets=ROOT/'assets';all_receipts=[];results=[]
    gh('release','create',TAG,'--repo',REPO,'--target',os.environ['GITHUB_SHA'],'--title','FlyGraph G3W: actual G3 warm-start with longer matched training','--notes','Draft until both fixed6000-update warm-starts complete; fresh optimizers, not exact original G3 resume.','--draft','--prerelease')
    for variant in ('full','hybrid'):
        for ext in ('safetensors','config.json'):gh('release','download',SOURCE_TAG,'--repo',REPO,'--pattern',f'{variant}_seed7301.{ext}','--dir',str(ROOT/'parents'))
        pp=ROOT/'parents'/f'{variant}_seed7301.safetensors';assert sha(pp)==BASE_SHA[variant]
        cfg=json.loads((ROOT/'parents'/f'{variant}_seed7301.config.json').read_text());torch.manual_seed(7701);random.seed(7701)
        m=Model(Config(**cfg));m.load_state_dict(load_file(str(pp)))
        initial={s:base.evaluate(m,tok,base.rows(s)) for s in base.SPLITS};initial_val=base.val(m,vr)
        put(ROOT/'metrics'/f'{variant}_initial.json',dict(validation_nll=initial_val,tests=initial))
        opt=torch.optim.AdamW(m.parameters(),lr=.0005,betas=(.9,.95),weight_decay=.01);rng=random.Random(77311)
        logs=[];digest=hashlib.sha256();started=time.perf_counter()
        for step in range(1,STEPS+1):
            m.train();opt.zero_grad(set_to_none=True);k=keys[rng.randrange(len(keys))];rr=rng.choices(pools[k],k=32);x,y=base.batch(rr)
            ids=[z['id'] for z in rr];digest.update(('\n'.join(ids)+'\n').encode());t=time.perf_counter();loss=m(x,y);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
            assert math.isfinite(float(loss.detach())) and math.isfinite(gn)
            lr=.0005*min(1.,step/100)*(.15+.85*.5*(1+math.cos(math.pi*step/STEPS)))
            for group in opt.param_groups:group['lr']=lr
            opt.step();log=dict(step=step,loss=float(loss.detach()),lr=lr,grad_norm=gn,seconds=time.perf_counter()-t,input_tokens=sum(len(z['tokens'])-1 for z in rr),supervised_tokens=int(y.ne(-100).sum()),padded_tokens=x.numel(),batch_ids=ids);logs.append(log)
            with (ROOT/'metrics'/f'{variant}_train.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
            if step%500==0:print('TRAIN',variant,step,log['loss'],flush=True)
            if step%1000==0:
                opt.zero_grad(set_to_none=True);native=assets/f'{variant}_step{step:04d}.resume.pt'
                state=dict(schema='G3W-v1',config=cfg,model=m.state_dict(),optimizer=opt.state_dict(),step=step,parent_updates=2000,fresh_optimizer=True,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),sampler_rng=rng.getstate(),batch_sequence_sha256=digest.hexdigest(),logs=logs,data_hashes=json.loads((G3/'data/manifest.json').read_text())['hashes'])
                tmp=native.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,native)
                assert all(torch.isfinite(v).all() for v in m.state_dict().values() if v.is_floating_point())
                receipts=upload([native]);all_receipts.extend(receipts)
                progress=ROOT/'reports/PROGRESS.json';put(progress,dict(status='completed_checkpoint_only',variant=variant,step=step,total_updates_including_parent=2000+step,expected_new_updates=6000,actual_asset=receipts[0],fresh_optimizer=True));push([progress])
        sf=assets/f'{variant}_step6000.safetensors';save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},str(sf));cf=assets/f'{variant}.config.json';put(cf,cfg)
        m2=Model(Config(**cfg));m2.load_state_dict(load_file(str(sf)));m.eval();m2.eval();xx=torch.tensor([tok.encode('Return the final integer.\nCalculate: 2 + 2.\nAnswer:',bos=True)])
        with torch.inference_mode():assert torch.equal(m(xx)[0],m2(xx)[0])
        tests={s:base.evaluate(m,tok,base.rows(s)) for s in base.SPLITS};final_val=base.val(m,vr)
        res=dict(status='completed',variant=variant,seed=7301,new_training_seed=7701,new_updates=6000,ancestor_updates=2000,parameters=report(m),initial_validation_nll=initial_val,final_validation_nll=final_val,initial=initial,tests=tests,input_tokens=sum(z['input_tokens'] for z in logs),supervised_tokens=sum(z['supervised_tokens'] for z in logs),padded_tokens=sum(z['padded_tokens'] for z in logs),training_seconds=sum(z['seconds'] for z in logs),elapsed_seconds=time.perf_counter()-started,batch_sequence_sha256=digest.hexdigest(),safetensors_sha256=sha(sf),native_sha256=sha(native),reload_logits_equal=True,not_exact_original_G3_resume=True)
        put(ROOT/'metrics'/f'{variant}_FINAL.json',res);gz=assets/f'{variant}_RAW_RESULTS.json.gz';gz.write_bytes(gzip.compress(json.dumps(res).encode(),mtime=0));all_receipts.extend(upload([sf,cf,gz]));results.append(res)
        summary=ROOT/'reports'/f'{variant}_COMPLETED.json';put(summary,{k:v for k,v in res.items() if k not in ('initial','tests')});push([summary]);print('FG_ARM',variant,{s:v['correct'] for s,v in tests.items()},flush=True)
    assert results[0]['batch_sequence_sha256']==results[1]['batch_sequence_sha256']
    a,b=results
    rate=lambda r,s:r[s]['correct']/r[s]['n']
    within={r['variant']:{s:100*(rate(r['tests'],s)-rate(r['initial'],s)) for s in base.SPLITS} for r in results}
    h1=within['hybrid']['test_iid']>=10 and (within['hybrid']['test_surface']+within['hybrid']['test_extrapolation'])>0 and results[1]['tests']['test_counterfactual']['both_correct']>=results[1]['initial']['test_counterfactual']['both_correct']
    diff={s:100*(rate(b['tests'],s)-rate(a['tests'],s)) for s in base.SPLITS}
    final=dict(status='completed',hypothesis='Longer matched training from actual saved weights improves generative transfer; hybrid compared with equally continued full-attention control.',models=[{k:v for k,v in r.items() if k not in ('initial','tests')} for r in results],counts={r['variant']:{stage:{s:{k:v for k,v in v.items() if k!='rows'} for s,v in r[stage].items()} for stage in ('initial','tests')} for r in results},within_gain_pp=within,hybrid_minus_full_pp=diff,optimization_gate_met=h1,hybrid_advantage_gate_met=(diff['test_iid']>=5 and diff['test_surface']+diff['test_extrapolation']>0),seed_count=1,same_seed_as_G3_parent=True,not_general_Qwen_parity=True,not_300M=True,assets=all_receipts)
    fp=ROOT/'metrics/FINAL_RESULT.json';put(fp,final)
    package=assets/'SOURCE_DATA_RESULTS.tar.gz'
    with tarfile.open(package,'w:gz') as tar:
        for folder in [G3/'src',G3/'configs',G3/'data',ROOT/'src',ROOT/'configs',ROOT/'reports',ROOT/'metrics']:
            for p in sorted(folder.rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname=str(p.relative_to(ROOT.parent)))
    all_receipts.extend(upload([package]));mf=assets/'MANIFEST.json';put(mf,dict(status='completed',workflow_run=os.environ['GITHUB_RUN_ID'],source_commit=os.environ['GITHUB_SHA'],tag=TAG,assets=all_receipts,all_remote_bytes_verified=True,native_optimizer_rng=True,new_optimizer_at_warm_start=True));upload([mf])
    gh('release','edit',TAG,'--repo',REPO,'--draft=false','--prerelease','--notes','Two completed6000-update warm-starts from real G3 seed7301 weights. Fresh AdamW explicitly used. Native optimizer/RNG saved every1000updates and safetensors final. All uploaded bytes re-downloaded and SHA256 verified. Synthetic small models, not general intelligence or300M. See raw results.')
    np=ROOT/'reports/NEXT_STATE.json';put(np,dict(status='completed',tag=TAG,do_not_repeat=True,models=2,new_steps_each=6000,prior_steps=2000,seed_count=1,need_next='Replicate promising direction on additional independently trained parent seeds, then diverse reasoning/code with uncontaminated held tests. No300Mscale merely for parameter count.'))
    push([fp,np]);print('FG_FINAL',json.dumps({'within':within,'hybrid_vs_full':diff,'optimization_gate':h1,'tag':TAG}),flush=True)
if __name__=='__main__':main()
