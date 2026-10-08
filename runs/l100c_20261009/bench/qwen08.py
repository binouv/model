"""Finite public-CPU reference inference only. No FlyGraph training or gold input.
Native open thinking, full BF16 text weights, no solver; preserve all stop reasons.
"""
from __future__ import annotations
import hashlib,json,os,subprocess,tempfile,time,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parent
REPO='binouv/model';BRANCH='flygraph/l100c-qwen08-20261009'
WEIGHTS='https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-BF16.gguf'
WEIGHT_SHA='cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5'
LLAMA='7fe450e19305b828c199d602c23a8337aaa1f03b'
FIXTURE='a22d13f80917af652ab0d1579fef255630166114f45b221596fbff6a8f882eb9'

def api(method,path,obj=None):
 assert os.environ.get('GITHUB_REPOSITORY')==REPO
 req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=None if obj is None else json.dumps(obj).encode(),headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'},method=method)
 with urllib.request.urlopen(req,timeout=90) as f:return json.load(f)

def publish(path):
 head=api('GET','/git/ref/heads/'+BRANCH)['object']['sha'];tree=api('GET','/git/commits/'+head)['tree']['sha']
 tr=api('POST','/git/trees',{'base_tree':tree,'tree':[{'path':'runs/l100c_20261009/bench/'+path.name,'mode':'100644','type':'blob','content':path.read_text()}]})['sha']
 co=api('POST','/git/commits',{'tree':tr,'parents':[head],'message':'L100C: preserve actual Qwen0.8 BF16 thinking responses; no FlyGraph training'})['sha'];api('PATCH','/git/refs/heads/'+BRANCH,{'sha':co,'force':False});print('RESPONSE_PUSH',co,flush=True)

def main():
 p=R/'prompts20.jsonl';assert hashlib.sha256(p.read_bytes()).hexdigest()==FIXTURE
 rows=[json.loads(s) for s in p.read_text().splitlines()]
 assert len(rows)==20 and all(set(z)=={'id','family','lang','prompt'} for z in rows)
 outputs=[];meta={'model':'Qwen/Qwen3.5-0.8B','format':'unsloth BF16 GGUF text weights','weights_sha256':WEIGHT_SHA,'llama_commit':LLAMA,'thinking':True,'native_prefill':'<think>\n','n_predict':32768,'context':40960,'fixture_sha256':FIXTURE,'workflow_run':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA'),'tools':False,'trained_here':False,'not_an_external_broad_benchmark':True}
 with tempfile.TemporaryDirectory(prefix='l100c-qwen08-') as td:
  td=Path(td);src=td/'llama';weights=td/'model.gguf'
  subprocess.run(['git','clone','--depth','1','--branch','v0.5.0','https://github.com/ggml-org/llama.cpp.git',str(src)],check=True,timeout=300)
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==LLAMA
  subprocess.run(['cmake','-S',str(src),'-B',str(src/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'],check=True,timeout=300)
  subprocess.run(['cmake','--build',str(src/'build'),'-j','4','--target','llama-server'],check=True,timeout=900)
  h=hashlib.sha256()
  with urllib.request.urlopen(WEIGHTS,timeout=180) as f,weights.open('wb') as o:
   while chunk:=f.read(8<<20):o.write(chunk);h.update(chunk)
  assert h.hexdigest()==WEIGHT_SHA
  meta['weights_bytes']=weights.stat().st_size
  sampling={'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repeat_penalty':1.0};meta['sampling']=sampling
  with (td/'server.log').open('w') as log:
   proc=subprocess.Popen([str(src/'build/bin/llama-server'),'-m',str(weights),'-c','40960','-t','4','-ngl','0','--host','127.0.0.1','--port','18080','--no-webui'],stdout=log,stderr=subprocess.STDOUT)
   try:
    for _ in range(240):
     if proc.poll() is not None:raise RuntimeError((td/'server.log').read_text()[-2000:])
     try:
      with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as rr:
       if rr.status==200:break
     except Exception:time.sleep(1)
    else:raise RuntimeError('server not ready')
    for i,z in enumerate(rows):
     prompt='<|im_start|>user\n'+z['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
     payload={'prompt':prompt,'n_predict':32768,'seed':9501+i,'cache_prompt':False,**sampling}
     req=urllib.request.Request('http://127.0.0.1:18080/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});start=time.perf_counter()
     with urllib.request.urlopen(req,timeout=1800) as f:resp=json.load(f)
     text=resp['content'];closed='</think>' in text;final=text.split('</think>',1)[1].strip() if closed else None
     outputs.append({**z,'generated':text,'final_text':final,'thinking_closed':closed,'nonempty_thinking':bool(text.split('</think>',1)[0].strip()),'stop_type':resp.get('stop_type'),'tokens_predicted':resp.get('tokens_predicted'),'timings':resp.get('timings'),'seconds':time.perf_counter()-start})
     out=R/'QWEN08_PARTIAL.json';out.write_text(json.dumps({'status':'partial_not_aggregate','metadata':meta,'n':len(outputs),'rows':outputs},ensure_ascii=False,indent=2))
     print('QWEN08',i+1,outputs[-1]['tokens_predicted'],outputs[-1]['stop_type'],repr(final),flush=True)
     if (i+1)%4==0:publish(out)
    result={'status':'completed','metadata':meta,'n':len(outputs),'all_eos':all(x['stop_type']=='eos' for x in outputs),'all_thinking_closed':all(x['thinking_closed'] for x in outputs),'censored_ids':[x['id'] for x in outputs if x['stop_type']!='eos' or not x['thinking_closed']],'rows':outputs}
    out=R/'QWEN08_COMPLETE.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2));publish(out)
   except Exception as e:
    out=R/'QWEN08_ERROR.json';out.write_text(json.dumps({'status':'incomplete','type':type(e).__name__,'message':str(e),'completed_requests':len(outputs),'no_partial_accuracy_claim':True}));publish(out);raise
   finally:
    proc.terminate()
    try:proc.wait(15)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
if __name__=='__main__':main()
