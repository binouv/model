"""Recover the missing8requests, NEVER reinterpret a timed-out32/40run as complete.
The original32EOSresponses are immutable and verified by SHA256. Same model,
backend, sampling, per-example seeds, prefix and token cap; no answer-based selection.
"""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,tempfile,time,urllib.request
from pathlib import Path
from qwen_thinking import URL,SHA,LLAMA,FIXTURE,PREFIX,run
PREVIOUS='https://raw.githubusercontent.com/binouv/model/4721dd181dd799d2e35852ecf543a7f589e1b6c4/runs/g3_architecture/bench/results_36198379877.json'
PREVIOUS_SHA='4e959ae79dbdb4b1dbcba706000b2903b532f6074869c1312a56b6ff73fe8b88'
def main():
 p=argparse.ArgumentParser();p.add_argument('--prompts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if hashlib.sha256(a.prompts.read_bytes()).hexdigest()!=FIXTURE:raise ValueError('fixture changed')
 cases=[json.loads(s) for s in a.prompts.read_text().splitlines()]
 with urllib.request.urlopen(PREVIOUS,timeout=60) as r:raw=r.read()
 if hashlib.sha256(raw).hexdigest()!=PREVIOUS_SHA:raise ValueError('old result changed')
 old=json.loads(raw);outputs=old['rows'];assert len(outputs)==32 and len(cases)==40
 assert old['metadata']['model_sha256']==SHA and old['metadata']['thinking'] is True
 for i,z in enumerate(outputs):
  assert z['id']==cases[i]['id'] and z['prompt']==cases[i]['prompt'] and z['stop_type']=='eos' and z['thinking_closed']
 sampling=old['metadata']['sampling'];assert sampling=={'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repeat_penalty':1.0}
 with tempfile.TemporaryDirectory(prefix='fg-q4-resume-') as td:
  root=Path(td);source=root/'llama';weights=root/'weights.gguf'
  run(['git','clone','--depth','1','--branch','v0.5.0','https://github.com/ggml-org/llama.cpp.git',str(source)])
  if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=LLAMA:raise ValueError('backend changed')
  run(['cmake','-S',str(source),'-B',str(source/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
  run(['cmake','--build',str(source/'build'),'-j','4','--target','llama-server'])
  h=hashlib.sha256()
  with urllib.request.urlopen(URL,timeout=180) as f,weights.open('wb') as out:
   while b:=f.read(8<<20):out.write(b);h.update(b)
  if h.hexdigest()!=SHA:raise ValueError('weights changed')
  f=(root/'server.log').open('w');proc=subprocess.Popen([str(source/'build/bin/llama-server'),'-m',str(weights),'-c','40960','-t','4','-ngl','0','--host','127.0.0.1','--port','18080','--no-webui'],stdout=f,stderr=subprocess.STDOUT)
  meta={**old['metadata'],'resume_workflow_run':os.environ['GITHUB_RUN_ID'],'resume_source_commit':os.environ['GITHUB_SHA'],'original_completed_requests':32,'new_requests':8,'prior_result_sha256':PREVIOUS_SHA,'single_fixture_completed_in_two_chunks':True}
  try:
   for _ in range(240):
    if proc.poll() is not None:raise RuntimeError('server exited')
    try:
     with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as r:
      if r.status==200:break
    except Exception:time.sleep(1)
   else:raise RuntimeError('startup timeout')
   for i in range(32,40):
    z=cases[i];prompt='<|im_start|>user\n'+PREFIX+z['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
    body={'prompt':prompt,'n_predict':32768,'seed':7401+i,'cache_prompt':False,**sampling}
    req=urllib.request.Request('http://127.0.0.1:18080/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=1800) as r:response=json.load(r)
    text=response['content'];closed='</think>' in text;final=text.split('</think>',1)[1].strip() if closed else None
    row={**z,'generated':text,'final_text':final,'thinking_closed':closed,'nonempty_thinking':bool(text.split('</think>',1)[0].strip()),'stop_type':response.get('stop_type'),'tokens_predicted':response.get('tokens_predicted'),'timings':response.get('timings'),'seconds':time.perf_counter()-t};outputs.append(row)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.with_suffix('.partial.json').write_text(json.dumps({'status':'incomplete_not_a_benchmark_result','metadata':meta,'rows':outputs},ensure_ascii=False,indent=2))
    print('FG_RESUME_PROGRESS',i+1,row['tokens_predicted'],row['stop_type'],repr(final),flush=True)
   censored=[z['id'] for z in outputs if z['stop_type']!='eos' or not z['thinking_closed']]
   result={'status':'completed','metadata':meta,'n':40,'rows':outputs,'censored_ids':censored,'valid_for_uncensored_comparison':not censored}
   raw=json.dumps(result,ensure_ascii=False,indent=2).encode();a.out.write_bytes(raw)
   compact={**result,'raw_file_sha256':hashlib.sha256(raw).hexdigest(),'rows':[{k:v for k,v in z.items() if k!='generated'} for z in outputs]}
   a.out.with_suffix('.compact.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2));print('FG_RESUME_COMPLETE',len(censored),flush=True)
  finally:
   proc.terminate()
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   f.close()
if __name__=='__main__':main()
