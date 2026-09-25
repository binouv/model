"""Finite Qwen3.5-4B thinking evaluation; no gold, tools or output execution."""
from __future__ import annotations
import argparse,gzip,hashlib,json,os,subprocess,tempfile,time,urllib.request
from pathlib import Path
URL='https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf'
SHA='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
LLAMA='7fe450e19305b828c199d602c23a8337aaa1f03b'
FIXTURE='ddef92124f5fe5f9058711480ceba522472606f06d4e2568aadf232f7317828d'
PREFIX='Return the final integer.\n'
def run(args,cwd=None):subprocess.run(args,cwd=cwd,check=True,timeout=900)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prompts',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
 if hashlib.sha256(a.prompts.read_bytes()).hexdigest()!=FIXTURE:raise ValueError('fixture changed')
 rows=[json.loads(s) for s in a.prompts.read_text().splitlines()]
 if len(rows)!=40 or any(set(z)!={'id','family','lang','prompt'} for z in rows):raise ValueError('fixture must contain40prompts and no labels')
 with tempfile.TemporaryDirectory(prefix='fg-g3-think-') as td:
  root=Path(td);source=root/'llama';weights=root/'weights.gguf'
  run(['git','clone','--depth','1','--branch','v0.5.0','https://github.com/ggml-org/llama.cpp.git',str(source)])
  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
  if revision!=LLAMA:raise ValueError('inference commit changed')
  run(['cmake','-S',str(source),'-B',str(source/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
  run(['cmake','--build',str(source/'build'),'-j','4','--target','llama-server'])
  h=hashlib.sha256();count=0
  with urllib.request.urlopen(URL,timeout=180) as f,weights.open('wb') as out:
   while b:=f.read(8<<20):out.write(b);h.update(b);count+=len(b)
  if h.hexdigest()!=SHA:raise ValueError('model SHA256 mismatch')
  sampling={'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repeat_penalty':1.0}
  meta={'model':'Qwen/Qwen3.5-4B','quantization':'Q4_K_M','model_sha256':SHA,'model_bytes':count,'llama_commit':revision,'thinking':True,'thinking_prefill':'<think>\n','sampling':sampling,'max_new_tokens':32768,'context':40960,'fixture_sha256':FIXTURE,'instruction_prefix':PREFIX,'tools':False,'not_BF16_capability_ceiling':True,'run_id':os.environ.get('GITHUB_RUN_ID'),'source_commit':os.environ.get('GITHUB_SHA')}
  log=root/'server.log';f=log.open('w');p=subprocess.Popen([str(source/'build/bin/llama-server'),'-m',str(weights),'-c','40960','-t','4','-ngl','0','--host','127.0.0.1','--port','18080','--no-webui'],stdout=f,stderr=subprocess.STDOUT);outputs=[]
  try:
   for i in range(240):
    if p.poll() is not None:raise RuntimeError(log.read_text()[-4000:])
    try:
     with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as r:
      if r.status==200:break
    except Exception:time.sleep(1)
   else:raise RuntimeError('server not ready')
   for i,z in enumerate(rows):
    prompt='<|im_start|>user\n'+PREFIX+z['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n';assert '</think>' not in prompt
    payload={'prompt':prompt,'n_predict':32768,'seed':7401+i,'cache_prompt':False,**sampling}
    req=urllib.request.Request('http://127.0.0.1:18080/completion',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=900) as r:response=json.load(r)
    text=response['content'];closed='</think>' in text;final=text.split('</think>',1)[1].strip() if closed else None
    row={**z,'generated':text,'final_text':final,'thinking_closed':closed,'nonempty_thinking':bool(text.split('</think>',1)[0].strip()),'stop_type':response.get('stop_type'),'tokens_predicted':response.get('tokens_predicted'),'timings':response.get('timings'),'seconds':time.perf_counter()-t};outputs.append(row)
    print('FG_THINK_PROGRESS',i+1,row['tokens_predicted'],row['stop_type'],repr(final),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.with_suffix('.partial.json').write_text(json.dumps({'status':'running','metadata':meta,'rows':outputs},ensure_ascii=False))
   censored=[z['id'] for z in outputs if z['stop_type']=='limit' or not z['thinking_closed']]
   result={'status':'completed','metadata':meta,'n':len(outputs),'censored_ids':censored,'all_thinking_closed':all(z['thinking_closed'] for z in outputs),'all_ended_eos':all(z['stop_type']=='eos' for z in outputs),'rows':outputs,'valid_for_uncensored_comparison':not censored}
   a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));a.output.with_suffix('.json.gz').write_bytes(gzip.compress(a.output.read_bytes(),mtime=0))
   print('FG_THINK_COMPLETE',json.dumps({'n':len(outputs),'censored':len(censored),'all_eos':result['all_ended_eos'],'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}),flush=True)
  finally:
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   f.close()
if __name__=='__main__':main()
