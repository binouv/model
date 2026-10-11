"""Finite Qwen inference only; never trains FlyGraph or sends gold answers."""
from __future__ import annotations
import hashlib,json,os,subprocess,tempfile,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO='binouv/model'
BRANCH='flygraph/l100v-reference-20261011'
FIXTURE='1144d1b3eb23b7f33d223782d76e4fcc3dad839ea880a681fdf323d47dbb0a90'
MODEL_URL='https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-BF16.gguf'
MODEL_SHA='9a7bed4041b7975e0f71fa34670d1e9025213bc92905ac0db75d36c4fa3fa623'
LLAMA='7fe450e19305b828c199d602c23a8337aaa1f03b'
SAMPLING={'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.,'presence_penalty':1.5,'repeat_penalty':1.0}
def api(method,path,obj=None):
    assert os.environ['GITHUB_REPOSITORY']==REPO
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,
      data=None if obj is None else json.dumps(obj).encode(),method=method,
      headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as f:return json.load(f)
def publish(paths):
    # Shards write disjoint paths. Every retry reads the new head and preserves its tree.
    for attempt in range(12):
        try:
            head=api('GET','/git/ref/heads/'+BRANCH)['object']['sha']
            base=api('GET','/git/commits/'+head)['tree']['sha']
            entries=[{'path':str(p.relative_to(ROOT.parents[2])),'mode':'100644','type':'blob','content':p.read_text()} for p in paths]
            tree=api('POST','/git/trees',{'base_tree':base,'tree':entries})['sha']
            commit=api('POST','/git/commits',{'tree':tree,'parents':[head],'message':'L100V: preserve actual Qwen thinking responses; no training'})['sha']
            api('PATCH','/git/refs/heads/'+BRANCH,{'sha':commit,'force':False})
            print('PUSH_REFERENCE',commit,flush=True);return
        except urllib.error.HTTPError as e:
            if e.code not in (409,422,429,500,502,503) or attempt==11:raise
            time.sleep(min(15,attempt+1))
def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2));return path
def execute(cmd):subprocess.run(cmd,check=True,timeout=1200)
def main():
    assert os.environ['GITHUB_REPOSITORY']==REPO
    assert os.environ['GITHUB_REF_NAME']==BRANCH
    shard=int(os.environ['SHARD']);assert 0<=shard<4
    prompt_file=ROOT/'PROMPTS.jsonl'
    assert hashlib.sha256(prompt_file.read_bytes()).hexdigest()==FIXTURE
    rows=[json.loads(x) for x in prompt_file.read_text().splitlines()]
    assert len(rows)==22 and all(set(x)=={'id','family','lang','prompt'} for x in rows)
    indexed=list(enumerate(rows))[shard::4]
    output=ROOT/'outputs'/f'shard_{shard}';output.mkdir(parents=True,exist_ok=True)
    metadata={'model':'Qwen3.5-0.8B','precision':'original BF16 GGUF, not quantized Q4',
      'model_sha256':MODEL_SHA,'llama_commit':LLAMA,'fixture_sha256':FIXTURE,
      'thinking':True,'open_think_prefill':True,'context':40960,'max_new_tokens':32768,
      'sampling':SAMPLING,'seed_base':110362,'shard':shard,'shards':4,
      'planned_all':22,'planned_shard':len(indexed),'own_model_training':False,
      'prompt_prefix':'','tools':False,'source_commit':os.environ['GITHUB_SHA'],
      'workflow_run':os.environ['GITHUB_RUN_ID'],'request_wall_timeout_seconds':1800,
      'note':'Wall time failures and truncated thinking remain censored, never scored as wrong.'}
    meta=save(output/'METADATA.json',metadata);publish([meta])
    with tempfile.TemporaryDirectory() as tmp:
        td=Path(tmp);src=td/'llama';weights=td/'model.gguf'
        execute(['git','clone','--depth','1','--branch','v0.5.0','https://github.com/ggml-org/llama.cpp.git',str(src)])
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()==LLAMA
        execute(['cmake','-S',str(src),'-B',str(src/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
        execute(['cmake','--build',str(src/'build'),'-j','4','--target','llama-server'])
        hh=hashlib.sha256();size=0
        with urllib.request.urlopen(MODEL_URL,timeout=180) as f,weights.open('wb') as out:
            while True:
                part=f.read(8<<20)
                if not part:break
                out.write(part);hh.update(part);size+=len(part)
        assert hh.hexdigest()==MODEL_SHA
        metadata['weight_bytes']=size;save(meta,metadata)
        handle=(output/'server.log').open('w');server=None;done=[]
        def start_server():
            proc=subprocess.Popen([str(src/'build/bin/llama-server'),'-m',str(weights),'-c','40960','-t','4','-ngl','0','--host','127.0.0.1','--port','18080','--no-webui'],stdout=handle,stderr=subprocess.STDOUT)
            for _ in range(240):
                if proc.poll() is not None:raise RuntimeError('server failed')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as r:
                        if r.status==200:return proc
                except Exception:time.sleep(1)
            proc.terminate();proc.wait(timeout=15);raise RuntimeError('server startup timeout')
        def stop_server(proc):
            if proc is None:return
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        try:
            server=start_server()
            for index,row in indexed:
                seed=110362+index
                prompt='<|im_start|>user\n'+row['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
                body={'prompt':prompt,'n_predict':32768,'seed':seed,'cache_prompt':False,**SAMPLING}
                req=urllib.request.Request('http://127.0.0.1:18080/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                started=time.perf_counter();item={**row,'index':index,'seed':seed}
                try:
                    with urllib.request.urlopen(req,timeout=1800) as f:response=json.load(f)
                    text=response['content'];closed='</think>' in text
                    item.update({'status':'response_received','generated':text,'final_text':text.split('</think>',1)[1].strip() if closed else None,
                      'thinking_closed':closed,'nonempty_thinking':bool(text.split('</think>',1)[0].strip()),
                      'stop_type':response.get('stop_type'),'tokens_predicted':response.get('tokens_predicted'),
                      'timings':response.get('timings'),'raw_response':response})
                    item['uncensored']=closed and item['stop_type']=='eos'
                except Exception as e:
                    item.update({'status':'request_failed','error':type(e).__name__+': '+str(e),'uncensored':False,'final_text':None})
                    stop_server(server);server=None
                item['seconds']=time.perf_counter()-started;done.append(item)
                pp=save(output/f'row_{index:02d}.json',item)
                progress=save(output/'PROGRESS.json',{'status':'partial_reference','attempted':len(done),'planned_shard':len(indexed),'uncensored':sum(x['uncensored'] for x in done)})
                publish([pp,progress,meta]);print('CASE',index,item['status'],item.get('tokens_predicted'),repr(item['final_text']),flush=True)
                if server is None and index!=indexed[-1][0]:server=start_server()
            final=save(output/'FINAL.json',{'status':'completed_attempts','n':len(done),'rows':done,'metadata':metadata,
              'censored_ids':[x['id'] for x in done if not x['uncensored']],
              'uncensored_comparison_valid':all(x['uncensored'] for x in done),
              'not_used_for_training':True,'not_a_broad_intelligence_benchmark':True})
            publish([final])
        finally:stop_server(server);handle.close()
if __name__=='__main__':main()
