"""Finite public-CPU BF16 Qwen0.8 thinking reference. No training or gold inputs."""
from __future__ import annotations
import hashlib,json,os,subprocess,tempfile,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO='binouv/model'
BRANCH='flygraph/l100p-qwen08-reference-20261009'
MODEL_URL='https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-BF16.gguf'
MODEL_SHA='9a7bed4041b7975e0f71fa34670d1e9025213bc92905ac0db75d36c4fa3fa623'
LLAMA='7fe450e19305b828c199d602c23a8337aaa1f03b'
FIXTURE='74bfffeb6cf3014b58d14cc1402fb8e9fca28818a0962d5deeb9140e792de26c'
PREFIX='Solve the task and return F=<integer> as the final answer.\n'
def execute(args,cwd=None):subprocess.run(args,cwd=cwd,check=True,timeout=900)
def gh_api(method,path,obj=None):
    assert os.environ['GITHUB_REPOSITORY']==REPO
    h={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','Content-Type':'application/json'}
    req=urllib.request.Request('https://api.github.com/repos/'+REPO+path,headers=h,data=None if obj is None else json.dumps(obj).encode(),method=method)
    with urllib.request.urlopen(req,timeout=90) as f:return json.load(f)
def push_files(paths):
    head=gh_api('GET','/git/ref/heads/'+BRANCH)['object']['sha']
    tree=gh_api('GET','/git/commits/'+head)['tree']['sha']
    entries=[{'path':str(p.relative_to(ROOT.parents[2])),'mode':'100644','type':'blob','content':p.read_text()} for p in paths]
    tree=gh_api('POST','/git/trees',{'base_tree':tree,'tree':entries})['sha']
    commit=gh_api('POST','/git/commits',{'tree':tree,'parents':[head],'message':'L100P: save actual Qwen0.8 BF16 thinking reference outputs; no training'})['sha']
    gh_api('PATCH','/git/refs/heads/'+BRANCH,{'sha':commit,'force':False})
    print('PUSH_REFERENCE',commit,flush=True)
def main():
    assert os.environ['GITHUB_REPOSITORY']==REPO
    p=ROOT/'PROMPTS.jsonl';assert hashlib.sha256(p.read_bytes()).hexdigest()==FIXTURE
    rows=[json.loads(z) for z in p.read_text().splitlines()]
    assert len(rows)==20 and all(set(z)=={'id','family','lang','prompt'} for z in rows)
    output=ROOT/'outputs';output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);source=td/'llama';weights=td/'model.gguf'
        execute(['git','clone','--depth','1','--branch','v0.5.0','https://github.com/ggml-org/llama.cpp.git',str(source)])
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==LLAMA
        execute(['cmake','-S',str(source),'-B',str(source/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
        execute(['cmake','--build',str(source/'build'),'-j','4','--target','llama-server'])
        hh=hashlib.sha256();size=0
        with urllib.request.urlopen(MODEL_URL,timeout=180) as f,weights.open('wb') as out:
            while bb:=f.read(8<<20):out.write(bb);hh.update(bb);size+=len(bb)
        assert hh.hexdigest()==MODEL_SHA
        metadata={'model':'Qwen3.5-0.8B','weights':'BF16 GGUF','sha256':MODEL_SHA,'weight_bytes':size,
          'llama_commit':LLAMA,'thinking':True,'native_open_think_prefill':True,'max_new_tokens':32768,
          'context':40960,'fixture_sha256':FIXTURE,'instruction_prefix':PREFIX,'tools':False,
          'workflow_run':os.environ['GITHUB_RUN_ID'],'source_commit':os.environ['GITHUB_SHA'],
          'sampling':{'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.0,'presence_penalty':1.5,'repeat_penalty':1.0}}
        (output/'METADATA.json').write_text(json.dumps(metadata,indent=2))
        log=output/'server.log';handle=log.open('w')
        server=subprocess.Popen([str(source/'build/bin/llama-server'),'-m',str(weights),'-c','40960','-t','4','-ngl','0','--host','127.0.0.1','--port','18080','--no-webui'],stdout=handle,stderr=subprocess.STDOUT)
        done=[]
        try:
            for _ in range(240):
                if server.poll() is not None:raise RuntimeError('llama server failed')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as r:
                        if r.status==200:break
                except Exception:time.sleep(1)
            else:raise RuntimeError('server startup timeout')
            for i,z in enumerate(rows):
                prompt='<|im_start|>user\n'+PREFIX+z['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n'
                assert '</think>' not in prompt
                body={'prompt':prompt,'n_predict':32768,'seed':92201+i,'cache_prompt':False,**metadata['sampling']}
                req=urllib.request.Request('http://127.0.0.1:18080/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                start=time.perf_counter()
                with urllib.request.urlopen(req,timeout=900) as f:response=json.load(f)
                text=response['content'];closed='</think>' in text
                item={**z,'generated':text,'final_text':text.split('</think>',1)[1].strip() if closed else None,
                      'thinking_closed':closed,'nonempty_thinking':bool(text.split('</think>',1)[0].strip()),
                      'stop_type':response.get('stop_type'),'tokens_predicted':response.get('tokens_predicted'),
                      'seconds':time.perf_counter()-start,'timings':response.get('timings')}
                done.append(item);pp=output/f'row_{i:02d}.json';pp.write_text(json.dumps(item,ensure_ascii=False,indent=2))
                progress=output/'PROGRESS.json';progress.write_text(json.dumps({'status':'partial_reference','completed_requests':len(done),'planned_requests':20}))
                push_files([pp,progress,output/'METADATA.json'])
                print('REFERENCE_CASE',i+1,item['stop_type'],item['tokens_predicted'],repr(item['final_text']),flush=True)
            censored=[x['id'] for x in done if x['stop_type']!='eos' or not x['thinking_closed']]
            result={'status':'completed_all_requests','n':len(done),'metadata':metadata,'rows':done,
                    'censored_ids':censored,'uncensored_comparison_valid':not censored,
                    'not_used_for_training':True,'not_a_general_intelligence_benchmark':True}
            pp=output/'FINAL.json';pp.write_text(json.dumps(result,ensure_ascii=False,indent=2));push_files([pp])
        finally:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait()
            handle.close()
if __name__=='__main__':main()
