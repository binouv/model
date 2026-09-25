"""One finite CPU integration benchmark. No paid API, no generated-code execution.
Quantized Qwen4B and an eight-token non-thinking budget are NOT its capability ceiling.
Only raw prompts from the fixture are sent; gold labels remain outside this program.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,tempfile,time,urllib.request
from pathlib import Path
MODEL_URL='https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf'
MODEL_SHA='00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
TAG='v0.5.0'

def command(args,cwd=None):
    subprocess.run(args,cwd=cwd,check=True,timeout=650)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prompts',type=Path,required=True);a=ap.parse_args()
    rows=[json.loads(s) for s in a.prompts.read_text().splitlines()]
    if len(rows)!=40 or len({z['id'] for z in rows})!=40:raise ValueError('expected40 unique IDs')
    if any(set(z)!={'id','family','lang','prompt'} for z in rows):raise ValueError('fixture must not contain gold')
    with tempfile.TemporaryDirectory(prefix='flygraph-qwen-smoke-') as td:
        root=Path(td);source=root/'llama.cpp';model=root/'model.gguf'
        command(['git','clone','--depth','1','--branch',TAG,'https://github.com/ggml-org/llama.cpp.git',str(source)])
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        command(['cmake','-S',str(source),'-B',str(source/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
        command(['cmake','--build',str(source/'build'),'-j','4','--target','llama-server'])
        h=hashlib.sha256();count=0
        req=urllib.request.Request(MODEL_URL,headers={'User-Agent':'FlyGraph-research-smoke'})
        with urllib.request.urlopen(req,timeout=180) as f,model.open('wb') as out:
            while b:=f.read(8<<20):out.write(b);h.update(b);count+=len(b)
        with model.open('rb') as f:magic=f.read(4)
        if h.hexdigest()!=MODEL_SHA or magic!=b'GGUF':raise ValueError('model checksum or format mismatch')
        meta={'model_id':'Qwen/Qwen3.5-4B','conversion':'unsloth/Qwen3.5-4B-GGUF Q4_K_M','sha256':h.hexdigest(),'bytes':count,'llama_commit':commit,'llama_tag':TAG,'thinking':False,'max_new_tokens':8,'tools':False,'fixture_sha256':hashlib.sha256(a.prompts.read_bytes()).hexdigest()}
        print('FG_METADATA '+json.dumps(meta),flush=True)
        log=root/'server.log';f=log.open('w')
        p=subprocess.Popen([str(source/'build/bin/llama-server'),'-m',str(model),'-c','512','-t','4','-ngl','0','--host','127.0.0.1','--port','18080','--no-webui'],stdout=f,stderr=subprocess.STDOUT)
        try:
            ready=False
            for _ in range(180):
                if p.poll() is not None:raise RuntimeError('server exited: '+log.read_text()[-4000:])
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18080/health',timeout=2) as q:
                        if q.status==200:ready=True;break
                except Exception:time.sleep(1)
            if not ready:raise RuntimeError('server startup timeout')
            for z in rows:
                # Official single-turn Qwen3.5 no-thinking template, empty system message omitted.
                prompt='<|im_start|>user\n'+z['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                body={'prompt':prompt,'n_predict':8,'temperature':0.0,'seed':7301,'cache_prompt':False}
                request=urllib.request.Request('http://127.0.0.1:18080/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                t=time.perf_counter()
                with urllib.request.urlopen(request,timeout=90) as q:response=json.load(q)
                out={'id':z['id'],'family':z['family'],'lang':z['lang'],'prompt':z['prompt'],'generated':response['content'],'elapsed_sec':time.perf_counter()-t,'response':response}
                print('FG_PREDICTION '+json.dumps(out,ensure_ascii=False),flush=True)
            print('FG_COMPLETE '+json.dumps({'n':40,'all_predictions_completed':True,'metadata':meta}),flush=True)
        finally:
            p.terminate()
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()
if __name__=='__main__':main()
