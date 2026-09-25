"""Held-out generative evaluation. Gold labels never enter model.forward.
One ablation: same weights with zero graph messages. No other model variants.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,re,time
from pathlib import Path
import numpy as np
import torch
from checkpoint_io import load_checkpoint,save_checkpoint,digest
from tokenizer import Tokenizer
from model import Config,FlyGraphLM,parameter_report
ROOT=Path(__file__).resolve().parents[1]

@torch.no_grad()
def lm_eval(model,split='test',n=4,seq=128,disable_edges=False):
    rows=[];data=ROOT/'data'
    for kind in ['python_source','english_docstring','memory_update','memory_chain','arithmetic','multistep','code_trace','sorting','ru_prose']:
        p=data/f'{split}_{kind}.bin';a=np.memmap(p,dtype=np.uint16,mode='r');L=min(seq,len(a)-1)
        rng=np.random.default_rng(7192)
        starts=np.arange(0,len(a)-L,L+1,dtype=np.int64)
        # Fixed, non-overlapping held windows. Do not count duplicated tokens as
        # independent held blocks. Selected before the first test evaluation.
        starts=rng.permutation(starts)[:n]
        for i,start0 in enumerate(starts):
            start=int(start0)
            t=torch.tensor(np.array(a[start:start+L+1],dtype=np.int64))[None]
            loss,lg=model(t[:,:-1],t[:,1:],disable_graph_edges=disable_edges)
            rows.append({'kind':kind,'block':i,'start':start,'tokens':L,'nll':float(loss),'token_accuracy':float(lg.argmax(-1).eq(t[:,1:]).float().mean())})
    by={}
    for kind in sorted({z['kind'] for z in rows}):
        rr=[z for z in rows if z['kind']==kind];nt=sum(z['tokens'] for z in rr)
        by[kind]={'tokens':nt,'blocks':len(rr),'nll':sum(z['nll']*z['tokens'] for z in rr)/nt,'token_accuracy':sum(z['token_accuracy']*z['tokens'] for z in rr)/nt}
    return {'split':split,'by_kind':by,'macro_nll':sum(z['nll'] for z in by.values())/len(by),'blocks':rows}

def normalized(s):return re.sub(r'\s+',' ',s.strip()).rstrip('.')

def generation_probes():
    rows=[json.loads(s) for s in (ROOT/'data/test_generation.jsonl').read_text().splitlines()]
    selected=[]
    for kind in ['arithmetic','multistep','memory_update','memory_chain','list_sum','code_trace','sorting']:
        selected.extend([z for z in rows if z['kind']==kind][:3])
    return selected

@torch.no_grad()
def eval_generation(model,tok,ablation=False,max_new=40):
    rows=[]
    for z in generation_probes():
        ids=torch.tensor([tok.encode(z['prompt'],bos=True)]);t0=time.perf_counter()
        out=model.generate(ids,max_new_tokens=max_new,disable_graph_edges=ablation)
        completion=tok.decode(out[0,ids.shape[1]:].tolist())
        # Strict teacher-completion metric and separate final-answer parsing.
        strict=normalized(completion)==normalized(z['gold_completion'])
        if z['kind'] in ['arithmetic','multistep','list_sum']:
            ms=re.findall(r'(?:Final|Итог):\s*(-?\d+)',completion)
            answer=ms[-1] if ms else None
        elif z['kind']=='memory_chain':
            ms=re.findall(r'Final:\s*([A-Za-z]+)',completion);answer=ms[-1] if ms else None
        elif z['kind']=='memory_update':
            ms=re.match(r'\s*([A-Za-z]+)',completion);answer=ms.group(1) if ms else None
        elif z['kind']=='code_trace':
            ms=re.match(r'\s*(-?\d+)\b',completion);answer=ms.group(1) if ms else None
        elif z['kind']=='sorting':
            ms=re.search(r'\[[\d,\s]+\]',completion);answer=ms.group(0) if ms else None
        else:answer=None
        rows.append({**z,'generated':completion,'strict_completion_correct':strict,'parsed_answer':answer,'answer_correct':answer is not None and normalized(answer)==normalized(z['final_answer']),'generated_tokens':out.shape[1]-ids.shape[1],'elapsed_sec':time.perf_counter()-t0})
    return {'n':len(rows),'answer_accuracy':sum(z['answer_correct'] for z in rows)/max(1,len(rows)),'strict_completion_accuracy':sum(z['strict_completion_correct'] for z in rows)/max(1,len(rows)),'rows':rows,'note':'Synthetic prompt families; not GSM8K/MBPP and not a Qwen comparison.'}

@torch.no_grad()
def audit_model(model):
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');ids=torch.tensor([tok.encode('Memory: oak=blue, pine=red. What is oak?\nAnswer:',bos=True)])
    all_logits=model(ids);changed=ids.clone();changed[:,-3:]=torch.tensor([[300,301,302]])
    causal=(all_logits[:,:-3]-model(changed)[:,:-3]).abs().max().item()
    a,cache=model(ids[:,:-1],use_cache=True);b,cache=model(ids[:,-1:],past=cache,use_cache=True)
    cached=(all_logits[:,-1:]-b).abs().max().item()
    edge_effect=(all_logits-model(ids,disable_graph_edges=True)).abs().max().item()
    finite=True
    for p in model.parameters():
        # Whole-tensor finite check, one tensor at a time.
        if not bool(torch.isfinite(p).all()):finite=False;break
    return {'causal_prefix_max_abs_diff':causal,'cached_last_token_max_abs_diff':cached,'graph_zero_message_max_abs_logit_diff':edge_effect,'all_parameter_values_finite':finite,**parameter_report(model)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--mode',choices=['evaluate','export','verify_export','generation'],default='evaluate');ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)  # eval() alone does not disable autograd
    m,idx=load_checkpoint(args.checkpoint);m.eval();tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    if args.mode=='export':
        z=save_checkpoint(m,ROOT/'release_bf16',step=idx['step'],training={**idx['training'],'precision':'BF16 export from FP32 trained checkpoint; not bit-exact resume'},dtype=torch.bfloat16)
        (ROOT/'metrics/bf16_export.json').write_text(json.dumps(z,indent=2));print('EXPORT_COMPLETE',flush=True);return
    if args.mode=='generation':
        rows=eval_generation(m,tok,False,40)
        (ROOT/'metrics/generation_main.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in rows.items() if k!='rows'}),flush=True)
        rows=eval_generation(m,tok,True,40)
        (ROOT/'metrics/generation_no_graph_messages.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('GENERATION_COMPLETE',flush=True);return
    if args.mode=='verify_export':
        aa=audit_model(m);ev=lm_eval(m,n=1)
        (ROOT/'metrics/bf16_reload_audit.json').write_text(json.dumps({'audit':aa,'test_small':ev},indent=2));print('BF16_RELOAD_COMPLETE',flush=True);return
    audit=audit_model(m);(ROOT/'metrics/final_model_audit.json').write_text(json.dumps(audit,indent=2));print('AUDIT',json.dumps(audit),flush=True)
    for arm,off in [('main',False),('no_graph_messages',True)]:
        z=lm_eval(m,disable_edges=off);(ROOT/f'metrics/test_{arm}.json').write_text(json.dumps(z,indent=2));print('TEST',arm,z['macro_nll'],flush=True)
    # Context-length support is an engineering forward test, not long-context capability.
    zz={}
    for T in [256,512,1024]:
        x=torch.arange(T)[None]%8192;t0=time.perf_counter();o=m(x)
        zz[str(T)]={'shape':list(o.shape),'finite':bool(torch.isfinite(o).all()),'seconds':time.perf_counter()-t0};del o
    (ROOT/'metrics/context_forward_tests.json').write_text(json.dumps(zz,indent=2));print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
