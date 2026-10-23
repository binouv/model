import hashlib,json,math,os,random,re,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from model import Model,Config
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def atomic_json(p,obj):
    p=Path(p);temp=p.with_name(p.name+'.tmp')
    with temp.open('w') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def setup():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
def rows(name):return [json.loads(s) for s in (ROOT/'data'/name).read_text().splitlines()]
def load(weights):
    c=Config(**json.loads((ROOT/'checkpoints/config.json').read_text()));c.checkpoint_blocks=False
    m=Model(c);sd=load_file(str(weights));m.load_state_dict(sd,strict=True)
    assert sum(p.numel() for p in m.parameters())==100028328
    assert len(sd)==135 and all(torch.isfinite(p).all() for p in m.parameters())
    return m,Tokenizer.load(ROOT/'data/tokenizer.json')
def encode(tok,r):
    p=tok.encode(r['prompt'],bos=True);a=tok.encode(r['completion'],eos=True)
    return torch.tensor([p+a[:-1]]),torch.tensor([[-100]*(len(p)-1)+a]),len(a)
def parse(text,n):
    q=re.fullmatch(r'\s*([A-D])\s*[.)]?\s*',text)
    return ord(q[1])-65 if q and ord(q[1])-65<n else None
@torch.inference_mode()
def score_labels(m,tok,r):
    ids=torch.tensor([tok.encode(r['prompt'],bos=True)]);base,cache=m(ids,last_only=True)
    scores=[];with_eos=[]
    for i in range(len(r['choices'])):
        tokens=tok.encode(' '+chr(65+i));lg=base;cc=cache;score=0.
        for t in tokens:
            score+=float(lg[0,-1].float().log_softmax(-1)[t]);lg,cc=m(torch.tensor([[t]]),cache=cc,last_only=True)
        scores.append(score);with_eos.append(score+float(lg[0,-1].float().log_softmax(-1)[tok.EOS]))
    return scores,with_eos
@torch.inference_mode()
def evaluate(m,tok,split,arm,generation=True,limit=None,write=True):
    m.eval();started=time.perf_counter();data=rows(split+'_mcq.jsonl');ad=rows(split+'_alpaca.jsonl')
    if limit:data=[r for task in ['openbookqa','piqa','arc_challenge'] for r in [x for x in data if x['task']==task][:limit]];ad=ad[:limit]
    out=[];av=[]
    for r in data:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            scores,eos_scores=score_labels(m,tok,r)
            seq=m.generate(torch.tensor([tok.encode(r['prompt'],bos=True)]),max_new_tokens=16,eos_id=tok.EOS)[0].tolist() if generation else []
        body=seq[:seq.index(tok.EOS)] if tok.EOS in seq else seq;text=tok.decode(body)
        out.append({'id':r['id'],'task':r['task'],'gold':r['gold_index'],'prompt':r['prompt'],'label_log_probs':scores,'label_with_eos_log_probs':eos_scores,'predicted_label':int(np.argmax(scores)),'generation':text,'new_token_ids':seq,'parsed_generation':parse(text,len(r['choices'])) if generation else None,'eos':tok.EOS in seq,'budget_exhausted':len(seq)==16 and tok.EOS not in seq})
    for r in ad:
        x,y,n=encode(tok,r)
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=float(m(x,targets=y))
        av.append({'id':r['id'],'nll':loss,'target_tokens':n})
    summary={'arm':arm,'split':split,'seconds':time.perf_counter()-started,'n_mcq':len(out),'n_alpaca':len(av),'generation_evaluated':generation,'by_task':{}}
    for task in ['openbookqa','piqa','arc_challenge']:
        pp=[p for p in out if p['task']==task];n=len(pp)
        summary['by_task'][task]={'n':n,'label_correct':sum(p['predicted_label']==p['gold'] for p in pp),'label_accuracy':sum(p['predicted_label']==p['gold'] for p in pp)/n,'generation_correct':sum(p['parsed_generation']==p['gold'] for p in pp),'generation_accuracy':sum(p['parsed_generation']==p['gold'] for p in pp)/n,'valid_generation':sum(p['parsed_generation'] is not None for p in pp),'gold_label_eos_nll':-sum(p['label_with_eos_log_probs'][p['gold']] for p in pp)/n}
    summary['alpaca_nll']=sum(p['nll']*p['target_tokens'] for p in av)/sum(p['target_tokens'] for p in av)
    # Equal task weights: general language, trained science QA, physical QA.
    # ARC dev is a transfer diagnostic, excluded from checkpoint selection.
    summary['selection_objective']=(summary['alpaca_nll']+summary['by_task']['openbookqa']['gold_label_eos_nll']+summary['by_task']['piqa']['gold_label_eos_nll'])/3
    if write:
        atomic_json(ROOT/f'metrics/{arm}_{split}_summary.json',summary)
        for suffix,rr in [('predictions',out),('alpaca',av)]:
            with (ROOT/f'metrics/{arm}_{split}_{suffix}.jsonl').open('x') as f:
                for p in rr:f.write(json.dumps(p,ensure_ascii=False)+'\n')
    return summary,out,av
