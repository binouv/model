"""R04/H1: frozen Qwen states vs lexical, structured and permuted-label probes.

NO model weights trained or modified. Only small independent diagnostic readouts.
Readout selection uses validation only; test scores cannot choose the winner.
Required complete run: 1200 train / 200 val / 300 test, seeds 11/22/33.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import re
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from h1_dataset import load_dataset


def file_sha(path: Path) -> str:
    with path.open('rb') as f:
        digest=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
        return digest.hexdigest()

PRIMARY_SEEDS=(11,22,33)


def lexical_features(examples: list[dict], dims: int = 2560) -> torch.Tensor:
    """Deterministic signed hash over word+number unigrams, bigrams, trigrams.

    Captures numeric glyphs, Cyrillic, token order and operation templates,
    but has no oracle values and no program executor.
    """
    out=torch.zeros((len(examples),dims),dtype=torch.float32)
    for i,row in enumerate(examples):
        toks=re.findall(r'[a-zA-Zа-яА-ЯёЁ]+|\d+|[%+*()=]',row['question'].lower())
        for n in (1,2,3):
            for j in range(len(toks)-n+1):
                gram=f'{n}:'+':'.join(toks[j:j+n])
                dig=hashlib.blake2b(gram.encode('utf-8'),digest_size=8,person=b'R04-H1-').digest()
                col=int.from_bytes(dig[:4],'little')%dims
                out[i,col]+=1 if dig[4]&1 else -1
    return F.normalize(out,p=2,dim=-1)*math.sqrt(dims)


def structured_features(examples: list[dict]) -> torch.Tensor:
    """Strong 'input-only' control: exact operands/operators, never answers."""
    out=torch.zeros((len(examples),66),dtype=torch.float32)
    for i,row in enumerate(examples):
        a,b,c=row['operands']
        out[i,a]=1
        out[i,20+b]=1
        out[i,40+c]=1
        out[i,60+(row['operator']=='weighted')]=1
        out[i,62+(row['domain']=='code')]=1
        out[i,64+(row['language']=='ru')]=1
    return out


class TwoDigitProbe(nn.Module):
    def __init__(self,feature_size:int,kind:str,hidden:int=160):
        super().__init__()
        if kind=='linear':
            self.net=nn.Sequential(nn.LayerNorm(feature_size),nn.Linear(feature_size,20))
        elif kind=='mlp':
            self.net=nn.Sequential(nn.LayerNorm(feature_size),nn.Linear(feature_size,hidden),
                                   nn.GELU(),nn.Linear(hidden,20))
        else:raise ValueError(kind)
        self.feature_size=feature_size
        self.kind=kind
    def forward(self,features):
        return self.net(features).reshape(-1,2,10)


@torch.inference_mode()
def classify(model,features,target,indices,rows):
    model.eval()
    if not indices:raise ValueError('Cannot classify empty split')
    x=features[indices]
    truth=target[indices]
    predictions=[]
    for k in range(0,len(indices),256):
        pred=model(x[k:k+256]).argmax(-1)
        predictions.append(pred.cpu())
    guessed=torch.cat(predictions,0)
    matches=guessed.eq(truth)
    exact=matches.all(-1)
    by_domain={d:float(exact[torch.tensor([rows[i]['domain']==d for i in indices],dtype=torch.bool)].float().mean()) for d in ('arithmetic','code')}
    by_language={lang:float(exact[torch.tensor([rows[i]['language']==lang for i in indices],dtype=torch.bool)].float().mean()) for lang in ('en','ru')}
    return {
        'exact':float(exact.float().mean()),'correct_count':int(exact.sum()),'n':len(indices),
        'digit0':float(matches[:,0].float().mean()),'digit1':float(matches[:,1].float().mean()),
        'by_domain':by_domain,'by_language':by_language,
    },guessed.tolist()


def run_one(name,kind,seed,features,target,train_indices,val_indices,rows,steps,batch,lr,device,
            output_root:Path,shuffle=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    size=features.size(1)
    model=TwoDigitProbe(size,kind).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    samples=torch.tensor(train_indices,device=device,dtype=torch.long)
    permuted=target.clone()
    if shuffle:
        rng=torch.Generator().manual_seed(seed+456)
        permuted[samples.cpu()]=permuted[samples.cpu()[torch.randperm(len(samples),generator=rng)]]
    feats=features.to(device)
    target_dev=permuted.to(device)
    generator=torch.Generator(device='cpu').manual_seed(seed+37)
    losses=[]
    t=time.perf_counter()
    for step in range(steps):
        ids=samples[torch.randint(len(samples),(batch,),generator=generator).to(device)]
        guess=model(feats[ids])
        loss=F.cross_entropy(guess.reshape(-1,10),target_dev[ids].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % max(1,steps//4)==0:
            losses.append({'step':step+1,'loss':round(float(loss.detach().cpu()),6)})
    if device.type=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-t
    model_cpu=model.cpu()
    val,_=classify(model_cpu,features,target,val_indices,rows)
    safe_name=f'{name}_{kind}_seed{seed}'
    file=output_root/'checkpoints'/f'{safe_name}.pt'
    file.parent.mkdir(parents=True,exist_ok=True)
    # Complete native optimizer state and RNG; keep each file below 90 MB.
    opt_cpu=optimizer.state_dict()
    for state in opt_cpu['state'].values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor):state[k]=v.detach().cpu()
    torch.save({
        'model':model_cpu.state_dict(), 'optimizer':opt_cpu,
        'torch_rng':torch.get_rng_state(),
        'numpy_rng':np.random.get_state(), 'python_rng':random.getstate(),
        'torch_cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'train_seed':seed,'name':name,'kind':kind,'steps':steps,'batch':batch,'lr':lr,
        'shuffle_train_labels':shuffle,'feature_size':size,
    },file)
    if file.stat().st_size>=90_000_000:
        raise RuntimeError('Single checkpoint exceeds 90,000,000-byte Git part ceiling')
    info={'name':name,'kind':kind,'seed':seed,'validation':val,'train_seconds':seconds,
          'trace':losses,'parameters':sum(p.numel() for p in model_cpu.parameters()),
          'checkpt':{'path':file.relative_to(output_root).as_posix(),
                     'sha256':file_sha(file),'bytes':file.stat().st_size}}
    print('trained',safe_name,'val',val['exact'],'params',info['parameters'],flush=True)
    del model,optimizer,feats,target_dev
    if device.type=='cuda':torch.cuda.empty_cache()
    return info


def restore(record,features,targets,indices,rows,out):
    cp=out/record['checkpt']['path']
    if file_sha(cp)!=record['checkpt']['sha256']:raise RuntimeError(f'Checkpoint hash mismatch: {cp}')
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    size=saved['feature_size']
    net=TwoDigitProbe(size,saved['kind'])
    net.load_state_dict(saved['model'],strict=True)
    measurements,pred=classify(net,features,targets,indices,rows)
    return measurements,pred


def main():
    ap=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--data',type=Path,default=root/'data')
    ap.add_argument('--cache',type=Path,default=root/'cache')
    ap.add_argument('--out',type=Path,default=root/'results'/'h1')
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--batch',type=int,default=96)
    ap.add_argument('--lr',type=float,default=.0015)
    ap.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    ap.add_argument('--seeds',type=int,nargs='*',default=list(PRIMARY_SEEDS))
    args=ap.parse_args()
    if tuple(args.seeds)!=PRIMARY_SEEDS:
        raise RuntimeError('Scientific H1 requires all three fixed seeds (11,22,33).')
    rows,manifest=load_dataset(args.data)
    source=args.cache/'h1_features.safetensors'
    feature_manifest_file=args.cache/'features_manifest.json'
    if not source.is_file() or not feature_manifest_file.is_file():
        raise SystemExit('First run GPU extraction script to completion.')
    source_manifest=json.loads(feature_manifest_file.read_text(encoding='utf-8'))
    if source_manifest['dataset_sha256']!=manifest['dataset_sha256_canonical'] or source_manifest['features_sha256']!=file_sha(source):
        raise RuntimeError('Teacher feature cache does not match generated dataset or SHA256')
    with safe_open(str(source),framework='pt',device='cpu') as handle:
        metadata=handle.metadata()
        if metadata.get('dataset_sha256')!=manifest['dataset_sha256_canonical']:
            raise RuntimeError('Safetensors cache dataset mismatch')
        feats={name:handle.get_tensor(name).float().contiguous() for name in ('layer16','layer24','last')}
    if any(x.shape!=(1700,2560) for x in feats.values()):raise RuntimeError('Unexpected feature shape')
    feats['hash']=lexical_features(rows)
    feats['structured']=structured_features(rows)
    labels=torch.tensor([r['digits'] for r in rows],dtype=torch.long)
    split_indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val','test')}
    if [len(split_indices[s]) for s in ('train','val','test')] != [1200,200,300]:
        raise RuntimeError('Split sizes mismatch, refusing H1')
    args.out.mkdir(parents=True,exist_ok=True)
    specs=[]
    for name in ('layer16','layer24','last','hash','structured'):
        for kind in ('linear','mlp'):
            specs.append((name,kind,False))
    specs.append(('last','mlp',True))  # label-permutation negative control
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the registered GPU experiment')
    device=torch.device(args.device)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    records=[]
    for name,kind,shuffle in specs:
        for seed in PRIMARY_SEEDS:
            slug=name+'_shuffle' if shuffle else name
            record=run_one(slug,kind,seed,feats[name],labels,split_indices['train'],split_indices['val'],
                           rows,args.steps,args.batch,args.lr,device,args.out,shuffle)
            records.append(record)
    # All test-selection decisions are made using ONLY validation scores.
    select={}
    for name in ('teacher_layers','input_controls','negative_controls'):
        candidates=({'teacher_layers':('layer16','layer24','last'),
            'input_controls':('hash','structured'),
            'negative_controls':('last_shuffle',)})[name]
        choices={}
        for candidate,kind in ((a,b) for a in candidates for b in ('linear','mlp')):
            subset=[r for r in records if r['name']==candidate and r['kind']==kind]
            if len(subset)!=3:continue
            choices[f'{candidate}:{kind}']=sum(r['validation']['exact'] for r in subset)/len(subset)
        selected=max(sorted(choices),key=choices.get)
        select[name]={'winner':selected,'validation_mean_exact':choices[selected],
                      'all_validation_mean_exact':choices}
    print('SELECTED ON VAL, before any test evaluation:',select,flush=True)
    selected_records={key:[r for r in records if f"{r['name']}:{r['kind']}"==x['winner']]
                      for key,x in select.items()}
    scored={}
    preds={}
    for category,models in selected_records.items():
        per_seed=[]
        preds[category]={}
        chosen_name=models[0]['name'].replace('_shuffle','')
        for record in models:
            metrics,output=restore(record,feats[chosen_name],labels,split_indices['test'],rows,args.out)
            per_seed.append({'seed':record['seed'],'metrics':metrics})
            preds[category][str(record['seed'])]=output
        scored[category]={'selected':select[category]['winner'],'by_seed':per_seed,
                          'mean_test_exact':sum(x['metrics']['exact'] for x in per_seed)/3}
    teacher_path=args.cache/'head_test_receipt.json'
    head_baseline=None
    teacher_examples={}
    if teacher_path.exists():
        head_baseline=json.loads(teacher_path.read_text(encoding='utf-8'))
        if head_baseline['dataset_sha256']!=manifest['dataset_sha256_canonical']:
            raise RuntimeError('Head teacher baseline provenance mismatch')
        predictions=args.cache/'head_test_predictions.jsonl'
        if head_baseline['predictions_file_sha256']!=file_sha(predictions):
            raise RuntimeError('Head baseline response file tampered')
        teacher_examples={x['id']:x for x in map(json.loads,predictions.read_text(encoding='utf-8').splitlines())}
    writer_path=args.cache/'writer_test_receipt.json'
    writer_baseline=json.loads(writer_path.read_text(encoding='utf-8')) if writer_path.exists() else None
    if writer_baseline and writer_baseline['dataset_sha256']!=manifest['dataset_sha256_canonical']:
        raise RuntimeError('Writer baseline provenance mismatch')
    confirmations=[]
    for i,seed in enumerate(PRIMARY_SEEDS):
        main=scored['teacher_layers']['by_seed'][i]['metrics']['exact']
        lexical=scored['input_controls']['by_seed'][i]['metrics']['exact']
        negative=scored['negative_controls']['by_seed'][i]['metrics']['exact']
        confirmations.append({'seed':seed,'probe':main,'input_control':lexical,'shuffle':negative,
             'criteria':main>=.70 and main-lexical>=.15 and main-negative>=.20})
    controls_complete = head_baseline is not None and writer_baseline is not None
    gate=controls_complete and head_baseline['exact_all_test']>=.80 and sum(x['criteria'] for x in confirmations)>=2
    metrics={
        'status':'completed_full_h1_protocol' if controls_complete else 'probes_completed_baselines_pending',
        'h1_conclusion':'CONFIRMED_under_registered_gates' if gate else ('NOT_CONFIRMED_under_registered_gates' if controls_complete else 'INCOMPLETE_STANDALONE_BASELINES'),
        'dataset_sha256':manifest['dataset_sha256_canonical'],
        'frozen_teacher_revision':source_manifest['revision'],
        'pre_answer_feature_sha256':source_manifest['features_sha256'],
        'architecture':'Two independent ten-class digits, full exact requires both; no answer tokens in frozen teacher input.',
        'test_selection_protocol':'mean validation accuracy across three seeds chooses architecture/layer and strongest input baseline, BEFORE test.',
        'training_settings':{'steps':args.steps,'batch':args.batch,'lr':args.lr,'seeds':list(PRIMARY_SEEDS),'optimizer':'AdamW','device':str(device)},
        'test_size':len(split_indices['test']),
        'interpretation': ('two_or_more_teacher_target_tokens' if head_baseline and head_baseline['answer_has_2plus_token_rate']>=.95 else 'insufficient_tokenization_evidence_for_multi_token_claim'),
        'validation_winners':select,'test':scored,'gate_per_seed':confirmations,
        'teacher_baseline':head_baseline,'writer_only_baseline':writer_baseline,
        'limitations':['Answer is two independent single digits, not arbitrary unconstrained future text.',
             'Frozen pretrained hidden vectors were quantized to float16 on disk.',
             'A positive answer probe does not establish 99% of reasoning FLOPs are already computed.',
             'Only selected models were tested; non-selected models have checkpoint files and val metrics.',
             'Synthetic RU/EN arithmetic/program-output questions, not general code generation.'],
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'},
    }
    outfile=args.out/'h1_metrics.json'
    # Raw per-example oracle + all three seeds of selected models; teacher outputs never leaked into probe training.
    perexample=[]
    for j,index in enumerate(split_indices['test']):
        ex=rows[index]
        detail={'id':ex['id'],'oracle':ex['digits'],'domain':ex['domain'],
                'language':ex['language'],'head_greedy':teacher_examples.get(ex['id']),
                'selected_readouts':{name:{seed:entries[j] for seed,entries in pred.items()} for name,pred in preds.items()}}
        perexample.append(detail)
    (args.out/'h1_test_predictions.jsonl').write_text(
        ''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in perexample),encoding='utf-8')
    metrics['test_predictions_sha256']=file_sha(args.out/'h1_test_predictions.jsonl')
    metrics['checkpoints']=[r['checkpt'] for r in records]
    metrics['val_all_runs']=[r for r in records]
    outfile.write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':metrics['status'],'conclusion':metrics['h1_conclusion'],
                      'teacher':head_baseline['exact_all_test'] if head_baseline else None,
                      'seed_gates':confirmations,'winners':{k:v['winner'] for k,v in select.items()}},indent=2),flush=True)


if __name__=='__main__':main()