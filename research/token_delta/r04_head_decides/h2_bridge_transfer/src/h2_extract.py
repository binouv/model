"""Frozen Qwen3.5-4B: pre-answer states and ALL greedy outputs on fresh H2 data.

Teacher gets only question. No gold/oracle answer enters teacher input.
Reproducible resumable paired safetensors + JSONL shards, BF16 head states.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import torch
from safetensors.torch import save_file
from safetensors import safe_open

from h2_dataset import load, sha256_file, canonical_sha

R04 = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R04/'src'))
from gpu_smoke import pinned_model, safe_chat_prompt  # noqa: E402


def parse_digits(raw: str):
    m=re.fullmatch(r'\s*([0-9])\s+([0-9])\s*',raw,flags=re.DOTALL)
    return [int(m.group(1)),int(m.group(2))] if m else None


def tokenizer_for(spec):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(spec['hf_id'],revision=spec['revision'],trust_remote_code=False)
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    return tok


def prompt_ids(tok,questions,max_prompt_tokens):
    s=[safe_chat_prompt(tok,q) for q in questions]
    enc=tok(s,add_special_tokens=False,padding=True,return_tensors='pt',truncation=False)
    if enc['input_ids'].shape[1]>max_prompt_tokens:
        raise RuntimeError('Prompt exceeds preregistered length ceiling')
    return {k:v.to('cuda:0') for k,v in enc.items()}


def save_pair(shard_tensor,shard_json,feature,records,meta):
    shard_tensor.parent.mkdir(parents=True,exist_ok=True)
    atmp=shard_tensor.with_suffix('.partial.safetensors')
    btmp=shard_json.with_suffix('.partial.jsonl')
    save_file({'head_last':feature.detach().cpu().contiguous()},str(atmp),metadata=meta)
    with btmp.open('w',encoding='utf-8',newline='\n') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    atmp.replace(shard_tensor)
    btmp.replace(shard_json)
    return {'tensor':shard_tensor.name,'tensor_sha256':sha256_file(shard_tensor),
            'records':shard_json.name,'records_sha256':sha256_file(shard_json),
            'count':len(records),'start':int(meta['start']),'end':int(meta['end'])}


def load_pair(shard_tensor,shard_json,expected,ids):
    if not shard_tensor.is_file() or not shard_json.is_file():
        raise RuntimeError('Incomplete paired shard: move partial data elsewhere and rerun')
    with safe_open(str(shard_tensor),framework='pt',device='cpu') as f:
        meta=f.metadata()
        for k,v in expected.items():
            if meta.get(k)!=v:raise RuntimeError(f'Shard provenance mismatch {k} at {shard_tensor}')
        emb=f.get_tensor('head_last')
    records=[json.loads(x) for x in shard_json.read_text(encoding='utf-8').splitlines()]
    if len(records)!=len(ids) or emb.shape!=(len(ids),2560) or not torch.isfinite(emb).all():
        raise RuntimeError('Saved shard tensor/rows corrupted')
    if [r['id'] for r in records]!=ids:
        raise RuntimeError('Saved shard order mismatch')
    return emb,records


def run(args):
    from huggingface_hub import HfApi
    import transformers
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('CUDA/BF16 unavailable; this is a real GPU experiment')
    rows,manifest=load(args.data)
    if args.pilot:
        if args.pilot not in (4,8,16):raise ValueError('pilot sample count must be 4/8/16')
        rows=[x for x in rows if x['split']=='train'][:args.pilot]
    else:
        if len(rows)!=1700:raise RuntimeError('Full H2 requires all 1700 examples')
    if len({x['id'] for x in rows})!=len(rows):raise RuntimeError('Duplicate examples')
    lock=json.loads(args.lock.read_text(encoding='utf-8'))
    spec=lock['head']
    rev=HfApi().model_info(spec['hf_id'],revision=spec['revision']).sha
    if rev!=spec['revision']:raise RuntimeError('Head revision is not pinned')
    if args.pilot and 'pilot' not in str(args.out).lower():
        raise ValueError('Pilot output must have a distinct path containing pilot')
    args.out.mkdir(parents=True,exist_ok=True)
    print('Frozen Qwen head:',spec['hf_id'],rev,flush=True)
    model=pinned_model('head',spec)
    tok=tokenizer_for(spec)
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    shard_info=[]
    all_tensors=[]
    all_records=[]
    for start in range(0,len(rows),args.shard):
        end=min(start+args.shard,len(rows))
        chunk=rows[start:end]
        shard_tensor=args.out/'shards'/f'head_{start:05d}_{end:05d}.safetensors'
        shard_json=args.out/'shards'/f'answers_{start:05d}_{end:05d}.jsonl'
        record_ids=[r['id'] for r in chunk]
        provenance={'dataset_sha256':manifest['canonical_sha256'],
                    'head_revision':rev,'start':str(start),'end':str(end),
                    'question_keys_sha256':hashlib.sha256(json.dumps(record_ids).encode()).hexdigest(),
                    'pilot':str(bool(args.pilot)).lower()}
        if shard_tensor.exists() and shard_json.exists():
            hidden,records=load_pair(shard_tensor,shard_json,provenance,record_ids)
            print('verified existing shard',start,end,flush=True)
        else:
            if shard_tensor.exists() or shard_json.exists():
                raise RuntimeError('Only one paired shard exists; do not resume stale data')
            features=[]
            records=[]
            for b in range(0,len(chunk),args.batch):
                block=chunk[b:b+args.batch]
                enc=prompt_ids(tok,[r['question'] for r in block],args.max_prompt_tokens)
                with torch.inference_mode():
                    out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
                    assert len(out.hidden_states)==33
                    feat=out.hidden_states[-1][:,-1,:].detach().to(torch.bfloat16).cpu().contiguous()
                    if not torch.isfinite(feat).all():raise RuntimeError('Non-finite Qwen hidden state')
                    features.append(feat)
                    del out
                for item in block:
                    sample=prompt_ids(tok,[item['question']],args.max_prompt_tokens)
                    torch.cuda.synchronize()
                    ts=time.perf_counter()
                    with torch.inference_mode():
                        generated=model.generate(**sample,do_sample=False,use_cache=True,
                             max_new_tokens=args.max_answer_tokens,pad_token_id=tok.pad_token_id)
                    torch.cuda.synchronize()
                    sec=time.perf_counter()-ts
                    ids=generated[0,sample['input_ids'].shape[1]:]
                    raw=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    plain=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    parsed=parse_digits(plain)
                    stopset=set()
                    for t in ('<|im_end|>','<|endoftext|>'):
                        token_id=tok.convert_tokens_to_ids(t)
                        if isinstance(token_id,int) and token_id>=0:stopset.add(token_id)
                    last=int(ids[-1]) if len(ids) else -1
                    records.append({'id':item['id'],'split':item['split'],'domain':item['domain'],
                        'language':item['language'],'group':item['group'],
                        'teacher_raw':raw,'teacher_plain':plain,'teacher_parsed':parsed,
                        'teacher_valid':parsed is not None,'teacher_correct':parsed==item['digits'],
                        'oracle':item['digits'],'completion_token_count':int(len(ids)),
                        'truncated':len(ids)>=args.max_answer_tokens and last not in stopset,
                        'gpu_generation_seconds':sec,'head_revision':rev})
                    del sample,generated,ids
                del enc
            hidden=torch.cat(features,dim=0)
            if hidden.shape!=(len(chunk),2560):raise RuntimeError('Teacher feature shape mismatch')
            save_pair(shard_tensor,shard_json,hidden,records,provenance)
            if torch.cuda.max_memory_reserved()/2**30 >= args.max_vram_gib:
                raise RuntimeError('Exceeded conservative GPU memory ceiling')
            print('saved teacher states+answers',end,'/',len(rows),flush=True)
        shard_info.append({'start':start,'end':end,
            'tensor_path':shard_tensor.relative_to(args.out).as_posix(),
            'tensor_sha256':sha256_file(shard_tensor),
            'answers_path':shard_json.relative_to(args.out).as_posix(),
            'answers_sha256':sha256_file(shard_json)})
        all_tensors.append(hidden)
        all_records+=records
    if [x['id'] for x in all_records]!=[x['id'] for x in rows]:
        raise RuntimeError('Teacher records misaligned to dataset')
    combined=torch.cat(all_tensors,dim=0)
    if combined.shape!=(len(rows),2560):raise RuntimeError('Combined teacher tensor shape mismatch')
    final_feat=args.out/'h2_head_features.safetensors'
    tmp_feat=args.out/'head_features.partial.safetensors'
    meta={'dataset_sha256':manifest['canonical_sha256'],
        'head_revision':rev,'pilot':str(bool(args.pilot)).lower(),
        'count':str(len(rows))}
    save_file({'head_last':combined},str(tmp_feat),metadata=meta)
    tmp_feat.replace(final_feat)
    final_text=args.out/'teacher_outputs.jsonl'
    tmp_text=args.out/'teacher_outputs.partial.jsonl'
    with tmp_text.open('w',encoding='utf-8',newline='\n') as f:
        for line in all_records:f.write(json.dumps(line,ensure_ascii=False,sort_keys=True)+'\n')
    tmp_text.replace(final_text)
    correct=sum(bool(x['teacher_correct']) for x in all_records)
    ntest=sum(x['split']=='test' for x in all_records)
    test_correct=sum(x['split']=='test' and x['teacher_correct'] for x in all_records)
    report={
        'status':'completed_frozen_head_pilot_no_science' if args.pilot else 'completed_frozen_head_features_and_all_outputs',
        'dataset_sha256':manifest['canonical_sha256'],
        'head_revision':rev,'count':len(rows),'pilot':bool(args.pilot),
        'all_correct':correct,'test_count':ntest,'test_correct':test_correct,
        'test_exact':test_correct/ntest if ntest else None,
        'head_feature_shape':list(combined.shape),'head_feature_dtype':str(combined.dtype),
        'features_sha256':sha256_file(final_feat),
        'teacher_outputs_sha256':sha256_file(final_text),
        'shards':shard_info,'wall_seconds':time.perf_counter()-begin,
        'max_vram_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'model':'Qwen/Qwen3.5-4B','generation':'no_thinking_greedy_12',
        'software':{'python':platform.python_version(),'torch':torch.__version__,
           'transformers':transformers.__version__,'cuda':torch.version.cuda},
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Pre-answer teacher states; no original question is sent to writer in H2. '
               'The correct oracle only enters metrics, not head prompts.',
    }
    (args.out/'feature_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','count','test_exact','features_sha256',
        'max_vram_reserved_gib']},indent=2),flush=True)
    del model,tok,combined
    gc.collect();torch.cuda.empty_cache()
    return report


def main():
    p=argparse.ArgumentParser()
    base=Path(__file__).resolve().parents[1]
    p.add_argument('--data',type=Path,default=base/'data')
    p.add_argument('--out',type=Path,default=base/'cache')
    p.add_argument('--lock',type=Path,default=base.parent/'MODEL_LOCK.json')
    p.add_argument('--pilot',type=int,default=0)
    p.add_argument('--shard',type=int,default=50)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--max-prompt-tokens',type=int,default=256)
    p.add_argument('--max-answer-tokens',type=int,default=12)
    p.add_argument('--max-vram-gib',type=float,default=22.0)
    args=p.parse_args()
    if args.max_answer_tokens!=12:
        raise RuntimeError('H2 registered teacher targets use greedy 12; alternatives are separate runs')
    run(args)

if __name__=='__main__':main()
