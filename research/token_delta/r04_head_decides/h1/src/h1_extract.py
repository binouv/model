"""R04/H1 extract PRE-ANSWER Qwen3.5-4B states on RTX 3090, no training.

The model is frozen; outputs are token-position-0 decision features ONLY from
prompt prefill, with no oracle, correct answers, teacher CoT or generated tokens
in model input. Sharded safetensors permit interruption/resume. Teacher greedy
baseline is evaluated on TEST only after extraction, never used for probes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import torch

from h1_dataset import load_dataset, strict_parse_answer

# Existing R04 tested GPU loader and assistant-start protocol live one level above /h1.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from gpu_smoke import pinned_model, safe_chat_prompt

LAYER_INDEX = {"layer16": 16, "layer24": 24, "last": 32}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_info():
    import transformers
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_total_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
    }


def prompt_batch(tokenizer, examples: list[dict], max_tokens: int):
    prompts = [safe_chat_prompt(tokenizer, e['question']) for e in examples]
    tokens = tokenizer(prompts, return_tensors='pt', padding=True,
                       add_special_tokens=False, truncation=False)
    if tokens.input_ids.shape[1] > max_tokens:
        raise ValueError(f'Prompt has {tokens.input_ids.shape[1]} tokens; max {max_tokens}')
    return {k: v.to('cuda:0') for k, v in tokens.items()}


def load_pinned_tokenizer(spec):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec['hf_id'], revision=spec['revision'],
                                        trust_remote_code=False)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def save_shard(path: Path, tensors: dict, dataset_sha: str, model_sha: str,
               start: int, end: int):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {'dataset_sha256':dataset_sha, 'model_revision':model_sha,
            'start':str(start), 'end':str(end), 'meaning':'before answer token generation'}
    tmp = path.with_suffix('.partial.safetensors')
    save_file({k:v.detach().contiguous().cpu() for k,v in tensors.items()}, str(tmp), metadata=meta)
    tmp.replace(path)
    return {'name': path.name, 'sha256': file_sha(path), 'bytes': path.stat().st_size}


def validated_shard(path: Path, dataset_sha: str, model_sha: str,
                    start: int, end: int, hidden: int):
    from safetensors import safe_open
    with safe_open(str(path), framework='pt', device='cpu') as file:
        meta=file.metadata()
        expect={'dataset_sha256':dataset_sha,'model_revision':model_sha,
                'start':str(start),'end':str(end)}
        if any(meta.get(key)!=val for key,val in expect.items()):
            raise RuntimeError(f'Stale/corrupt shard metadata: {path}')
        arrays={key:file.get_tensor(key) for key in LAYER_INDEX}
    for key,x in arrays.items():
        if x.shape!=(end-start,hidden) or not torch.isfinite(x).all():
            raise RuntimeError(f'Invalid shard tensor {path}: {key}, {x.shape}')
    return arrays


def extract(args, model, tokenizer, examples, dataset_manifest, lock):
    from safetensors.torch import save_file
    import transformers
    folder=args.out
    folder.mkdir(parents=True,exist_ok=True)
    rev=lock['head']['revision']
    dsha=dataset_manifest['dataset_sha256_canonical']
    h=lock['head']['text_hidden_size']
    if (folder/'features_manifest.json').is_file():
        receipt=json.loads((folder/'features_manifest.json').read_text(encoding='utf-8'))
        features=folder/'h1_features.safetensors'
        if receipt['dataset_sha256'] != dsha or receipt['revision'] != rev or not features.is_file() or receipt['features_sha256']!=file_sha(features):
            raise RuntimeError('Existing features mismatch. Never reuse stale data.')
        print('Already completed features verified; not recomputing.')
        return receipt
    torch.cuda.reset_peak_memory_stats()
    shards=[]
    total_tokens=0
    begin=time.perf_counter()
    count=len(examples)
    for start in range(0,count,args.shard):
        end=min(count,start+args.shard)
        shard_path=folder/'shards'/f'head_{start:05d}_{end:05d}.safetensors'
        if shard_path.exists():
            vals=validated_shard(shard_path,dsha,rev,start,end,h)
            print('verified saved shard',shard_path.name,flush=True)
        else:
            gathered={k:[] for k in LAYER_INDEX}
            for local in range(start,end,args.batch):
                part=examples[local:min(end,local+args.batch)]
                batch=prompt_batch(tokenizer,part,args.max_prompt_tokens)
                total_tokens+=int(batch['attention_mask'].sum().item())
                with torch.inference_mode():
                    outputs=model(**batch,use_cache=False,output_hidden_states=True,return_dict=True)
                    assert len(outputs.hidden_states)==lock['head']['text_num_hidden_layers']+1
                    for name,index in LAYER_INDEX.items():
                        val=outputs.hidden_states[index][:,-1,:].float().cpu()
                        gathered[name].append(val)
                del outputs,batch
            vals={name:torch.cat(v,dim=0).contiguous().to(torch.bfloat16) for name,v in gathered.items()}
            save_shard(shard_path,vals,dsha,rev,start,end)
            print(f'Saved {end}/{count} prompt-only features',flush=True)
        shards.append({'path':str(shard_path),'sha256':file_sha(shard_path),
                       'bytes':shard_path.stat().st_size,'start':start,'end':end})
    all_layers={name:[] for name in LAYER_INDEX}
    for item in shards:
        vals=validated_shard(Path(item['path']),dsha,rev,item['start'],item['end'],h)
        for key in all_layers:all_layers[key].append(vals[key])
    features={key:torch.cat(part,dim=0) for key,part in all_layers.items()}
    for key,x in features.items():
        if x.shape!=(count,h):raise RuntimeError(f'Feature shape not complete: {key} {x.shape}')
    temp=folder/'h1_features.partial.safetensors'
    complete=folder/'h1_features.safetensors'
    save_file(features,str(temp),metadata={'dataset_sha256':dsha,'revision':rev,
              'layer_indices':json.dumps(LAYER_INDEX,sort_keys=True),'pre_answer_only':'true'})
    temp.replace(complete)
    torch.cuda.synchronize()
    receipt={
        'status':'completed_pre_answer_feature_extraction_not_h1_proof',
        'dataset_sha256':dsha,'revision':rev,'layers':LAYER_INDEX,
        'features_sha256':file_sha(complete),'features_bytes':complete.stat().st_size,
        'feature_dtype':'bfloat16_preserved_from_frozen_head',
        'shape':{k:list(v.shape) for k,v in features.items()},
        'shards':shards,'computed_prompt_tokens_this_invocation':total_tokens,
        'wall_seconds':time.perf_counter()-begin,
        'peak_reserved_gib':round(torch.cuda.max_memory_reserved()/2**30,3),
        'runtime':runtime_info(),
        'note':'BF16 states are retained exactly in BF16 safetensors; extra model accuracy is not implied.',
    }
    (folder/'features_manifest.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return receipt


def baseline(args,model,tokenizer,examples,manifest,lock, role='head'):
    output=args.out/f'{role}_test_predictions.jsonl'
    receipt_path=args.out/f'{role}_test_receipt.json'
    if output.exists() or receipt_path.exists():
        if output.exists() and receipt_path.exists():
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt['dataset_sha256']==manifest['dataset_sha256_canonical'] and receipt['model_sha']==lock[role]['revision'] and receipt['predictions_file_sha256']==file_sha(output):
                print('Already verified teacher baseline',role,flush=True)
                return receipt
        raise RuntimeError('Baseline artifacts partial/mismatch; move elsewhere before rerun')
    test=[x for x in examples if x['split']=='test']
    records=[]
    elapsed=0.
    tokens_out=0
    token_hist=Counter()
    for ex in test:
        token_hist[len(tokenizer(ex['answer'],add_special_tokens=False).input_ids)]+=1
    for start in range(0,len(test),args.batch):
        part=test[start:start+args.batch]
        batch=prompt_batch(tokenizer,part,args.max_prompt_tokens)
        torch.cuda.synchronize()
        before=time.perf_counter()
        with torch.inference_mode():
            ids=model.generate(**batch,do_sample=False,use_cache=True,
                  max_new_tokens=args.max_answer_tokens,
                  pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        elapsed+=time.perf_counter()-before
        responses=ids[:,batch['input_ids'].shape[1]:]
        tokens_out+=int(responses.numel())
        for i,ex in enumerate(part):
            raw=tokenizer.decode(responses[i],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            parsed=strict_parse_answer(raw)
            records.append({'id':ex['id'],'split':'test','domain':ex['domain'],'language':ex['language'],
                 'oracle':ex['digits'],'raw':raw,'parsed':parsed,'valid_output':parsed is not None,
                 'exact':parsed==ex['digits']})
        print(f'{role} greedy baseline {min(len(test),start+args.batch)}/{len(test)}',flush=True)
        del ids,responses,batch
    tmp=output.with_suffix('.partial.jsonl')
    with tmp.open('w',encoding='utf-8',newline='\n') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    tmp.replace(output)
    denom=len(records)
    receipt={'status':'completed_greedy_baseline','role':role,'model_sha':lock[role]['revision'],
         'dataset_sha256':manifest['dataset_sha256_canonical'],'n_test':denom,
         'exact_all_test':sum(r['exact'] for r in records)/denom,
         'valid_output_rate':sum(r['valid_output'] for r in records)/denom,
         'domain_accuracy':{key:sum(r['exact'] for r in records if r['domain']==key)/sum(r['domain']==key for r in records) for key in ('code','arithmetic')},
         'lang_accuracy':{key:sum(r['exact'] for r in records if r['language']==key)/sum(r['language']==key for r in records) for key in ('en','ru')},
         'answer_token_count_histogram':dict(token_hist),'answer_has_2plus_token_rate':sum(n for token_count,n in token_hist.items() if token_count>=2)/denom,
         'generated_tokens_with_pad_including_early_stop':tokens_out,
         'cuda_generation_seconds':elapsed,
         'predictions_file_sha256':file_sha(output),
         'note':'This is one frozen model greedy decode, full output strictly parsed, test only.'}
    receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return receipt


def main():
    ap=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--lock',type=Path,default=root.parent/'MODEL_LOCK.json')
    ap.add_argument('--data',type=Path,default=root/'data')
    ap.add_argument('--out',type=Path,default=root/'cache')
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--shard',type=int,default=50)
    ap.add_argument('--max-prompt-tokens',type=int,default=256)
    ap.add_argument('--max-answer-tokens',type=int,default=12)
    ap.add_argument('--skip-teacher-baseline',action='store_true')
    ap.add_argument('--run-writer-baseline',action='store_true')
    ap.add_argument('--max-peak-gib',type=float,default=22)
    args=ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('RTX 3090 CUDA/BF16 not available; this is a GPU extraction script.')
    lock=json.loads(args.lock.read_text(encoding='utf-8'))
    examples,manifest=load_dataset(args.data)
    from huggingface_hub import HfApi
    info=HfApi().model_info(lock['head']['hf_id'],revision=lock['head']['revision'])
    if info.sha!=lock['head']['revision']:
        raise RuntimeError('Head model SHA mismatch')
    print('Load frozen pretrained head',lock['head']['hf_id'],info.sha,flush=True)
    model=pinned_model('head',lock['head'])
    tokenizer=load_pinned_tokenizer(lock['head'])
    torch.cuda.reset_peak_memory_stats()
    receipt=extract(args,model,tokenizer,examples,manifest,lock)
    print('Feature extraction',receipt['status'],'peak',receipt['peak_reserved_gib'],flush=True)
    if not args.skip_teacher_baseline:
        out=baseline(args,model,tokenizer,examples,manifest,lock,'head')
        print('Frozen HEAD test exact',out['exact_all_test'],flush=True)
    peak=torch.cuda.max_memory_reserved()/2**30
    if peak>=args.max_peak_gib:
        raise RuntimeError(f'Head exceeded safe peak {peak:.2f} >= {args.max_peak_gib:.2f} GiB')
    del model,tokenizer
    gc.collect();torch.cuda.empty_cache()
    if args.run_writer_baseline:
        info=HfApi().model_info(lock['writer']['hf_id'],revision=lock['writer']['revision'])
        if info.sha!=lock['writer']['revision']:
            raise RuntimeError('Writer revision mismatch')
        model=pinned_model('writer',lock['writer'])
        tokenizer=load_pinned_tokenizer(lock['writer'])
        out=baseline(args,model,tokenizer,examples,manifest,lock,'writer')
        print('Frozen WRITER-only test exact',out['exact_all_test'],flush=True)
        if torch.cuda.max_memory_reserved()/2**30>=args.max_peak_gib:
            raise RuntimeError('Writer exceeded VRAM ceiling')
    print('R04 H1 features prepared; H1 proof requires independent probe training and thresholds.',flush=True)


if __name__=='__main__':
    main()