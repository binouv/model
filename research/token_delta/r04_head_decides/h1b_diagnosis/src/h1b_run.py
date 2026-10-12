"""GPU H1b diagnostic on NEW (non-H1) questions; frozen Qwen3.5-4B only.

Four predefined arms reveal formatting errors, generation budget, extra
serial reasoning and one-step capabilities. Never re-labels original H1.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

import torch

THIS=Path(__file__).resolve()
R04=THIS.parents[2]
H1=R04/'h1'/'src'
sys.path.insert(0,str(H1))
sys.path.insert(0,str(R04/'src'))
from h1_extract import load_pinned_tokenizer  # noqa: E402
from gpu_smoke import pinned_model, safe_chat_prompt  # noqa: E402
from h1_dataset import strict_parse_answer  # noqa: E402
from h1b_dataset import load_dataset,one_step_question  # noqa: E402

ARMS={
    'no_think_12': {'thinking':False,'max_tokens':12,'do_sample':False,'one_step':False},
    'no_think_96': {'thinking':False,'max_tokens':96,'do_sample':False,'one_step':False},
    'thinking_384':{'thinking':True,'max_tokens':384,'do_sample':True,'one_step':False},
    'one_step_32': {'thinking':False,'max_tokens':32,'do_sample':False,'one_step':True},
}
RE_PAIR_END=re.compile(r'(?<!\d)([0-9])\s+([0-9])(?!\d)\s*[.!。]*\s*$', re.DOTALL)
RE_EXPLICIT_PQ=re.compile(r'(?:p|р)\s*=\s*([0-9])\s*[,;\s]+(?:q|к)\s*=\s*([0-9])\s*[.!。]*\s*$',re.IGNORECASE)
RE_SINGLE_END=re.compile(r'(?<!\d)([0-9])(?!\d)\s*[.!。]*\s*$',re.DOTALL)


def strip_stop_tokens(raw:str)->str:
    s=raw
    for stop in ('<|im_end|>','<|endoftext|>','<|fim_suffix|>'):
        s=s.replace(stop,'')
    return s.strip()


def final_answer_segment(raw:str,thinking:bool)->tuple[str,str]:
    """Never score digits found only inside an incomplete thinking trace."""
    source=strip_stop_tokens(raw)
    if thinking:
        if '</think>' in source:
            return source.rsplit('</think>',1)[1].strip(),'thinking_closed'
        return '', 'no_thinking_close_marker'
    if '<think>' in source or '</think>' in source:
        return '', 'unexpected_thinking_marker'
    return source,'nonthinking'


def parse_final(raw:str,thinking:bool,one_step:bool=False)->tuple[list[int]|None,str]:
    segment,status=final_answer_segment(raw,thinking)
    if not segment:return None,status
    # Strict final-only parser: never fish numbers out of a whole CoT or code block.
    if one_step:
        m=RE_SINGLE_END.search(segment)
        return ([int(m.group(1))] if m else None),status
    for pat in (RE_PAIR_END,RE_EXPLICIT_PQ):
        m=pat.search(segment)
        if m:return [int(m.group(1)),int(m.group(2))],status
    return None,status


def hash_file(path:Path)->str:
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            sha.update(block)
    return sha.hexdigest()


def score_rows(records:list[dict],length:int)->dict:
    if len(records)!=length*len(ARMS):
        raise ValueError('Require full matrix of examples/arms')
    aggregates={}
    for arm in ARMS:
        selected=[r for r in records if r['arm']==arm]
        if len(selected)!=length or len({r['id'] for r in selected})!=length:
            raise ValueError('Duplicate or missing diagnostic examples')
        by_group={}
        for category in ('domain','language','operator'):
            by_group[category]={}
            for value in sorted(set(r[category] for r in selected)):
                rows=[r for r in selected if r[category]==value]
                by_group[category][value]={
                    'n':len(rows),
                    'semantic_exact':sum(r['semantic_exact'] for r in rows)/len(rows),
                    'strict_exact':sum(r['strict_exact'] for r in rows)/len(rows),
                    'valid_semantic_output':sum(r['parsed'] is not None for r in rows)/len(rows),
                }
        aggregates[arm]={
            'n':length,
            'semantic_exact_count':sum(r['semantic_exact'] for r in selected),
            'semantic_exact':sum(r['semantic_exact'] for r in selected)/length,
            'strict_exact':sum(r['strict_exact'] for r in selected)/length,
            'valid_semantic_output':sum(r['parsed'] is not None for r in selected)/length,
            'truncated_rate':sum(r['truncated'] for r in selected)/length,
            'mean_generated_tokens':sum(r['generated_tokens'] for r in selected)/length,
            'mean_cuda_generation_s':sum(r['cuda_seconds'] for r in selected)/length,
            'by_group':by_group,
        }
        if arm!='one_step_32':
            aggregates[arm]['digit0']=sum(r['digit0_correct'] for r in selected)/length
            aggregates[arm]['digit1']=sum(r['digit1_correct'] for r in selected)/length
    a=aggregates['no_think_12']['semantic_exact']
    b=aggregates['no_think_96']['semantic_exact']
    c=aggregates['thinking_384']['semantic_exact']
    return {
       'arms':aggregates,
       'diagnostic_gates_predeclared':{
          'format_error_major_contributor': (
              aggregates['no_think_12']['semantic_exact']-aggregates['no_think_12']['strict_exact'])>=.20,
          'extra_non_thinking_tokens_help': b-a>=.20,
          'serial_thinking_help': c-b>=.25 and c>=.50,
          'thinking_mode_reaches_80pct': c>=.80,
          'thinking_mode_still_below_50pct': c<.50,
       },
       'not_a_replacement_for_original_h1':True,
    }


def build_conditions():
    return list(ARMS.items())


def load_partial(out:Path, expected_sha:str, max_items:int)->dict:
    path=out/'diagnosis.partial.jsonl'
    if not path.exists():return {}
    rows=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    found={}
    for row in rows:
        if row['dataset_sha256']!=expected_sha or row['n_planned']!=max_items:
            raise ValueError('Cannot reuse incomplete run with another dataset/subset')
        key=(row['id'],row['arm'])
        if key in found:raise ValueError(f'Duplicate partial record {key}')
        found[key]=row
    return found


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('CUDA + BF16 required; no CPU pretend-teacher inference')
    data,manifest=load_dataset(args.data)
    if args.max_items not in (16,128):
        raise SystemExit('--max-items must be 16 (pilot) or 128 (full registered diagnostic)')
    if args.max_items==16:
        subset=[]
        for domain in ('arithmetic','code'):
            for language in ('en','ru'):
                subset.extend([r for r in data if r['domain']==domain and r['language']==language][:4])
    else:
        subset=data
    assert len(subset)==args.max_items
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'diagnosis_metrics.json').exists() or (args.out/'diagnosis_predictions.jsonl').exists():
        raise FileExistsError('A completed result already exists in --out; use a new output directory')
    from huggingface_hub import HfApi
    lock=json.loads(args.lock.read_text(encoding='utf-8'))
    spec=lock['head']
    resolved=HfApi().model_info(spec['hf_id'],revision=spec['revision']).sha
    if resolved!=spec['revision']:raise ValueError('Unpinned head weights')
    print('Frozen Qwen head',spec['hf_id'],resolved,flush=True)
    torch.cuda.reset_peak_memory_stats()
    model=pinned_model('head',spec)
    tok=load_pinned_tokenizer(spec)
    already=load_partial(args.out,manifest['new_dataset_sha256'],args.max_items)
    partial=args.out/'diagnosis.partial.jsonl'
    t0=time.perf_counter()
    with partial.open('a',encoding='utf-8',newline='\n') as writer:
      for k,ex in enumerate(subset):
        for arm,cfg in build_conditions():
          if (ex['id'],arm) in already:continue
          prompt_text=one_step_question(ex) if cfg['one_step'] else ex['question']
          encoded=tok.apply_chat_template(
              [{'role':'user','content':prompt_text}],tokenize=False,
              add_generation_prompt=True,enable_thinking=cfg['thinking'])
          ids=tok(encoded,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0')
          if ids.shape[1]>256:raise RuntimeError('Diagnostic prompt over 256 token limit')
          torch.manual_seed(20261013+k)
          torch.cuda.manual_seed_all(20261013+k)
          opts={
             'do_sample':cfg['do_sample'],
             'max_new_tokens':cfg['max_tokens'],
             'use_cache':True,
             'pad_token_id':tok.pad_token_id,
          }
          if cfg['do_sample']:
              opts.update(temperature=.6,top_p=.95,top_k=20)
          torch.cuda.synchronize();start=time.perf_counter()
          with torch.inference_mode():
              generated=model.generate(input_ids=ids,**opts)
          torch.cuda.synchronize();elapsed=time.perf_counter()-start
          if torch.cuda.max_memory_reserved()/2**30>=22:
              raise RuntimeError('H1b peak reserved VRAM >=22GiB, diagnostic stopped without a completion receipt')
          reply_tokens=generated[0,ids.shape[1]:]
          raw=tok.decode(reply_tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
          plain=tok.decode(reply_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
          expected=ex['digits'][:1] if cfg['one_step'] else ex['digits']
          parsed,status=parse_final(raw,cfg['thinking'],cfg['one_step'])
          strict=(None if cfg['thinking'] or cfg['one_step'] else strict_parse_answer(plain))
          # For thinking, strict full original H1 is undefined; do not redefine it.
          exact=parsed==expected
          last_id=int(reply_tokens[-1]) if len(reply_tokens) else None
          stop_ids=set()
          for token in ('<|im_end|>','<|endoftext|>'):
              v=tok.convert_tokens_to_ids(token)
              if isinstance(v,int) and v>=0:stop_ids.add(v)
          if isinstance(model.generation_config.eos_token_id,list):
              stop_ids.update(int(i) for i in model.generation_config.eos_token_id)
          elif model.generation_config.eos_token_id is not None:
              stop_ids.add(int(model.generation_config.eos_token_id))
          truncated=len(reply_tokens)>=cfg['max_tokens'] and last_id not in stop_ids
          row={
            'id':ex['id'],'arm':arm,'domain':ex['domain'],'language':ex['language'],
            'operator':ex['operator'],
            'dataset_sha256':manifest['new_dataset_sha256'],'n_planned':args.max_items,
            'oracle':expected,'parsed':parsed,'parse_status':status,
            'strict_parsed':strict,'strict_exact':strict==expected if strict is not None else False,
            'semantic_exact':exact,
            'digit0_correct':int(parsed is not None and parsed[0]==expected[0]),
            'digit1_correct':int(not cfg['one_step'] and parsed is not None and parsed[1]==expected[1]),
            'raw':raw,'plain':plain,'generated_tokens':len(reply_tokens),
            'truncated':truncated,'cuda_seconds':elapsed,
            'model_sha':resolved,
          }
          writer.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
          writer.flush()
          already[(ex['id'],arm)]=row
          del ids,generated,reply_tokens
        if (k+1)%8==0:print('H1b diagnostic',k+1,'/',len(subset),'four arms',flush=True)
    keys={(e['id'],name) for e in subset for name in ARMS}
    if set(already)!=keys:
        raise RuntimeError('Diagnostic incomplete, partial file retained; no completed receipt')
    ordered=[already[(e['id'],name)] for e in subset for name in ARMS]
    scores=score_rows(ordered,len(subset))
    whole=args.out/'diagnosis_predictions.jsonl'
    os.replace(partial,whole)
    metrics={
       'run_status':'completed_h1b_diagnostic_not_original_h1',
       'test_population':'fresh cases disjoint from all 1700 original H1 cases',
       'n_cases':len(subset), 'pilot_only':len(subset)<128,
       'teacher':'Qwen3.5-4B','model_revision':resolved,
       'dataset_sha256':manifest['new_dataset_sha256'],
       'original_h1_dataset_sha256':manifest['old_dataset_sha256'],
       'strict_h1_gates_unchanged':True,
       'generation_config':ARMS,
       'accuracy':scores,
       'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
       'wall_seconds':time.perf_counter()-t0,
       'timestamp_utc':datetime.now(timezone.utc).isoformat(),
       'software':{'python':platform.python_version(),'torch':torch.__version__},
       'predictions_sha256':hash_file(whole),
    }
    (args.out/'diagnosis_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({arm:v['semantic_exact'] for arm,v in scores['arms'].items()},indent=2))
    print('H1b diagnostic complete, not a change to original H1 status')


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--data',type=Path,default=root/'data')
    parser.add_argument('--out',type=Path,default=root/'results'/'pilot')
    parser.add_argument('--lock',type=Path,default=root.parent/'MODEL_LOCK.json')
    parser.add_argument('--max-items',type=int,default=16)
    args=parser.parse_args();run(args)

if __name__=='__main__':main()