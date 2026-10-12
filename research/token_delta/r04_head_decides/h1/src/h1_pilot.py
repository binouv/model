"""Non-scientific H1 GPU shape/VRAM/generation pilot, not a research result."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import time

import torch
from h1_dataset import load_dataset,strict_parse_answer
from h1_extract import load_pinned_tokenizer,prompt_batch,runtime_info
from gpu_smoke import pinned_model


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--lock',type=Path,default=root.parent/'MODEL_LOCK.json')
    p.add_argument('--data',type=Path,default=root/'data')
    p.add_argument('--output',type=Path,default=root/'results'/'h1_pilot_gpu.json')
    p.add_argument('--samples',type=int,default=8)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, this is an RTX 3090 GPU pilot.')
    lock=json.loads(args.lock.read_text(encoding='utf-8'))
    rows,manifest=load_dataset(args.data)
    test=[x for x in rows if x['split']=='test']
    examples=test[:args.samples]
    if not 1<=len(examples)<=16:raise ValueError('Pilot supports 1..16 cases')
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    model=pinned_model('head',lock['head'])
    tok=load_pinned_tokenizer(lock['head'])
    inputs=prompt_batch(tok,examples,256)
    torch.cuda.synchronize()
    with torch.inference_mode():
        ans=model(**inputs,use_cache=False,output_hidden_states=True,return_dict=True)
        layers={name:list(ans.hidden_states[idx][:,-1,:].shape) for name,idx in {'layer16':16,'layer24':24,'last':32}.items()}
        decoded=model.generate(**inputs,max_new_tokens=12,do_sample=False,pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    responses=decoded[:,inputs['input_ids'].size(-1):]
    pred=[]
    for i,ex in enumerate(examples):
        raw=tok.decode(responses[i],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        guess=strict_parse_answer(raw)
        pred.append({'id':ex['id'],'oracle':ex['digits'],'raw':raw,'prediction':guess,
                     'correct':guess==ex['digits']})
    peak=torch.cuda.max_memory_reserved()/2**30
    if peak>=22:raise RuntimeError('Pilot peak reserved memory exceeds safety limit 22 GiB')
    report={'status':'passed_engineering_pilot_not_h1_test',
        'dataset_sha256':manifest['dataset_sha256_canonical'],
        'model_revision':lock['head']['revision'],'layers':layers,
        'pilot_n':len(examples),'gpu_peak_reserved_gib':round(peak,3),
        'seconds_including_load':time.perf_counter()-started,
        'runtime':runtime_info(), 'predictions':pred}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()