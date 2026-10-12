"""R04-H2: frozen writer with trainable soft-prefix bridge; Qwen head frozen.

Original H2 quality is judged against BOTH the teacher's answers and oracle.
No test-only correct teacher examples are included in training. Oracle source
is an explicit separate upper-bound control, not a realistic head input.
This code does NOT benchmark full-system speed (H4).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from h2_dataset import load,sha256_file

R04=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R04/'src'))
from latent_bridge import LatentBridge  # noqa: E402

HEAD_REV='c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97'
WRITER_REV='23eda19ab0239bc12350efba411911b8eb89f486'
TRAIN_STEPS=400
BATCH_SIZE=4
LEARNING_RATE=0.0003
PRIMARY_SEEDS=(11,22,33)
MAX_NEW_TOKENS=12
SPLIT_BYTES=80_000_000


def lexical_features(rows,dims=2560):
    """Only question surface text, never oracle or head response."""
    out=torch.zeros(len(rows),dims,dtype=torch.float32)
    pat=r'[a-zA-Zа-яА-ЯёЁ]+|\d+|[%+*()=]'
    for i,item in enumerate(rows):
        t=re.findall(pat,item['question'].lower())
        for n in (1,2,3):
            for j in range(len(t)-n+1):
                token=f'{n}:'+':'.join(t[j:j+n])
                h=hashlib.blake2b(token.encode('utf-8'),digest_size=8,person=b'R04-H2--').digest()
                index=int.from_bytes(h[:4],'little')%dims
                out[i,index]+=1 if (h[4]&1) else -1
    return F.normalize(out,dim=-1)*math.sqrt(dims)


def oracle_feature(rows,dim=2560):
    """Explicitly privileged features; not available to a realistic head."""
    x=torch.zeros(len(rows),dim,dtype=torch.float32)
    for i,row in enumerate(rows):
        p,q=row['digits']
        x[i,p]=1.0
        x[i,10+q]=1.0
    return x


def parse_digits(text):
    m=re.fullmatch(r'\s*([0-9])\s+([0-9])\s*',text,re.DOTALL)
    return [int(m.group(1)),int(m.group(2))] if m else None


def read_cache(out:Path, rows, dataset_sha:str, pilot:bool):
    receipt=json.loads((out/'feature_manifest.json').read_text(encoding='utf-8'))
    source=out/'h2_head_features.safetensors'
    teachers=out/'teacher_outputs.jsonl'
    if receipt['dataset_sha256']!=dataset_sha or receipt['head_revision']!=HEAD_REV:
        raise RuntimeError('Teacher cache provenance mismatch')
    if bool(receipt['pilot'])!=pilot:
        raise RuntimeError('Pilot/full cache confusion')
    if sha256_file(source)!=receipt['features_sha256'] or sha256_file(teachers)!=receipt['teacher_outputs_sha256']:
        raise RuntimeError('Teacher cache SHA mismatch')
    with safe_open(str(source),framework='pt',device='cpu') as f:
        metadata=f.metadata()
        if metadata.get('dataset_sha256')!=dataset_sha or metadata.get('head_revision')!=HEAD_REV:
            raise RuntimeError('Teacher tensor metadata mismatch')
        state=f.get_tensor('head_last').float()
    records=[json.loads(x) for x in teachers.read_text(encoding='utf-8').splitlines()]
    if state.shape!=(len(records),2560) or not torch.isfinite(state).all():
        raise RuntimeError('Teacher states corrupt')
    by_id={r['id']:r for r in rows}
    if len(records)!=len({r['id'] for r in records}):raise RuntimeError('Duplicate teacher IDs')
    for record in records:
        ref=by_id[record['id']]
        if record['oracle']!=ref['digits'] or record['split']!=ref['split']:
            raise RuntimeError('Teacher outputs misaligned to oracle data')
    if not pilot and len(records)!=1700:raise RuntimeError('Complete scientific H2 requires 1700 teacher outputs')
    if pilot and any(r['split']!='train' for r in records):raise RuntimeError('Pilot test data leakage')
    selected=[by_id[r['id']] for r in records]
    return selected,records,state,receipt


def writer_targets(tok, records):
    """All train labels are teacher-produced text, including wrong answers.

    The eos marker is appended separately; gold oracle is NEVER used here.
    """
    end_id=tok.convert_tokens_to_ids('<|im_end|>')
    if not isinstance(end_id,int) or end_id<0:
        raise RuntimeError('Writer tokenizer missing required im_end token')
    sequences=[]
    for r in records:
        raw=r['teacher_plain']
        ids=tok.encode(raw,add_special_tokens=False)
        if len(ids)>MAX_NEW_TOKENS+2:
            raise RuntimeError('Teacher target unexpectedly long')
        sequences.append(ids+[end_id])
    return sequences


def make_teacher_forcing_batch(soft_plan, start_ids, target_seqs, embedding):
    """Causal alignment: logits at K+header-1 predict first target.

    Input includes the target embeddings for teacher forcing. No target is
    present at inference. Right padding is masked and ignored in CE.
    """
    if soft_plan.ndim!=3 or start_ids.ndim!=2:raise ValueError('Shapes')
    batch,k,hidden=soft_plan.shape
    if batch!=len(target_seqs) or start_ids.shape[0]!=batch:raise ValueError('Batch size')
    header=embedding(start_ids)
    length=max(len(x) for x in target_seqs)
    pad_id=0
    tensors=torch.full((batch,length),pad_id,device=soft_plan.device,dtype=torch.long)
    mask=torch.zeros(batch,length,device=soft_plan.device,dtype=torch.bool)
    for i,seq in enumerate(target_seqs):
        if not seq:raise ValueError('Empty target')
        tensors[i,:len(seq)]=torch.tensor(seq,device=soft_plan.device,dtype=torch.long)
        mask[i,:len(seq)]=True
    # Ensure padded target embeds are harmless; cross-entropy masks loss too.
    embeddings=torch.cat((soft_plan.to(header.dtype),header,embedding(tensors)),dim=1)
    attention=torch.cat((torch.ones(batch,k+header.shape[1],device=soft_plan.device,dtype=torch.long),
                          mask.to(torch.long)),dim=1)
    gold=tensors.masked_fill(~mask,-100)
    return embeddings,attention,gold


def shifted_ce(logits,gold,k:int,header_len:int):
    start=k+header_len-1
    selected=logits[:,start:start+gold.shape[1],:].float()
    if selected.shape[:2]!=gold.shape:raise RuntimeError('Incorrect causal alignment')
    return F.cross_entropy(selected.reshape(-1,selected.shape[-1]),gold.reshape(-1),ignore_index=-100)


def write_checkpoint(directory:Path, filename:str, bundle:dict):
    """Save actual bytes; split >80MB for normal Git publication (<90MB).

    Validate SHA and replay assembled bytes before original staging is removed.
    """
    directory.mkdir(parents=True,exist_ok=True)
    stage=directory/(filename+'.staging')
    torch.save(bundle,stage)
    full_sha=sha256_file(stage)
    size=stage.stat().st_size
    manifest={'name':filename,'bytes':size,'sha256':full_sha,'parts':[]}
    if size<=SPLIT_BYTES:
        final=directory/filename
        stage.replace(final)
        manifest['parts'].append({'name':filename,'bytes':size,'sha256':full_sha})
    else:
        with stage.open('rb') as f:
            i=0
            while True:
                chunk=f.read(SPLIT_BYTES)
                if not chunk:break
                part=directory/(filename+f'.part{i:03d}')
                part.write_bytes(chunk)
                manifest['parts'].append({'name':part.name,'bytes':len(chunk),
                       'sha256':hashlib.sha256(chunk).hexdigest()})
                i+=1
        if sum(x['bytes'] for x in manifest['parts'])!=size:
            raise RuntimeError('Checkpoint part counts failed')
        h=hashlib.sha256()
        for p in manifest['parts']:
            full=directory/p['name']
            if sha256_file(full)!=p['sha256']:raise RuntimeError('Part corruption')
            with full.open('rb') as inp:
                for block in iter(lambda:inp.read(2**20),b''):h.update(block)
        if h.hexdigest()!=full_sha:raise RuntimeError('Reassembly checksum mismatch')
        stage.unlink()
    manifest_path=directory/(filename+'.manifest.json')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def restore_checkpoint(directory:Path, manifest:dict):
    import io
    whole=b''
    # H2 checkpoints are small; a strict byte-exact join proves the full state.
    for item in manifest['parts']:
        part=directory/item['name']
        if part.stat().st_size!=item['bytes'] or sha256_file(part)!=item['sha256']:
            raise RuntimeError('Missing/corrupted checkpoint part')
        whole+=part.read_bytes()
    if len(whole)!=manifest['bytes'] or hashlib.sha256(whole).hexdigest()!=manifest['sha256']:
        raise RuntimeError('Checkpoint reconstruction SHA failed')
    return torch.load(io.BytesIO(whole),map_location='cpu',weights_only=False)


def plan_prefix(model, source, idx, feats, dev):
    return model(feats[idx].to(dev))


def generate_writer(writer,tok,plan,constant_start,max_tokens=MAX_NEW_TOKENS,mode='hf'):
    """No user prompt or style ID. Full-prefix decode is quality reference.

    'hf' uses efficient generation with inputs_embeds; first three outputs may
    be compared with the reference decoder by the user before a scientific run.
    """
    wembed=writer.get_input_embeddings()
    batch=len(plan)
    if batch!=1:raise ValueError('Evaluation generates one example at a time')
    header=wembed(constant_start)
    input_embeds=torch.cat((plan.to(header.dtype),header),dim=1)
    stop_ids=[]
    for stop in ('<|im_end|>','<|endoftext|>'):
        n=tok.convert_tokens_to_ids(stop)
        if isinstance(n,int) and n>=0:stop_ids.append(n)
    if not stop_ids:raise RuntimeError('Tokenizer lacks stop markers')
    if mode=='hf':
        with torch.inference_mode():
            returned=writer.generate(inputs_embeds=input_embeds,
                attention_mask=torch.ones(input_embeds.shape[:2],dtype=torch.long,device=input_embeds.device),
                max_new_tokens=max_tokens,do_sample=False,use_cache=True,
                eos_token_id=stop_ids,pad_token_id=tok.pad_token_id)
        predicted=returned[0].tolist()
        if not 1<=len(predicted)<=max_tokens+1:
            raise RuntimeError(f'Unexpected HF input-embeds generated token count: {len(predicted)}')
        # Transformers may prepend one dummy BOS for inputs_embeds. A verification
        # versus manual reference is mandatory in pilot; do not silently strip it.
        if len(predicted)>max_tokens:
            raise RuntimeError('HF returned an extra token; use --decode-mode manual and inspect')
    elif mode=='manual':
        pred=[]
        with torch.inference_mode():
            for step in range(max_tokens):
                out=writer(inputs_embeds=input_embeds,use_cache=False,return_dict=True)
                nxt=int(out.logits[0,-1,:].argmax(-1))
                pred.append(nxt)
                if nxt in stop_ids:break
                prev=torch.tensor([[nxt]],device=input_embeds.device)
                input_embeds=torch.cat((input_embeds,wembed(prev)),dim=1)
        predicted=pred
    else:raise ValueError(mode)
    if any(x in stop_ids for x in predicted):
        predicted=predicted[:next(i for i,x in enumerate(predicted) if x in stop_ids)+1]
    raw=tok.decode(predicted,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    plain=tok.decode(predicted,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return raw,plain,predicted


@torch.inference_mode()
def standalone_writer_baseline(writer,tok,rows,max_tokens=MAX_NEW_TOKENS):
    """Strong independent 0.8B question-conditioned baseline; never used to train bridge."""
    records=[]
    for item in rows:
        text=tok.apply_chat_template([{'role':'user','content':item['question']}],
             tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(text,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda:0')
        generated=writer.generate(input_ids=ids,do_sample=False,use_cache=True,
                  max_new_tokens=max_tokens,pad_token_id=tok.pad_token_id)
        tail=generated[0,ids.size(1):]
        raw=tok.decode(tail,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        plain=tok.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        parsed=parse_digits(plain)
        records.append({'id':item['id'],'oracle':item['digits'],
                        'raw':raw,'plain':plain,'parsed':parsed,
                        'exact':parsed==item['digits'],
                        'domain':item['domain'],'language':item['language'],
                        'generated_tokens':len(tail)})
    return records


def score_predictions(rows,records,predictions):
    from collections import defaultdict
    if len(rows)!=len(predictions):raise RuntimeError('Missing outputs')
    exact=[parse_digits(p['writer_plain'])==r['digits'] for p,r in zip(predictions,rows)]
    mimic=[p['writer_plain'].strip()==t['teacher_plain'].strip() for p,t in zip(predictions,records)]
    correct=[bool(r['teacher_correct']) for r in records]
    n=len(rows)
    ncorrect=sum(correct)
    counts={}
    for label in ('arithmetic','code','en','ru'):
        indices=[i for i,r in enumerate(rows) if r['domain']==label or r['language']==label]
        counts[label]={'n':len(indices),'writer_oracle_exact':sum(exact[i] for i in indices)/len(indices) if indices else None}
    return {'n':n,'writer_oracle_count':sum(exact), 'writer_oracle_exact':sum(exact)/n,
       'teacher_oracle_count':ncorrect,'teacher_oracle_exact':ncorrect/n,
       'teacher_text_mimic':sum(mimic)/n,
       'teacher_correct_subset_n':ncorrect,
       'writer_accuracy_on_teacher_correct':sum(exact[i] for i in range(n) if correct[i])/ncorrect if ncorrect else None,
       'writer_accuracy_on_teacher_wrong':sum(exact[i] for i in range(n) if not correct[i])/(n-ncorrect) if ncorrect<n else None,
       'strata':counts}


def main():
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from huggingface_hub import HfApi
    p=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--data',type=Path,default=root/'data')
    p.add_argument('--cache',type=Path,default=root/'cache')
    p.add_argument('--out',type=Path,default=root/'results'/'full')
    p.add_argument('--lock',type=Path,default=root.parent/'MODEL_LOCK.json')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--steps',type=int,default=TRAIN_STEPS)
    p.add_argument('--batch',type=int,default=BATCH_SIZE)
    p.add_argument('--lr',type=float,default=LEARNING_RATE)
    p.add_argument('--decode-mode',choices=('hf','manual'),default='hf')
    p.add_argument('--max-vram-gib',type=float,default=22.)
    args=p.parse_args()
    if (args.steps,args.batch,args.lr)!=(TRAIN_STEPS,BATCH_SIZE,LEARNING_RATE):
        if not args.pilot:raise RuntimeError('Scientific H2 requires fixed 400 steps/batch4/lr=0.0003')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('Real frozen writer CUDA/BF16 required')
    if args.pilot:
        if 'pilot' not in str(args.out).lower():raise RuntimeError('Pilot output must include pilot')
    else:
        if 'pilot' in str(args.out).lower():raise RuntimeError('Full H2 must not write to pilot')
    rows,ds=load(args.data)
    selected,records,headstates,teacher_receipt=read_cache(args.cache,rows,ds['canonical_sha256'],args.pilot)
    if args.pilot and len(selected) not in (4,8,16):raise RuntimeError('Pilot cache expected 4/8/16 train samples')
    if not args.pilot and len(selected)!=1700:raise RuntimeError('Missing full teacher dataset')
    lock=json.loads(args.lock.read_text(encoding='utf-8'))
    writer_spec=lock['writer']
    assert writer_spec['revision']==WRITER_REV
    if HfApi().model_info(writer_spec['hf_id'],revision=WRITER_REV).sha!=WRITER_REV:
        raise RuntimeError('Writer revision mismatch')
    print('Load frozen 0.8B writer',flush=True)
    writer=AutoModelForCausalLM.from_pretrained(writer_spec['hf_id'],revision=WRITER_REV,
             dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',
             trust_remote_code=False,use_safetensors=True).eval()
    writer.requires_grad_(False)
    if any(x.requires_grad for x in writer.parameters()):raise RuntimeError('Writer not frozen')
    tok=AutoTokenizer.from_pretrained(writer_spec['hf_id'],revision=WRITER_REV,trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    fixed_header='<|im_start|>assistant\n'
    header_ids=tok(fixed_header,add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0')
    if header_ids.shape[1]<1:raise RuntimeError('Missing constant assistant start marker')
    teacher_seqs=writer_targets(tok,records)
    feats={'head':headstates,'hash':lexical_features(selected),'oracle':oracle_feature(selected)}
    # Before any scientific run, independently verify that HF inputs_embeds
    # generation is identical to manual greedy full-prefix forward on this
    # exact frozen model. Avoid claiming meaningful quality if token alignment
    # differs across Transformers versions.
    if args.decode_mode=='hf':
        torch.manual_seed(42)
        check_bridge=LatentBridge(head_size=2560,writer_size=1024,plan_tokens=4,
                                  inner_size=512).to('cuda:0').eval()
        with torch.inference_mode():
            pfx=check_bridge(feats['head'][:1].to('cuda:0'))
            manual=generate_writer(writer,tok,pfx,header_ids,max_tokens=MAX_NEW_TOKENS,mode='manual')
            cached=generate_writer(writer,tok,pfx,header_ids,max_tokens=MAX_NEW_TOKENS,mode='hf')
        if manual[2]!=cached[2]:
            raise RuntimeError('HF inputs_embeds generation differs from manual causal decode. '
                               'Do not report H2 results; rerun with --decode-mode manual.')
        del check_bridge,pfx
        print('Verified HF cached token generation equals manual prefix recomputation',flush=True)
    idx={s:[i for i,r in enumerate(selected) if r['split']==s] for s in ('train','val','test')}
    if not args.pilot and tuple(len(idx[k]) for k in ('train','val','test'))!=(1200,200,300):
        raise RuntimeError('Split lengths do not match')
    if args.pilot and (idx['val'] or idx['test']):raise RuntimeError('Pilot contains evaluation/test samples')
    torch.cuda.reset_peak_memory_stats()
    args.out.mkdir(parents=True,exist_ok=True)
    registry=[('head',4,s) for s in PRIMARY_SEEDS]+[('hash',4,s) for s in PRIMARY_SEEDS]
    registry += [('head',k,11) for k in (1,8,16)]
    registry += [('oracle',4,s) for s in PRIMARY_SEEDS]
    if args.pilot:registry=[('head',4,11),('hash',4,11),('oracle',4,11)]
    all_reports=[]
    all_predictions=[]
    run_receipts=args.out/'run_receipts'
    run_receipts.mkdir(exist_ok=True)
    for source,k,seed in registry:
        label=f'{source}_K{k}_seed{seed}'
        perrun=run_receipts/(label+'.json')
        perpred=run_receipts/(label+'.predictions.jsonl')
        if perrun.exists() or perpred.exists():
            if not perrun.is_file() or not perpred.is_file():
                raise RuntimeError('Partial H2 run receipt: inspect and quarantine before resuming '+label)
            recorded=json.loads(perrun.read_text(encoding='utf-8'))
            if (recorded['label']!=label or recorded['dataset_sha256']!=ds['canonical_sha256']
                or recorded['teacher_features_sha256']!=teacher_receipt['features_sha256']
                or recorded['predictions_sha256']!=sha256_file(perpred)
                or recorded['pilot']!=args.pilot):
                raise RuntimeError('Stale/corrupt H2 partial completed run '+label)
            restore_checkpoint(args.out/'checkpoints',recorded['report']['checkpoint'])
            old_predictions=[json.loads(x) for x in perpred.read_text(encoding='utf-8').splitlines()]
            all_reports.append(recorded['report'])
            all_predictions.extend(old_predictions)
            print('Verified completed H2 run receipt, skipping retraining',label,flush=True)
            continue
        print('H2 train',label,flush=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        bridge=LatentBridge(head_size=2560,writer_size=1024,
                            plan_tokens=k,inner_size=512).to('cuda:0')
        optimizer=torch.optim.AdamW(bridge.parameters(),lr=args.lr,weight_decay=.01)
        generator=torch.Generator().manual_seed(seed+712)
        nsteps=3 if args.pilot else args.steps
        t0=time.perf_counter()
        losses=[]
        for step in range(nsteps):
            ids=torch.tensor(idx['train'],dtype=torch.long)[torch.randint(len(idx['train']),
                          (args.batch,),generator=generator)].tolist()
            bridge.train()
            z=bridge(feats[source][ids].to('cuda:0'))
            batch_header=header_ids.repeat(len(ids),1)
            if source=='oracle':
                sequences=[tok.encode(selected[i]['answer'],add_special_tokens=False)+
                            [tok.convert_tokens_to_ids('<|im_end|>')] for i in ids]
            else:
                sequences=[teacher_seqs[i] for i in ids]
            embeddings,mask,gold=make_teacher_forcing_batch(z,batch_header,sequences,
                                                            writer.get_input_embeddings())
            output=writer(inputs_embeds=embeddings,attention_mask=mask,use_cache=False,
                          return_dict=True)
            loss=shifted_ce(output.logits,gold,k,batch_header.size(1))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in writer.parameters()):raise RuntimeError('Writer accumulated gradients')
            valid=sum(p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.parameters())
            if valid==0:raise RuntimeError('Bridge got no gradients')
            nn.utils.clip_grad_norm_(bridge.parameters(),1.0)
            optimizer.step()
            if (step+1)%max(1,nsteps//4)==0:
                losses.append([step+1,float(loss.detach().cpu())])
            del z,output,embeddings,mask,gold,loss
        torch.cuda.synchronize()
        duration=time.perf_counter()-t0
        bridge.eval()
        out_rows=[]
        eval_indices=idx['train'][:8] if args.pilot else idx['test']
        with torch.inference_mode():
            for count,i in enumerate(eval_indices):
                z=bridge(feats[source][i:i+1].to('cuda:0'))
                raw,plain,pred=generate_writer(writer,tok,z,header_ids,max_tokens=MAX_NEW_TOKENS,
                                              mode=args.decode_mode)
                out_rows.append({'id':selected[i]['id'],'source':source,'K':k,'seed':seed,
                       'oracle':selected[i]['digits'],'teacher_plain':records[i]['teacher_plain'],
                       'teacher_correct':records[i]['teacher_correct'],'writer_raw':raw,
                       'writer_plain':plain,'writer_parsed':parse_digits(plain),
                       'generated_tokens':pred,'target_uses_oracle':source=='oracle'})
                if (count+1)%50==0:print(label,'generated',count+1,'/',len(eval_indices),flush=True)
        eval_r=[selected[i] for i in eval_indices]
        eval_t=[records[i] for i in eval_indices]
        metrics=score_predictions(eval_r,eval_t,out_rows)
        # INTERVENTIONS on identical fixed eval population, no train/test leakage.
        # Full run: 300 shuffled/zero/contradictory samples may be time-consuming,
        # but all are mandatory for H2 success claims.
        zero_predictions=[]
        shuffled_predictions=[]
        contradictory_predictions=[]
        perm=torch.randperm(len(eval_indices),generator=torch.Generator().manual_seed(seed+49)).tolist()
        with torch.inference_mode():
            for j,i in enumerate(eval_indices):
                x=feats[source][i:i+1].to('cuda:0')
                z=bridge(x)
                controls={'zero':torch.zeros_like(z),
                    'shuffle':bridge(feats[source][eval_indices[perm[j]]:eval_indices[perm[j]]+1].to('cuda:0')),
                    'contradictory':-z}
                for key,plan in controls.items():
                    _,plain,_=generate_writer(writer,tok,plan,header_ids,MAX_NEW_TOKENS,
                                              mode=args.decode_mode)
                    target=(zero_predictions if key=='zero' else
                            shuffled_predictions if key=='shuffle' else contradictory_predictions)
                    target.append({'id':selected[i]['id'],'source':source,'K':k,'seed':seed,
                       'control':key,'oracle':selected[i]['digits'],'teacher_plain':records[i]['teacher_plain'],
                       'teacher_correct':records[i]['teacher_correct'],
                       'writer_plain':plain,'writer_parsed':parse_digits(plain)})
        def acc(v):return sum(x['writer_parsed']==x['oracle'] for x in v)/len(v)
        metrics['controls']={'zero_oracle_exact':acc(zero_predictions),
                             'shuffled_oracle_exact':acc(shuffled_predictions),
                             'negated_latent_oracle_exact':acc(contradictory_predictions)}
        # Save full weight bytes and optimizer/RNG as a resumable native checkpoint.
        bundle={'bridge':bridge.cpu().state_dict(), 'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'source':source,'seed':seed,'K':k,'steps':nsteps,'batch':args.batch,'lr':args.lr,
                'dataset_sha256':ds['canonical_sha256'],
                'teacher_feature_sha256':teacher_receipt['features_sha256'],
                'teacher_outputs_sha256':teacher_receipt['teacher_outputs_sha256'],
                'head_revision':HEAD_REV,'writer_revision':WRITER_REV,
                'experiment_status':'pilot' if args.pilot else 'completed_h2_readout'}
        manifest=write_checkpoint(args.out/'checkpoints',label+'.pt',bundle)
        restored=restore_checkpoint(args.out/'checkpoints',manifest)
        if restored['K']!=k or restored['seed']!=seed:
            raise RuntimeError('Checkpoint metadata restore failed')
        report={'source':source,'K':k,'seed':seed,'train_seconds':duration,
                'train_loss_trace':losses,'bridge_parameters':sum(p.numel() for p in bridge.parameters()),
                'metrics':metrics,'checkpoint':manifest,'n_eval':len(eval_indices)}
        run_predictions=out_rows+zero_predictions+shuffled_predictions+contradictory_predictions
        predtmp=perpred.with_suffix('.partial.jsonl')
        with predtmp.open('w',encoding='utf-8',newline='\n') as f:
            for row in run_predictions:
                f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        predtmp.replace(perpred)
        receipt={'label':label,'pilot':args.pilot,'dataset_sha256':ds['canonical_sha256'],
                'teacher_features_sha256':teacher_receipt['features_sha256'],
                'predictions_sha256':sha256_file(perpred), 'report':report}
        receipt_tmp=perrun.with_suffix('.partial.json')
        receipt_tmp.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        receipt_tmp.replace(perrun)
        all_reports.append(report)
        all_predictions.extend(run_predictions)
        print('H2 evaluated and checkpointed',label,metrics['writer_oracle_exact'],metrics['teacher_text_mimic'],flush=True)
        del bridge,optimizer,bundle,restored
        gc.collect();torch.cuda.empty_cache()
    out_jsonl=args.out/'predictions.jsonl'
    with out_jsonl.open('w',encoding='utf-8',newline='\n') as f:
        for row in all_predictions:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    # A question-conditioned 0.8B-only baseline is a DIFFERENT control: it is
    # allowed to see the question, whereas the H2 writer never does.
    if not args.pilot:
        test_rows=[selected[i] for i in idx['test']]
        standalone=standalone_writer_baseline(writer,tok,test_rows)
        if len(standalone)!=300:raise RuntimeError('Missing standalone writer baseline')
        independent_path=args.out/'writer_only_question_baseline.jsonl'
        with independent_path.open('w',encoding='utf-8',newline='\n') as f:
            for item in standalone:f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n')
        writer_only={'n':len(standalone),'exact_count':sum(x['exact'] for x in standalone),
                     'exact':sum(x['exact'] for x in standalone)/len(standalone),
                     'predictions_sha256':sha256_file(independent_path)}
    else:
        writer_only={'status':'pilot_does_not_use_test'}
    # Primary K4 comparison across 3 seeds, original threshold unchanged.
    results={'status':'pilot_engineering_only_no_scientific_result' if args.pilot else 'completed_h2_original_quality_experiment',
             'dataset_sha256':ds['canonical_sha256'],
             'teacher_feature_sha256':teacher_receipt['features_sha256'],
             'teacher_outputs_sha256':teacher_receipt['teacher_outputs_sha256'],
             'head_revision':HEAD_REV,'writer_revision':WRITER_REV,
             'training_only_bridge':True,'full_teacher_and_writer_frozen':True,
             'no_question_or_style_at_writer_inference':True,
             'oracle_source_is_privileged_upper_bound':True,
             'baseline_teacher_accuracy':teacher_receipt.get('test_exact'),
             'method':'frozen_writer_soft_prefix_supervised_on_teacher_text',
             'writer_decode_mode':args.decode_mode,
             'writer_only_question_baseline':writer_only,
             'not_end_to_end_speed_measurement':True,
             'model_runs':all_reports,'predictions_sha256':sha256_file(out_jsonl),
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
             'software':{'torch':torch.__version__,'transformers':transformers.__version__,
                         'python':platform.python_version(),'cuda':torch.version.cuda},
             'original_gates':{'all_test_exact':.75,'head_correct_exact':.80,
                'lexical_advantage_pp':.15,'shuffled_advantage_pp':.30},
             'scientific_limitation':'Teacher 4B ~16.7% on earlier H1 two-step benchmark; fidelity alone cannot satisfy oracle quality gate.',
            }
    gate_per_seed={}
    for seed in PRIMARY_SEEDS:
        primary=next((r for r in all_reports if r['source']=='head' and r['K']==4 and r['seed']==seed),None)
        lexical=next((r for r in all_reports if r['source']=='hash' and r['K']==4 and r['seed']==seed),None)
        if primary is None or lexical is None:
            if not args.pilot:raise RuntimeError('Missing primary or hash control')
            continue
        a=primary['metrics'];b=lexical['metrics']
        subset=a['writer_accuracy_on_teacher_correct']
        conditions={
          'all_test_75pct':a['writer_oracle_exact']>=.75,
          'teacher_correct_80pct':subset is not None and subset>=.80,
          'hash_advantage_15pp':a['writer_oracle_exact']-b['writer_oracle_exact']>=.15,
          'shuffle_advantage_30pp':a['writer_oracle_exact']-a['controls']['shuffled_oracle_exact']>=.30,
        }
        gate_per_seed[str(seed)]={'all':bool(all(conditions.values())),
                                  'conditions':conditions,
                                  'head_accuracy':a['writer_oracle_exact'],
                                  'head_teacher_correct_n':a['teacher_correct_subset_n'],
                                  'hash_accuracy':b['writer_oracle_exact']}
    results['h2_primary_gate_by_seed']=gate_per_seed
    results['h2_registered_conclusion']='PILOT_NO_CLAIM' if args.pilot else (
          'CONFIRMED_under_registered_gates' if sum(v['all'] for v in gate_per_seed.values())>=2
          else 'NOT_CONFIRMED_under_registered_gates')
    if torch.cuda.max_memory_reserved()/2**30>=args.max_vram_gib:
        raise RuntimeError('Completed H2 exceeds conservative 22 GiB cap; no completed receipt')
    (args.out/'metrics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('H2 quality complete; H3/H4 are separate experiments.',flush=True)

if __name__=='__main__':main()
