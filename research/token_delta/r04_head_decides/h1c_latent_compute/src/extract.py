"""R04-H1c GPU: cache only *pre-answer* hidden states of frozen Qwen3.5-4B."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"h1"/"src"))
from h1_extract import (pinned_model, prompt_batch, load_pinned_tokenizer,
                        file_sha, save_shard, runtime_info)
from dataset import load

MODEL_SHA="c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97"


def run(data:Path,out:Path,lockfile:Path,batch_size:int=4,shard:int=50,max_tokens:int=256):
    from safetensors.torch import save_file
    from safetensors import safe_open
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 GPU unavailable; use your RTX3090 machine")
    if batch_size<1 or shard<1:raise ValueError("invalid batch or shard")
    rows,dataset=load(data)
    lock=json.loads(lockfile.read_text(encoding="utf-8"))
    spec=lock["head"]
    if spec["revision"]!=MODEL_SHA or spec["text_hidden_size"]!=2560:
        raise RuntimeError("H1c locked teacher differs from H1/R04")
    out.mkdir(parents=True,exist_ok=True)
    dsha=dataset["dataset_sha256_canonical"]
    full=out/"h1c_last_features.safetensors"
    receipt=out/"feature_manifest.json"
    if full.exists() or receipt.exists():
        if not (full.exists() and receipt.exists()):
            raise RuntimeError("Partially published full feature cache")
        info=json.loads(receipt.read_text(encoding="utf-8"))
        if info["dataset_sha256"]!=dsha or info["teacher_revision"]!=MODEL_SHA or info["features_sha256"]!=file_sha(full):
            raise RuntimeError("Invalid/stale full teacher feature cache")
        with safe_open(str(full),framework="pt",device="cpu") as f:
            x=f.get_tensor("last")
            if x.shape!=(1700,2560) or x.dtype!=torch.bfloat16:
                raise RuntimeError("Stored hidden cache invalid")
        return info
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    model=pinned_model("head",spec)
    tokenizer=load_pinned_tokenizer(spec)
    shards=[]
    processed=0
    for begin in range(0,len(rows),shard):
        end=min(len(rows),begin+shard)
        path=out/"shards"/f"state_{begin:05d}_{end:05d}.safetensors"
        if path.exists():
            with safe_open(str(path),framework="pt",device="cpu") as f:
                meta=f.metadata()
                if any(meta.get(k)!=v for k,v in {"dataset_sha256":dsha,
                        "model_revision":MODEL_SHA,"start":str(begin),"end":str(end)}.items()):
                    raise RuntimeError("H1c shard metadata mismatch")
                chunk=f.get_tensor("last")
        else:
            payload=[]
            for cursor in range(begin,end,batch_size):
                examples=rows[cursor:min(end,cursor+batch_size)]
                inputs=prompt_batch(tokenizer,examples,max_tokens)
                with torch.inference_mode():
                    output=model(**inputs,use_cache=False,output_hidden_states=True,return_dict=True)
                    # left-pad: last index is always final nonpad token
                    feature=output.hidden_states[-1][:,-1,:].detach().to("cpu",dtype=torch.bfloat16)
                assert feature.shape==(len(examples),2560)
                assert torch.isfinite(feature).all()
                payload.append(feature)
                del inputs,output
            chunk=torch.cat(payload)
            # save_shard persists last (plus prior other keys if present) with SHA-marked metadata
            save_shard(path,{"last":chunk},dsha,MODEL_SHA,begin,end)
        if chunk.shape!=(end-begin,2560):raise RuntimeError("Invalid shard")
        shards.append({"path":path.name,"bytes":path.stat().st_size,"sha256":file_sha(path),"start":begin,"end":end})
        processed=end
        print("H1c prompt-only feature extraction",processed,"/",len(rows),flush=True)
    parts=[]
    for s in shards:
        path=out/"shards"/s["path"]
        with safe_open(str(path),framework="pt",device="cpu") as f:
            meta=f.metadata()
            if meta["dataset_sha256"]!=dsha or meta["model_revision"]!=MODEL_SHA:
                raise RuntimeError("Invalid shard metadata")
            x=f.get_tensor("last")
        parts.append(x)
    all_data=torch.cat(parts)
    assert all_data.shape==(1700,2560) and all_data.dtype==torch.bfloat16
    tmp=out/"h1c_last_features.partial.safetensors"
    save_file({"last":all_data.contiguous()},str(tmp),metadata={
        "dataset_sha256":dsha,"model_revision":MODEL_SHA,
        "context":"pre-answer; no oracle, answer or generated assistant text"})
    tmp.replace(full)
    torch.cuda.synchronize()
    result={"status":"completed_pre_answer_feature_extraction_only_not_a_scientific_pass",
      "dataset_sha256":dsha,"teacher_revision":MODEL_SHA,
      "features_sha256":file_sha(full),"features_bytes":full.stat().st_size,
      "shape":{"last":[1700,2560]},"dtype":"bfloat16",
      "wall_seconds":time.perf_counter()-start,
      "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
      "runtime":runtime_info(),"shards":shards,
      "teacher_frozen":True,"oracle_in_teacher_input":False}
    receipt.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    del model,tokenizer
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    p.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1]/"cache")
    p.add_argument("--lock",type=Path,default=ROOT/"MODEL_LOCK.json")
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--shard",type=int,default=50)
    p.add_argument("--max-prompt-tokens",type=int,default=256)
    args=p.parse_args()
    output=run(args.data,args.out,args.lock,args.batch,args.shard,args.max_prompt_tokens)
    print(json.dumps({k:output[k] for k in ("status","features_sha256","wall_seconds","peak_reserved_gib")},indent=2))

if __name__=="__main__":main()