"""CLI for both published local BF16 models. No teacher, solver, or pickle input."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from runtime import ROOT,load_model,generate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['curriculum','shuffled'],default='curriculum')
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt');group.add_argument('--prompt-file',type=Path)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--max-new-tokens',type=int,default=96)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',default='cpu');ap.add_argument('--fp32-matmul',action='store_true')
    a=ap.parse_args()
    if not 1<=a.threads<=64:ap.error('threads must be 1..64')
    torch.set_num_threads(a.threads)
    prompt=a.prompt_file.read_text(encoding='utf-8') if a.prompt_file else a.prompt
    m,t,s=load_model(a.root,a.arm,a.device);start=time.perf_counter()
    out=generate(m,t,prompt,a.max_new_tokens,not a.fp32_matmul)
    out.update(arm=a.arm,seconds=time.perf_counter()-start,weights_sha256=s['models'][a.arm]['sha256'],status='research_checkpoint_not_general_assistant')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
