"""Generate text from a saved G1 checkpoint; optional EXPLICIT memory tool input."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
from checkpoint_io import load_checkpoint
from tokenizer import Tokenizer
from protected_memory import ProtectedMemory
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--prompt',required=True)
    ap.add_argument('--tokenizer',default=str(ROOT/'data/tokenizer.json'))
    ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--max-new-tokens',type=int,default=64);ap.add_argument('--temperature',type=float,default=0.0)
    ap.add_argument('--memory-file');ap.add_argument('--memory-keys',nargs='*',default=[])
    args=ap.parse_args();torch.set_num_threads(args.threads)
    model,idx=load_checkpoint(args.checkpoint,device=args.device);model.eval()
    tok=Tokenizer.load(args.tokenizer);prompt=args.prompt
    if args.memory_file:
        memory=ProtectedMemory.load(args.memory_file)
        prompt='Explicit memory records: '+memory.serialize(args.memory_keys)+'\n'+prompt
    ids=torch.tensor([tok.encode(prompt,bos=True)],device=args.device)
    out=model.generate(ids,max_new_tokens=args.max_new_tokens,temperature=args.temperature)
    print(json.dumps({'checkpoint_step':idx['step'],'prompt':prompt,'completion':tok.decode(out[0,ids.shape[1]:].cpu().tolist()),'memory_tool_supplied':bool(args.memory_file),'stage':'bootstrap, not a production assistant'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
