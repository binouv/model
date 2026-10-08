"""Unconstrained local generation, with no hidden arithmetic/memory solver."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import torch
from common import ROOT,Tokenizer
from checkpoint_io import load

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--checkpoint',required=True,type=Path)
 p=ap.add_mutually_exclusive_group(required=True);p.add_argument('--prompt');p.add_argument('--prompt-file',type=Path)
 ap.add_argument('--tokenizer',type=Path,default=ROOT/'data/tokenizer.json')
 ap.add_argument('--threads',type=int,default=4);ap.add_argument('--max-new-tokens',type=int,default=64)
 ap.add_argument('--device',default='cpu');a=ap.parse_args()
 if not 1<=a.max_new_tokens<=2048:raise ValueError('max-new-tokens out of range')
 torch.set_num_threads(a.threads);m,mf=load(a.checkpoint,device=a.device);m.eval();tok=Tokenizer.load(a.tokenizer)
 prompt=a.prompt if a.prompt is not None else a.prompt_file.read_text(encoding='utf-8');ids=torch.tensor([tok.encode(prompt,bos=True)],device=a.device)
 if ids.shape[1]+a.max_new_tokens>m.config.max_context:raise ValueError('Prompt plus output budget exceeds configured context; no truncation')
 t=time.perf_counter()
 with torch.inference_mode(),torch.autocast(a.device.split(':')[0],dtype=torch.bfloat16):out=m.generate(ids,max_new_tokens=a.max_new_tokens)[0].tolist()
 eos=tok.EOS in out;out=out[:out.index(tok.EOS)] if eos else out
 print(json.dumps({'checkpoint_step':mf['step'],'prompt':prompt,'generated':tok.decode(out),'emitted_eos':eos,'new_tokens':len(out),'seconds':time.perf_counter()-t,'model_stage':'research checkpoint, not a production assistant','tools':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
