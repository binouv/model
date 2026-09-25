"""Unrestricted generation from a saved G2R checkpoint. No embedded answer solver."""
from pathlib import Path
import argparse,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'base_src'))
from checkpoint_io import load_checkpoint
from tokenizer import Tokenizer
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--prompt',required=True);ap.add_argument('--tokenizer',default=str(ROOT/'data/tokenizer.json'));ap.add_argument('--threads',type=int,default=4);ap.add_argument('--max-new-tokens',type=int,default=32);a=ap.parse_args()
 torch.set_num_threads(a.threads);m,idx=load_checkpoint(a.checkpoint,dtype=torch.float32);m.eval();tok=Tokenizer.load(a.tokenizer)
 ids=torch.tensor([tok.encode(a.prompt,bos=True)])
 with torch.inference_mode():out=m.generate(ids,max_new_tokens=a.max_new_tokens)
 text=tok.decode(out[0,ids.shape[1]:].tolist());print(json.dumps({'completion':text,'step':idx['step'],'training':idx['training'],'not_pretrained_assistant':True},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
