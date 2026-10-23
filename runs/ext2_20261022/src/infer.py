"""Offline FlyGraph inference with the exact inherited tokenizer and model."""
import argparse,json
from pathlib import Path
import torch
from common import setup,load
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--prompt',required=True);p.add_argument('--max-new-tokens',type=int,default=64);a=p.parse_args();setup();m,tok=load(a.weights)
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):seq=m.generate(torch.tensor([tok.encode(a.prompt,bos=True)]),max_new_tokens=a.max_new_tokens)[0].tolist()
    text=tok.decode(seq[:seq.index(2)] if 2 in seq else seq)
    print(json.dumps({'completion':text,'new_token_ids':seq,'eos':2 in seq},ensure_ascii=False))
if __name__=='__main__':main()
