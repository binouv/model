"""Unrestricted inference from real safetensors release; no answer solver."""
import argparse,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from model import Model,Config,report
from tokenizer import Tokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--config',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--prompt',required=True);p.add_argument('--max-new',type=int,default=32);a=p.parse_args()
 torch.set_num_threads(4);cfg=Config(**json.loads(Path(a.config).read_text()));m=Model(cfg);m.load_state_dict(load_file(a.weights));m.eval();tok=Tokenizer.load(a.tokenizer)
 ids=torch.tensor([tok.encode(a.prompt,bos=True)])
 if len(ids[0])+a.max_new>cfg.max_context:raise ValueError('requested generation exceeds configured test context')
 out=m.generate(ids,max_new=a.max_new)[0].tolist();eos=2 in out
 if eos:out=out[:out.index(2)]
 print(json.dumps({'generated':tok.decode(out),'eos':eos,'parameters':report(m),'scope':'compact synthetic-task architecture probe, not a general assistant'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
