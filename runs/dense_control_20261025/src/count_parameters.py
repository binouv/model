"""Meta-device parameter counts: no production tensors or training allocated."""
import argparse,json
from pathlib import Path
import torch
from model import Config,Model,parameter_report
ROOT=Path(__file__).resolve().parents[1]
def count(config):
 c=Config(**json.loads(config.read_text()))
 with torch.device('meta'):m=Model(c)
 return parameter_report(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 obj={v.stem:count(v) for v in ([a.config] if a.config else sorted((ROOT/'configs').glob('*.json')))}
 if a.output:a.output.write_text(json.dumps(obj,indent=2)+'\n')
 print(json.dumps({k:{n:v[n] for n in ['parameters','tensor_count','embedding_parameters','block_parameters','trained']} for k,v in obj.items()},indent=2))
if __name__=='__main__':main()
