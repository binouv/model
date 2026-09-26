"""CPU inference for a released G8C probe; no embedded solver or gold input."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from model import Model,Config,ByteTokenizer,EOS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--prompt',required=True)
    ap.add_argument('--expected-sha256',required=True)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--max-new-bytes',type=int,default=128)
    ap.add_argument('--raw-prompt',action='store_true')
    a=ap.parse_args()
    if hashlib.sha256(a.weights.read_bytes()).hexdigest()!=a.expected_sha256:
        raise ValueError('weight SHA256 mismatch')
    if not 1<=a.max_new_bytes<=128:raise ValueError('use 1..128 bytes for this finite probe')
    torch.set_num_threads(a.threads)
    m=Model(Config(**json.loads(a.config.read_text())));m.load_state_dict(load_file(str(a.weights)));m.eval()
    tok=ByteTokenizer();prompt=a.prompt if a.raw_prompt else a.prompt+'\nFinish with F=<integer>.\n'
    ids=torch.tensor([tok.encode(prompt,bos=True)])
    start=time.perf_counter()
    with torch.inference_mode():out=m.generate(ids,torch.tensor([ids.shape[1]]),max_new=a.max_new_bytes)[0].tolist()
    eos=EOS in out;out=out[:out.index(EOS)] if eos else out
    print(json.dumps({'prompt':prompt,'generated':tok.decode(out),'eos':eos,'generated_bytes':len(out),'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in m.parameters()),'stage':'small synthetic research probe, not a general assistant','tools_used':False},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
