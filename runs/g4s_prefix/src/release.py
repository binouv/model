import hashlib,json,shutil,sys
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from model import Model,Config
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 work=ROOT/'results';assets=work/'release_assets';assets.mkdir(parents=True,exist_ok=True);items=[]
 final=json.loads((work/'metrics/FINAL_RESULT.json').read_text())
 for r in final['runs']:
  v,s=r['variant'],r['seed'];cp=work/'checkpoints'/f'{v}_s{s}'/'checkpoint.pt';native=assets/f'{v}_s{s}.checkpoint.pt';shutil.copy2(cp,native)
  z=torch.load(cp,weights_only=False);m=Model(Config(**z['config']));m.load_state_dict(z['model'])
  st={k:x.detach().cpu().contiguous() for k,x in m.state_dict().items()};sf=assets/f'{v}_s{s}.safetensors';save_file(st,str(sf))
  check=load_file(str(sf));m2=Model(Config(**z['config']));m2.load_state_dict(check)
  items.append({'variant':v,'seed':s,'native_checkpoint':native.name,'native_bytes':native.stat().st_size,'native_sha256':sha(native),'optimizer_rng_in_native':True,'safetensors':sf.name,'safetensors_bytes':sf.stat().st_size,'safetensors_sha256':sha(sf),'safetensors_reload':True,'parameters':sum(p.numel() for p in m.parameters())})
 manifest={'status':'completed_release_payload','run':'G4S-PrefixHybrid-Repro-20260926','items':items,'final_result_sha256':sha(work/'metrics/FINAL_RESULT.json'),'notes':['native checkpoint contains actual model+optimizer+torch/python/sampler RNG','safetensors is model-only','all six runs are fixed step800 endpoints','small mechanism probes, not Qwen parity']}
 (assets/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
