"""Add explicit native tensor counts/dtypes without changing checkpoint bytes."""
import collections,json
import torch
from safetensors.torch import load_file
from common import ROOT,sha,atomic_json

def inventory(tensors):
 return dict(tensor_count=len(tensors),elements=sum(t.numel() for t in tensors),tensor_dtypes=dict(collections.Counter(str(t.dtype) for t in tensors)))
def main():
 p=ROOT/'artifacts/MANIFEST.json';meta=json.loads(p.read_text())
 for c in meta['components']:
  source=ROOT/f"checkpoints/step_{c['step']:04d}"/c['filename'];assert sha(source)==c['sha256']
  if c['role'].endswith('model'):
   sd=load_file(str(source));info=dict(schema='flygraph.native-tensors.v1',model=inventory(list(sd.values())));assert info['model']['tensor_count']==135 and info['model']['elements']==100028328;del sd
  else:
   state=torch.load(source,weights_only=True);opt=state['optimizer']['state'];tensors=[v for s in opt.values() for v in s.values() if isinstance(v,torch.Tensor)];oi=inventory(tensors);oi['parameter_states']=len(opt);info=dict(schema='flygraph.native-tensors.v1',optimizer=oi,torch_rng=dict(dtype=str(state['torch_rng'].dtype),elements=state['torch_rng'].numel()));assert len(opt)==20;del state,tensors
  c['native_tensor_inventory']=info;mp=ROOT/c['directory']/'MANIFEST.json';m=json.loads(mp.read_text());m['native_tensor_inventory']=info;atomic_json(mp,m)
 atomic_json(p,meta);return meta
if __name__=='__main__':print(json.dumps(main(),indent=2))
