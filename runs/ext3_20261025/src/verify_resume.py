import json,random
import numpy as np
import torch
from safetensors.torch import load_file
from common import ROOT,setup,load,rows,sha,atomic_json
from objective import configure
from train import train_update

def restore_rng(s):
 torch.set_rng_state(s['torch_rng']);random.setstate(s['python_rng']);nr=s['numpy_rng'];np.random.set_state((nr['kind'],np.array(nr['keys'],dtype=np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
def main():
 setup();cfg=json.loads((ROOT/'config.json').read_text());m,tok=load(ROOT/'checkpoints/step_0064/model.safetensors');params=configure(m);s=torch.load(ROOT/'checkpoints/step_0064/training_state.pt',weights_only=True);assert s['sampler_cursor']==192 and s['data_manifest_sha256']==sha(ROOT/'data/manifest.json');opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False);opt.load_state_dict(s['optimizer']);restore_rng(s);del s
 data=rows('train.jsonl');logs={r['step']:r for r in [json.loads(x) for x in (ROOT/'logs/train.jsonl').read_text().splitlines()]}
 for step in range(65,129):
  v=train_update(m,opt,data[3*(step-1):3*step],step,cfg,tok);assert v['loss']==logs[step]['loss'] and v['grad_norm']==logs[step]['grad_norm'],step
 expected=load_file(str(ROOT/'verification/step_0128_model.safetensors'));assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items())
 atomic_json(ROOT/'metrics/resume_verification.json',dict(status='passed',from_step=64,to_step=128,all64_losses_and_gradient_norms_exact=True,all135_native_tensors_bitwise_equal=True,expected_model_sha256=sha(ROOT/'verification/step_0128_model.safetensors'),final_weights_not_modified=True))
 print('64 updates and135 native tensors replay exactly',flush=True)
if __name__=='__main__':main()
