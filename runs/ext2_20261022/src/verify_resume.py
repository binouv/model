"""Replay actual published training samples from step64; no invented fixture."""
import json,random,hashlib
import numpy as np
import torch
from safetensors.torch import load_file
from common import ROOT,setup,load,rows,encode,atomic_json,sha
from train import train_update
def main():
    setup();cfg=json.loads((ROOT/'config.json').read_text());d=ROOT/'checkpoints/step_0064';m,tok=load(d/'model.safetensors')
    state=torch.load(d/'training_state.pt',weights_only=True);assert state['step']==64 and state['sampler_cursor']==256
    assert state['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);nr=state['numpy_rng'];np.random.set_state((nr['kind'],np.array(nr['keys'],dtype=np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
    del state;data=rows('train.jsonl');logs={r['step']:r for r in [json.loads(s) for s in (ROOT/'logs/train.jsonl').read_text().splitlines()]};equality=[]
    for step in range(65,129):
        batch=[encode(tok,r) for r in data[4*(step-1):4*step]];result=train_update(m,opt,batch,step,cfg)
        same=result['loss']==logs[step]['loss'] and result['grad_norm']==logs[step]['grad_norm'];equality.append(same)
        assert same,(step,result['loss'],logs[step]['loss'])
    expected=load_file(str(ROOT/'verification/step_0128_model.safetensors'))
    tensor_equal=all(torch.equal(v,expected[k]) for k,v in m.state_dict().items());assert tensor_equal
    atomic_json(ROOT/'metrics/resume_verification.json',{'replayed_updates':64,'from_completed_step':64,'to_completed_step':128,'all_64_loss_and_grad_norm_equal':all(equality),'all_135_native_tensors_bitwise_equal':tensor_equal,'expected_model_sha256':sha(ROOT/'verification/step_0128_model.safetensors'),'no_final_or_selected_checkpoint_modified':True,'fixture':'same external training records used in actual updates65..128','status':'passed'})
    print('Native resume replay passed:64 losses/gradients and135 tensors exactly equal',flush=True)
if __name__=='__main__':main()
