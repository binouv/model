"""FP32 forward/cache/gradient equivalence on real external dev records."""
import argparse,datetime,json,time
from pathlib import Path
import torch
import model
from chunked_delta import chunked_delta_rule,original_delta_rule
from common import ROOT,setup,load,rows,atomic_json,sha

ATOL=1e-3;RTOL=1e-4

def diff(a,b):
 err=float((a-b).abs().max());ok=bool(torch.allclose(a,b,atol=ATOL,rtol=RTOL));return dict(max_abs=err,passed=ok)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weights',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'metrics/chunked_delta_verification.json');args=ap.parse_args();setup();cfg=json.loads((ROOT/'config.json').read_text());weights=args.weights or (ROOT/cfg['EXT3_path']).resolve();assert sha(weights)==cfg['EXT3_sha256'];m,tok=load(weights);m.eval();rr=rows('dev_mcq.jsonl');chosen=[next(r for r in rr if r['task']==task) for task in ['openbookqa','piqa','arc_challenge']];longest=max(rr,key=lambda r:len(tok.encode(r['prompt'],bos=True)))
 if longest['id'] not in {r['id'] for r in chosen}:chosen.append(longest)
 records=[]
 with torch.inference_mode():
  for r in chosen:
   x=torch.tensor([tok.encode(r['prompt'],bos=True)]);outputs=[];times=[]
   for fn in [original_delta_rule,chunked_delta_rule]:
    model.delta_rule=fn;t=time.perf_counter();lg,cache=m(x,last_only=True);times.append(time.perf_counter()-t);nextid=int(lg[0,-1].argmax());nextlg,ncache=m(torch.tensor([[nextid]]),cache=cache,last_only=True);outputs.append((lg,cache,nextlg,nextid))
   a,b=outputs;states=[diff(ca,cb) for block,ca,cb in zip(m.blocks,a[1],b[1]) if block.kind=='d'];record=dict(id=r['id'],tokens=len(x[0]),logits=diff(a[0],b[0]),next_step_logits=diff(a[2],b[2]),same_next_token=a[3]==b[3],delta_state_checks=states,seconds_original_prefill=times[0],seconds_chunk64_prefill=times[1]);records.append(record);print(json.dumps(record),flush=True)
 # Actual dev prefix used as a numerical fixture, not as a scored/trained task.
 r=longest;p=tok.encode(r['prompt'],bos=True)[:128];x=torch.tensor([p]);target=torch.full_like(x,-100);target[0,-1]=tok.encode(r['completion'])[0];grads=[];losses=[]
 for fn in [original_delta_rule,chunked_delta_rule]:
  model.delta_rule=fn;m.zero_grad(set_to_none=True);loss=m(x,targets=target);loss.backward();losses.append(float(loss.detach()));grads.append({n:p.grad.detach().clone() for n,p in m.named_parameters()})
 gchecks={n:diff(grads[0][n],grads[1][n]) for n in grads[0]};model.delta_rule=original_delta_rule
 result=dict(status='passed' if all(r['logits']['passed'] and r['next_step_logits']['passed'] and r['same_next_token'] and all(s['passed'] for s in r['delta_state_checks']) for r in records) and all(g['passed'] for g in gchecks.values()) else 'failed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),weights_sha256=cfg['EXT3_sha256'],precision='FP32',chunk_size=64,atol=ATOL,rtol=RTOL,forward_cache=records,gradient_fixture=dict(id=r['id'],prefix_tokens=len(p),losses=losses,tensor_count=len(gchecks),checks=gchecks),neural_optimizer_updates=0,new_learned_parameters=0,note='External dev numerical fixtures only; no quality benchmark result and no exact-bitwise resume claim. Timing includes no warmup and is a single observation, not a throughput estimate.')
 atomic_json(args.output,result);assert result['status']=='passed';print('CHUNKED_VERIFIED',flush=True)
if __name__=='__main__':main()
