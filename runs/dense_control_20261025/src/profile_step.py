"""Measure forward/backward only; never performs an optimizer update.

Use the tiny config first. Full-size profiling is a future command, not a result
of the current smoke. Runtime/VRAM predictions for other devices are not made.
"""
import argparse,datetime,json,resource,statistics,time
from pathlib import Path
import torch
from model import Config,Model,parameter_report
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=ROOT/'configs/tiny_smoke.json');ap.add_argument('--sequence-length',type=int,default=32);ap.add_argument('--iterations',type=int,default=3);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.iterations<1 or a.sequence_length<2:raise ValueError('positive iterations and sequence length >=2 required')
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(2026102505);c=Config(**json.loads(a.config.read_text()));tok=Tokenizer.load(ROOT/'data/tokenizer.json')
 if c.vocab_size!=tok.vocab_size:raise ValueError('No matching tokenizer for this config: 32k template is not runnable with the 8k tokenizer')
 rr=[json.loads(s) for s in (ROOT/'data/external_dev_fixtures.jsonl').read_text().splitlines()];seq=[tok.encode(r['prompt']+r['completion'],bos=True,eos=True) for r in rr]
 if a.sequence_length+1>min(map(len,seq)):raise ValueError('Requested source prefix exceeds an original fixture; choose a shorter profile length')
 ids=torch.tensor([s[:a.sequence_length] for s in seq]);targets=torch.tensor([s[1:a.sequence_length+1] for s in seq]);m=Model(c).train();times=[];losses=[]
 for step in range(a.iterations+1):
  m.zero_grad(set_to_none=True);start=time.perf_counter();loss=m(ids,targets=targets);loss.backward();elapsed=time.perf_counter()-start
  if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):raise RuntimeError('non-finite forward/backward')
  if step:times.append(elapsed);losses.append(float(loss.detach()))
 result=dict(status='passed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),device='cpu',precision='float32',parameters=parameter_report(m)['parameters'],batch=len(ids),tokens_per_sequence=a.sequence_length,warmup_iterations=1,measured_iterations=a.iterations,seconds=times,median_seconds=statistics.median(times),median_input_tokens_per_second=ids.numel()/statistics.median(times),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,losses=losses,optimizer_updates=0,trained=False,quality_evaluated=False,note='Forward/backward only on fixed source prefixes. Excludes optimizer, data loading, checkpoint and evaluation overhead; not end-to-end training throughput. No extrapolation to GPU.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
