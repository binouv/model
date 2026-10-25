"""Numerical engineering smoke on external dev fixtures, not model evaluation."""
import argparse,copy,dataclasses,datetime,hashlib,json,math,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from model import Config,Model,KVCache,parameter_report,ARCHITECTURE
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
SEED=2026102505

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'metrics/smoke.json');ap.add_argument('--weights',type=Path,default=ROOT/'artifacts/tiny_UNTRAINED_initial.safetensors');ap.add_argument('--readback',action='store_true');a=ap.parse_args();started=time.perf_counter();torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
 cfg=Config(**json.loads((ROOT/'configs/tiny_smoke.json').read_text()));tok=Tokenizer.load(ROOT/'data/tokenizer.json');assert tok.vocab_size==cfg.vocab_size;rr=[json.loads(s) for s in (ROOT/'data/external_dev_fixtures.jsonl').read_text().splitlines()];seq=[tok.encode(r['prompt'],bos=True) for r in rr];n=min(48,*map(len,seq));assert n>32;x=torch.tensor([s[:n] for s in seq],dtype=torch.long);m=Model(cfg).eval();checks={};details={}
 def record(name,condition,detail=None):
  checks[name]=bool(condition)
  if detail is not None:details[name]=detail
  if not condition:raise AssertionError(name)
 def near(name,aa,bb,atol=2e-5,rtol=1e-4):record(name,torch.allclose(aa,bb,atol=atol,rtol=rtol),dict(max_abs=float((aa-bb).abs().max()),atol=atol,rtol=rtol))
 def rejects(name,fn):
  try:fn()
  except (ValueError,TypeError,RuntimeError):record(name,True)
  else:record(name,False)
 with torch.no_grad():
  full,cache=m(x);record('logit_and_cache_shapes',full.shape==(2,n,cfg.vocab_size) and cache.position==n and all(k.shape==(2,cfg.kv_heads,n,cfg.dim//cfg.heads) and v.shape==k.shape for k,v in cache.layers));record('finite_logits',torch.isfinite(full).all())
  changed=x.clone();changed[:,n//2:]=x.flip(0)[:,n//2:];record('causal_fixture_has_changed_future',not torch.equal(changed,x));other,_=m(changed);near('causal_prefix_invariant_to_future',full[:,:n//2],other[:,:n//2],atol=1e-6,rtol=1e-5)
  cached=None;out=[]
  for i in range(n):lg,cached=m(x[:,i:i+1],cache=cached);out.append(lg)
  near('one_token_cache_matches_full',torch.cat(out,1),full)
  boundaries=[0,11,18,31,n];cached=None;out=[]
  for left,right in zip(boundaries,boundaries[1:]):
   lg,cached=m(x[:,left:right],cache=cached);out.append(lg)
  near('multitoken_cache_matches_full',torch.cat(out,1),full)
  _,prefix=m(x[:,:11]);snapshot=[(k.clone(),v.clone()) for k,v in prefix.layers];m(x[:,11:18],cache=prefix)
  record('cache_input_not_mutated',prefix.position==11 and all(torch.equal(k,kk) and torch.equal(v,vv) for (k,v),(kk,vv) in zip(prefix.layers,snapshot)))
  last,_=m(x,last_only=True);near('last_only_matches_full',last,full[:,-1:])
  nocache,none=m(x,use_cache=False);near('use_cache_false_logits',nocache,full);record('use_cache_false_returns_none',none is None)
  record('tied_same_parameter_and_storage',m.output_weight is m.embed.weight and m.output_weight.data_ptr()==m.embed.weight.data_ptr() and sum(p is m.embed.weight for p in m.parameters())==1)
  rejects('reject_inconsistent_cache_position',lambda:m(x[:,:1],cache=KVCache(cache.layers,n+1)))
  rejects('reject_inconsistent_cache_layers',lambda:m(x[:,:1],cache=KVCache(cache.layers[:-1],n)))
  rejects('reject_over_context',lambda:m(x[:,0:1].repeat(1,cfg.max_context+1)))
  rejects('reject_empty_sequence',lambda:m(x[:,:0]));rejects('reject_invalid_dimensions',lambda:Config(dim=65,heads=4));rejects('reject_foreign_architecture_marker',lambda:Config(architecture='flygraph_l100m_hybrid'))
  generation=m.generate(x[:,:11],max_new_tokens=6,eos_id=None);manual=x[:,:11].clone()
  for _ in range(6):lg,_=m(manual,last_only=True);manual=torch.cat((manual,lg[:,-1].argmax(-1)[:,None]),1)
  record('generation_cache_matches_full_greedy',torch.equal(generation,manual[:,11:]))
  record('generation_zero_budget',m.generate(x,max_new_tokens=0).shape==(2,0))
  near_limit=x[:,0:1].repeat(1,cfg.max_context-1);record('generation_respects_context_budget',m.generate(near_limit,max_new_tokens=9,eos_id=None).shape==(2,1))
 # Actual adjacent source tokens, loss aligned by the caller; no synthetic gold.
 inputs=x[:,:-1];targets=x[:,1:].clone();targets[:,:9]=-100;m.train();m.zero_grad(set_to_none=True);loss=m(inputs,targets=targets);loss.backward()
 record('finite_loss_and_all_gradients',torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 record('nonzero_gradient_each_tensor',all(bool(p.grad.ne(0).any()) for p in m.parameters()))
 details['loss']=float(loss.detach());gradients={name:p.grad.detach().clone() for name,p in m.named_parameters()}
 with torch.no_grad():
  lg,_=m(inputs,use_cache=False);dense_loss=torch.nn.functional.cross_entropy(lg.flatten(0,1).float(),targets.flatten(),ignore_index=-100);near('masked_loss_matches_dense',loss.detach(),dense_loss,atol=1e-6)
 rejects('reject_no_supervision',lambda:m(inputs,targets=torch.full_like(targets,-100)))
 rejects('reject_target_shape_mismatch',lambda:m(inputs,targets=targets[:,:-1]))
 cp=Model(dataclasses.replace(cfg,checkpoint_blocks=True));cp.load_state_dict(m.state_dict(),strict=True);cp.train();cp_loss=cp(inputs,targets=targets);cp_loss.backward();near('checkpointed_loss_matches',loss.detach(),cp_loss.detach(),atol=1e-6)
 errors=[]
 for name,p in cp.named_parameters():
  record('checkpoint_gradient_'+name,torch.allclose(p.grad,gradients[name],atol=2e-6,rtol=1e-4));errors.append(float((p.grad-gradients[name]).abs().max()))
 details['checkpoint_gradient_max_abs']=max(errors)
 # No optimizer exists or runs. Saved bytes are explicitly UNTRAINED init only.
 if a.readback:
  sd=load_file(str(a.weights));record('saved_init_matches_seed',all(torch.equal(sd[k],v) for k,v in m.state_dict().items()))
 else:save_file(m.state_dict(),str(a.weights),metadata={'architecture':ARCHITECTURE,'status':'UNTRAINED_INITIALIZATION_ONLY','seed':str(SEED),'optimizer_updates':'0'})
 restored=Model(cfg);restored.load_state_dict(load_file(str(a.weights)),strict=True);m.eval();restored.eval()
 with torch.no_grad():record('native_reload_exact_logits',torch.equal(m(x)[0],restored(x)[0]))
 record('native_reload_retains_tie',restored.output_weight.data_ptr()==restored.embed.weight.data_ptr())
 result=dict(status='passed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=SEED,device='cpu',precision='float32',checks=checks,details=details,check_count=len(checks),seconds=time.perf_counter()-started,parameters=parameter_report(m),source_fixture_ids=[r['id'] for r in rr],fixture_tokens=n,optimizer_updates=0,backward_passes=2,full_size_model_initialized=False,quality_evaluated=False,untrained_weight_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),untrained_weight_bytes=a.weights.stat().st_size)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['checks','details','parameters']},indent=2))
if __name__=='__main__':main()
