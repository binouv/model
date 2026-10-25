"""Minimal CPU FP32 LM runner with exact native resume and bounded tokens.

No claim of pretrained quality. A new dense model starts from scratch; this is
not an EXT3 continuation. Every window resets causal state. No data downloads.
"""
import argparse,dataclasses,datetime,hashlib,json,os,random,tempfile,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from model import Config,Model,parameter_report
from tokenizer import Tokenizer
from checkpoint_parts import split as split_parts,verify as verify_parts
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def rng_state():
 n=np.random.get_state();return dict(python=random.getstate(),torch=torch.get_rng_state(),numpy=dict(kind=n[0],keys=n[1].tolist(),position=n[2],has_gauss=n[3],cached_gaussian=n[4]))
def restore_rng(r):
 random.setstate(r['python']);torch.set_rng_state(r['torch']);n=r['numpy'];np.random.set_state((n['kind'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
class Sampler:
 def __init__(self,n,seed):
  self.rng=random.Random(seed);self.n=n;self.epoch=0;self.order=list(range(n));self.rng.shuffle(self.order);self.index=0;self.offset=0
 def state(self):return dict(n=self.n,epoch=self.epoch,order=self.order.copy(),index=self.index,offset=self.offset,rng=self.rng.getstate())
 def load(self,s):
  if s['n']!=self.n or sorted(s['order'])!=list(range(self.n)) or not 0<=s['index']<self.n or s['offset']<0:raise ValueError('invalid sampler state')
  self.epoch=s['epoch'];self.order=s['order'];self.index=s['index'];self.offset=s['offset'];self.rng.setstate(s['rng'])
 def next(self,data,max_tokens):
  doc=self.order[self.index];tokens=data[doc]['tokens'];start=self.offset;n=min(max_tokens,len(tokens)-1-start)
  if n<=0:raise ValueError('invalid sampler offset or exhausted budget')
  x=tokens[start:start+n];y=tokens[start+1:start+n+1];self.offset+=n
  if self.offset==len(tokens)-1:
   self.offset=0;self.index+=1
   if self.index==self.n:self.index=0;self.epoch+=1;self.rng.shuffle(self.order)
  return torch.tensor([x]),torch.tensor([y]),dict(id=data[doc]['id'],record_index=doc,offset=start,input_tokens=n)

def load_data(path,tok):
 out=[]
 for index,line in enumerate(Path(path).read_text().splitlines()):
  r=json.loads(line);text=r['text'] if 'text' in r else r['prompt']+r['completion'];tt=tok.encode(text,bos=True,eos=True)
  if len(tt)<2:raise ValueError('empty encoded document')
  out.append(dict(id=str(r.get('id',index)),tokens=tt))
 if not out:raise ValueError('empty training data')
 return out

def tensor_inventory(obj):
 counts={};elements=0;tensors=0
 def visit(x):
  nonlocal elements,tensors
  if isinstance(x,torch.Tensor):tensors+=1;elements+=x.numel();counts[str(x.dtype)]=counts.get(str(x.dtype),0)+1
  elif isinstance(x,dict):
   for v in x.values():visit(v)
  elif isinstance(x,(list,tuple)):
   for v in x:visit(v)
 visit(obj);return dict(tensor_count=tensors,elements=elements,dtypes=counts)

class Runner:
 def __init__(self,model_config,train_config,data_path,tokenizer_path):
  self.cfg=json.loads(Path(train_config).read_text());self.mc=Config(**json.loads(Path(model_config).read_text()));c=self.cfg
  required=['seed','learning_rate','weight_decay','max_steps','token_budget','sequence_length','checkpoint_every','wall_limit_seconds','run_scope']
  if any(k not in c for k in required):raise ValueError('incomplete training config')
  if min(c['max_steps'],c['token_budget'],c['sequence_length'],c['checkpoint_every'],c['wall_limit_seconds'])<1 or c['sequence_length']>self.mc.max_context:raise ValueError('invalid finite run budget')
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
  tok=Tokenizer.load(tokenizer_path)
  if tok.vocab_size!=self.mc.vocab_size:raise ValueError('tokenizer vocabulary does not match model')
  self.data=load_data(data_path,tok);self.sampler=Sampler(len(self.data),c['seed']);self.model=Model(self.mc).train();self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=c['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=c['weight_decay'],foreach=False)
  self.identity=dict(model_config=dataclasses.asdict(self.mc),train_config=c,data_sha256=sha(data_path),tokenizer_sha256=sha(tokenizer_path),source_sha256={name:sha(ROOT/'src'/name) for name in ['train.py','model.py','tokenizer.py','checkpoint_parts.py']},torch_version=str(torch.__version__),precision='CPU FP32 weights/AdamW, no AMP',schedule='constant LR, exact config and optimizer param_groups saved')
  self.step=0;self.tokens=0;self.trace=[]
 def state(self):return dict(schema='flygraph.dense-native-train.v1',identity=self.identity,optimizer=self.optimizer.state_dict(),step=self.step,tokens=self.tokens,sampler=self.sampler.state(),rng=rng_state(),trace=self.trace,scaler=None,cuda_rng=None,scheduler=dict(kind='constant',lr=self.cfg['learning_rate']),gradient_accumulation=1)
 def save(self,out,status):
  out=Path(out);out.mkdir(parents=True,exist_ok=True);target=out/f'step_{self.step:06d}'
  if target.exists():raise FileExistsError(target)
  stage=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=out));weights=stage/'model.safetensors';statepath=stage/'training_state.pt';save_file(self.model.state_dict(),str(weights));state=self.state();torch.save(state,statepath)
  for p in [weights,statepath]:
   with p.open('rb') as f:os.fsync(f.fileno())
  saved=load_file(str(weights));assert all(torch.equal(saved[k],v) for k,v in self.model.state_dict().items());loaded=torch.load(statepath,map_location='cpu',weights_only=True);assert loaded['step']==self.step and loaded['identity']==self.identity
  manifests={}
  for role,p in [('model',weights),('training_state',statepath)]:
   component=split_parts(p,stage/(role+'_parts'),identity='dense_runner_20261025:'+self.cfg['run_scope']);component.update(completed_optimizer_steps=self.step,parent_checkpoint=None,model_parameter_count=parameter_report(self.model)['parameters'],tensor_inventory=tensor_inventory(saved if role=='model' else state));write_json(stage/(role+'_parts')/'MANIFEST.json',component);verify_parts(stage/(role+'_parts'));manifests[role]=component
  manifest=dict(schema='flygraph.dense-checkpoint.v1',status=status,scope=self.cfg['run_scope'],step=self.step,input_tokens=self.tokens,model_sha256=sha(weights),training_state_sha256=sha(statepath),components=manifests,identity=self.identity,optimizer_updates_complete=True)
  write_json(stage/'MANIFEST.json',manifest);os.rename(stage,target)
  fd=os.open(out,os.O_RDONLY);os.fsync(fd);os.close(fd);write_json(out/'LATEST.json',dict(checkpoint=target.name,step=self.step,tokens=self.tokens,status=status));return target
 def restore(self,path):
  path=Path(path);manifest=json.loads((path/'MANIFEST.json').read_text())
  if manifest['identity']!=self.identity:raise ValueError('config/data/tokenizer/code/runtime identity changed; exact resume refused')
  if sha(path/'model.safetensors')!=manifest['model_sha256'] or sha(path/'training_state.pt')!=manifest['training_state_sha256']:raise ValueError('checkpoint hash mismatch')
  state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
  if state['identity']!=self.identity:raise ValueError('native state identity mismatch')
  self.model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True);self.optimizer.load_state_dict(state['optimizer']);self.step=state['step'];self.tokens=state['tokens'];self.trace=state['trace'];self.sampler.load(state['sampler']);restore_rng(state['rng'])
 def run(self,out,stop_after=None):
  out=Path(out)
  if out.exists() and any(out.iterdir()):raise FileExistsError('use a new output directory; no existing run overwrite')
  out.mkdir(parents=True,exist_ok=True);write_json(out/'IDENTITY.json',self.identity);started=time.monotonic();last_saved=-1;reason=None
  while self.step<self.cfg['max_steps'] and self.tokens<self.cfg['token_budget']:
   if stop_after is not None and self.step>=stop_after:reason='controlled_pause';break
   if time.monotonic()-started>=self.cfg['wall_limit_seconds']:reason='wall_limit_controlled_stop';break
   x,y,record=self.sampler.next(self.data,min(self.cfg['sequence_length'],self.cfg['token_budget']-self.tokens));self.optimizer.zero_grad(set_to_none=True);loss=self.model(x,targets=y)
   if not torch.isfinite(loss):raise RuntimeError('non-finite loss before update')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True);self.optimizer.step();self.step+=1;self.tokens+=x.numel();row=dict(step=self.step,input_tokens_total=self.tokens,loss=float(loss.detach()),grad_norm=float(norm),lr=self.optimizer.param_groups[0]['lr'],**record);self.trace.append(row)
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
   if self.step%self.cfg['checkpoint_every']==0 and self.step<self.cfg['max_steps'] and self.tokens<self.cfg['token_budget']:self.save(out,'recovery_in_progress');last_saved=self.step
  complete=self.step>=self.cfg['max_steps'] or self.tokens>=self.cfg['token_budget'];reason=reason or ('token_budget' if self.tokens>=self.cfg['token_budget'] else 'fixed_steps');status='completed_engineering_smoke' if complete and self.cfg['run_scope']=='tiny_engineering_smoke' else 'completed_training_pilot' if complete else 'interrupted_recoverable'
  if last_saved!=self.step:checkpoint=self.save(out,status)
  else:checkpoint=out/f'step_{self.step:06d}'
  summary=dict(status=status,complete=complete,reason=reason,step=self.step,input_tokens=self.tokens,checkpoint=str(checkpoint.resolve()),model_parameters=parameter_report(self.model)['parameters'],scope=self.cfg['run_scope'],quality_evaluated=False,seconds=time.monotonic()-started,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write_json(out/'SUMMARY.json',summary);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-config',type=Path,required=True);p.add_argument('--train-config',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int);a=p.parse_args();r=Runner(a.model_config,a.train_config,a.data,a.tokenizer)
 if a.resume:r.restore(a.resume)
 print(json.dumps(r.run(a.output,a.stop_after),indent=2))
if __name__=='__main__':main()
