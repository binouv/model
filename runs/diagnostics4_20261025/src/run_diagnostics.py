"""Frozen-model scoring controls and explicit linear-probe diagnostics."""
import collections,datetime,json,math,time
import numpy as np
import torch
from safetensors.torch import save_file
from common import ROOT,setup,load,rows,sha,atomic_json,parse

@torch.inference_mode()
def feature_and_labels(m,tok,r,bf16=True):
 capture=[]
 def hook(module,args,out):capture.append(out[0,-1].detach().float().clone())
 handle=m.norm.register_forward_hook(hook)
 with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):lg,_=m(torch.tensor([tok.encode(r['prompt'],bos=True)]),last_only=True)
 handle.remove();ix=[tok.encode(' '+chr(65+i))[0] for i in range(len(r['choices']))];scores=lg[0,-1,ix].float().log_softmax(-1)
 return capture[0],scores.tolist()
@torch.inference_mode()
def answer_score(m,tok,prompt,answer):
 p=tok.encode(prompt,bos=True);a=tok.encode(' '+answer);assert a and len(p)+len(a)<=2048
 x=torch.tensor([p+a[:-1]]);y=torch.tensor([[-100]*(len(p)-1)+a])
 with torch.autocast('cpu',dtype=torch.bfloat16):loss=float(m(x,targets=y))
 return -loss,-loss*len(a),len(a)
def feature_split(m,tok,data):
 xx=[];ss=[]
 for r in data:
  x,s=feature_and_labels(m,tok,r);xx.append(x);ss.append(s)
 return torch.stack(xx),ss

def fit_ridge(X,y,classes,lam):
 X=X.double();mu=X.mean(0);sd=X.std(0,unbiased=False).clamp_min(1e-6);z=(X-mu)/sd;z=torch.cat([z,torch.ones((len(z),1),dtype=torch.float64)],1);Y=torch.nn.functional.one_hot(y,num_classes=classes).double();alpha=torch.linalg.solve(z@z.T+lam*torch.eye(len(z),dtype=torch.float64),Y);W=z.T@alpha
 return {'mean':mu,'std':sd,'weight':W}
def predict(head,X):
 z=(X.double()-head['mean'])/head['std'];z=torch.cat([z,torch.ones((len(z),1),dtype=torch.float64)],1);return (z@head['weight']).argmax(1)
def score_count(pred,rr):return sum(int(p)==r['gold_index'] for p,r in zip(pred,rr))
def main():
 setup();cfg=json.loads((ROOT/'config.json').read_text());start=time.perf_counter();allsets={name:rows(name+'.jsonl') for name in ['probe_train','tiny_train','seen_control','dev_mcq','test_mcq']};models={'EXT2':(ROOT/cfg['EXT2_path']).resolve(),'EXT3':(ROOT/cfg['EXT3_path']).resolve()};allraw={};features={};heads={};probe_summary={}
 # The fixed probe/model pipeline is frozen before any new holdout outputs.
 for arm,path in models.items():
  assert sha(path)==cfg[arm+'_sha256'];m,tok=load(path);m.eval();out=[]
  for split in ['seen_control','test_mcq']:
   fpath=ROOT/f'metrics/{arm}_{split}_scoring.jsonl'
   with fpath.open('x') as f:
    for j,r in enumerate(allsets[split]):
     feat,bf=feature_and_labels(m,tok,r);_,fp=feature_and_labels(m,tok,r,bf16=False)
     conditional='User: Answer the question briefly.\nQuestion: '+r['question']+'\nAssistant:';null='User: Answer the question briefly.\nAssistant:'
     cond=[answer_score(m,tok,conditional,c) for c in r['choices']];uncond=[answer_score(m,tok,null,c) for c in r['choices']]
     vals={'label_bf16':bf,'label_fp32':fp,'answer_mean':[v[0] for v in cond],'answer_sum':[v[1] for v in cond],'answer_calibrated_mean':[a[0]-b[0] for a,b in zip(cond,uncond)],'answer_calibrated_sum':[a[1]-b[1] for a,b in zip(cond,uncond)],'answer_without_question_mean':[v[0] for v in uncond]}
     with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):seq=m.generate(torch.tensor([tok.encode(r['prompt'],bos=True)]),max_new_tokens=16)[0].tolist()
     body=seq[:seq.index(2)] if 2 in seq else seq;generation=tok.decode(body);row=dict(id=r['id'],task=r['task'],split=split,gold=r['gold_index'],scores=vals,predictions={k:int(np.argmax(v)) for k,v in vals.items()},candidate_token_counts=[v[2] for v in cond],generation=generation,token_ids=seq,parsed_generation=parse(generation,len(r['choices'])))
     out.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
     if j%32==0:print('SCORING',arm,split,j+1,'/',len(allsets[split]),flush=True)
  allraw[arm]=out
  if arm=='EXT3':
   for split in ['probe_train','dev_mcq','test_mcq']:
    X,ss=feature_split(m,tok,allsets[split]);features[split]=X;save_file({'last_hidden_state':X},str(ROOT/f'artifacts/{split}_features.safetensors'));print('FEATURES',split,len(X),flush=True)
   # Additional numerical/cache control on existing new dev examples only.
   from common import score_labels,encode
   protocol=[]
   for task in ['openbookqa','piqa','arc_challenge']:
    r=next(x for x in allsets['dev_mcq'] if x['task']==task);x,y,n=encode(tok,r)
    with torch.inference_mode():full=float(m(x,targets=y));_,ss=score_labels(m,tok,r)
    diff=abs(full+ss[r['gold_index']]/n);assert diff<1e-4;protocol.append(dict(id=r['id'],cached_vs_full_target_nll_difference=diff))
   atomic_json(ROOT/'metrics/scoring_protocol_verification.json',dict(status='passed',checks=protocol))
   # Gate statistics on one complete held dev prompt, not a memory benchmark.
   hooks=[];gate_stats={}
   for i,b in enumerate(m.blocks):
    if b.kind!='d':continue
    def hook(module,args,out,index=i):
     raw=out.detach().float();H=m.config.heads;g=raw[...,:H];beta=raw[...,H:].sigmoid();logdecay=-torch.nn.functional.softplus(g-4);half=math.log(2)/(-logdecay);gate_stats[str(index)]=dict(n=int(logdecay.numel()),decay_per_token_quantiles=torch.quantile(logdecay.exp().flatten(),torch.tensor([.1,.5,.9])).tolist(),nominal_half_life_tokens_quantiles=torch.quantile(half.flatten(),torch.tensor([.1,.5,.9])).tolist(),beta_quantiles=torch.quantile(beta.flatten(),torch.tensor([.1,.5,.9])).tolist())
    hooks.append(b.mix.gates.register_forward_hook(hook))
   with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):m(torch.tensor([tok.encode(allsets['dev_mcq'][0]['prompt'],bos=True)]),last_only=True)
   for h in hooks:h.remove()
   atomic_json(ROOT/'metrics/gate_statistics.json',dict(example_id=allsets['dev_mcq'][0]['id'],layers=gate_stats,scope='One public dev prompt; nominal decay is not an end-to-end memory horizon because beta writes and full attention layers also act.'))
  del m
 # Ridge heads are diagnostic readouts of frozen EXT3 features, not a new LLM.
 ids={r['id']:i for i,r in enumerate(allsets['probe_train'])};bundle={};selection={};raw_probe=[]
 for task,classes in [('openbookqa',4),('piqa',2)]:
  ti=[i for i,r in enumerate(allsets['probe_train']) if r['task']==task];di=[i for i,r in enumerate(allsets['dev_mcq']) if r['task']==task];X=features['probe_train'][ti];y=torch.tensor([allsets['probe_train'][i]['gold_index'] for i in ti]);DX=features['dev_mcq'][di];dr=[allsets['dev_mcq'][i] for i in di];scores=[];best=None
  for lam in cfg['ridge_lambdas']:
   head=fit_ridge(X,y,classes,lam);k=score_count(predict(head,DX),dr);scores.append(dict(regularization=lam,dev_correct=k,n=len(dr)))
   if best is None or k>best[0] or k==best[0] and lam>best[1]:best=(k,lam,head)
  _,lam,head=best;heads[task]=head;selection[task]=dict(selected_regularization=lam,candidates=scores,tie_break='largest regularization among equal dev accuracy')
  for k,v in head.items():bundle[task+'_'+k]=v
  training_k=score_count(predict(head,X),[allsets['probe_train'][i] for i in ti]);tiny=[r for r in allsets['tiny_train'] if r['task']==task];tinyX=features['probe_train'][[ids[r['id']] for r in tiny]];tinyy=torch.tensor([r['gold_index'] for r in tiny]);tinyhead=fit_ridge(tinyX,tinyy,classes,cfg['tiny_ridge_lambda']);tiny_k=score_count(predict(tinyhead,tinyX),tiny)
  for k,v in tinyhead.items():bundle['tiny_'+task+'_'+k]=v
  probe_summary[task]=dict(train_correct=training_k,train_n=len(ti),tiny_train_correct=tiny_k,tiny_train_n=len(tiny))
 atomic_json(ROOT/'metrics/probe_selection.json',selection);save_file(bundle,str(ROOT/'artifacts/probe_weights.native_float64.safetensors'))
 for split in ['dev_mcq','test_mcq']:
  for i,r in enumerate(allsets[split]):
   source='piqa' if r['task']=='piqa' else 'openbookqa';pred=int(predict(heads[source],features[split][i:i+1])[0]);raw_probe.append(dict(split=split,id=r['id'],task=r['task'],head=source,gold=r['gold_index'],predicted=pred))
 with (ROOT/'metrics/probe_predictions.jsonl').open('x') as f:
  for r in raw_probe:f.write(json.dumps(r)+'\n')
 for task in ['openbookqa','piqa','arc_challenge']:
  for split in ['dev_mcq','test_mcq']:
   rr=[r for r in raw_probe if r['task']==task and r['split']==split];probe_summary.setdefault(task,{})[split]=dict(correct=sum(r['gold']==r['predicted'] for r in rr),n=len(rr))
 summaries={}
 for arm,rr in allraw.items():
  summaries[arm]={}
  for split in ['seen_control','test_mcq']:
   summaries[arm][split]={}
   for task in dict.fromkeys(r['task'] for r in rr if r['split']==split):
    data=[r for r in rr if r['split']==split and r['task']==task];n=len(data);methods={k:dict(correct=sum(r['predictions'][k]==r['gold'] for r in data),n=n) for k in data[0]['predictions']}
    methods['generation16']=dict(correct=sum(r['parsed_generation']==r['gold'] for r in data),valid=sum(r['parsed_generation'] is not None for r in data),n=n)
    summaries[arm][split][task]=dict(methods=methods,bf16_fp32_label_disagreements=sum(r['predictions']['label_bf16']!=r['predictions']['label_fp32'] for r in data),conditioned_vs_question_ablated_answer_disagreements=sum(r['predictions']['answer_mean']!=r['predictions']['answer_without_question_mean'] for r in data))
 atomic_json(ROOT/'metrics/results.json',dict(completed=True,neural_optimizer_updates=0,frozen_model_hashes={a:sha(p) for a,p in models.items()},scoring=summaries,probe=probe_summary,probe_selection=selection,wall_seconds=time.perf_counter()-start,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_of_general_superiority_established=False,scope='Exploratory diagnostics; ridge heads are supervised task-specific readouts, not generative FlyGraph improvements; no baseline API calls.'))
 print('DIAGNOSTICS_FINISHED',json.dumps(probe_summary),flush=True)
if __name__=='__main__':main()
