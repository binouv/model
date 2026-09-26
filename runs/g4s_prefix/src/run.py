import argparse,collections,hashlib,json,math,os,random,statistics,time
from pathlib import Path
import torch
from model import *
from data import build
SPLITS=('test_iid','test_surface','test_extrapolation','test_counterfactual')
def rows(root,s):return [json.loads(x) for x in (root/f'{s}.jsonl').read_text().splitlines()]
def batch(rr):
 T=max(len(z['tokens']) for z in rr)-1;x=torch.zeros((len(rr),T),dtype=torch.long);y=torch.full_like(x,-100);p=torch.tensor([z['prompt_tokens'] for z in rr])
 for i,z in enumerate(rr):t=torch.tensor(z['tokens']);n=len(t)-1;x[i,:n]=t[:-1];y[i,z['prompt_tokens']-1:n]=t[z['prompt_tokens']:]
 return x,y,p
@torch.inference_mode()
def val(m,rr):
 s=n=0
 for j in range(0,len(rr),32):x,y,p=batch(rr[j:j+32]);k=int(y.ne(-100).sum());s+=float(m(x,y,p))*k;n+=k
 return s/n
@torch.inference_mode()
def evaluate(m,rr):
 tok=ByteTokenizer();groups=collections.defaultdict(list);out=[];t0=time.perf_counter()
 for z in rr:groups[z['prompt_tokens']].append(z)
 for plen,gg in groups.items():
  for j in range(0,len(gg),32):
   bb=gg[j:j+32];ids=torch.tensor([tok.encode(z['prompt'],bos=True) for z in bb]);pr=m.generate(ids,torch.full((len(bb),),plen))
   for z,q in zip(bb,pr.tolist()):
    eos=EOS in q;q=q[:q.index(EOS)] if eos else q;text=tok.decode(q);out.append({'id':z['id'],'family':z['family'],'pair_id':z['pair_id'],'gold':z['answer'],'generated':text,'token_ids':q,'eos':eos,'exact':text==z['answer']})
 pairs=collections.defaultdict(list)
 for z in out:
  if z['pair_id'] is not None:pairs[z['pair_id']].append(z)
 return {'n':len(out),'correct':sum(z['exact'] for z in out),'accuracy':sum(z['exact'] for z in out)/len(out),'eos':sum(z['eos'] for z in out),'both_correct_pairs':sum(all(x['exact'] for x in q) for q in pairs.values()),'pair_count':len(pairs),'by_family':{f:{'n':sum(z['family']==f for z in out),'correct':sum(z['family']==f and z['exact'] for z in out)} for f in ('arithmetic','conditional','code_trace','list_reasoning','memory_update')},'seconds':time.perf_counter()-t0,'rows':out}
def train(root,variant,seed,steps=800):
 torch.manual_seed(seed);random.seed(seed);m=Model(Config(variant=variant));assert report(m)['total_parameters']==469648;tr=rows(root/'data','train');vr=rows(root/'data','validation');pool=collections.defaultdict(list)
 for z in tr:pool[(z['family'],len(z['tokens']))].append(z)
 keys=[k for k,v in sorted(pool.items()) if len(v)>=4];rng=random.Random(75311);opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);logs=[];bh=hashlib.sha256();start=time.perf_counter()
 for step in range(1,steps+1):
  m.train();opt.zero_grad(set_to_none=True);key=keys[rng.randrange(len(keys))];rr=rng.choices(pool[key],k=16)
  for z in rr:bh.update((z['id']+'\n').encode())
  x,y,p=batch(rr);ts=time.perf_counter();loss=m(x,y,p);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));lr=.001*min(1,step/80)*(.15+.85*.5*(1+math.cos(math.pi*step/steps)))
  for g in opt.param_groups:g['lr']=lr
  opt.step();logs.append({'step':step,'loss':float(loss),'grad_norm':gn,'lr':lr,'seconds':time.perf_counter()-ts,'family':key[0],'length':key[1]})
 tests={s:evaluate(m,rows(root/'data',s)) for s in SPLITS};vn=val(m,vr);d=root/'checkpoints'/f'{variant}_s{seed}';d.mkdir(parents=True,exist_ok=True);pt=d/'checkpoint.pt';torch.save({'schema':'G4S-v1','config':m.config.__dict__,'model':m.state_dict(),'optimizer':opt.state_dict(),'step':steps,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'sampler_rng':rng.getstate(),'batch_sequence_sha256':bh.hexdigest()},pt);sha=hashlib.sha256(pt.read_bytes()).hexdigest();m2=Model(Config(variant=variant));m2.load_state_dict(torch.load(pt,weights_only=False)['model']);a=evaluate(m,rows(root/'data','test_iid')[:16])['rows'];b=evaluate(m2,rows(root/'data','test_iid')[:16])['rows'];assert [(z['generated'],z['token_ids']) for z in a]==[(z['generated'],z['token_ids']) for z in b]
 met={'status':'completed','variant':variant,'seed':seed,'steps':steps,'parameters':report(m),'validation_nll':vn,'tests':{s:{k:v for k,v in q.items() if k!='rows'} for s,q in tests.items()},'batch_sequence_sha256':bh.hexdigest(),'median_step_seconds':statistics.median(z['seconds'] for z in logs),'mean_step_seconds':statistics.mean(z['seconds'] for z in logs),'elapsed_seconds':time.perf_counter()-start,'checkpoint_sha256':sha,'checkpoint_bytes':pt.stat().st_size,'reload_probe_cases':16,'reload_mismatches':0,'all_losses_finite':all(math.isfinite(z['loss']) and math.isfinite(z['grad_norm']) for z in logs)}; (root/'metrics').mkdir(exist_ok=True);(root/'metrics'/f'{variant}_s{seed}.json').write_text(json.dumps({**met,'raw_tests':tests,'train_log':logs},indent=2));return met
def campaign(root,seeds,steps=800):
 root.mkdir(parents=True,exist_ok=True);build(root/'data');runs=[]
 for seed in seeds:
  for v in ('causal_hybrid','prefix_hybrid'):print('TRAIN',v,seed,flush=True);runs.append(train(root,v,seed,steps))
 agg={'status':'completed','hypothesis':'Bidirectional attention within the fully observed prompt improves free-generated reasoning while answer generation remains causal.','seeds':seeds,'runs':runs}
 for v in ('causal_hybrid','prefix_hybrid'):
  vv=[x for x in runs if x['variant']==v];agg[v]={s:{'mean_accuracy':sum(x['tests'][s]['accuracy'] for x in vv)/len(vv),'correct_by_seed':[x['tests'][s]['correct'] for x in vv],'both_correct_pairs_by_seed':[x['tests'][s]['both_correct_pairs'] for x in vv]} for s in SPLITS}
 diffs=[]
 for seed in seeds:
  a=next(x for x in runs if x['seed']==seed and x['variant']=='causal_hybrid')['tests']['test_iid']['accuracy'];b=next(x for x in runs if x['seed']==seed and x['variant']=='prefix_hybrid')['tests']['test_iid']['accuracy'];diffs.append(100*(b-a))
 iid=sum(diffs)/len(diffs);se=lambda v:(agg[v]['test_surface']['mean_accuracy']+agg[v]['test_extrapolation']['mean_accuracy'])/2;se_gain=100*(se('prefix_hybrid')-se('causal_hybrid'));ca=sum(agg['causal_hybrid']['test_counterfactual']['both_correct_pairs_by_seed'])/(40*len(seeds));cb=sum(agg['prefix_hybrid']['test_counterfactual']['both_correct_pairs_by_seed'])/(40*len(seeds));reg=100*(ca-cb);g={'mean_iid_gain_pp_ge_5':iid>=5,'pooled_surface_extrapolation_gain_positive':se_gain>0,'counterfactual_pair_regression_le_2pp':reg<=2,'positive_iid_direction_at_least_2seeds':sum(x>0 for x in diffs)>=2};agg['comparison']={'iid_gain_pp':iid,'iid_gain_pp_by_seed':diffs,'pooled_surface_extrapolation_gain_pp':se_gain,'counterfactual_pair_regression_pp':reg,'gates':g,'confirmed':all(g.values())};(root/'metrics'/'FINAL_RESULT.json').write_text(json.dumps(agg,indent=2));return agg
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/mnt/data/flygraph_g4s'));ap.add_argument('--steps',type=int,default=800);ap.add_argument('--seeds',default='7501,7502,7503');a=ap.parse_args();torch.set_num_threads(min(8,os.cpu_count() or 4));torch.set_num_interop_threads(1);print(json.dumps(campaign(a.root,list(map(int,a.seeds.split(','))),a.steps),indent=2))
