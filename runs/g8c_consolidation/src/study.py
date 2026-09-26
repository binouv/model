"""G8C: finite consolidation of real G5S CPU checkpoints, not a fresh replay.

Parent schema/weights/optimizer/RNG are restored. Stage 2 changes only its
objective and LR schedule; source, data and checkpoint provenance are explicit.
No solver or labels enter the neural generation interface.
"""
from __future__ import annotations
import collections,copy,csv,gzip,hashlib,json,math,os,random,re,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from model import Model,Config,ByteTokenizer,EOS
import data
SPLITS=('iid','surface','extrapolation','counterfactual')
ARMS=('verified_trace','final_only')
SEEDS=(7601,7602,7603)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)
def state_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def answer(c):
 f=c['family']
 if f=='arithmetic':
  a,b=c['values'];return a+b if c['op']=='+' else a-b if c['op']=='-' else a*b
 if f=='conditional':
  x,k,a,b=c['values'];return x+(a if x<k else -b)
 if f=='code_trace':return c['start']+sum(a if op=='+' else -a for op,a in c['program'])
 if f=='list_reasoning':
  s=sorted(c['values']);return sum(s) if c['kind']=='sum' else s[0] if c['kind']=='min' else s[-1]
 return next(v for k,v in reversed(c['writes']) if k==c['query'])
def pack(z):
 p=z['prompt']+'\nFinish with F=<integer>.\n';tok=ByteTokenizer();pt=tok.encode(p,bos=True)
 return {**z,'prompt':p,'tokens':pt+tok.encode('F='+z['answer'],eos=True),'prompt_tokens':len(pt)}
def read_rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines()]
def prepare(root=ROOT):
 root=Path(root);old=root/'data/parent';old.mkdir(parents=True,exist_ok=True);man=data.build(old)
 original=json.loads((root/'base/preregistered.json').read_text())
 for split,digest in original['data_sha256'].items():
  if sha(old/f'{split}.jsonl')!=digest:raise ValueError('parent data did not reproduce: '+split)
 denied={z['canonical'] for p in old.glob('*.jsonl') for z in read_rows(p)};occupied=set(denied)
 new=root/'data/fresh';new.mkdir(parents=True,exist_ok=True);counts={};tok=ByteTokenizer()
 for split,seed,ood,variant in [('iid',88301,False,0),('surface',88302,False,1),('extrapolation',88303,True,1)]:
  rng=random.Random(seed);rows=[]
  while len(rows)<400:
   c=data.make(rng,data.FAMS[len(rows)%5],ood);g=data.canonical(c)
   if g in occupied:continue
   z=data.row(c,variant,tok)
   if len(pack(z)['tokens'])>256:continue
   assert int(z['answer'])==answer(c)
   rows.append(z);occupied.add(g)
  (new/f'{split}.jsonl').write_text(''.join(json.dumps(z,separators=(',',':'))+'\n' for z in rows));counts[split]=len(rows)
 rng=random.Random(88304);rows=[]
 while len(rows)<400:
  f='arithmetic' if len(rows)<200 else 'memory_update';c=data.make(rng,f);g=data.canonical(c)
  if g in occupied:continue
  alt=copy.deepcopy(c)
  if f=='arithmetic':alt['op']=rng.choice([x for x in '+-*' if x!=c['op']])
  else:alt['query']=rng.choice([k for k in 'abcd' if k!=c['query']])
  if answer(c)==answer(alt):continue
  pair='g8cf'+str(len(rows)//2)
  aa=[data.row(cc,0,tok,pair) for cc in [c,alt]]
  if max(len(pack(z)['tokens']) for z in aa)>256:continue
  assert all(int(z['answer'])==answer(z['case']) for z in aa)
  rows+=aa;occupied.add(g)
 (new/'counterfactual.jsonl').write_text(''.join(json.dumps(z,separators=(',',':'))+'\n' for z in rows));counts['counterfactual']=len(rows)
 result={'status':'completed_data_audit','parent_sha256_matched':True,'parent_canonical_groups':len(denied),'fresh_groups':len(occupied)-len(denied),'fresh_cases':sum(counts.values()),'parent_overlap':0,'language':'English','held_seeds':[88301,88302,88303,88304],'fresh_sha256':{s:sha(new/f'{s}.jsonl') for s in SPLITS},'scope':'fresh relative to available G5S datasets; no claim about absent G7S dataset or natural-language benchmarks'}
 put(root/'reports/DATA_AUDIT.json',result);return result

def batch(rr):
 n=max(len(z['tokens']) for z in rr)-1;x=torch.zeros((len(rr),n),dtype=torch.long);y=torch.full_like(x,-100);p=torch.tensor([z['prompt_tokens'] for z in rr])
 for i,z in enumerate(rr):
  t=torch.tensor(z['tokens']);end=len(t)-1;x[i,:end]=t[:-1];y[i,p[i]-1:end]=t[p[i]:]
 return x,y,p

def parse(text):
 if text.count('F=')!=1:return None
 m=re.fullmatch(r'(?:T=[^;\n]+;)?F=(-?\d+)',text.strip())
 return int(m.group(1)) if m else None

@torch.inference_mode()
def predictions(m,prompts):
 # Only strings are accepted. Gold, cases and trace targets stay outside.
 tok=ByteTokenizer();tt=[tok.encode(p,bos=True) for p in prompts]
 if len({len(t) for t in tt})!=1:raise ValueError('equal prompt lengths required')
 start=time.perf_counter();out=m.generate(torch.tensor(tt),torch.full((len(tt),),len(tt[0])),max_new=128)
 rows=[]
 for seq in out.tolist():
  eos=EOS in seq;n=seq.index(EOS) if eos else len(seq);seq=seq[:n]
  rows.append({'text':tok.decode(seq),'ids':seq,'eos':eos})
 return rows,time.perf_counter()-start

@torch.inference_mode()
def evaluate(m,rr):
 m.eval();groups=collections.defaultdict(list);result=[];seconds=0.
 for z in rr:groups[z['prompt_tokens']].append(z)
 for _,group in sorted(groups.items()):
  for j in range(0,len(group),16):
   bb=group[j:j+16];out,t=predictions(m,[z['prompt'] for z in bb]);seconds+=t
   for z,pr in zip(bb,out):
    val=parse(pr['text']);ok=pr['eos'] and val is not None and val==int(z['answer'])
    result.append({**pr,'id':z['id'],'family':z['family'],'pair_id':z['pair_id'],'gold':z['answer'],'exact':ok,'parsed':val,'strict_final_only':pr['eos'] and pr['text'].strip()=='F='+z['answer']})
 pairs=collections.defaultdict(list)
 for z in result:
  if z['pair_id'] is not None:pairs[z['pair_id']].append(z)
 if not all(len(p)==2 for p in pairs.values()):raise ValueError('incomplete pair')
 return {'status':'completed','n':len(result),'correct':sum(z['exact'] for z in result),'accuracy':sum(z['exact'] for z in result)/len(result),'both_correct_pairs':sum(all(z['exact'] for z in p) for p in pairs.values()),'pair_n':len(pairs),'truncated':sum(not z['eos'] for z in result),'generation_seconds':seconds,'generated_bytes':sum(len(z['ids']) for z in result),'rows':sorted(result,key=lambda z:z['id'])}

def update(m,opt,rr,lr):
 m.train();opt.zero_grad(set_to_none=True);x,y,p=batch(rr)
 for g in opt.param_groups:g['lr']=lr
 start=time.perf_counter();loss=m(x,y,p);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.))
 if not math.isfinite(float(loss.detach())) or not math.isfinite(gn):raise ValueError('nonfinite update')
 opt.step();return {'loss':float(loss.detach()),'grad_norm':gn,'lr':lr,'seconds':time.perf_counter()-start,'input_tokens':sum(len(z['tokens'])-1 for z in rr),'supervised_tokens':int(y.ne(-100).sum()),'padded_tokens':x.numel(),'batch_ids':[z['id'] for z in rr]}

def restore(st):
 if st['schema'] not in ('G5S_v1','G8C_v1'):raise ValueError('unknown checkpoint')
 m=Model(Config(**st['config']));m.load_state_dict(st['model']);opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);opt.load_state_dict(copy.deepcopy(st['optimizer']))
 rng=random.Random();rng.setstate(st['sampler_rng']);random.setstate(st['python_rng']);torch.set_rng_state(st['torch_rng'])
 return m,opt,rng

def save(m,opt,rng,arm,seed,step,logs,parent_hash):
 cp=ROOT/'checkpoints'/f'{arm}_s{seed}'/f'step{step:04d}';cp.mkdir(parents=True,exist_ok=False)
 state={'schema':'G8C_v1','config':m.config.__dict__,'arm':arm,'seed':seed,'step':step,'parent_step':1000,'stage2_steps':step-1000,'parent_checkpoint_sha256':parent_hash,'model':m.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':rng.getstate(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'log':logs,'preregistered_sha256':sha(ROOT/'configs/preregistered.json')}
 torch.save(state,cp/'resume.pt');save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},str(cp/'model.safetensors'));put(cp/'config.json',m.config.__dict__)
 put(cp/'MANIFEST.json',{'parameters':469648,'stage2_step':step-1000,'files':{n:{'bytes':(cp/n).stat().st_size,'sha256':sha(cp/n)} for n in ['resume.pt','model.safetensors','config.json']}});(cp/'COMPLETE').write_text('complete\n');return cp

def pools_for_training():
 rr=[pack(z) for z in read_rows(ROOT/'data/parent/train.jsonl')];pools=collections.defaultdict(list)
 for z in rr:pools[z['family'],z['prompt_tokens']].append(z)
 keys=[k for k,v in sorted(pools.items()) if len(v)>=4]
 return keys,pools

def train(arm,seed,parent_file,publish=lambda *args:None):
 final=ROOT/'metrics'/f'{arm}_s{seed}.json'
 if final.exists():raise FileExistsError('refuse duplicate completed run')
 st=torch.load(parent_file,weights_only=False,map_location='cpu')
 if st['step']!=1000 or st['arm']!=arm or st['seed']!=seed:raise ValueError('wrong parent checkpoint')
 parent_hash=sha(parent_file);m,opt,rng=restore(st);keys,pools=pools_for_training()
 assert sum(p.numel() for p in m.parameters())==469648
 # Sampling starts exactly where the persisted G5S sampler stopped.
 # G5S keys use the same family/prompt-length grouping in both objective arms.
 init_hash=state_hash(m);logs=[];eval_rows={s:[pack(z) for z in read_rows(ROOT/f'data/fresh/{s}.jsonl')] for s in SPLITS}
 before={s:evaluate(m,z) for s,z in eval_rows.items()}
 put(ROOT/'metrics'/f'{arm}_s{seed}_before.json',before)
 for j in range(1,1001):
  key=keys[rng.randrange(len(keys))];bb=rng.choices(pools[key],k=16)
  lr=.0003*(.2+.8*.5*(1+math.cos(math.pi*j/1000)))
  rec=update(m,opt,bb,lr);rec['global_step']=1000+j;logs.append(rec)
  with (ROOT/'metrics'/f'{arm}_s{seed}.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  if j%100==0:print('TRAIN',arm,seed,j,round(rec['loss'],4),flush=True)
  if j%250==0:
   cp=save(m,opt,rng,arm,seed,1000+j,logs,parent_hash)
   put(ROOT/'reports/NEXT_STATE.json',{'status':'running','stage':'G8C','active_arm':arm,'active_seed':seed,'last_durable_step':1000+j,'checkpoint':str(cp),'do_not_duplicate':True,'not_a_completed_grid':True})
   publish(cp,arm,seed,j,None)
 tests={s:evaluate(m,z) for s,z in eval_rows.items()}
 # Reload entire inference state. Don't claim model API tests as solving accuracy.
 n=Model(Config(**m.config.__dict__));n.load_state_dict(load_file(str(cp/'model.safetensors')))
 a=evaluate(m,eval_rows['iid'][:16])['rows'];b=evaluate(n,eval_rows['iid'][:16])['rows'];assert a==b
 resume=torch.load(cp/'resume.pt',weights_only=False);hashes=[]
 for _ in range(2):
  n,oo,rrng=restore(resume);key=keys[rrng.randrange(len(keys))];bb=rrng.choices(pools[key],k=16);update(n,oo,bb,logs[-1]['lr']);hashes.append(state_hash(n))
 assert hashes[0]==hashes[1]
 assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
 result={'status':'completed','arm':arm,'seed':seed,'parameters':469648,'parent_steps':1000,'additional_steps':1000,'parent_checkpoint_sha256':parent_hash,'parent_weight_state_sha256':init_hash,'final_weight_state_sha256':state_hash(m),'parent_optimizer_restored':True,'parent_rng_restored':True,'LR_schedule_change':'registered cosine .0003 to .00006; not bitwise continuation of original1000step recipe','stage2_batch_sha256':hashlib.sha256('\n'.join(x for z in logs for x in z['batch_ids']).encode()).hexdigest(),'stage2_input_tokens':sum(z['input_tokens'] for z in logs),'stage2_supervised_tokens':sum(z['supervised_tokens'] for z in logs),'stage2_padded_tokens':sum(z['padded_tokens'] for z in logs),'train_seconds':sum(z['seconds'] for z in logs),'all_parameter_values_finite':True,'reload_cases':16,'reload_mismatches':0,'next_update_replay_verified':True,'checkpoint_manifest':json.loads((cp/'MANIFEST.json').read_text()),'checkpoint':str(cp),'before':before,'tests':tests}
 put(final,result);publish(cp,arm,seed,1000,result)
 print('DONE',arm,seed,{s:tests[s]['correct'] for s in SPLITS},flush=True);return result

def paired(a,b):
 aa={z['id']:z['exact'] for z in a};bb={z['id']:z['exact'] for z in b};assert set(aa)==set(bb)
 w=sum(aa[k] and not bb[k] for k in aa);l=sum(bb[k] and not aa[k] for k in aa);n=w+l
 p=min(1.,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1.
 return {'wins':w,'losses':l,'delta_pp':100*(w-l)/len(aa),'exact_two_sided_p':p}
def aggregate():
 rr=[json.loads((ROOT/'metrics'/f'{arm}_s{s}.json').read_text()) for s in SEEDS for arm in ARMS]
 assert len(rr)==6 and all(z['status']=='completed' and z['additional_steps']==1000 for z in rr)
 assert len({z['stage2_batch_sha256'] for z in rr})==1
 vals={};ps={}
 for arm in ARMS:
  r=[z for z in rr if z['arm']==arm]
  vals[arm]={s:{'mean_accuracy':sum(z['tests'][s]['accuracy'] for z in r)/3,'correct_by_seed':[z['tests'][s]['correct'] for z in r],'n_unique':400,'both_correct_pairs_by_seed':[z['tests'][s]['both_correct_pairs'] for z in r],'pair_n':r[0]['tests'][s]['pair_n'],'truncated_total':sum(z['tests'][s]['truncated'] for z in r)} for s in SPLITS}
 for s in SEEDS:
  a=next(z for z in rr if z['seed']==s and z['arm']==ARMS[0]);b=next(z for z in rr if z['seed']==s and z['arm']==ARMS[1])
  ps[str(s)]={sp:paired(a['tests'][sp]['rows'],b['tests'][sp]['rows']) for sp in SPLITS}
 gains={sp:100*(vals[ARMS[0]][sp]['mean_accuracy']-vals[ARMS[1]][sp]['mean_accuracy']) for sp in SPLITS}
 pairdiff=100*(sum(vals[ARMS[0]]['counterfactual']['both_correct_pairs_by_seed'])-sum(vals[ARMS[1]]['counterfactual']['both_correct_pairs_by_seed']))/600
 gates={'iid_ge5pp':gains['iid']>=5,'surface_extrapolation_positive':gains['surface']+gains['extrapolation']>0,'cf_pair_drop_le2pp':pairdiff>=-2,'iid_positive_at_least2seeds':sum(ps[str(s)]['iid']['delta_pp']>0 for s in SEEDS)>=2}
 result={'run':'G8C','status':'completed','training_continuations':6,'new_random_initializations':0,'ancestral_initialization_seeds':list(SEEDS),'parameters_each':469648,'additional_updates_each':1000,'results':vals,'paired_by_seed':ps,'gains_pp':gains,'counterfactual_pair_gain_pp':pairdiff,'gates':gates,'verdict':'EFFECT_CRITERIA_MET_REQUIRES_EXTERNAL_VALIDATION' if all(gates.values()) else 'NOT_CONFIRMED_IN_REGISTERED_PROTOCOL','paired_case_count_per_seed':1600,'unique_test_cases':1600,'same_cases_repeated_across_seeds':True,'all_stage2_batches_match':True,'scope':'Small synthetic English mechanism test, not a300-800M model or Qwen parity; parent trace training had extra tokens/FLOPs.'}
 put(ROOT/'metrics/FINAL.json',result)
 with (ROOT/'metrics/FINAL.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['arm','seed','split','n','correct','accuracy','both_correct_pairs','truncated'])
  for z in rr:
   for sp,t in z['tests'].items():w.writerow([z['arm'],z['seed'],sp,t['n'],t['correct'],t['accuracy'],t['both_correct_pairs'],t['truncated']])
 put(ROOT/'reports/NEXT_STATE.json',{'status':'completed','run':'G8C','goal_not_achieved':True,'verdict':result['verdict'],'do_not_repeat':['G3','G5S_replay','G8C'],'latest_checkpoint_steps':2000,'actual_new_weight_bytes':'see release receipt, not this manifest','next_step':'If OOD remains weak, change supervision/representation rather than extending this numeric-template curriculum. Register a single matched mechanism before scaling.'})
 (ROOT/'reports/REPORT.md').write_text('# G8C completed continuation\n\n'+json.dumps(result,indent=2)+'\n\nParent optimizer/RNG and model bytes restored; only stage2 LR/objective changed as registered. All1600questions are fresh relative to the available G5S case groups. Raw outputs include before/after,EOS and every gold value for audit, never as model input. See six native checkpoints for optimizer/RNG. No external language/code benchmark and no new Qwen run.\n')
 return result
