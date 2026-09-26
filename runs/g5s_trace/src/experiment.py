"""Registered paired trace objective on exact saved G4S cases, no inference solver."""
import argparse,collections,copy,csv,gzip,hashlib,json,math,os,random,re,time
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
from model import *
ROOT=Path(__file__).resolve().parents[1];SPLITS=('test_iid','test_surface','test_extrapolation','test_counterfactual');SEEDS=(7601,7602,7603);ARMS=('verified_trace','final_only')
def put(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_hash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()
def independent_answer(c):
 f=c['family']
 if f=='arithmetic':
  a,b=c['values'];return a+b if c['op']=='+' else a-b if c['op']=='-' else a*b
 if f=='conditional':
  x,k,a,b=c['values'];return x+(a if x<k else -b)
 if f=='code_trace':return c['start']+sum(a if op=='+' else -a for op,a in c['program'])
 if f=='list_reasoning':
  v=sorted(c['values']);return sum(v) if c['kind']=='sum' else v[0] if c['kind']=='min' else v[-1]
 return next(v for k,v in reversed(c['writes']) if k==c['query'])
def trace(c):
 f=c['family']
 if f=='arithmetic':
  a,b=c['values'];op=c['op'];ans={'+':a+b,'-':a-b,'*':a*b}[op];return f'{a}{op}{b}={ans}',ans
 if f=='conditional':
  x,k,a,b=c['values'];cond=x<k;v=x+a if cond else x-b;return f'{x}<{k}:{int(cond)},{x}{"+" if cond else "-"}{a if cond else b}={v}',v
 if f=='code_trace':
  v=c['start'];ss=[str(v)]
  for op,a in c['program']:v=v+a if op=='+' else v-a;ss.append(str(v))
  return ','.join(ss),v
 if f=='list_reasoning':
  kind=c['kind'];ss=[];v=0 if kind=='sum' else c['values'][0]
  for a in c['values']:
   v=v+a if kind=='sum' else (min(v,a) if kind=='min' else max(v,a));ss.append(str(v))
  return ','.join(ss),v
 q=c['query'];vv=[v for k,v in c['writes'] if k==q];return q+':'+','.join(map(str,vv)),vv[-1]
def load(sp,arm):
 tok=ByteTokenizer();out=[]
 for z in map(json.loads,(ROOT/'data'/f'{sp}.jsonl').read_text().splitlines()):
  tr,a=trace(z['case']);assert a==independent_answer(z['case'])==int(z['answer'])
  prompt=z['prompt']+'\nFinish with F=<integer>.\n';target=('T='+tr+';' if arm=='verified_trace' else '')+'F='+str(a)
  pt=tok.encode(prompt,bos=True);ct=tok.encode(target,eos=True);assert len(ct)<=128
  out.append({**z,'prompt':prompt,'target':target,'trace':tr,'tokens':pt+ct,'prompt_tokens':len(pt)})
 return out
def batch(rr):
 n=max(len(z['tokens']) for z in rr)-1;x=torch.zeros((len(rr),n),dtype=torch.long);y=torch.full_like(x,-100);p=torch.tensor([z['prompt_tokens'] for z in rr])
 for i,z in enumerate(rr):
  t=torch.tensor(z['tokens']);nn=len(t)-1;x[i,:nn]=t[:-1];y[i,p[i]-1:nn]=t[p[i]:]
 return x,y,p
def parse_final(text):
 # A unique terminal F is required. No gold-dependent extraction.
 match=re.fullmatch(r'(?:T=[^;\n]+;)?F=(-?\d+)',text)
 return int(match.group(1)) if match else None
@torch.inference_mode()
def generate(m,prompts):
 tok=ByteTokenizer();ids=[tok.encode(p,bos=True) for p in prompts];assert len({len(x) for x in ids})==1
 out=m.generate(torch.tensor(ids),torch.full((len(ids),),len(ids[0])),max_new=128)
 rr=[]
 for x in out.tolist():
  eos=EOS in x;x=x[:x.index(EOS)] if eos else x;rr.append({'generated':tok.decode(x),'token_ids':x,'eos':eos})
 return rr
@torch.inference_mode()
def evaluate(m,rr):
 groups=collections.defaultdict(list);out=[];t=time.perf_counter()
 for z in rr:groups[z['prompt_tokens']].append(z)
 for plen,gg in groups.items():
  for j in range(0,len(gg),16):
   bb=gg[j:j+16];pred=generate(m,[z['prompt'] for z in bb])
   for z,a in zip(bb,pred):
    p=parse_final(a['generated']);valid=a['generated']=='T='+z['trace']+';F='+z['answer']
    out.append({**a,'id':z['id'],'family':z['family'],'pair_id':z['pair_id'],'gold':z['answer'],'parsed_final':p,'exact':p is not None and p==int(z['answer']),'exact_canonical_trace':valid})
 pairs=collections.defaultdict(list)
 for x in out:
  if x['pair_id']:pairs[x['pair_id']].append(x)
 assert all(len(p)==2 for p in pairs.values())
 return {'status':'completed','n':len(out),'correct':sum(x['exact'] for x in out),'accuracy':sum(x['exact'] for x in out)/len(out),'trace_correct':sum(x['exact_canonical_trace'] for x in out),'eos_count':sum(x['eos'] for x in out),'both_correct_pairs':sum(all(z['exact'] for z in p) for p in pairs.values()),'pair_count':len(pairs),'seconds':time.perf_counter()-t,'generated_bytes':sum(len(x['token_ids']) for x in out),'rows':out}
def save(m,opt,rng,arm,seed,step,logs):
 dest=ROOT/'checkpoints'/f'{arm}_s{seed}'/f'step{step:04d}';dest.mkdir(parents=True,exist_ok=False);tmp=dest/'resume.pt.tmp'
 torch.save({'schema':'G5S_v1','config':m.config.__dict__,'arm':arm,'seed':seed,'step':step,'total_steps':1000,'model':m.state_dict(),'optimizer':opt.state_dict(),'sampler_rng':rng.getstate(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'log':logs,'preregistered_sha':sha(ROOT/'configs/preregistered.json')},tmp);os.replace(tmp,dest/'resume.pt')
 save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},str(dest/'model.safetensors'));put(dest/'config.json',m.config.__dict__)
 put(dest/'MANIFEST.json',{'step':step,'parameters':469648,'files':{n:{'bytes':(dest/n).stat().st_size,'sha256':sha(dest/n)} for n in ['model.safetensors','resume.pt','config.json']},'optimizer_rng':True});(dest/'COMPLETE').write_text('complete\n');return dest

def train(arm,seed):
 out=ROOT/'metrics'/f'{arm}_s{seed}.json'
 if out.exists():raise FileExistsError('Already completed; refusing duplicate')
 torch.manual_seed(seed);random.seed(seed);m=Model(Config());init=state_hash(m);rr=load('train',arm);pool=collections.defaultdict(list)
 for z in rr:pool[(z['family'],z['prompt_tokens'])].append(z)
 keys=[k for k,v in sorted(pool.items()) if len(v)>=4];rng=random.Random(76311);opt=torch.optim.AdamW(m.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);logs=[];bh=hashlib.sha256();t=time.perf_counter()
 for step in range(1,1001):
  m.train();opt.zero_grad(set_to_none=True);key=keys[rng.randrange(len(keys))];bb=rng.choices(pool[key],k=16);x,y,p=batch(bb);t0=time.perf_counter();ll=m(x,y,p);ll.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.));assert math.isfinite(float(ll)) and math.isfinite(gn)
  lr=.001*min(1,step/100)*(.15+.85*.5*(1+math.cos(math.pi*step/1000)))
  for g in opt.param_groups:g['lr']=lr
  opt.step();row={'step':step,'loss':float(ll.detach()),'lr':lr,'grad_norm':gn,'seconds':time.perf_counter()-t0,'batch_ids':[z['id'] for z in bb],'input_tokens':sum(len(z['tokens'])-1 for z in bb),'loss_tokens':int(y.ne(-100).sum()),'padded_tokens':x.numel()};logs.append(row)
  for z in bb:bh.update((z['id']+'\n').encode())
  with (ROOT/'metrics'/f'{arm}_s{seed}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if step%100==0:print('TRAIN',arm,seed,step,round(row['loss'],4),round(row['seconds'],3),flush=True)
  if step%200==0:
   opt.zero_grad(set_to_none=True);cp=save(m,opt,rng,arm,seed,step,logs);put(ROOT/'reports/NEXT_STATE.json',{'status':'running','active_arm':arm,'active_seed':seed,'last_durable_step':step,'checkpoint':str(cp),'do_not_duplicate':True})
 tests={sp:evaluate(m,load(sp,arm)) for sp in SPLITS};st=torch.load(cp/'resume.pt',weights_only=False);n=Model(Config());n.load_state_dict(load_file(str(cp/'model.safetensors')))
 a=evaluate(m,load('test_iid',arm)[:12])['rows'];b=evaluate(n,load('test_iid',arm)[:12])['rows'];assert a==b
 assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
 # Verify exact independent next-update replay from saved optimizer and RNG.
 resume_hash=[]
 for _ in range(2):
  n=Model(Config());n.load_state_dict(st['model']);oo=torch.optim.AdamW(n.parameters(),lr=.001,betas=(.9,.95),weight_decay=.01);oo.load_state_dict(copy.deepcopy(st['optimizer']));r=random.Random();r.setstate(st['sampler_rng']);torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);k=keys[r.randrange(len(keys))];xx,yy,pp=batch(r.choices(pool[k],k=16));n.train();oo.zero_grad(set_to_none=True);l=n(xx,yy,pp);l.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),1.);oo.step();resume_hash.append(state_hash(n))
 assert resume_hash[0]==resume_hash[1]
 result={'status':'completed','arm':arm,'seed':seed,'parameters':469648,'steps':1000,'input_tokens':sum(x['input_tokens'] for x in logs),'loss_tokens':sum(x['loss_tokens'] for x in logs),'train_seconds':sum(x['seconds'] for x in logs),'elapsed_seconds':time.perf_counter()-t,'initial_state_sha256':init,'final_state_sha256':state_hash(m),'batch_sequence_sha256':bh.hexdigest(),'checkpoint':str(cp),'checkpoint_manifest':json.loads((cp/'MANIFEST.json').read_text()),'reload_cases':12,'reload_mismatches':0,'resume_next_update_verified':True,'tests':tests}
 put(out,result);print('COMPLETE',arm,seed,{s:v['correct'] for s,v in tests.items()},flush=True)
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--seed',choices=SEEDS,type=int,required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);train(a.arm,a.seed)
