import collections,hashlib,json,random
from pathlib import Path
from model import ByteTokenizer
FAMS=('arithmetic','conditional','code_trace','list_reasoning','memory_update')
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def canonical(c):
 f=c['family']
 if f in ('arithmetic','list_reasoning'):return h({'family':f,'values':sorted(c['values'])})
 if f=='memory_update':
  ks=sorted({k for k,_ in c['writes']});return h({'family':f,'histories':{k:[v for kk,v in c['writes'] if kk==k] for k in ks}})
 return h(c)
def solve(c):
 f=c['family']
 if f=='arithmetic':a,b=c['values'];return {'+':a+b,'-':a-b,'*':a*b}[c['op']]
 if f=='conditional':x,k,a,b=c['values'];return x+a if x<k else x-b
 if f=='code_trace':
  x=c['start']
  for op,a in c['program']:x=x+a if op=='+' else x-a
  return x
 if f=='list_reasoning':v=c['values'];return {'sum':sum(v),'min':min(v),'max':max(v)}[c['kind']]
 mem={}
 for k,v in c['writes']:mem[k]=v
 return mem[c['query']]
def make(r,f,ood=False):
 lo,hi=(128,255) if ood else (0,127)
 if f=='arithmetic':return {'family':f,'values':[r.randint(lo,hi),r.randint(lo,hi)],'op':r.choice('+-*')}
 if f=='conditional':return {'family':f,'values':[r.randint(lo,hi) for _ in range(4)]}
 if f=='code_trace':return {'family':f,'start':r.randint(lo,hi),'program':[[r.choice('+-'),r.randint(lo,hi)] for _ in range(4 if ood else 2)]}
 if f=='list_reasoning':return {'family':f,'values':[r.randint(lo,hi) for _ in range(7 if ood else 4)],'kind':r.choice(['sum','min','max'])}
 ks=list('abcd');w=[[k,r.randint(lo,hi)] for k in ks]
 for _ in range(6 if ood else 3):w.append([r.choice(ks),r.randint(lo,hi)])
 r.shuffle(w);return {'family':f,'writes':w,'query':r.choice(ks)}
def render(c,v=0):
 f=c['family']
 if f=='arithmetic':
  a,b=c['values'];e=f'{a} {c["op"]} {b}';return f'Calculate {e}.\nAnswer:' if v==0 else f'What integer is {e}?\nAnswer:'
 if f=='conditional':
  x,k,a,b=c['values'];return f'x={x}. If x<{k}, add {a}; else subtract {b}.\nAnswer:' if v==0 else f'Start at {x}. Below {k}: +{a}. Otherwise: -{b}.\nAnswer:'
 if f=='code_trace':
  s='x = '+str(c['start'])+'\n'+''.join(f'x {op}= {a}\n' for op,a in c['program'])+'print(x)';return ('What integer does this code print?\n' if v==0 else 'Trace this program and return only the printed integer.\n')+s+'\nAnswer:'
 if f=='list_reasoning':
  lab={'sum':'sum','min':'minimum','max':'maximum'}[c['kind']];return f'Find the {lab} of {c["values"]}.\nAnswer:' if v==0 else f'Numbers: {c["values"]}. Return their {lab}.\nAnswer:'
 w='; '.join(f'{k}={x}' for k,x in c['writes']);return f'Writes in time order: {w}. Latest value of {c["query"]}?\nAnswer:' if v==0 else f'Apply these updates chronologically: {w}. Read current {c["query"]}.\nAnswer:'
def row(c,v,tok,pid=None):
 p=render(c,v);a=str(solve(c));pt=tok.encode(p,bos=True);return {'id':h({'case':c,'variant':v,'pair':pid}),'canonical':canonical(c),'family':c['family'],'prompt':p,'answer':a,'case':c,'tokens':pt+tok.encode(a,eos=True),'prompt_tokens':len(pt),'pair_id':pid}
def build(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);tok=ByteTokenizer();occ=set();man={};spec=[('train',8192,76011,0,0),('validation',400,76012,0,0),('test_iid',160,76013,0,0),('test_surface',80,76014,0,1),('test_extrapolation',80,76015,1,1)]
 for split,n,seed,ood,v in spec:
  r=random.Random(seed);out=[];attempt=0
  while len(out)<n:
   attempt+=1
   if attempt>2000000:raise RuntimeError('unique-case space exhausted')
   c=make(r,FAMS[len(out)%5],bool(ood));g=canonical(c)
   if g in occ:continue
   z=row(c,v,tok)
   if len(z['tokens'])>256:continue
   out.append(z);occ.add(g)
  p=root/f'{split}.jsonl';p.write_text(''.join(json.dumps(z,separators=(',',':'))+'\n' for z in out));man[split]={'n':n,'attempts':attempt,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 r=random.Random(76016);out=[];pairs=0
 while pairs<40:
  f='arithmetic' if pairs<20 else 'memory_update';c=make(r,f);g=canonical(c)
  if g in occ:continue
  c2=json.loads(json.dumps(c))
  if f=='arithmetic':c2['op']=r.choice([x for x in '+-*' if x!=c['op']])
  else:
   ks=sorted({k for k,_ in c['writes']});c2['query']=r.choice([k for k in ks if k!=c['query']])
  if solve(c)==solve(c2):continue
  pid=f'cf{pairs:03d}';a,b=row(c,0,tok,pid),row(c2,0,tok,pid)
  if max(len(a['tokens']),len(b['tokens']))>256:continue
  out += [a,b];occ.add(g);pairs+=1
 p=root/'test_counterfactual.jsonl';p.write_text(''.join(json.dumps(z,separators=(',',':'))+'\n' for z in out));man['test_counterfactual']={'n':80,'pairs':40,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 seen={}
 for split,_,_,_,_ in spec:
  for z in map(json.loads,(root/f'{split}.jsonl').read_text().splitlines()):
   assert str(solve(z['case']))==z['answer'];assert z['canonical'] not in seen;seen[z['canonical']]=split
 for z in map(json.loads,p.read_text().splitlines()):
  assert str(solve(z['case']))==z['answer'];assert z['canonical'] not in seen or seen[z['canonical']]=='test_counterfactual';seen[z['canonical']]='test_counterfactual'
 (root/'manifest.json').write_text(json.dumps({'status':'completed','tokenizer':'byte259','cross_split_canonical_overlap':0,'chronology':'per-key write order retained','normal_range':[0,127],'extrapolation_range':[128,255],'splits':man},indent=2));return man
