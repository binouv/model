"""Fresh counterfactual pairs; offline labels never used by model inference.
All generated cases are disjoint from available ancestor case groups. Pair
members, commutations and order-preserving memory interleavings share split.
"""
from __future__ import annotations
import collections,copy,json,random,sys,zipfile
from pathlib import Path
from common import *
import parent_data as old
FAMILIES=old.FAMILIES

def group(c):
 f=c['family']
 if f in ('arithmetic','list_reasoning'):x={'family':f,'values':sorted(c['values'])}
 elif f=='memory_update':x={'family':f,'histories':{k:[v for kk,v in c['writes'] if kk==k] for k in sorted({k for k,v in c['writes']})}}
 elif f=='code_trace':x={'family':f,'start':c['start'],'operands':sorted(a for op,a in c['program'])}
 elif f=='conditional':x={'family':f,'x':c['values'][0],'branches':c['values'][2:]}
 else:raise ValueError('unknown case family')
 return hash_obj(x)

def independent(c):
 f=c['family']
 if f=='arithmetic':
  a,b=c['values'];return sum((a,b)) if c['op']=='+' else sum((a,-b)) if c['op']=='-' else sum(a for _ in range(b))
 if f=='code_trace':return sum([c['start']]+[a if op=='+' else -a for op,a in c['program']])
 if f=='conditional':
  x,k,a,b=c['values'];return sum((x,a if x<k else -b))
 if f=='list_reasoning':
  v=sorted(c['values']);return sum(v) if c['kind']=='sum' else v[0] if c['kind']=='min' else v[-1]
 return next(v for k,v in reversed(c['writes']) if k==c['query'])

def pair(rng,f,ood=False):
 c=old.make_case(rng,f,ood);b=copy.deepcopy(c)
 if f=='arithmetic':b['op']=rng.choice([op for op in '+-*' if op!=c['op']])
 elif f=='code_trace':
  j=rng.randrange(len(b['program']));b['program'][j][0]='-' if b['program'][j][0]=='+' else '+'
 elif f=='conditional':
  x,k,a,d=c['values'];b['values'][1]=x if x<k else x+1
 elif f=='list_reasoning':b['kind']=rng.choice([k for k in ['sum','min','max'] if k!=c['kind']])
 else:b['query']=rng.choice([k for k in 'abcd' if k!=c['query']])
 assert group(c)==group(b)
 return c,b

def pack(c,tok,lang,variant,pid,member):
 p=old.render(c,lang,variant);answer=independent(c)
 assert answer==old.solve(c)
 completion='F='+str(answer);pt=tok.encode(p,bos=True);ct=tok.encode(completion,eos=True)
 assert tok.encode(p+completion,bos=True,eos=True)==pt+ct
 return {'id':hash_obj({'case':c,'lang':lang,'variant':variant}),'group':group(c),'pair_id':pid,'member':member,'family':c['family'],'lang':lang,'variant':variant,'case':c,'prompt':p,'answer':str(answer),'completion':completion,'tokens':pt+ct,'prompt_tokens':len(pt)}

def build(parent,legacy,out=ROOT/'data'):
 out=Path(out);out.mkdir(exist_ok=True);tok=Tokenizer.load(out/'tokenizer.json');denied=set();source_counts={}
 for p in sorted(Path(parent).glob('reasoning_*.jsonl')):
  rr=read(p);source_counts[p.name]=len(rr)
  denied.update(group(z['case']) for z in rr)
 with zipfile.ZipFile(legacy) as zz:
  n=0
  for name in zz.namelist():
   if '/data/' not in name or not name.endswith('.jsonl'):continue
   for line in zz.read(name).decode().splitlines():
    z=json.loads(line)
    if 'case' in z:
     denied.add(group(z['case']));n+=1
  source_counts['G8C_archive_case_records']=n
 occupied=set(denied);spec=[('train',4096,91011,0,False),('validation',100,91012,0,False),('test_iid',200,91013,0,False),('test_surface',100,91014,1,False),('test_extrapolation',100,91015,1,True)]
 manifests={};rejected=collections.Counter();written={}
 for split,npairs,seed,variant,ood in spec:
  rng=random.Random(seed);rows=[];attempt=0
  while len(rows)//2<npairs:
   attempt+=1
   if attempt>1000000:raise RuntimeError('case space exhausted')
   f=FAMILIES[(len(rows)//2)%5];a,b=pair(rng,f,ood);g=group(a)
   if g in occupied:rejected[split+':overlap']+=1;continue
   aa,bb=independent(a),independent(b)
   if aa==bb:rejected[split+':same_answer']+=1;continue
   # Match answer token lengths to prevent rank loss exploiting answer length.
   if len(tok.encode('F='+str(aa),eos=True))!=len(tok.encode('F='+str(bb),eos=True)):
    rejected[split+':answer_length']+=1;continue
   lang='ru' if (len(rows)//2)%2 else 'en';pid=split+f'_p{len(rows)//2:05d}'
   pp=[pack(c,tok,lang,variant,pid,j) for j,c in enumerate((a,b))]
   if max(len(z['tokens']) for z in pp)>129:
    rejected[split+':overlength']+=1;continue
   occupied.add(g);rows.extend(pp)
  text=''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in rows)
  p=out/(split+'.jsonl');p.write_text(text);written[split]=rows
  manifests[split]={'examples':len(rows),'pairs':npairs,'seed':seed,'sha256':sha(p),'lang':dict(collections.Counter(z['lang'] for z in rows)),'tokens':sum(len(z['tokens'])-1 for z in rows)}
 sets={s:{z['group'] for z in rr} for s,rr in written.items()}
 assert all(not a&b for i,a in enumerate(sets.values()) for b in list(sets.values())[i+1:])
 assert all(not denied&s for s in sets.values())
 for s,rr in written.items():
  assert all(z['group']==group(z['case']) and int(z['answer'])==independent(z['case']) for z in rr)
  assert all(rr[i]['answer']!=rr[i+1]['answer'] for i in range(0,len(rr),2))
 audit={'status':'completed_data_audit','splits':manifests,'rejected':dict(rejected),'ancestor_source_record_counts':source_counts,'excluded_ancestor_groups':len(denied),'observed_cross_split_group_overlap':0,'ancestor_group_overlap':0,'max_sequence_tokens':128,'answer_token_lengths_matched':True,'same_lexical_tokenizer_sha256':sha(out/'tokenizer.json'),'note':'English/Russian synthetic paired probes. Range 0..255, extrapolation256..511 plus length. All answers independent offline replay. Source-file split exclusion is not proof against every functional equivalence.'}
 put(out/'MANIFEST.json',audit);return audit

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--parent-data',required=True);ap.add_argument('--legacy-zip',required=True);args=ap.parse_args()
 print(json.dumps(build(args.parent_data,args.legacy_zip),ensure_ascii=False,indent=2))
