"""New frozen public-data splits; cyclic answer orders are lossless permutations."""
import collections,datetime,hashlib,json,random,re,unicodedata,zipfile
from pathlib import Path
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'ext2_20261022';RAW=ROOT.parent/'ext_20261018';SEED=20261025

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).lower()).strip()
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def serialize(task,r,split,i):
 if task=='piqa':q=r['goal'];choices=[r['sol1'],r['sol2']];gold=r['label']
 else:q=r['question']['stem'];choices=[x['text'] for x in r['question']['choices']];gold=[x['label'] for x in r['question']['choices']].index(r['answerKey'])
 letters=[chr(65+k) for k in range(len(choices))];user='Answer the question. Reply with only the letter '+', '.join(letters)+'.\nQuestion: '+q+'\n'+'\n'.join(l+'. '+c for l,c in zip(letters,choices))
 return dict(id=task+':'+split+':'+str(r.get('id',i)),task=task,source_split=split,source_index=i,question=q,choices=choices,gold_index=gold,user_text=user,prompt='User: '+user+'\nAssistant:',completion=' '+letters[gold],original=r)
def main():
 rng=random.Random(SEED);tok=Tokenizer.load(ROOT/'data/tokenizer.json');z=zipfile.ZipFile(RAW/'data/OpenCompassData-core-20240207.zip')
 previous=[]
 for name in ['train.jsonl','dev_mcq.jsonl','test_mcq.jsonl','dev_alpaca.jsonl','test_alpaca.jsonl']:previous.extend(read(OLD/'data'/name))
 excluded_questions={norm(r['question']) for r in previous if 'question' in r}
 excluded_questions.update(norm(r['question']) for r in read(RAW/'data/arc_test128.jsonl'))
 pools={};members={}
 paths={'openbookqa_dev':'data/openbookqa/Main/dev.jsonl','openbookqa_test':'data/openbookqa/Main/test.jsonl','piqa_dev':'data/piqa/dev.jsonl','arc_challenge_dev':'data/ARC/ARC-c/ARC-Challenge-Dev.jsonl','arc_challenge_test':'data/ARC/ARC-c/ARC-Challenge-Test.jsonl'}
 for key,path in paths.items():
  b=z.read(path);members[path]=hashlib.sha256(b).hexdigest();rr=[json.loads(s) for s in b.splitlines()];task,split=key.rsplit('_',1)
  if task=='piqa':
   b=z.read('data/piqa/dev-labels.lst');members['data/piqa/dev-labels.lst']=hashlib.sha256(b).hexdigest();yy=list(map(int,b.splitlines()));rr=[dict(r,label=y) for r,y in zip(rr,yy)]
  pool=[serialize(task,r,split,i) for i,r in enumerate(rr) if task=='piqa' or len(r['question']['choices'])==4];rng.shuffle(pool);pools[key]=pool
 seen=set(excluded_questions);dev=[];test=[]
 for task in ['openbookqa','piqa','arc_challenge']:
  for split,n,dest in [('dev',32,dev),('test',128,test)]:
   key=task+('_dev' if task=='piqa' else '_'+split);count=0
   for r in pools[key]:
    q=norm(r['question'])
    if q in seen:continue
    if len(tok.encode(r['prompt'],bos=True))+16>2048:continue
    dest.append(r);seen.add(q);count+=1
    if count==n:break
   assert count==n,(task,split,count)
 held={norm(r['question']) for r in dev+test};data=read(OLD/'data/train.jsonl');by={}
 for task in ['openbookqa','piqa','alpaca']:
  pool=[r for r in data if r['task']==task and ('question' not in r or norm(r['question']) not in held)]
  if task=='alpaca':pool=[r for r in pool if not any(q in norm(r['user_text']+' '+r['completion']) for q in held)]
  rng.shuffle(pool);assert len(pool)>=768;by[task]=pool[:768]
 train=[]
 for i in range(768):train.extend([by['openbookqa'][i],by['piqa'][i],by['alpaca'][i]])
 oldids={r['id'] for r in previous if r['task']=='alpaca'}
 oldids.update(r['id'] for n in ['train.jsonl','validation.jsonl'] for r in read(RAW/'data'/n))
 oldusers={norm(r['user_text']) for r in previous if r['task']=='alpaca'}
 raw=json.loads((RAW/'data/alpaca_raw.json').read_text());idx=list(range(len(raw)));rng.shuffle(idx);fresh=[];seenusers=set(oldusers)
 prior_ids={int(str(r['id']).split(':')[-1]) for n in ['train.jsonl','validation.jsonl'] for r in read(RAW/'data'/n)}
 for i in prior_ids:seenusers.add(norm(raw[i]['instruction']+('\n'+raw[i]['input'] if raw[i]['input'] else '')))
 for i in idx:
  r=raw[i];user=r['instruction']+('\n'+r['input'] if r['input'] else '');key=norm(user)
  if f'alpaca:{i}' in oldids or key in seenusers or not r['output'].strip():continue
  row=dict(id=f'alpaca:{i}',task='alpaca',source_index=i,user_text=user,prompt='User: '+user+'\nAssistant:',completion=' '+r['output'],original=r)
  if len(tok.encode(row['prompt'],bos=True))+len(tok.encode(row['completion'],eos=True))>512:continue
  fresh.append(row);seenusers.add(key)
  if len(fresh)==96:break
 assert len(fresh)==96
 files={}
 for name,rr in [('train.jsonl',train),('dev_mcq.jsonl',dev),('test_mcq.jsonl',test),('dev_alpaca.jsonl',fresh[:32]),('test_alpaca.jsonl',fresh[32:])]:
  p=ROOT/'data'/name
  with p.open('x') as f:
   for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  files[name]=dict(count=len(rr),sha256=sha(p))
 # All new MCQ holdouts excluded from every prior EXT2 MCQ split; Alpaca test
 # excludes prior train/dev/test IDs and normalized instructions, not only IDs.
 trainqs={norm(r['question']) for r in train if 'question' in r};assert not trainqs&held
 manifest=dict(seed=SEED,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive_sha256=sha(RAW/'data/OpenCompassData-core-20240207.zip'),source_release='OpenCompassData-core-20240207 / 0.2.2.rc1',source_members=members,original_ext2_data_manifest_sha256=sha(OLD/'data/manifest.json'),alpaca_revision='761dc5bfbdeeffa89b8bff5d038781a4055f796a',alpaca_raw_sha256=sha(RAW/'data/alpaca_raw.json'),licenses={'alpaca':'CC-BY-NC-4.0','openbookqa':'Apache-2.0','piqa':'AFL-3.0','arc_challenge':'CC-BY-SA-4.0'},files=files,train_unique_source_examples=2304,train_overlap_with_prior_training='allowed and disclosed; source records sampled from EXT2 training only',new_mcq_holdout_intersection_with_prior_ext2_mcq_train_dev_test_and_arc128=0,new_alpaca_holdouts_exclude_prior_ext_and_ext2_train_dev_test=True,piqa_test='reserved new disjoint official validation subset; not hidden official test',augmentation='cyclic permutations of original unchanged alternatives; gold index remapped; no new questions, answers, or rationales',historical_pretraining_contamination='unknown; semantic/paraphrase overlap not excluded')
 (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(files,indent=2))
if __name__=='__main__':main()
