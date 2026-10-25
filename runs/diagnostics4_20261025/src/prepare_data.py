import collections,datetime,hashlib,json,random,re,unicodedata,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'ext_20261018';E2=ROOT.parent/'ext2_20261022';E3=ROOT.parent/'ext3_20261025';SEED=2026102504

def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).lower()).strip()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def make(task,r,split,i):
 if task=='piqa':q=r['goal'];cc=[r['sol1'],r['sol2']];y=r['label']
 else:q=r['question']['stem'];cc=[c['text'] for c in r['question']['choices']];y=[c['label'] for c in r['question']['choices']].index(r['answerKey'])
 labels=[chr(65+i) for i in range(len(cc))];u='Answer the question. Reply with only the letter '+', '.join(labels)+'.\nQuestion: '+q+'\n'+'\n'.join(a+'. '+b for a,b in zip(labels,cc))
 return dict(id=task+':'+split+':'+str(r.get('id',i)),task=task,source_split=split,source_index=i,question=q,choices=cc,gold_index=y,prompt='User: '+u+'\nAssistant:',completion=' '+labels[y],original=r)
def main():
 rng=random.Random(SEED);seen=[]
 for parent in [E2,E3]:
  for name in ['train.jsonl','dev_mcq.jsonl','test_mcq.jsonl']:seen+=rows(parent/'data'/name)
 excluded={norm(r['question']) for r in seen if 'question' in r};excluded.update(norm(r['question']) for r in rows(OLD/'data/arc_test128.jsonl'))
 # These records were seen by both selected EXT2 and selected EXT3(step640).
 available=rows(E3/'data/train.jsonl')[:3*640];train=[];control=[];tiny=[]
 for task,nlabels in [('openbookqa',4),('piqa',2)]:
  for label in range(nlabels):
   pool=[r for r in available if r['task']==task and r['gold_index']==label];rng.shuffle(pool);assert len(pool)>=256//nlabels;part=pool[:256//nlabels];train+=part;control+=part[:32//nlabels];tiny+=part[:16//nlabels]
 rng.shuffle(train);rng.shuffle(control);rng.shuffle(tiny)
 z=zipfile.ZipFile(OLD/'data/OpenCompassData-core-20240207.zip');pools={};members={}
 paths={'openbookqa_dev':'data/openbookqa/Main/dev.jsonl','openbookqa_test':'data/openbookqa/Main/test.jsonl','piqa_dev':'data/piqa/dev.jsonl','arc_challenge_dev':'data/ARC/ARC-c/ARC-Challenge-Dev.jsonl','arc_challenge_test':'data/ARC/ARC-c/ARC-Challenge-Test.jsonl'}
 for key,path in paths.items():
  data=z.read(path);members[path]=hashlib.sha256(data).hexdigest();rr=[json.loads(s) for s in data.splitlines()];task,split=key.rsplit('_',1)
  if task=='piqa':
   b=z.read('data/piqa/dev-labels.lst');members['data/piqa/dev-labels.lst']=hashlib.sha256(b).hexdigest();rr=[dict(r,label=y) for r,y in zip(rr,map(int,b.splitlines()))]
  pool=[make(task,r,split,i) for i,r in enumerate(rr) if task=='piqa' or len(r['question']['choices'])==4];rng.shuffle(pool);pools[key]=pool
 # Reject complete long stems occurring in known prior Alpaca instruction data.
 raw=json.loads((OLD/'data/alpaca_raw.json').read_text());oldids={int(str(r['id']).split(':')[-1]) for r in rows(OLD/'data/train.jsonl')};texts=[norm(r['user_text']+' '+r['completion']) for r in rows(E2/'data/train.jsonl') if r['task']=='alpaca']+[norm(raw[i]['instruction']+' '+raw[i]['input']+' '+raw[i]['output']) for i in oldids];combined='\n'.join(texts)
 dev=[];test=[];occupied=set(excluded);rejected=0
 for task in ['openbookqa','piqa','arc_challenge']:
  for split,n,dest in [('dev',32,dev),('test',64,test)]:
   pool=pools[task+('_dev' if task=='piqa' else '_'+split)];count=0
   for r in pool:
    q=norm(r['question'])
    if q in occupied:continue
    if len(q)>=20 and q in combined:rejected+=1;continue
    if any(q in t and all(norm(c) in t for c in r['choices']) for t in texts):rejected+=1;continue
    dest.append(r);occupied.add(q);count+=1
    if count==n:break
   assert count==n,(task,split,count)
 files={}
 for name,data in [('probe_train.jsonl',train),('seen_control.jsonl',control),('tiny_train.jsonl',tiny),('dev_mcq.jsonl',dev),('test_mcq.jsonl',test)]:
  p=ROOT/'data'/name
  with p.open('x') as f:
   for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  files[name]=dict(n=len(data),sha256=sha(p))
 manifest=dict(seed=SEED,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive_sha256=sha(OLD/'data/OpenCompassData-core-20240207.zip'),source_release='OpenCompassData-core-20240207 /0.2.2.rc1',source_members=members,files=files,train_seen_by_both_models=True,selected_EXT3_training_cursor_exclusion='only source rows before3*640',balanced_probe_train_per_task=256,seen_control_per_task=32,tiny_train_per_task=16,new_dev_per_task=32,new_test_per_task=64,all_new_holdouts_exclude_previous_EXT2_EXT3_mcq_train_dev_test_and_ARC128=True,known_prior_alpaca_long_stem_or_full_item_exclusions=rejected,piqa_test='new reserved official validation subset; not hidden official test',historical_pretraining_overlap='unknown',no_generated_question_or_answer_content=True)
 (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(files,indent=2))
if __name__=='__main__':main()
