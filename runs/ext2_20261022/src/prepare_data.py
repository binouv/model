"""Freeze only published records; no invented questions, targets, or rationales."""
import collections,hashlib,json,random,re,unicodedata,zipfile
from pathlib import Path
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'ext_20261018'
SEED=20261022
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).lower()).strip()
def prompt(user):return 'User: '+user+'\nAssistant:'
def mcq(task,r,split,index):
    if task in ['openbookqa','arc_challenge']:
        question=r['question']['stem'];choices=[c['text'] for c in r['question']['choices']]
        answer=[c['label'] for c in r['question']['choices']].index(r['answerKey'])
    else:question=r['goal'];choices=[r['sol1'],r['sol2']];answer=r['label']
    letters=[chr(65+i) for i in range(len(choices))]
    user='Answer the question. Reply with only the letter '+', '.join(letters)+'.\nQuestion: '+question+'\n'+'\n'.join(f'{l}. {c}' for l,c in zip(letters,choices))
    return {'id':task+':'+split+':'+str(r.get('id',index)),'task':task,'source_split':split,'source_index':index,'question':question,'choices':choices,'gold_index':answer,'user_text':user,'prompt':prompt(user),'completion':' '+letters[answer],'original':r}
def write(name,rows):
    p=ROOT/'data'/name
    with p.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return {'count':len(rows),'sha256':sha(p),'bytes':p.stat().st_size}
def main():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');z=zipfile.ZipFile(OLD/'data/OpenCompassData-core-20240207.zip')
    sources={};pools={}
    paths={'openbookqa_train':'data/openbookqa/Main/train.jsonl','openbookqa_dev':'data/openbookqa/Main/dev.jsonl','openbookqa_test':'data/openbookqa/Main/test.jsonl','piqa_train':'data/piqa/train.jsonl','piqa_dev':'data/piqa/dev.jsonl','arc_challenge_dev':'data/ARC/ARC-c/ARC-Challenge-Dev.jsonl','arc_challenge_test':'data/ARC/ARC-c/ARC-Challenge-Test.jsonl'}
    for key,path in paths.items():
        data=z.read(path);sources[path]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        raw=[json.loads(s) for s in data.splitlines()];task,split=key.rsplit('_',1)
        if task=='piqa':
            lp='data/piqa/'+split+'-labels.lst';lb=z.read(lp);labels=list(map(int,lb.splitlines()));assert len(labels)==len(raw)
            sources[lp]={'sha256':hashlib.sha256(lb).hexdigest(),'bytes':len(lb)}
            raw=[dict(r,label=y) for r,y in zip(raw,labels)]
        pools[key]=[mcq(task,r,split,i) for i,r in enumerate(raw) if task=='piqa' or len(r['question']['choices'])==4]
    rng=random.Random(SEED);dev=[];test=[]
    for task in ['openbookqa','arc_challenge']:
        for split,n,dst in [('dev',64,dev),('test',256,test)]:
            pool=pools[task+'_'+split][:];rng.shuffle(pool);dst.extend(pool[:n])
    pi=pools['piqa_dev'][:];rng.shuffle(pi);dev.extend(pi[:64]);test.extend(pi[64:320])
    # Remove repeated question text from held-out evaluation before any test run.
    occupied={norm(r['question']) for r in dev+test};seen_test=set();replaced_test_duplicates=0
    for i,r in enumerate(test):
        key=norm(r['question'])
        if key in seen_test:
            pool=pools[r['task']+('_dev' if r['task']=='piqa' else '_test')]
            replacement=next(x for x in pool if norm(x['question']) not in occupied)
            test[i]=replacement;key=norm(replacement['question']);occupied.add(key);replaced_test_duplicates+=1
        seen_test.add(key)
    assert not ({norm(r['question']) for r in dev}&seen_test)
    held_questions={norm(r['question']) for r in dev+test}
    held_questions.update(norm(json.loads(s)['question']) for s in (OLD/'data/arc_test128.jsonl').read_text().splitlines())
    removed=collections.Counter({'test_duplicate_questions_replaced_before_evaluation':replaced_test_duplicates});train_by={}
    def token_info(r):
        r['prompt_tokens']=len(tok.encode(r['prompt'],bos=True));r['answer_tokens']=len(tok.encode(r['completion'],eos=True))
        return r['prompt_tokens']+r['answer_tokens']<=512
    for task in ['openbookqa','piqa']:
        seen=set();pool=[]
        for r in pools[task+'_train']:
            q=norm(r['question'])
            if q in held_questions or q in seen:removed[task+'_overlap_or_duplicate']+=1;continue
            seen.add(q)
            if not token_info(r):removed[task+'_over_length']+=1;continue
            pool.append(r)
        rng.shuffle(pool);assert len(pool)>=4096,(task,len(pool));train_by[task]=pool[:4096]
    raw=json.loads((OLD/'data/alpaca_raw.json').read_text());old_seen={json.loads(s)['id'] for name in ['train.jsonl','validation.jsonl'] for s in (OLD/'data'/name).read_text().splitlines()}
    alp=[];seen=set()
    for i,r in enumerate(raw):
        user=r['instruction']+('\n'+r['input'] if r['input'] else '');key=norm(user)
        if key in seen:removed['alpaca_duplicate']+=1;continue
        seen.add(key)
        row={'id':f'alpaca:{i}','task':'alpaca','source_index':i,'user_text':user,'prompt':prompt(user),'completion':' '+r['output'],'original':r}
        if not r['output'].strip() or not token_info(row):removed['alpaca_empty_or_over_length']+=1;continue
        alp.append(row)
    rng.shuffle(alp);fresh=[r for r in alp if r['id'] not in old_seen]
    adev=fresh[:128];atest=fresh[128:256];held_ids={r['id'] for r in adev+atest};held_inputs={norm(r['user_text']) for r in adev+atest}
    # Exact normalized cross-source substring contamination screen. No model
    # outputs or test outcomes influence selection. Unknown historic pretraining
    # contamination and paraphrases remain outside what this check proves.
    needles=sorted(held_questions,key=len,reverse=True)
    for r in alp:
        if r['id'] in held_ids:continue
        text=norm(r['user_text']+' '+r['original']['output'])
        if any(q in text for q in needles):removed['alpaca_held_question_substring']+=1;continue
        train_by.setdefault('alpaca',[]).append(r)
        if len(train_by['alpaca'])==8192:break
    assert len(train_by['alpaca'])==8192
    assert not ({norm(r['user_text']) for r in train_by['alpaca']}&held_inputs)
    for task in ['openbookqa','piqa']:
        assert not ({norm(r['question']) for r in train_by[task]}&held_questions)
    # Two general instruction examples and one example from each QA dataset per
    # update. All are unmodified source questions/targets plus serialization.
    train=[]
    for i in range(4096):train.extend([train_by['alpaca'][2*i],train_by['openbookqa'][i],train_by['alpaca'][2*i+1],train_by['piqa'][i]])
    for r in dev+test+adev+atest:token_info(r)
    assert all(r['prompt_tokens']+16<=2048 for r in dev+test)
    files={}
    for name,rows in [('train.jsonl',train),('dev_mcq.jsonl',dev),('test_mcq.jsonl',test),('dev_alpaca.jsonl',adev),('test_alpaca.jsonl',atest)]:files[name]=write(name,rows)
    manifest={'seed':SEED,'created_before_new_model_training_or_test_evaluation':True,'external_archive':{'release':'OpenCompassData-core-20240207 / 0.2.2.rc1','url':'https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip','sha256':sha(OLD/'data/OpenCompassData-core-20240207.zip'),'members':sources},'alpaca':{'revision':'761dc5bfbdeeffa89b8bff5d038781a4055f796a','sha256':sha(OLD/'data/alpaca_raw.json'),'url':'https://github.com/tatsu-lab/stanford_alpaca','license':'CC-BY-NC-4.0, research/noncommercial','new_holdouts_exclude_old_1088_records':True},'licenses':{'openbookqa':'Apache-2.0, https://github.com/allenai/OpenBookQA/blob/main/LICENSE','piqa':'AFL-3.0, https://github.com/ybisk/ybisk.github.io/blob/master/piqa/README.md','arc_challenge':'CC-BY-SA-4.0, ARC-V1-Feb2018'},'splits':{'train':{'alpaca':8192,'openbookqa_official_train':4096,'piqa_official_train':4096},'dev':{'alpaca_new_disjoint':128,'openbookqa_official_dev':64,'piqa_official_dev_disjoint_subset':64,'arc_challenge_official_dev':64},'test':{'alpaca_new_disjoint':128,'openbookqa_official_test':256,'piqa_official_dev_remaining_disjoint_subset':256,'arc_challenge_official_test':256}},'selection':{'max_train_tokens':512,'minimum_nonempty_answer_tokens':1,'no_text_truncation':True,'no_model_outcome_filter':True,'old_arc128_not_used_for_selection_or_training':True,'removed':dict(removed)},'overlap_checks':{'train_vs_dev_test_normalized_same_source_inputs':0,'selected_alpaca_train_contains_selected_mcq_held_or_old_arc128_question':0,'historic_pretraining_contamination':'unknown','paraphrase_leakage':'not exhaustively established'},'files':files}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':files,'removed':dict(removed),'short_answers_under_8_tokens':sum(r['answer_tokens']<8 for r in train),'input_tokens':sum(r['prompt_tokens']+r['answer_tokens']-1 for r in train)},indent=2))
if __name__=='__main__':main()
