"""Verified synthetic skills. Solvers and state labels are data/evaluation ONLY.
Semantic groups are assigned before wording, and train/test groups cannot overlap.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from common import ROOT, Tokenizer, put, sha, rows, write_rows
FAMILIES=('arithmetic','conditional','code_trace','list_reasoning','memory_update','code_generation')

def canonical(c):
    f=c['family']
    if f=='arithmetic':z=(f,c['op'],sorted((c['a'],c['b'])) if c['op']=='+' else [c['a'],c['b']])
    elif f=='memory_update':
        # Remove inter-key ordering while preserving the history of each key.
        histories={}
        for k,v in c['writes']:histories.setdefault(k,[]).append(v)
        z=(f,sorted(histories.items()))  # query and phrasing do NOT split a bank
    elif f=='code_generation':
        a,b=1,0
        for op,v in c['ops']:
            if op=='*':a*=v;b*=v
            elif op=='+':b+=v
            else:b-=v
        z=(f,a,b)  # exclude semantically equivalent affine programs
    elif f=='list_reasoning':z=(f,sorted(c['values']))
    else:z=c
    return hashlib.sha256(json.dumps(z,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def solve(c):
    f=c['family']
    if f=='arithmetic':
        a,b=c['a'],c['b'];v=a+b if c['op']=='+' else a-b
        s=f"T={a}{c['op']}{b}={v};F={v}"
        start=s.index('=',2)+1;spans=[(start,start+len(str(v)))]
        return str(v),s,spans
    if f=='conditional':
        a,t,k=c['a'],c['threshold'],c['delta'];truth=a>t;v=a+k if truth else a-k
        s=f'T={int(truth)},{v};F={v}';start=s.index(',')+1
        return str(v),s,[(2,3),(start,start+len(str(v)))]
    if f in ('code_trace','list_reasoning'):
        if f=='code_trace':
            acc=c['start'];states=[acc]
            for op,n in c['ops']:acc=acc+n if op=='+' else acc-n;states.append(acc)
            first=1
        else:
            acc=0;states=[]
            for n in c['values']:acc+=n;states.append(acc)
            first=1 # first listed value is copied, subsequent sums computed
        s='T=';spans=[]
        for i,n in enumerate(states):
            if i:s+=','
            lo=len(s);s+=str(n)
            if i>=first:spans.append((lo,len(s)))
        return str(acc),s+f';F={acc}',spans
    if f=='memory_update':
        vals=[v for k,v in c['writes'] if k==c['query']]
        state={}
        for k,v in c['writes']:state[k]=v
        assert vals and state[c['query']]==vals[-1]
        s='T='+c['query']+',';spans=[]
        for i,v in enumerate(vals):
            if i:s+=','
            lo=len(s);s+=str(v);spans.append((lo,len(s)))
        return str(vals[-1]),s+f';F={vals[-1]}',spans
    if f=='code_generation':
        code='def solve(x):\n'
        for op,n in c['ops']:code+=f'    x {op}= {n}\n'
        code+='    return x'
        return code,code,[]
    raise ValueError(f)

def sample(rng,f,level):
    # Difficulty separates surface transfer from larger-value/longer-operation OOD.
    lo,hi=(0,9) if level==0 else (10,59) if level==1 else (60,149)
    c={'family':f}
    if f=='arithmetic':c.update(a=rng.randint(lo,hi),b=rng.randint(lo,hi),op=rng.choice('+-'))
    elif f=='conditional':c.update(a=rng.randint(lo,hi),threshold=rng.randint(lo,hi),delta=rng.randint(1,9 if level<2 else 19))
    elif f=='code_trace':c.update(start=rng.randint(lo,hi),ops=[[rng.choice('+-'),rng.randint(1,9)] for _ in range(1 if level==0 else 2 if level==1 else 4)])
    elif f=='list_reasoning':c.update(values=[rng.randint(lo,hi) for _ in range(2 if level==0 else 3 if level==1 else 5)])
    elif f=='memory_update':
        keys='ab' if level==0 else 'abcd'
        ws=[[k,rng.randint(lo,hi)] for k in keys]
        ws += [[rng.choice(keys),rng.randint(lo,hi)] for _ in range(1 if level==0 else 3 if level==1 else 8)]
        rng.shuffle(ws);c.update(writes=ws,query=rng.choice(keys))
    else:
        nops=1 if level==0 else rng.choice((2,3)) if level==1 else 4
        c.update(ops=[[rng.choice(('+','-','*')),rng.randint(1,9 if level<2 else 15)] for _ in range(nops)])
    return c

def render(c,lang,variant=0):
    f=c['family'];ru=lang=='ru'
    if f=='arithmetic':
        a,b=c['a'],c['b'];op=c['op']
        if variant:s=f'Вычисли результат: {a} {op} {b}.' if ru else f'Compute the result: {a} {op} {b}.'
        else:s=f'Сколько будет {a} {op} {b}?' if ru else f'What is {a} {op} {b}?'
    elif f=='conditional':
        a,t,k=c['a'],c['threshold'],c['delta']
        if ru:s=f'Число {a}. Если оно больше {t}, прибавь {k}, иначе вычти {k}. Какой результат?'
        else:s=f'Start with {a}. If it is greater than {t}, add {k}; otherwise subtract {k}. What is the result?'
        if variant:s=('Выполни условие. ' if ru else 'Apply the rule. ')+s
    elif f=='code_trace':
        code='x='+str(c['start'])+'\n'+'\n'.join(f'x{op}={n}' for op,n in c['ops'])
        s=('Чему равно x после программы?\n' if ru else 'What is x after this program?\n')+code
        if variant:s=('Проследи вычисления:\n' if ru else 'Trace these statements:\n')+code+('\nНайди x.' if ru else '\nFind x.')
    elif f=='list_reasoning':
        arr=', '.join(map(str,c['values']))
        s=(f'Найди сумму чисел [{arr}].' if ru else f'Find the sum of [{arr}].')
        if variant:s=f'Сложи все элементы: {arr}.' if ru else f'Add every entry: {arr}.'
    elif f=='memory_update':
        text='; '.join(k+'='+str(v) for k,v in c['writes'])
        s=(f'Записи по времени: {text}. Последнее значение {c["query"]}?' if ru else f'Writes in time order: {text}. Latest value of {c["query"]}?')
        if variant:s=(f'Обновления: {text}. Прочитай {c["query"]}.' if ru else f'Apply updates chronologically: {text}. Read {c["query"]}.')
    else:
        bits=[]
        for op,n in c['ops']:
            bits.append(({'*':'умножь x на','+':'прибавь к x','-':'вычти из x'} if ru else {'*':'multiply x by','+':'add to x','-':'subtract from x'})[op]+' '+str(n))
        if ru:s='Напиши функцию Python solve(x). По порядку: '+', затем '.join(bits)+'. Верни результат. Только код.'
        else:s='Write a Python function solve(x). In order: '+', then '.join(bits)+'. Return the result. Code only.'
        if variant:
            s=('Требуется функция solve(x): ' if ru else 'Implement solve(x): ')+('; '.join(bits))+('. Верни итог, ответь кодом Python.' if ru else '. Return the final value, answer in Python code.')
    if f!='code_generation':s+=('\nПокажи вычисления, закончи F=<число>.' if ru else '\nShow working and end with F=<integer>.')
    return s+('\nОтвет:\n' if ru else '\nAnswer:\n')

def make_row(c,lang,variant,stage,pair=None):
    answer,completion,spans=solve(c);p=render(c,lang,variant);g=canonical(c)
    return {'id':hashlib.sha256((g+'|'+p).encode()).hexdigest(),'group':g,'family':c['family'],'lang':lang,
      'level':stage,'prompt':p,'answer':answer,'completion':completion,'computed_spans':spans,'case':c,'pair_id':pair}

def pack(row,tok):
    prefix=tok.encode(row['prompt'],bos=True);completion=tok.encode(row['completion'],eos=True)
    tokens=prefix+completion;weights=[1.]*len(completion)
    for lo,hi in row['computed_spans']:
        a=len(tok.encode(row['completion'][:lo]));b=len(tok.encode(row['completion'][:hi]))
        # Token boundaries must be exact, not a guessed character/token ratio.
        if tok.encode(row['completion'][:hi])!=completion[:b]:raise ValueError('non-prefix tokenization')
        for i in range(a,b):weights[i]=8.
    return {'tokens':tokens,'prefix':len(prefix),'weights':weights,'id':row['id'],'family':row['family']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ancestor-data',type=Path,required=True);args=ap.parse_args()
    if (ROOT/'data/MANIFEST.json').exists():raise FileExistsError('Immutable data already exists')
    rng=random.Random(110341);tok=Tokenizer.load(ROOT/'data/tokenizer.json');occupied=set();known=0
    for file in args.ancestor_data.glob('reasoning_*.jsonl'):
        for row in rows(file):
            try:occupied.add(canonical(row['case']));known+=1
            except (KeyError,TypeError):pass
    # Keep new split assignment deterministic, before rendering/paraphrasing.
    datasets={};groups={}
    for split,n,level,var in [('elementary',256,0,0),('composed',512,1,0),('validation',20,1,0),('iid',40,1,0),('surface',20,1,1),('ood',20,2,0)]:
        rr=[];gg=set()
        for f in FAMILIES:
            count=0;attempts=0
            while count<n:
                attempts+=1
                if attempts>200000:raise RuntimeError('Exhausted unique groups '+split+' '+f)
                c=sample(rng,f,level);g=canonical(c)
                # Small elementary affine/digit spaces necessarily repeat within train.
                if split=='elementary':
                    if g in occupied and g not in gg and f not in ('arithmetic','code_generation'):continue
                elif g in occupied:continue
                occupied.add(g);gg.add(g)
                rr.append(make_row(c,'ru' if count%2 else 'en',var,level));count+=1
        datasets[split]=rr;groups[split]=gg
    cf=[];cf_groups=set()
    while len(cf)<160:
        c=sample(rng,'memory_update',1);g=canonical(c)
        if g in occupied:continue
        latest=dict(c['writes']);keys=list(latest);rng.shuffle(keys)
        pairs=[(a,b) for a in keys for b in keys if a!=b and latest[a]!=latest[b]]
        if not pairs:continue
        a,b=rng.choice(pairs);occupied.add(g);cf_groups.add(g);pair='cf-'+str(len(cf)//2)
        for q in (a,b):cf.append(make_row({**c,'query':q},'ru' if (len(cf)//2)%2 else 'en',0,1,pair))
    datasets['counterfactual']=cf;groups['counterfactual']=cf_groups
    intersections={a+':'+b:len(groups[a]&groups[b]) for a in groups for b in groups if a<b}
    assert not any(intersections.values()),intersections
    for name,rr in datasets.items():write_rows(ROOT/'data'/f'{name}.jsonl',rr)
    # Fixed stratified pilot before training; gold is NOT sent to external inference.
    pilot=[]
    for split in ('iid','surface','ood'):
        for f in FAMILIES:
            subset=[z for z in datasets[split] if z['family']==f]
            pilot.append(subset[0])
    pilot+=cf[:4]
    write_rows(ROOT/'bench/reference_cases.jsonl',pilot)
    write_rows(ROOT/'bench/prompts.jsonl',[{k:r[k] for k in ('id','family','lang','prompt')} for r in pilot])
    stat={n:{'n':len(rr),'groups':len(groups[n]),'max_tokens':max(len(pack(z,tok)['tokens']) for z in rr)} for n,rr in datasets.items()}
    put(ROOT/'data/MANIFEST.json',{'seed':110341,'statistics':stat,'intersections':intersections,'available_ancestor_rows_inspected':known,
      'unknown_EC_training_groups':'Exact EC train dataset is not available in this container; no decontamination guarantee against that unavailable corpus.',
      'files':{p.name:sha(p) for p in (ROOT/'data').glob('*.jsonl')},'tokenizer_sha256':sha(ROOT/'data/tokenizer.json'),
      'elementary_repetitions':'Small atomic spaces can repeat within train. No new held group occurs in either train partition.'})
    print(json.dumps(stat,indent=2))
if __name__=='__main__':main()
