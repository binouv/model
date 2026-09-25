"""Verified synthetic instruction curriculum. Gold cases never enter inference.
Semantic problem groups (including translations/operator variants) share a split.
Not a substitute for natural-language pretraining or an external reasoning benchmark.
"""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
FAMILIES=('arithmetic','conditional','code_trace','list_reasoning','memory_update')

def stable_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def semantic_group(case):
    c=dict(case)
    # Prevent alternative queries/operations on the same underlying data crossing splits.
    for key in ['op','query','kind']:c.pop(key,None)
    return stable_hash(c)

def solve(c):
    f=c['family']
    if f=='arithmetic':
        a,b=c['values'];return {'+':a+b,'-':a-b,'*':a*b}[c['op']]
    if f=='conditional':
        x,k,a,b=c['values'];return x+a if x<k else x-b
    if f=='code_trace':
        x=c['start']
        for op,a in c['program']:x=x+a if op=='+' else x-a if op=='-' else x*a
        return x
    if f=='list_reasoning':
        v=c['values'];return {'sum':sum(v),'min':min(v),'max':max(v)}[c['kind']]
    if f=='memory_update':
        mem={}
        for k,v in c['writes']:mem[k]=v
        return mem[c['query']]
    raise ValueError(f)

def make_case(rng,f,ood=False):
    lo,hi=(48,95) if ood else (0,47)
    vals=lambda n:[rng.randint(lo,hi) for _ in range(n)]
    if f=='arithmetic':return dict(family=f,values=vals(2),op=rng.choice(['+','-','*']))
    if f=='conditional':return dict(family=f,values=vals(4))
    if f=='code_trace':return dict(family=f,start=rng.randint(lo,hi),program=[[rng.choice(['+','-']),rng.randint(lo,hi)] for _ in range(3 if ood else 2)])
    if f=='list_reasoning':return dict(family=f,values=vals(6 if ood else 3),kind=rng.choice(['sum','min','max']))
    keys=list('abcd');q=rng.choice(keys);writes=list(zip(keys,vals(4)))
    for _ in range(4 if ood else 2):writes.append((rng.choice(keys),rng.randint(lo,hi)))
    # Interleave key histories, preserving updates but removing path/storage order cue.
    hist={k:[(kk,v) for kk,v in writes if kk==k] for k in keys};ordered=[]
    while hist:
        k=rng.choice(list(hist));ordered.append(hist[k].pop(0))
        if not hist[k]:del hist[k]
    return dict(family=f,writes=ordered,query=q)

def render(c,lang,variant=0):
    ru=lang=='ru';f=c['family'];v=variant%3
    if f=='arithmetic':
        a,b=c['values'];expr=f'{a} {c["op"]} {b}'
        templates=['Вычисли: {}.','Чему равно {}?','Найди значение выражения {}.'] if ru else ['Calculate: {}.','What is {}?','Evaluate the expression {}.']
        text=templates[v].format(expr)
    elif f=='conditional':
        x,k,a,b=c['values']
        text=(f'x={x}. Если x<{k}, прибавь {a}; иначе вычти {b}.' if ru else f'x={x}. If x<{k}, add {a}; otherwise subtract {b}.') if v!=2 else (f'Начни с {x}. При значении меньше {k} добавь {a}, в противном случае отними {b}.' if ru else f'Begin at {x}. Increase by {a} when below {k}; otherwise decrease by {b}.')
    elif f=='code_trace':
        text=('Какое число напечатает Python?\n' if ru else 'What number does Python print?\n') if v!=2 else ('Определи вывод кода:\n' if ru else 'Determine the code output:\n')
        text+=f'x = {c["start"]}\n'+''.join(f'x {op}= {a}\n' for op,a in c['program'])+'print(x)'
    elif f=='list_reasoning':
        labels={'sum':'сумму','min':'минимум','max':'максимум'} if ru else {'sum':'sum','min':'minimum','max':'maximum'}
        text=(f'Найди {labels[c["kind"]]}: {c["values"]}.' if ru else f'Find the {labels[c["kind"]]}: {c["values"]}.') if v!=2 else (f'Числа {c["values"]}. Верни {labels[c["kind"]]}.' if ru else f'Numbers {c["values"]}. Return their {labels[c["kind"]]}.')
    else:
        w='; '.join(f'{k}={v}' for k,v in c['writes'])
        text=(f'Записи по времени: {w}. Последнее значение {c["query"]}?' if ru else f'Writes in time order: {w}. Latest value of {c["query"]}?') if v!=2 else (f'Изменения: {w}. Прочитай актуальное значение ключа {c["query"]}.' if ru else f'Updates: {w}. Read the current value for key {c["query"]}.')
    # Explicit completion boundary; no chat template or hidden label tokens.
    return text+('\nОтвет:' if ru else '\nAnswer:')

def build(root,tokenizer):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    spec={'train':(4096,72111,0),'validation':(160,72211,1),'test_iid':(160,72311,2),'test_surface':(80,72411,2),'test_extrapolation':(80,72511,2)}
    groups={};report={};seen_prompts=set();all_semantic={}
    for split,(n,seed,partition) in spec.items():
        rng=random.Random(seed);rows=[];attempts=0
        while len(rows)<n:
            attempts+=1
            if attempts>200000:raise RuntimeError('insufficient unique semantic groups')
            f=FAMILIES[len(rows)%len(FAMILIES)];c=make_case(rng,f,split=='test_extrapolation');g=semantic_group(c)
            bucket=int(g[:8],16)%10
            got=0 if bucket<8 else bucket-7
            if got!=partition or g in all_semantic:continue
            lang=rng.choice(['ru','en']);variant=2 if split=='test_surface' else rng.randrange(2)
            p=render(c,lang,variant);a=solve(c);completion=f' {a}\n'
            if p in seen_prompts:continue
            pt=tokenizer.encode(p,bos=True);ct=tokenizer.encode(completion,eos=True)
            # No truncation, including the last answer token. Long test extrapolation kept.
            if len(pt)+len(ct)>96 and split in ['train','validation']:continue
            row={'id':stable_hash({'case':c,'lang':lang,'variant':variant}), 'group':g,'family':f,'lang':lang,'variant':variant,'prompt':p,'completion':completion,'answer':str(a),'case':c,'tokens':pt+ct,'prompt_tokens':len(pt)}
            rows.append(row);seen_prompts.add(p);all_semantic[g]=split
        groups[split]=rows;report[split]={'examples':len(rows),'attempts':attempts,'tokens':sum(len(z['tokens']) for z in rows),'max_length':max(len(z['tokens']) for z in rows),'ru':sum(z['lang']=='ru' for z in rows)}
    # Counterfactual pairs: change the queried key/operator while preserving other context.
    rng=random.Random(72611);pairs=[];pair_groups=set()
    while len(pairs)<80:
        f='arithmetic' if len(pairs)%4==0 else 'memory_update';c=make_case(rng,f);g=semantic_group(c)
        if int(g[:8],16)%10!=9 or g in all_semantic or g in pair_groups:continue
        cc=json.loads(json.dumps(c))
        if f=='arithmetic':cc['op']='-' if c['op']!='-' else '+'
        else:cc['query']=rng.choice([k for k in 'abcd' if k!=c['query']])
        if solve(c)==solve(cc):continue
        lang=rng.choice(['ru','en']);pair_id=g
        for arm,case in enumerate([c,cc]):
            p=render(case,lang,0);co=f' {solve(case)}\n';pt=tokenizer.encode(p,bos=True)
            pairs.append({'id':stable_hash({'pair_id':pair_id,'arm':arm}),'pair_id':pair_id,'group':g,'family':f,'lang':lang,'variant':0,'prompt':p,'completion':co,'answer':str(solve(case)),'case':case,'tokens':pt+tokenizer.encode(co,eos=True),'prompt_tokens':len(pt)})
        pair_groups.add(g)
    groups['test_counterfactual']=pairs
    report['test_counterfactual']={'examples':len(pairs),'pairs':len(pair_groups),'max_length':max(len(z['tokens']) for z in pairs)}
    for split,rows in groups.items():
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for s,rows in groups.items():
        for r in rows:assert str(solve(r['case']))==r['answer']
    memory=[r for rows in groups.values() for r in rows if r['family']=='memory_update']
    report.update({'semantic_group_cross_split_overlap':0,'same_case_surface_leakage':False,'all_gold_replayed':True,'fixed_G1_tokenizer':True,'external_teacher_used':False,'real_world_benchmark':False,'memory_last_global_write_shortcut':sum(str(z['case']['writes'][-1][1])==z['answer'] for z in memory)/len(memory),'seeds':{s:x[1] for s,x in spec.items()},'counterfactual_seed':72611})
    (root/'manifest.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    import argparse,sys
    ap=argparse.ArgumentParser();ap.add_argument('--tokenizer',required=True);ap.add_argument('--base-src',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    sys.path.insert(0,a.base_src);from tokenizer import Tokenizer
    print(json.dumps(build(a.out,Tokenizer.load(a.tokenizer)),indent=2))
