"""Local-only corpus: preserved CPython source + fresh verified RU/EN tasks.
No external teacher. Gold is constructed offline and never used by inference.
Natural-source file splits are inherited from G1; fresh semantic groups are
partitioned before rendering and exclude all available G8C/G5S case groups.
"""
from __future__ import annotations
import collections,copy,hashlib,json,random,zipfile
from pathlib import Path
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=('arithmetic','conditional','code_trace','list_reasoning','memory_update')

def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(c):
    f=c['family']
    if f in ('arithmetic','list_reasoning'):obj={'family':f,'values':sorted(c['values'])}
    elif f=='memory_update':obj={'family':f,'histories':{k:[v for kk,v in c['writes'] if kk==k] for k in sorted({k for k,_ in c['writes']})}}
    else:obj=c
    return sha(json.dumps(obj,sort_keys=True,separators=(',',':')).encode())
def solve(c):
    f=c['family']
    if f=='arithmetic':
        a,b=c['values'];return a+b if c['op']=='+' else a-b if c['op']=='-' else a*b
    if f=='conditional':
        x,k,a,b=c['values'];return x+a if x<k else x-b
    if f=='code_trace':
        x=c['start']
        for op,a in c['program']:x=x+a if op=='+' else x-a
        return x
    if f=='list_reasoning':
        a=c['values'];return sum(a) if c['kind']=='sum' else min(a) if c['kind']=='min' else max(a)
    d={}
    for k,v in c['writes']:d[k]=v
    return d[c['query']]
def independent_trace(c):
    f=c['family']
    if f=='arithmetic':
        a,b=c['values']; op=c['op']
        val=sum((a,b)) if op=='+' else sum((a,-b)) if op=='-' else sum([a]*b)
        return f'{a}{op}{b}={val}',val
    if f=='code_trace':
        values=[c['start']]
        for op,a in c['program']:values.append(sum((values[-1],a if op=='+' else -a)))
        return ','.join(map(str,values)),values[-1]
    if f=='conditional':
        x,k,a,b=c['values'];p=int(x<k);val=sum((x,a if p else -b))
        return f'{x}<{k}:{p},{x}{"+" if p else "-"}{a if p else b}={val}',val
    if f=='list_reasoning':
        a=c['values'];kind=c['kind'];v=0 if kind=='sum' else a[0];steps=[]
        for n in a:
            v=v+n if kind=='sum' else sorted((v,n))[0] if kind=='min' else sorted((v,n))[-1]
            steps.append(v)
        return ','.join(map(str,steps)),v
    values=[v for k,v in c['writes'] if k==c['query']]
    return c['query']+':'+','.join(map(str,values)),values[-1]
def make_case(rng,f,ood=False):
    lo,hi=(256,511) if ood else (0,255)
    if f=='arithmetic':return {'family':f,'values':[rng.randint(lo,hi),rng.randint(lo,hi)],'op':rng.choice('+-*')}
    if f=='conditional':return {'family':f,'values':[rng.randint(lo,hi) for _ in range(4)]}
    if f=='code_trace':return {'family':f,'start':rng.randint(lo,hi),'program':[[rng.choice('+-'),rng.randint(lo,hi)] for _ in range(4 if ood else 2)]}
    if f=='list_reasoning':return {'family':f,'values':[rng.randint(lo,hi) for _ in range(6 if ood else 4)],'kind':rng.choice(['sum','min','max'])}
    keys=list('abcd');writes=[[k,rng.randint(lo,hi)] for k in keys]
    writes += [[rng.choice(keys),rng.randint(lo,hi)] for _ in range(6 if ood else 3)]
    rng.shuffle(writes)
    return {'family':f,'writes':writes,'query':rng.choice(keys)}
def render(c,lang='en',variant=0):
    ru=lang=='ru';f=c['family']
    if f=='arithmetic':
        a,b=c['values'];s=f'{a} {c["op"]} {b}'
        p=(f'Вычисли: {s}.' if ru else f'Calculate: {s}.') if not variant else (f'Чему равно {s}?' if ru else f'What is the value of {s}?')
    elif f=='conditional':
        x,k,a,b=c['values']
        p=(f'x={x}. Если x<{k}, прибавь {a}, иначе вычти {b}.' if ru else f'x={x}. If x<{k}, add {a}; else subtract {b}.') if not variant else (f'Начни с {x}. Ниже {k}: +{a}. Иначе: -{b}.' if ru else f'Start at {x}. Below {k}: +{a}. Otherwise: -{b}.')
    elif f=='code_trace':
        code=f'x = {c["start"]}\n'+''.join(f'x {op}= {a}\n' for op,a in c['program'])+'print(x)'
        p=(('Что напечатает код?\n' if ru else 'What does this code print?\n') if not variant else ('Определи вывод:\n' if ru else 'Determine the output:\n'))+code
    elif f=='list_reasoning':
        name=({'sum':'сумма','min':'минимум','max':'максимум'} if ru else {'sum':'sum','min':'minimum','max':'maximum'})[c['kind']]
        p=(f'{c["values"]}. Найди: {name}.' if ru else f'Find the {name} of {c["values"]}.') if not variant else (f'Числа: {c["values"]}. Требуется {name}.' if ru else f'Numbers: {c["values"]}. Return their {name}.')
    else:
        w='; '.join(f'{k}={v}' for k,v in c['writes'])
        p=(f'Записи по времени: {w}. Последнее значение {c["query"]}?' if ru else f'Writes in time order: {w}. Latest value of {c["query"]}?') if not variant else (f'Обновления: {w}. Прочитай {c["query"]}.' if ru else f'Apply updates chronologically: {w}. Read {c["query"]}.')
    return p+'\n'+('Ответ:\n' if ru else 'Answer:\n')
def make_row(c,tok,rng,variant=0,pair=None,lang=None):
    lang=lang or rng.choice(['ru','en']);prompt=render(c,lang,variant)
    tr,a=independent_trace(c);assert a==solve(c)
    completion='T='+tr+';F='+str(a)
    text=prompt+completion;ids=tok.encode(text,bos=True,eos=True)
    # Tokenization of prompt followed by target must preserve the prefix.
    pt=tok.encode(prompt,bos=True)
    assert ids[:len(pt)]==pt
    return {'id':sha(text.encode()),'group':canonical(c),'family':c['family'],'kind':'verified_reasoning','lang':lang,'case':c,'prompt':prompt,'completion':completion,'answer':str(a),'canonical_trace':tr,'text':text,'tokens':ids,'prompt_tokens':len(pt),'pair_id':pair,'source':'locally-generated-verified-v1'}
def write_rows(path,rows):
    text=''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in rows)
    path.write_text(text,encoding='utf-8');return sha(path.read_bytes())
def main():
    ROOT.joinpath('data').mkdir(exist_ok=True);tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    old_zip=Path('/mnt/data/FlyGraph_G8C_FullCheckpointBundle_2026-09-26.zip')
    denied=set()
    with zipfile.ZipFile(old_zip) as z:
        for name in z.namelist():
            if '/data/' in name and name.endswith('.jsonl'):
                for line in z.read(name).decode().splitlines():
                    row=json.loads(line)
                    if 'case' in row:denied.add(canonical(row['case']))
    occupied=set(denied);fresh={};dropped=collections.Counter()
    specs=[('train',16384,82011,0,False),('validation',400,82012,0,False),('test_iid',200,82013,0,False),('test_surface',100,82014,1,False),('test_extrapolation',100,82015,1,True)]
    for split,n,seed,variant,ood in specs:
        rng=random.Random(seed);rows=[]
        while len(rows)<n:
            c=make_case(rng,FAMILIES[len(rows)%5],ood);g=canonical(c)
            if g in occupied:continue
            row=make_row(c,tok,rng,variant)
            if len(row['tokens'])>129:
                dropped[split+':'+row['lang']+':'+c['family']]+=1;continue
            occupied.add(g);rows.append(row)
        fresh[split]=rows
    rng=random.Random(82016);pairs=[]
    while len(pairs)<100:
        family='memory_update' if len(pairs)%4==0 else 'arithmetic'
        c=make_case(rng,family);g=canonical(c)
        if g in occupied:continue
        alt=copy.deepcopy(c)
        if family=='memory_update':alt['query']=rng.choice([k for k in 'abcd' if k!=c['query']])
        else:alt['op']=rng.choice([op for op in '+-*' if op!=c['op']])
        if solve(c)==solve(alt):continue
        lang=rng.choice(['ru','en']);pid='l100cf'+str(len(pairs)//2)
        rr=[make_row(cc,tok,rng,pair=pid,lang=lang) for cc in [c,alt]]
        if max(len(z['tokens']) for z in rr)>129:continue
        pairs+=rr;occupied.add(g)
    fresh['test_counterfactual']=pairs
    legacy=Path('/mnt/data/FlyGraph_G1_SourceData_2026-09-25.zip');natural={}
    with zipfile.ZipFile(legacy) as z:
        for split in ['train','validation','test']:
            rr=[]
            for line in z.read(f'flygraph_g1_310m/data/{split}.jsonl').decode().splitlines():
                row=json.loads(line)
                if row['kind'] not in ['python_source','english_docstring','ru_prose','en_prose']:continue
                row['tokens']=tok.encode(row['text'],bos=True,eos=True);row['family']='natural';row['prompt_tokens']=0
                rr.append(row)
            natural[split]=rr
    files={};stats={}
    for split,rows in fresh.items():
        name='reasoning_'+split+'.jsonl';files[name]=write_rows(ROOT/'data'/name,rows)
        stats[name]={'examples':len(rows),'tokens':sum(len(z['tokens']) for z in rows),'language_counts':dict(collections.Counter(z['lang'] for z in rows))}
    for split,rows in natural.items():
        name='natural_'+split+'.jsonl';files[name]=write_rows(ROOT/'data'/name,rows)
        stats[name]={'examples':len(rows),'tokens':sum(len(z['tokens']) for z in rows),'kinds':dict(collections.Counter(z['kind'] for z in rows))}
    train_groups={z['group'] for z in fresh['train']}
    for split,rows in fresh.items():
        if split!='train':assert not train_groups&{z['group'] for z in rows}
        assert all(solve(z['case'])==int(z['answer']) for z in rows)
    assert not denied&{z['group'] for rows in fresh.values() for z in rows}
    memory=[z for z in fresh['test_iid'] if z['family']=='memory_update']
    last=sum(int(z['case']['writes'][-1][1])==int(z['answer']) for z in memory)
    gen=[]
    # Fixed pre-training selection: 10 IID per family, 5 per family for surface/OOD,
    # plus 10 complete counterfactual pairs. All raw cases remain saved.
    for split,per in [('test_iid',10),('test_surface',5),('test_extrapolation',5)]:
        for f in FAMILIES:gen += [{**z,'split':split} for z in fresh[split] if z['family']==f][:per]
    gen += [{**z,'split':'test_counterfactual'} for z in pairs[:20]]
    files['generation_probes.jsonl']=write_rows(ROOT/'data/generation_probes.jsonl',gen)
    manifest={'status':'completed','files_sha256':files,'statistics':stats,'generation_n':len(gen),'max_training_context':128,'data_seed_range':[82011,82016],
    'excluded_legacy_canonical_groups':len(denied),'fresh_vs_legacy_overlap':0,'fresh_cross_split_group_overlap':0,'source_archives':{legacy.name:sha(legacy.read_bytes()),old_zip.name:sha(old_zip.read_bytes())},
    'tokenizer_sha256':sha((ROOT/'data/tokenizer.json').read_bytes()),'tokenizer_origin':'Existing G1 train-only lexical frequency vocabulary, UTF8-byte fallback, 8192. Not BPE; not newly trained.',
    'known_limitations':['Natural corpora are local CPython and small original explanatory paragraphs, NOT general web pretraining.','12 Russian train prose paragraphs inherited; synthetic tasks dominate Russian exposure.','Synthetic semantic families overlap across splits; this is not broad independent reasoning.','Very long synthetic prompts are rejected to fit fixed128-token training, with counts reported.','Natural split inherits original CPython source-file assignment; source and extracted docstrings can overlap within split.','Full G8C model/optimizer is not resumed: dimensions differ and current request explicitly changes size.'],
    'length_rejections':dict(dropped),'test_iid_last_global_write_shortcut':{'correct':last,'n':len(memory),'accuracy':last/len(memory)},'external_downloads':False,'gold_used_at_inference':False}
    (ROOT/'data/MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'completed','stats':stats,'generation_n':len(gen),'last_write_shortcut':manifest['test_iid_last_global_write_shortcut'],'rejections':dict(dropped)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
