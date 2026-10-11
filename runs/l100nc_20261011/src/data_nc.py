"""L100NC: natural-language-to-Python data; never imported by inference.
No executable operators or solution fragments are included in the condition.
Semantic fingerprints group exact affine/piecewise pointwise behaviour before
English/Russian rendering. These are synthetic tasks, not a broad benchmark.
"""
from __future__ import annotations
import copy, hashlib, json, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=('pipeline','branch','filter_map','aggregate')

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def op_value(x,op,k):
    if op=='add': return x+k
    if op=='sub': return x-k
    if op=='mul': return x*k
    raise ValueError('unknown op')

def expected(case, x):
    f=case['family']
    if f=='pipeline':
        for op,k in case['ops']: x=op_value(x,op,k)
        return x
    if f=='branch':
        op,k=case['yes'] if x<case['threshold'] else case['no']
        return op_value(x,op,k)
    pred=lambda v: v>case['threshold'] if case['cmp']=='gt' else v<case['threshold']
    if f=='filter_map': return [case['a']*v+case['b'] for v in x if pred(v)]
    if f=='aggregate': return sum(v+case['b'] for v in x if pred(v))
    raise ValueError('unknown family')

def semantic_key(c):
    # The registered scalar families are affine or one-breakpoint piecewise affine;
    # domain covers every registered breakpoint and enough adjacent integer inputs.
    if c['family'] in ('pipeline','branch'):
        signature=[expected(c,x) for x in range(-64,65)]
        return digest(['scalar',signature])
    # Pointwise response of list transforms; singleton plus empty determines
    # the order-preserving map/filter and additive fold families used here.
    return digest([c['family'],[expected(c,[x]) for x in range(-64,65)],expected(c,[])])

def make_case(r,f,long=False,outside=False):
    lo,hi=(23,35) if outside else (2,19)
    if f=='pipeline':
        n=3 if long else 2
        ops=[[r.choice(['add','sub','mul']),r.randint(lo,hi)] for _ in range(n)]
        if long and all(op!='mul' for op,k in ops):ops[0][0]='mul'
        return {'family':f,'ops':ops}
    if f=='branch':
        return {'family':f,'threshold':r.randint(-hi,hi),
                'yes':[r.choice(['add','sub','mul']),r.randint(lo,hi)],
                'no':[r.choice(['add','sub','mul']),r.randint(lo,hi)]}
    return {'family':f,'threshold':r.randint(-hi,hi),'cmp':r.choice(['lt','gt']),
            'a':r.randint(lo,min(hi,9)) if not outside else r.randint(lo,hi),'b':r.randint(lo,hi)}

def action(op,k,lang,style):
    if lang=='en':
        return ({'add':f'add {k}','sub':f'subtract {k}','mul':f'multiply by {k}'} if style==0 else
                {'add':f'increase it by {k}','sub':f'decrease it by {k}','mul':f'scale it by a factor of {k}'})[op]
    return ({'add':f'прибавь {k}','sub':f'вычти {k}','mul':f'умножь на {k}'} if style==0 else
            {'add':f'увеличь число на {k}','sub':f'уменьши число на {k}','mul':f'получи произведение числа и {k}'})[op]

def render(c,lang='en',style=0):
    f=c['family'];is_scalar=f in ('pipeline','branch')
    intro=('Write a Python function solve(x). ' if is_scalar else 'Write a Python function solve(xs). ')
    if lang=='ru':intro='Напиши функцию Python solve(x). ' if is_scalar else 'Напиши функцию Python solve(xs). '
    if f=='pipeline':
        acts=[action(o,k,lang,style) for o,k in c['ops']]
        body=('Starting with the input, '+', then '.join(acts)+'. Return the result.' if style==0 else
              'Apply these changes to the input in the given order: '+'; next '.join(acts)+'. Return the final number.') if lang=='en' else (
              'Начни с входного числа: '+', затем '.join(acts)+'. Верни результат.' if style==0 else
              'Измени входное число в указанном порядке: '+'; после этого '.join(acts)+'. Верни итоговое число.')
    elif f=='branch':
        yes=action(*c['yes'],lang,style);no=action(*c['no'],lang,style);t=c['threshold']
        body=(f'If the input is less than {t}, {yes}; otherwise {no}. Return the changed number.' if style==0 else
              f'Choose based on the original input: below {t}, {yes}; at least {t}, {no}. Return the outcome.') if lang=='en' else (
              f'Если входное число меньше {t}, {yes}; иначе {no}. Верни изменённое число.' if style==0 else
              f'Выбери действие по исходному числу: ниже {t}, {yes}; не ниже {t}, {no}. Верни итог.')
    else:
        t=c['threshold'];b=c['b'];cmp=c['cmp'];a=c['a']
        cond=('greater than' if cmp=='gt' else 'less than') if style==0 else ('above' if cmp=='gt' else 'below')
        condru=('больше' if cmp=='gt' else 'меньше') if style==0 else ('выше' if cmp=='gt' else 'ниже')
        if f=='filter_map':
            body=(f'Keep input numbers strictly {cond} {t}. For each retained number multiply by {a}, then add {b}. Return the list in the original order.') if lang=='en' else (
                 f'Оставь входные числа строго {condru} {t}. Каждое оставшееся число умножь на {a}, затем прибавь {b}. Верни список в исходном порядке.')
        else:
            body=(f'Select input numbers strictly {cond} {t}. Add {b} to every selected number and return their sum. Return zero if none are selected.') if lang=='en' else (
                 f'Выбери входные числа строго {condru} {t}. К каждому выбранному числу прибавь {b} и верни их сумму. Если таких чисел нет, верни ноль.')
    ending='\nReturn only Python code.\n' if lang=='en' else '\nВерни только код Python.\n'
    return intro+body+ending

def expression(op,k,x='x'):
    return f'({x} { {"add":"+","sub":"-","mul":"*"}[op] } {k})'

def solution(c):
    f=c['family']
    if f=='pipeline':
        x='x'
        for op,k in c['ops']:x=expression(op,k,x)
        return 'def solve(x):\n    return '+x+'\n'
    if f=='branch':
        return 'def solve(x):\n    return '+expression(*c['yes'])+f' if x < {c["threshold"]} else '+expression(*c['no'])+'\n'
    sym='>' if c['cmp']=='gt' else '<'
    if f=='filter_map':
        return f'def solve(xs):\n    return [x * {c["a"]} + {c["b"]} for x in xs if x {sym} {c["threshold"]}]\n'
    return f'def solve(xs):\n    return sum(x + {c["b"]} for x in xs if x {sym} {c["threshold"]})\n'

def test_inputs(c):
    if c['family'] in ('pipeline','branch'):
        return list(range(-64,65))+[-1000,1000,10001]
    t=c['threshold']
    return [[],[t],[t-1],[t+1],[t-1,t,t+1],[3,3,-5,100],[-1000,0,1000],
            list(range(-20,21)),list(range(20,-21,-1)),[t+1,t-1]*4]

def case_row(c,lang,style,pair=None):
    return {'id':digest([c,lang,style,pair]),'semantic':semantic_key(c),'case':c,'family':c['family'],
            'lang':lang,'style':style,'pair_id':pair,'prompt':render(c,lang,style),'target':solution(c)}

def build():
    destination=ROOT/'data';destination.mkdir(exist_ok=True)
    if (destination/'MANIFEST.json').exists():raise FileExistsError('refuse dataset overwrite')
    seen=set();sets={};attempts={}
    # Completely disjoint semantic groups across all sets; bilingual duplicates
    # occur inside each split only. Do not treat them as independent problems.
    for split,n,seed,style,long in [('train',2048,111051,0,False),('validation',48,111052,0,False),
              ('iid',80,111053,0,False),('surface',80,111054,1,False),('length',40,111055,1,True)]:
        rng=random.Random(seed);rr=[];tries=0;count=0
        while count<n:
            tries+=1
            if tries>200000:raise RuntimeError('case space exhausted')
            f='pipeline' if long else FAMILIES[count%len(FAMILIES)]
            c=make_case(rng,f,long);g=semantic_key(c)
            if g in seen:continue
            seen.add(g);rr.extend(case_row(c,l,style) for l in ('en','ru'));count+=1
        sets[split]=rr;attempts[split]=tries
    rng=random.Random(111056);rr=[];pairs=0
    while pairs<80:
        c=make_case(rng,FAMILIES[pairs%4]);d=copy.deepcopy(c)
        if c['family']=='pipeline':d['ops'][0][0]=rng.choice([x for x in ('add','sub','mul') if x!=c['ops'][0][0]])
        elif c['family']=='branch':d['yes'],d['no']=d['no'],d['yes']
        else:d['cmp']='lt' if d['cmp']=='gt' else 'gt'
        g,h=semantic_key(c),semantic_key(d)
        if g==h or g in seen or h in seen:continue
        seen.update([g,h]);lang='en' if (pairs//4)%2==0 else 'ru';pair=f'nccf{pairs:03d}'
        rr.extend([case_row(c,lang,0,pair),case_row(d,lang,0,pair)]);pairs+=1
    sets['counterfactual']=rr
    hashes={}
    for split,rr in sets.items():
        p=destination/f'{split}.jsonl';p.write_text(''.join(json.dumps(z,ensure_ascii=False,separators=(',',':'))+'\n' for z in rr),encoding='utf8')
        hashes[split]={'rows':len(rr),'groups':len({z['semantic'] for z in rr}),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    # Both-correct pairs are the unit for counterfactual metrics.
    manifest={'format':'L100NC-v1','splits':hashes,'cross_split_semantic_overlap':0,'source':'original procedural natural-language code tasks; no external teacher',
       'grouping':'exact affine/piecewise behaviour on every integer -64..64; singleton and empty signatures for list families',
       'ancestors':'EC complete original train unavailable, so ancestral decontamination is NOT established',
       'case_solver_is_post_generation_only':True,'language':'balanced English/Russian within non-CF splits','scope':'synthetic Python subset, not a universal language/code benchmark'}
    (destination/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print(json.dumps(hashes,indent=2))
    return manifest

if __name__=='__main__':build()
