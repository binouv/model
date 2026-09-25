from __future__ import annotations
import collections,hashlib,json,random,sys
from pathlib import Path
from g2_data import make_case,render,solve,FAMILIES
from tokenizer import Tokenizer
ROOT=Path(__file__).resolve().parents[1]
PREFIX='Return the final integer.\n'
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def canonical(c):
    c=json.loads(json.dumps(c));f=c['family']
    if f=='arithmetic':
        c['values']=sorted(c['values']);c.pop('op')
    elif f=='list_reasoning':c['values']=sorted(c['values']);c.pop('kind')
    elif f=='memory_update':
        keys=sorted({k for k,v in c['writes']});c={'family':f,'histories':{k:[v for kk,v in c['writes'] if k==kk] for k in keys}}
    return h(c)
def main():
    d=ROOT/'data';tok=Tokenizer.load(d/'tokenizer.json');forbidden=set();held=[]
    for path in sorted(d.glob('test_*.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()];held.extend(rows)
        for z in rows:
            forbidden.add(canonical(z['case']));z['prompt']=z['prompt'] if z['prompt'].startswith(PREFIX) else PREFIX+z['prompt'];p=tok.encode(z['prompt'],bos=True)
            z['tokens']=p+tok.encode(z['completion'],eos=True);z['prompt_tokens']=len(p)
        path.write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in rows))
    occupied=set(forbidden);stats={};sets={}
    for split,n,seed in [('validation',400,73111),('train',16384,73211)]:
        rng=random.Random(seed);rows=[];seen=set();gs=set();attempts=0
        while len(rows)<n:
            attempts+=1
            if attempts>2000000:raise RuntimeError('exhausted unique cases')
            f=FAMILIES[len(rows)%5];c=make_case(rng,f);g=canonical(c)
            if g in occupied:continue
            lang=rng.choice(['ru','en']);variant=rng.randrange(2);prompt=PREFIX+render(c,lang,variant)
            if prompt in seen:continue
            pt=tok.encode(prompt,bos=True);answer=str(solve(c));co=' '+answer+'\n';tt=pt+tok.encode(co,eos=True)
            if len(tt)>128:continue
            z={'id':h({'case':c,'lang':lang,'variant':variant}),'canonical':g,'family':f,'lang':lang,'prompt':prompt,'answer':answer,'completion':co,'case':c,'tokens':tt,'prompt_tokens':len(pt)}
            rows.append(z);seen.add(prompt);gs.add(g)
        occupied.update(gs);sets[split]=gs
        (d/f'{split}.jsonl').write_text(''.join(json.dumps(z,ensure_ascii=False)+'\n' for z in rows))
        stats[split]={'n':n,'canonical_groups':len(gs),'attempts':attempts,'tokens':sum(len(z['tokens']) for z in rows),'max_tokens':max(len(z['tokens']) for z in rows),'by_family':dict(collections.Counter(z['family'] for z in rows))}
    assert not sets['train']&sets['validation'] and not sets['train']&forbidden
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob('*.jsonl'))}
    audit={'status':'completed','stats':stats,'held_cases':len(held),'held_canonical_groups':len(forbidden),'train_held_canonical_overlap':0,'validation_held_overlap':0,'new_random_weights_no_G1_parameter_initialization':True,'G1_tokenizer_only_reused':True,'prefix':PREFIX,'hashes':hashes,'note':'All held suites remain synthetic; old G2 test fixtures reused without model-selection on them. Canonical grouping conservative, not a proof of semantic novelty for every possible equivalence.'}
    (d/'manifest.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
