"""Free generation only; restricted generated-code execution is confined to evaluator."""
from __future__ import annotations
import argparse,ast,collections,re,time
from common import *
from engine import generate
XS=(-17,-9,-2,-1,0,1,2,5,11,23,97,251)

def strip_fence(s):
    s=s.strip()
    m=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',s,re.S)
    return m.group(1).strip() if m else s

def numeric(s):
    # One terminal explicitly-labelled answer; no last-integer rescue.
    s=s.strip()
    if s.count('F=')!=1:return None
    m=re.search(r'(?:^|[;\s])F=(-?\d+)\s*$',s)
    return int(m.group(1)) if m else None

def execute_code(s):
    s=strip_fence(s)
    if len(s)>4000:raise ValueError('Code too long')
    t=ast.parse(s)
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,ast.AugAssign,ast.Name,ast.Load,ast.Store,
      ast.BinOp,ast.UnaryOp,ast.USub,ast.UAdd,ast.Constant,ast.Add,ast.Sub,ast.Mult)
    nn=list(ast.walk(t))
    if len(nn)>160 or any(not isinstance(n,allowed) for n in nn):raise ValueError('Disallowed AST')
    if len(t.body)!=1 or not isinstance(t.body[0],ast.FunctionDef):raise ValueError('Only one function permitted')
    fn=t.body[0]
    if fn.name!='solve' or fn.decorator_list or fn.returns or fn.args.defaults or fn.args.kw_defaults or fn.args.kwarg or fn.args.vararg:raise ValueError('Unsupported function')
    if fn.args.posonlyargs or fn.args.kwonlyargs or [a.arg for a in fn.args.args]!=['x']:raise ValueError('Expected solve(x)')
    for n in nn:
        if isinstance(n,ast.Name) and n.id!='x':raise ValueError('Only local x permitted')
        if isinstance(n,ast.Constant) and (type(n.value) is not int or abs(n.value)>10000):raise ValueError('Invalid constant')
        if isinstance(n,ast.FunctionDef) and n is not fn:raise ValueError('Nested function')
    env={'__builtins__':{}};exec(compile(t,'<generated>','exec'),env)
    result=[env['solve'](x) for x in XS]
    if not all(type(v) is int for v in result):raise ValueError('Noninteger result')
    return result

def expected_code(c):
    out=[]
    for x in XS:
        for op,n in c['ops']:x=x+n if op=='+' else x-n if op=='-' else x*n
        out.append(x)
    return out

def score(text,eos,row):
    if row['family']=='code_generation':
        try:
            got=execute_code(text);want=expected_code(row['case']);return bool(eos and got==want),{'outputs':got,'expected':want,'error':None}
        except Exception as e:return False,{'error':type(e).__name__+': '+str(e)}
    v=numeric(text);return bool(eos and v is not None and v==int(row['answer'])),{'parsed':v}

def evaluate(m,tok,rr,device):
    groups=collections.defaultdict(list);out=[];tick=time.perf_counter()
    for r in rr:groups[len(tok.encode(r['prompt'],bos=True))].append(r)
    for _,group in sorted(groups.items()):
        for i in range(0,len(group),8):
            bb=group[i:i+8];pp=generate(m,tok,[r['prompt'] for r in bb],128,device)
            for r,p in zip(bb,pp):
                ok,detail=score(p['text'],p['eos'],r)
                out.append({**p,'id':r['id'],'family':r['family'],'lang':r['lang'],'prompt':r['prompt'],'gold':r['answer'],
                            'pair_id':r.get('pair_id'),'correct':ok,'scoring':detail})
    by={}
    for f in sorted({r['family'] for r in rr}):
        zz=[r for r in out if r['family']==f];by[f]={'n':len(zz),'correct':sum(r['correct'] for r in zz)}
    pairs=collections.defaultdict(list)
    for r in out:
        if r['pair_id'] is not None:pairs[r['pair_id']].append(r)
    if not all(len(r)==2 for r in pairs.values()):raise ValueError('Incomplete pairs')
    return {'status':'completed','n':len(out),'correct':sum(r['correct'] for r in out),'by_family':by,
     'pairs':len(pairs),'both_correct_pairs':sum(all(r['correct'] for r in zz) for zz in pairs.values()),
     'same_text_pairs':sum(zz[0]['text']==zz[1]['text'] for zz in pairs.values()),'truncated':sum(not r['eos'] for r in out),
     'generated_tokens':sum(len(r['token_ids']) for r in out),'seconds':time.perf_counter()-tick,'rows':sorted(out,key=lambda r:r['id'])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--config',type=Path)
    ap.add_argument('--name',required=True);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    m=load_model(args.model,args.config,args.device);tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    for split in ('iid','surface','ood','counterfactual'):
        p=ROOT/'metrics'/f'{args.name}_{split}.json'
        if p.exists():raise FileExistsError('Evaluation already exists: '+str(p))
        result=evaluate(m,tok,rows(ROOT/'data'/f'{split}.jsonl'),args.device);put(p,result)
        print('EVAL',args.name,split,result['correct'],result['n'],'PAIRS',result['both_correct_pairs'],result['pairs'],flush=True)
    p=ROOT/'metrics'/f'{args.name}_reference.json';result=evaluate(m,tok,rows(ROOT/'bench/reference_cases.jsonl'),args.device);put(p,result)
    print('DONE',args.name,flush=True)
if __name__=='__main__':main()
