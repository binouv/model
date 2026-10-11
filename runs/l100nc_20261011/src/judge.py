"""Post-generation execution of raw model code in a bounded Python subset.
Not imported by the model or trainer. No repair, retries, grammar decoding or
answer selection. AST allowlist + subprocess resource limits, no imports,
attributes, dynamic calls, loops, I/O or access to files/OS/network.
This is a constrained benchmark executor, NOT a general Python security sandbox.
"""
from __future__ import annotations
import ast,json,os,resource,subprocess,sys,tempfile
from pathlib import Path
ALLOWED=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Assign,
 ast.If,ast.IfExp,ast.Expr,ast.BinOp,ast.UnaryOp,ast.Name,ast.Constant,ast.Compare,
 ast.Load,ast.Store,ast.Add,ast.Sub,ast.Mult,ast.USub,ast.UAdd,ast.Lt,ast.Gt,
 ast.LtE,ast.GtE,ast.Eq,ast.NotEq,ast.List,ast.Tuple,ast.ListComp,ast.GeneratorExp,
 ast.comprehension,ast.Call,ast.BoolOp,ast.And,ast.Or,ast.Not)

def validate(code):
    if not isinstance(code,str) or len(code)>4096:raise ValueError('source length')
    tree=ast.parse(code)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):raise ValueError('one function only')
    fn=tree.body[0]
    if fn.name!='solve' or fn.decorator_list or fn.returns or len(fn.args.args)!=1 or fn.args.defaults or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:raise ValueError('invalid signature')
    if sum(1 for _ in ast.walk(tree))>200:raise ValueError('AST too large')
    for n in ast.walk(tree):
        if not isinstance(n,ALLOWED):raise ValueError('forbidden '+type(n).__name__)
        if isinstance(n,ast.FunctionDef) and n is not fn:raise ValueError('nested function')
        if isinstance(n,ast.Name) and (n.id.startswith('_') or n.id not in {'x','xs','solve','sum','y','v','a','b','result','values','numbers'}):raise ValueError('name rejected')
        if isinstance(n,ast.Call) and (not isinstance(n.func,ast.Name) or n.func.id!='sum' or len(n.args)!=1 or n.keywords):raise ValueError('call rejected')
        if isinstance(n,ast.Constant) and (type(n.value) not in (int,bool,type(None)) or (isinstance(n.value,int) and abs(n.value)>100000)):raise ValueError('constant rejected')
        if isinstance(n,ast.comprehension) and (n.is_async or not isinstance(n.iter,ast.Name) or n.iter.id!=fn.args.args[0].arg):raise ValueError('only iterate over input')
    if len(fn.body)>12:raise ValueError('too many statements')
    return tree

def worker(item):
    # Called in a separate isolated-mode interpreter with no inherited env.
    resource.setrlimit(resource.RLIMIT_CPU,(1,1));resource.setrlimit(resource.RLIMIT_AS,(128<<20,128<<20))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(16,16))
    source=item['code'];tree=validate(source)
    env={'__builtins__':{'sum':sum}};exec(compile(tree,'<generated>','exec'),env)
    results=[]
    for x in item['inputs']:
        y=env['solve'](x)
        if type(y) is not int and not (type(y) is list and len(y)<=100 and all(type(t) is int for t in y)):raise ValueError('result type')
        results.append(y)
    return results

def execute(code,inputs):
    try:validate(code)
    except Exception as e:return {'valid':False,'outputs':None,'error':type(e).__name__+': '+str(e)[:120]}
    try:
        with tempfile.TemporaryDirectory(prefix='fg_judge_') as td:
            r=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--worker'],
              input=json.dumps({'code':code,'inputs':inputs}),text=True,capture_output=True,
              cwd=td,env={'PATH':'/usr/bin:/bin','OMP_NUM_THREADS':'1'},timeout=3)
        if r.returncode:return {'valid':True,'outputs':None,'error':'worker exit '+str(r.returncode)}
        v=json.loads(r.stdout)
        return {'valid':True,'outputs':v['outputs'],'error':v.get('error')}
    except Exception as e:return {'valid':True,'outputs':None,'error':type(e).__name__}

if __name__=='__main__' and '--worker' in sys.argv:
    try:print(json.dumps({'outputs':worker(json.load(sys.stdin))}))
    except Exception as e:print(json.dumps({'outputs':None,'error':type(e).__name__+': '+str(e)[:120]}))
