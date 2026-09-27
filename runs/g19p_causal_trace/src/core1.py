import os,math,json,random,time,hashlib,re,tarfile
from pathlib import Path
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

torch.set_num_threads(max(1,min(4,os.cpu_count() or 1))); torch.use_deterministic_algorithms(True)
PAD,BOS,EOS,VOCAB,MAXLEN=258,256,257,259,128

def enc(s,bos=False,eos=False): return ([BOS] if bos else [])+list(s.encode('ascii'))+([EOS] if eos else [])
def dec(x): return bytes([i for i in x if 0<=i<256]).decode('ascii','ignore')
def hfile(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def hobj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
@dataclass(frozen=True)
class C: id:str; fam:str; p:str; tr:tuple; a:int; pair:str|None=None

def make_data(seed=190001):
 r=random.Random(seed); so,sf,sm=set(),set(),set()
 def op(fam,surf=False,extra=False,cid=''):
  while 1:
   x=r.randrange(10); cur=x; ds=[]; st=[]; mags=(3,) if extra else (1,2)
   for _ in range(4):
    q=[d for m in mags for d in (-m,m) if 0<=cur+d<=9]; d=r.choice(q); ds+=[d]; cur+=d; st+=[cur]
   k=(x,tuple(ds))
   if k in so or st==sorted(st) or st==sorted(st,reverse=True): continue
   so.add(k); break
  if fam=='arithmetic':
   p=(f"Start at {x}. Then "+', then '.join(('increase' if d>0 else 'decrease')+f' by {abs(d)}' for d in ds)+'. Return final digit.\n') if surf else f"ARITH x={x}; ops {' '.join(('+' if d>0 else '-')+str(abs(d)) for d in ds)}. Return final digit.\n"
  else:
   p=(f"Execute in order: x={x}; "+'; '.join(f"x = x {'+' if d>0 else '-'} {abs(d)}" for d in ds)+'. Return x.\n') if surf else f"CODE x={x};"+';'.join(f"x{'+' if d>0 else '-'}={abs(d)}" for d in ds)+'. Return x.\n'
  return C(cid,fam,p,tuple(st),st[-1])
 def fold(surf=False,extra=False,cid=''):
  while 1:
   m=3 if extra else 2; init=r.randrange(10); vs=[r.randrange(10) for _ in range(4)]; s=init; st=[]
   for v in vs: s=(m*s+v)%10; st+=[s]
   k=(m,init,tuple(vs))
   if k in sf or st==sorted(st): continue
   sf.add(k); break
  p=f"Begin s={init}. For each digit in [{','.join(map(str,vs))}], update s=({m}*s+digit) mod 10. Return the last s.\n" if surf else f"FOLD s={init}; v={','.join(map(str,vs))}; rule s=({m}*s+v)%10. Return final digit.\n"
  return C(cid,'list_reasoning',p,tuple(st),st[-1])
 def mem(surf=False,cid=''):
  while 1:
   A=[r.randrange(10) for _ in range(4)]; B=[r.randrange(10) for _ in range(4)]; k=(tuple(A),tuple(B))
   if k in sm or A[-1]==B[-1]: continue
   sm.add(k); break
  hist=' '.join(f'{k}={v}' for a,b in zip(A,B) for k,v in [('A',a),('B',b)])
  out=[]
  for q,tr in [('A',A),('B',B)]:
   p=f"Memory log: {hist}. What is the latest value at key {q}? Return one digit.\n" if surf else f"MEM {hist}; QUERY {q}. Return final digit.\n"
   out+=[C(cid+q,'memory_update',p,tuple(tr),tr[-1],cid)]
  return out
 def mix(n,pref,surf=False,extra=False):
  z=[]; fs=['arithmetic','code_trace','list_reasoning','memory_update']; i=0
  while len(z)<n:
   f=fs[i%4]; cid=f'{pref}{i:05d}'
   if f in ('arithmetic','code_trace'): z+=[op(f,surf,extra,cid)]
   elif f=='list_reasoning': z+=[fold(surf,extra,cid)]
   else: z+=[mem(surf,cid)[i%2]]
   i+=1
  return z[:n]
 d={'train':mix(512,'tr'),'validation':mix(64,'va'),'iid':mix(128,'ii'),'surface':mix(64,'su',True),'extrapolation':mix(64,'ex',False,True)}
 cf=[]
 for i in range(32): cf+=mem(False,f'cf{i:04d}')
 d['counterfactual']=cf; return d

def tgt(c,arm):
 t=['x']*4 if arm=='noop_trace' else [str(x) for x in (sorted(c.tr) if arm=='bag_trace' else c.tr)]
 return 'T='+','.join(t)+';F='+str(c.a)

class LM(nn.Module):
 def __init__(self):
  super().__init__(); self.e=nn.Embedding(VOCAB,128); self.r=nn.GRU(128,192,2,batch_first=True); self.n=nn.LayerNorm(192); self.o=nn.Linear(192,VOCAB,bias=False)
 def forward(self,x,h=None): y,h=self.r(self.e(x),h); return self.o(self.n(y)),h

def batch(cs,arm):
 xs=[]; ys=[]
 for c in cs:
  p=enc(c.p,True); o=enc(tgt(c,arm),eos=True); s=p+o; xs+=[s]; ys+=[[-100]*len(p)+o]
 T=max(map(len,xs)); return torch.tensor([x+[PAD]*(T-len(x)) for x in xs]),torch.tensor([y+[-100]*(T-len(y)) for y in ys])
def loss(m,cs,arm):
 x,y=batch(cs,arm); q,_=m(x); return F.cross_entropy(q[:,:-1].reshape(-1,VOCAB),y[:,1:].reshape(-1),ignore_index=-100)
