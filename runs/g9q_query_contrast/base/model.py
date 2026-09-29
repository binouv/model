from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
PAD,BOS,EOS,VOCAB=0,1,2,259
class ByteTokenizer:
 def encode(self,s,bos=False,eos=False):return ([BOS] if bos else [])+[3+b for b in s.encode()]+([EOS] if eos else [])
 def decode(self,z):return bytes([i-3 for i in z if i>=3]).decode('utf-8','replace')
@dataclass
class Config:
 variant:str='causal_hybrid';vocab:int=VOCAB;dim:int=96;heads:int=4;hidden:int=256;key_dim:int=12;max_context:int=256
class Norm(nn.Module):
 def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
 def forward(self,x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*self.weight
class Attention(nn.Module):
 def __init__(self,c):
  super().__init__();self.c=c;self.hd=c.dim//c.heads;self.qkv=nn.Linear(c.dim,3*c.dim,bias=False);self.out=nn.Linear(c.dim,c.dim,bias=False);self.register_buffer('freq',1/(10000**(torch.arange(0,self.hd,2).float()/self.hd)),persistent=False)
 def rope(self,x,start):
  t=torch.arange(start,start+x.shape[2],device=x.device);a=t[:,None]*self.freq[None];co,si=a.cos()[None,None],a.sin()[None,None];p,q=x[...,::2],x[...,1::2];return torch.stack((p*co-q*si,p*si+q*co),-1).flatten(-2)
 def forward(self,x,cache=None,prefix_lens=None,prefix_bidir=False):
  B,T,D=x.shape;H=self.c.heads;hd=self.hd;q,k,v=self.qkv(x).view(B,T,3,H,hd).permute(2,0,3,1,4).unbind(0);start=0 if cache is None else cache[0].shape[2];q,k=self.rope(q,start),self.rope(k,start)
  if cache is not None:k=torch.cat((cache[0],k),2);v=torch.cat((cache[1],v),2)
  if cache is not None and T==1:y=F.scaled_dot_product_attention(q,k,v,is_causal=False)
  elif start==0 and prefix_bidir and prefix_lens is not None:
   pos=torch.arange(T,device=x.device);qp=pos[None,:,None];kp=pos[None,None,:];pl=prefix_lens[:,None,None];allow=torch.where(qp<pl,kp<pl,kp<=qp);y=F.scaled_dot_product_attention(q,k,v,attn_mask=allow[:,None])
  elif start==0:y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
  else:
   kp=torch.arange(k.shape[2],device=x.device)[None];qp=torch.arange(start,start+T,device=x.device)[:,None];y=F.scaled_dot_product_attention(q,k,v,attn_mask=kp<=qp)
  return self.out(y.transpose(1,2).reshape(B,T,D)),(k,v)
def parallel_delta(q,k,v,g,beta,state=None):
 q,k,v,g,beta=(z.float() for z in (q,k,v,g,beta));B,H,T,K=k.shape;V=v.shape[-1]
 if state is None:state=torch.zeros((B,H,K,V),device=q.device)
 if T==1:
  d=state*g[:,:,0,None,None].exp();u=beta[:,:,0,None]*(v[:,:,0]-(k[:,:,0,:,None]*d).sum(-2));last=d+k[:,:,0,:,None]*u[:,:,None];return (q[:,:,0,:,None]*last).sum(-2).unsqueeze(2),last
 lp=g.cumsum(-1);ratio=(lp[...,None]-lp[...,None,:]).clamp_max(0).exp().tril();lower=((k@k.transpose(-1,-2))*ratio*beta[...,None]).tril(-1);I=torch.eye(T,device=q.device);rhs=beta[...,None]*(v-(k@state)*lp.exp()[...,None]);u=torch.linalg.solve_triangular(I+lower,rhs,upper=False,unitriangular=True);y=(q@state)*lp.exp()[...,None]+((q@k.transpose(-1,-2))*ratio).tril()@u;last=state*lp[:,:,-1,None,None].exp()+k.transpose(-1,-2)@(u*ratio[:,:,-1,:,None]);return y,last
class Delta(nn.Module):
 def __init__(self,c):
  super().__init__();H,K,V=c.heads,c.key_dim,c.dim//c.heads;self.c=c;self.q=nn.Linear(c.dim,H*K,bias=False);self.k=nn.Linear(c.dim,H*K,bias=False);self.v=nn.Linear(c.dim,H*V,bias=False);self.gates=nn.Linear(c.dim,2*H);self.read=nn.Linear(c.dim,c.dim,bias=False);self.out=nn.Linear(c.dim,c.dim,bias=False)
 def forward(self,x,cache=None):
  B,T,D=x.shape;H,K,V=self.c.heads,self.c.key_dim,D//self.c.heads;q=F.normalize(self.q(x).view(B,T,H,K).transpose(1,2),dim=-1);k=F.normalize(self.k(x).view(B,T,H,K).transpose(1,2),dim=-1);v=self.v(x).view(B,T,H,V).transpose(1,2);gg,bb=self.gates(x).view(B,T,2,H).permute(2,0,3,1).unbind(0);y,s=parallel_delta(q,k,v,-F.softplus(gg-4),bb.sigmoid(),cache);return self.out(y.transpose(1,2).reshape(B,T,D)*self.read(x).sigmoid()),s
class Block(nn.Module):
 def __init__(self,c,kind):super().__init__();self.kind=kind;self.n1=Norm(c.dim);self.n2=Norm(c.dim);self.mix=Attention(c) if kind=='a' else Delta(c);self.uv=nn.Linear(c.dim,2*c.hidden,bias=False);self.down=nn.Linear(c.hidden,c.dim,bias=False)
 def forward(self,x,cache=None,prefix_lens=None,prefix_bidir=False):
  y,s=self.mix(self.n1(x),cache,prefix_lens,prefix_bidir) if self.kind=='a' else self.mix(self.n1(x),cache);x=x+y;u,v=self.uv(self.n2(x)).chunk(2,-1);return x+self.down(F.silu(u)*v),s
class Model(nn.Module):
 def __init__(self,c):
  super().__init__();self.config=c;assert c.variant in ('causal_hybrid','prefix_hybrid');self.embed=nn.Embedding(c.vocab,c.dim);self.blocks=nn.ModuleList([Block(c,k) for k in 'dada']);self.norm=Norm(c.dim);self.apply(self._init)
 def _init(self,m):
  if isinstance(m,(nn.Linear,nn.Embedding)):
   nn.init.normal_(m.weight,std=.02)
   if getattr(m,'bias',None) is not None:nn.init.zeros_(m.bias)
 def forward(self,ids,targets=None,prefix_lens=None,cache=None,last_only=False):
  x=self.embed(ids);new=[]
  for i,b in enumerate(self.blocks):x,s=b(x,None if cache is None else cache[i],prefix_lens,self.config.variant=='prefix_hybrid');new.append(s)
  x=self.norm(x)
  if targets is not None:
   keep=targets.ne(-100);return F.cross_entropy(F.linear(x[keep],self.embed.weight),targets[keep])
  return F.linear(x[:,-1:] if last_only else x,self.embed.weight),new
 @torch.inference_mode()
 def generate(self,ids,prefix_lens,max_new=12):
  self.eval();logits,cache=self(ids,prefix_lens=prefix_lens,last_only=True);out=[];done=torch.zeros(ids.shape[0],dtype=torch.bool)
  for _ in range(max_new):
   z=logits[:,-1].argmax(-1);z=torch.where(done,torch.full_like(z,EOS),z);out.append(z);done|=z.eq(EOS)
   if bool(done.all()):break
   logits,cache=self(z[:,None],cache=cache,last_only=True)
  return torch.stack(out,1)
def report(m):
 n=sum(p.numel() for p in m.parameters());return {'total_parameters':n,'active_parameters':n,'variant':m.config.variant,'executed_blocks':4,'persistent_intertoken_delta_state':True,'prefix_bidirectional_attention':m.config.variant=='prefix_hybrid'}
