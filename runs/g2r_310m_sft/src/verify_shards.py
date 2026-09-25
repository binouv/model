"""Streaming integrity/finite-value verification without allocating a full model."""
from __future__ import annotations
import argparse,hashlib,json,math,struct
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def verify(path):
 path=Path(path)
 if not (path/'COMPLETE').is_file():raise ValueError('checkpoint incomplete')
 idx=json.loads((path/'index.json').read_text());seen={};floating_count=0;buffer_count=0;rows=[]
 for rec in idx['files']:
  name=rec['file']
  if Path(name).name!=name:raise ValueError('unsafe shard name')
  p=path/name
  if sha(p)!=rec['sha256']:raise ValueError('shard checksum mismatch: '+name)
  with p.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0]
   if n>10_000_000:raise ValueError('oversized header')
   head=json.loads(f.read(n));start=8+n
   for key,z in head.items():
    if key=='__metadata__':continue
    if key in seen:raise ValueError('duplicate tensor')
    seen[key]=name;dt=z['dtype'];shape=z['shape'];count=math.prod(shape);lo,hi=z['data_offsets']
    esize={'F32':4,'BF16':2,'F16':2,'I64':8,'I32':4}.get(dt)
    if esize is None or hi-lo!=esize*count or start+hi>p.stat().st_size:raise ValueError('tensor storage mismatch')
    finite=True
    if dt in ('F32','BF16','F16'):
     floating_count+=count;f.seek(start+lo);remain=hi-lo
     while remain:
      chunk=f.read(min(remain,1<<20))
      if not chunk:raise ValueError('truncated tensor')
      a=np.frombuffer(chunk,dtype='<u2' if dt=='BF16' else '<f4' if dt=='F32' else '<f2')
      if dt=='BF16':finite &= not bool(np.any((a & 0x7f80)==0x7f80))
      else:finite &= bool(np.isfinite(a).all())
      remain-=len(chunk)
    else:buffer_count+=count
    if not finite:raise ValueError('nonfinite weights: '+key)
    rows.append({'tensor':key,'dtype':dt,'shape':shape,'elements':count,'all_finite':finite})
 if seen!=idx['weight_map']:raise ValueError('index and actual tensor mapping disagree')
 if floating_count!=309993253:raise ValueError('unexpected parameter count')
 return {'status':'passed','path':str(path),'step':idx['step'],'shards':len(idx['files']),'weight_bytes':sum((path/z['file']).stat().st_size for z in idx['files']),'trainable_floating_parameter_values':floating_count,'nonfloating_buffer_values':buffer_count,'all_values_finite':True,'all_shard_sha256_match':True,'index_sha256':sha(path/'index.json'),'tensors':rows}

def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--out',required=True);x=a.parse_args();z=verify(x.checkpoint);Path(x.out).write_text(json.dumps(z,indent=2));print(json.dumps({k:v for k,v in z.items() if k!='tensors'},indent=2))
if __name__=='__main__':main()
