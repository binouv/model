"""L100C utilities. No gold or symbolic solver enters Model.generate."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base'))
from tokenizer import Tokenizer
from model import Model,Config

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def put(path,z):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False));os.replace(q,p)
def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def hash_obj(z):return hashlib.sha256(json.dumps(z,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
