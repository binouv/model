"""Explicit, persistent discrete records; no target labels or neural soft mixing.
This is an exposed tool interface, NOT evidence that the LM learned to use it.
"""
from __future__ import annotations
import json
from dataclasses import dataclass,asdict
from pathlib import Path
@dataclass(frozen=True)
class Record:
    key:str
    value:str
    revision:int
class ProtectedMemory:
    def __init__(self): self._records=[]
    def write(self,key:str,value:str)->Record:
        if not isinstance(key,str) or not isinstance(value,str): raise TypeError('string key/value required')
        r=Record(key,value,len(self._records)); self._records.append(r); return r
    def read(self,key:str):
        for r in reversed(self._records):
            if r.key==key:return r
        return None
    def serialize(self,keys:list[str])->str:
        rows=[]
        for key in keys:
            r=self.read(key)
            if r is not None: rows.append(asdict(r))
        return json.dumps(rows,ensure_ascii=False,sort_keys=True)
    def save(self,path): Path(path).write_text(json.dumps([asdict(r) for r in self._records],ensure_ascii=False))
    @classmethod
    def load(cls,path):
        m=cls()
        for row in json.loads(Path(path).read_text()):
            r=m.write(row['key'],row['value'])
            if r.revision!=row['revision']:raise ValueError('invalid revision chain')
        return m
