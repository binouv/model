"""Dependency-free, reversible lexical subwords with UTF-8 byte fallback.
Not BPE: full pieces are selected by train-only frequency, not pair merges.
Whitespace and individual digits are preserved. Arbitrary Unicode is supported.
"""
from __future__ import annotations
import json,re,collections
from pathlib import Path

PAT=re.compile(r" ?[^\W\d_]+|\d|_+|[^\w\s]+|\s+",re.UNICODE)
class Tokenizer:
    PAD=0; BOS=1; EOS=2
    def __init__(self,tokens):
        self.tokens=[bytes.fromhex(t) for t in tokens]
        self.lookup={s:i+259 for i,s in enumerate(self.tokens)}
        self.vocab_size=259+len(self.tokens)
    @classmethod
    def train(cls,texts,vocab_size=8192):
        counts=collections.Counter()
        for text in texts:
            for m in PAT.finditer(text):
                b=m.group().encode('utf-8')
                if 1<len(b)<=128: counts[b]+=1
        ordered=sorted(counts,key=lambda b:(-counts[b],b))[:vocab_size-259]
        if len(ordered)<vocab_size-259:
            # Add lossless byte pairs, only if the corpus has too few distinct pieces.
            seen=set(ordered)
            for a in range(256):
                for b in range(256):
                    x=bytes((a,b))
                    if x not in seen: ordered.append(x);seen.add(x)
                    if len(ordered)==vocab_size-259: break
                if len(ordered)==vocab_size-259: break
        return cls([s.hex() for s in ordered])
    def encode(self,text,bos=False,eos=False):
        out=[self.BOS] if bos else []
        # finditer fallback spans make the tokenizer lossless even for unusual Unicode.
        cursor=0
        for m in PAT.finditer(text):
            if m.start()>cursor: out.extend(x+3 for x in text[cursor:m.start()].encode('utf-8'))
            b=m.group().encode('utf-8'); k=self.lookup.get(b)
            if k is None: out.extend(x+3 for x in b)
            else: out.append(k)
            cursor=m.end()
        if cursor<len(text): out.extend(x+3 for x in text[cursor:].encode('utf-8'))
        if eos: out.append(self.EOS)
        return out
    def decode(self,ids):
        b=b''.join(bytes((int(i)-3,)) if 3<=int(i)<259 else self.tokens[int(i)-259] if int(i)>=259 else b'' for i in ids)
        return b.decode('utf-8',errors='replace')
    def save(self,path):
        Path(path).write_text(json.dumps({'type':'train_frequency_lexical_utf8_fallback_v1','special_ids':{'pad':0,'bos':1,'eos':2},'tokens_hex':[s.hex() for s in self.tokens]},ensure_ascii=False))
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text())['tokens_hex'])
