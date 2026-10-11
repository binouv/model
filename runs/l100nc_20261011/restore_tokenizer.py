"""Restore exactly the original lossless8192-token vocabulary, no new training."""
import hashlib,lzma
from pathlib import Path
root=Path(__file__).resolve().parent
parts=sorted((root/'data/tokenizer_xz').glob('*.xzpart'))
if len(parts)!=4:raise ValueError('four tokenizer parts required')
data=lzma.decompress(b''.join(p.read_bytes() for p in parts))
if hashlib.sha256(data).hexdigest()!='aa6f3b462d4d1bb9ace09d0421a1879500fea91127a347a29df3e46a6f58c500':raise ValueError('tokenizer SHA256 mismatch')
p=root/'data/tokenizer.json'
if p.exists() and p.read_bytes()!=data:raise ValueError('refuse different tokenizer overwrite')
p.write_bytes(data)
print('TOKENIZER_SHA256_VERIFIED')
