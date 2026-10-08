"""Restore the inherited vocabulary from already-published binary data fragments.
No model weights or executable source are encoded here. Never overwrite a
conflicting vocabulary. The training run used this exact pre-existing vocabulary.
"""
from pathlib import Path
import argparse,hashlib,lzma
ROOT=Path(__file__).resolve().parents[1]
PARTS=[(9000,'a18dda3213fb029b2ed9c76dcd3c2a5ef5be3eb3d72b627e834a396f8ea8c1a9'),(9000,'7285f2f229c4f21024ea8a3be647fe64881b0e2137b767430c8cd847d21cff12'),(9000,'63152c313dd397b98611c54054851672df89fb837f07b9c0a78dfc921f41607d'),(1368,'f04d6ad995e338e3c0ed7061dbb3978229d3f25eae1d72466a31d97d46d0a819')]
EXPECTED='aa6f3b462d4d1bb9ace09d0421a1879500fea91127a347a29df3e46a6f58c500'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'data/tokenizer.json');a=ap.parse_args();chunks=[]
 for i,(size,digest) in enumerate(PARTS):
  b=(ROOT/'data/tokenizer_xz'/f'part{i:03d}.xzpart').read_bytes()
  if len(b)!=size or hashlib.sha256(b).hexdigest()!=digest:raise ValueError('corrupt vocabulary fragment')
  chunks.append(b)
 b=lzma.decompress(b''.join(chunks))
 if hashlib.sha256(b).hexdigest()!=EXPECTED:raise ValueError('wrong vocabulary')
 if a.output.exists() and a.output.read_bytes()!=b:raise FileExistsError('refuse vocabulary replacement')
 a.output.parent.mkdir(exist_ok=True,parents=True)
 if not a.output.exists():a.output.write_bytes(b)
 print('VOCABULARY_SHA256_VERIFIED',len(b),EXPECTED)
if __name__=='__main__':main()
