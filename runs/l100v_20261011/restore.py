"""Restore the exact executed UTF-8 sources. Never executes decoded source."""
import hashlib,json,lzma
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPECTED='83bc17c576a1d647245664f810930a5b3cb9d7c9eb03e51b1eedf0c26bc1998d'
def main():
    raw=b''.join((ROOT/'source_parts'/f'source.{i:03d}.xzpart').read_bytes() for i in range(3))
    if len(raw)!=16180 or hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('source archive integrity failure')
    mapping=json.loads(lzma.decompress(raw))
    for name,text in mapping.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not isinstance(text,str):raise ValueError('invalid source entry')
        if path.exists() and path.read_bytes()!=text.encode():raise FileExistsError('refuse to replace modified file: '+name)
    for name,text in mapping.items():
        path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text.encode())
    print('Restored',len(mapping),'exact files; model weights and datasets are separate artifacts.')
if __name__=='__main__':main()
