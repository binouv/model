"""Restore byte-verified source and lossless primary span coordinates; stdlib only."""
import base64,hashlib,json,lzma
from pathlib import Path
root=Path(__file__).resolve().parent
packed=base64.b64decode(''.join(p.read_text() for p in sorted((root/'source_parts').glob('*.b64'))),validate=True)
assert hashlib.sha256(packed).hexdigest()=='a955c5d59c69ca8615a4f9121103075f777a4f4331b86b4d539ad123a4f85085'
source=json.loads(lzma.decompress(packed))
for name,text in source.items():
 path=(root/name).resolve()
 if not path.is_relative_to(root):raise ValueError('unsafe path')
 if path.exists() and path.read_text()!=text:raise FileExistsError(path)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
packed=base64.b64decode((root/'SPAN_COORDINATES.b64').read_text(),validate=True)
assert hashlib.sha256(packed).hexdigest()=='f83f898fec09146a2ef4ca5e41172387f4d18d380536c4b380991b09458b170f'
(root/'metrics').mkdir(exist_ok=True)
(root/'metrics/SPAN_COORDINATES.json').write_bytes(lzma.decompress(packed))
print('Verified restored source files:',len(source))
