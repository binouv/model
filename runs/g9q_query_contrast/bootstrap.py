import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
m=json.loads((ROOT/'configs/source_manifest.json').read_text())
for rel,z in m.items():
    src=ROOT/'source_parts'/z['gzip_file']; raw=src.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==z['gzip_sha256']
    plain=gzip.decompress(raw)
    assert hashlib.sha256(plain).hexdigest()==z['raw_sha256']
    dst=ROOT/rel; dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): assert dst.read_bytes()==plain
    else: dst.write_bytes(plain)
print('SOURCE_OK',len(m))
