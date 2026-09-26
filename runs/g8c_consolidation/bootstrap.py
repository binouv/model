"""Unpack exactly three original, checksum-verified source files."""
import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'configs/source_manifest.json').read_text())
for name,m in manifest.items():
 assert name in ('src/study.py','src/ci.py','tests/test_protocol.py')
 parts=sorted((ROOT/'source_parts').glob(Path(name).stem+'.part*'))
 compressed=b''.join(p.read_bytes() for p in parts)
 assert hashlib.sha256(compressed).hexdigest()==m['gzip_sha256'],name
 plain=gzip.decompress(compressed)
 assert len(plain)==m['bytes'] and hashlib.sha256(plain).hexdigest()==m['sha256'],name
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==plain,'refuse overwrite different source'
 else:p.write_bytes(plain)
print('SOURCE_SHA256_VERIFIED',len(manifest))
