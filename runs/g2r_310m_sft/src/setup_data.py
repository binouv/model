"""Reconstruct the exact G2 vocabulary and synthetic data entirely offline.

Small binary vocabulary fragments are real committed bytes, not model weights.
Refuse to replace any local artifact with different content. Verify all generated
JSONL files before installing them. No training or test inference is run here.
"""
from __future__ import annotations
import argparse, hashlib, json, lzma, os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PARTS = [
    (9000, 'a18dda3213fb029b2ed9c76dcd3c2a5ef5be3eb3d72b627e834a396f8ea8c1a9'),
    (9000, '7285f2f229c4f21024ea8a3be647fe64881b0e2137b767430c8cd847d21cff12'),
    (9000, '63152c313dd397b98611c54054851672df89fb837f07b9c0a78dfc921f41607d'),
    (1368, 'f04d6ad995e338e3c0ed7061dbb3978229d3f25eae1d72466a31d97d46d0a819'),
]
VOCAB_SHA = 'aa6f3b462d4d1bb9ace09d0421a1879500fea91127a347a29df3e46a6f58c500'

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=ROOT/'data')
    args = ap.parse_args()
    source = ROOT/'data'
    chunks = []
    for i, (size, checksum) in enumerate(PARTS):
        b = (source/'tokenizer_xz'/f'part{i:03d}.xzpart').read_bytes()
        if len(b) != size or digest(b) != checksum:
            raise ValueError(f'corrupt vocabulary part {i}')
        chunks.append(b)
    vocabulary = lzma.decompress(b''.join(chunks))
    if digest(vocabulary) != VOCAB_SHA:
        raise ValueError('vocabulary SHA256 mismatch')
    expected = json.loads((source/'EXPECTED_SHA256.json').read_text())
    sys.path.insert(0, str(ROOT/'base_src'))
    from tokenizer import Tokenizer
    from data import build
    with tempfile.TemporaryDirectory(prefix='flygraph-g2-data-') as td:
        tmp = Path(td)
        (tmp/'tokenizer.json').write_bytes(vocabulary)
        manifest = build(tmp, Tokenizer.load(tmp/'tokenizer.json'))
        verified = {'tokenizer.json': VOCAB_SHA, **expected}
        for name, checksum in verified.items():
            if Path(name).name != name or digest((tmp/name).read_bytes()) != checksum:
                raise ValueError('regenerated data differs: '+name)
            dest = args.output/name
            if dest.exists() and digest(dest.read_bytes()) != checksum:
                raise FileExistsError('refuse to overwrite different data: '+str(dest))
        args.output.mkdir(parents=True, exist_ok=True)
        for name in verified:
            dest = args.output/name
            if not dest.exists():
                staging = dest.with_suffix(dest.suffix+'.tmp')
                staging.write_bytes((tmp/name).read_bytes())
                os.replace(staging, dest)
        # Derived metadata, never counted as a model benchmark.
        metadata = args.output/'manifest.json'
        if not metadata.exists():
            metadata.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status': 'exact_offline_data_reconstruction_verified',
                      'files': len(verified), 'sha256': verified}, indent=2))

if __name__ == '__main__':
    main()
