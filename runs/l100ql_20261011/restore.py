"""Restore byte-exact L100QL sources from the tracked compact source package."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, lzma

ROOT = Path(__file__).resolve().parent
SHA256 = '3ef7db1c7bcb6546dbfa432230981ba791f6a684bb4e6604928d8d123009c5f7'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT)
    args = parser.parse_args()
    payload = b''.join((ROOT/'source_parts'/f'part{i:03d}.xzpart').read_bytes() for i in range(8))
    if len(payload) != 19824 or hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Source package size/SHA256 mismatch')
    sources = json.loads(lzma.decompress(payload))
    if not isinstance(sources, dict) or len(sources) != 13:
        raise ValueError('Unexpected source package schema')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    targets = []
    for name, content in sources.items():
        rel = PurePosixPath(name)
        if not isinstance(content, str) or rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe source path')
        path = out.joinpath(*rel.parts)
        if not path.resolve().is_relative_to(out):
            raise ValueError('Source destination escapes output directory')
        value = content.encode('utf-8')
        if path.exists() and path.read_bytes() != value:
            raise FileExistsError(f'Refusing to overwrite modified source: {path}')
        targets.append((path, value))
    for path, value in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    print(json.dumps({'restored_files': len(targets), 'source_sha256': SHA256, 'output': str(out)}))

if __name__ == '__main__':
    main()
