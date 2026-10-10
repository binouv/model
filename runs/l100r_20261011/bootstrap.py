"""Restore the exact published L100R source and records. No training or inference."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import lzma
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe relative path: ' + name)
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT) or target == ROOT:
        raise ValueError('Path escapes bundle: ' + name)
    return target


def bundle(spec: dict) -> dict:
    pieces = []
    for item in spec['parts']:
        data = safe_path(item['path']).read_bytes()
        if len(data) != item['bytes'] or digest(data) != item['sha256']:
            raise ValueError('Invalid part: ' + item['path'])
        pieces.append(data)
    data = b''.join(pieces)
    if len(data) != spec['bytes'] or digest(data) != spec['sha256']:
        raise ValueError('Invalid assembled bundle')
    decoder = lzma.LZMADecompressor(memlimit=512 << 20)
    raw = decoder.decompress(data, max_length=64 << 20)
    if not decoder.eof or decoder.unused_data:
        raise ValueError('Oversized or trailing compressed data')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Expected object')
    return value


def write_files(files: dict[str, bytes]) -> None:
    # Check all destinations before changing any of them.
    checked = [(safe_path(name), data) for name, data in files.items()]
    for target, data in checked:
        if target.exists() and (not target.is_file() or target.read_bytes() != data):
            raise FileExistsError('Refuse to overwrite modified file: ' + str(target))
    for target, data in checked:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.restore.tmp')
        with temporary.open('xb') as stream:
            stream.write(data)
        os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--source-only', action='store_true')
    group.add_argument('--records', action='store_true')
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'FlyGraph-repo-bundle-v1':
        raise ValueError('Unknown bundle schema')
    source = bundle(manifest['bundles']['source'])
    write_files({name: text.encode('utf-8') for name, text in source.items()})
    counts = {'source_files': len(source), 'record_files': 0, 'training_updates_executed': 0}
    if args.records:
        if args.data is None:
            parser.error('--records requires --data pointing to preserved source_data')
        source_data = args.data.resolve()
        if digest((source_data / 'MANIFEST.json').read_bytes()) != manifest['records_dictionary_manifest_sha256']:
            raise ValueError('Wrong data dictionary manifest')
        sys.path.insert(0, str(ROOT / 'src'))
        spec = importlib.util.spec_from_file_location('l100r_record_codec', ROOT / 'src/compact_records.py')
        if spec is None or spec.loader is None:
            raise RuntimeError('Cannot import verified record codec')
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        files = codec.decode(bundle(manifest['bundles']['records']), source_data, ROOT / 'data/tokenizer.json')
        write_files(files)
        counts['record_files'] = len(files)
    if counts['source_files'] != manifest['source_files'] or (args.records and counts['record_files'] != manifest['record_files']):
        raise ValueError('Unexpected restored file count')
    print(json.dumps({'status': 'verified', **counts}))


if __name__ == '__main__':
    main()
