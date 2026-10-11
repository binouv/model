"""Restore genuine L100GE source/data/raw predictions/native adapter checkpoints from Git binary parts.

Uses only stdlib; refuses mismatched files, paths escaping the run, or tar symlinks.
Does NOT pretend to restore the separately published 100M parent from metadata.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
BLOCK = 1 << 20


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(BLOCK), b''):
            h.update(chunk)
    return h.hexdigest()


def restore_archive(parts_path: Path, archive_path: Path) -> None:
    manifest = json.loads((parts_path / 'MANIFEST.json').read_text())
    if manifest['format'] != 'flygraph.checkpoint-parts.v1':
        raise ValueError('Invalid parts schema')
    assert manifest['filename'] == archive_path.name
    if archive_path.exists():
        if archive_path.stat().st_size != manifest['size_bytes'] or digest(archive_path) != manifest['sha256']:
            raise ValueError('Existing archive mismatch: ' + str(archive_path))
        return
    h = hashlib.sha256()
    tmpfd, tmpname = tempfile.mkstemp(prefix='.restore-', dir=ROOT / 'artifacts')
    temp = Path(tmpname)
    try:
        total = 0
        with os.fdopen(tmpfd, 'wb') as output:
            for index, part in enumerate(manifest['parts']):
                expected_name = f'{archive_path.name}.part{index:04d}'
                if part['name'] != expected_name or not 0 < part['size_bytes'] < 90_000_000:
                    raise ValueError('Unsafe part name/size')
                path = parts_path / part['name']
                if path.is_symlink() or not path.is_file() or path.stat().st_size != part['size_bytes']:
                    raise ValueError('Missing/corrupt part ' + part['name'])
                ph = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(BLOCK), b''):
                        ph.update(chunk); h.update(chunk); total += len(chunk); output.write(chunk)
                if ph.hexdigest() != part['sha256']:
                    raise ValueError('Part SHA256 mismatch ' + part['name'])
            output.flush(); os.fsync(output.fileno())
        if total != manifest['size_bytes'] or h.hexdigest() != manifest['sha256']:
            raise ValueError('Full archive SHA256 mismatch')
        os.link(temp, archive_path)  # no existing file overwritten
    finally:
        temp.unlink(missing_ok=True)


def extract_safe(archive_path: Path) -> None:
    with tarfile.open(archive_path, 'r:xz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('Non-file tar entry not permitted')
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError('Unsafe archive member path')
            path = ROOT.joinpath(*rel.parts)
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('Archive escapes experiment dir')
            with archive.extractfile(member) as stream:
                data = stream.read()
            if path.exists():
                if not path.is_file() or path.is_symlink() or path.stat().st_size != len(data) or digest(path) != hashlib.sha256(data).hexdigest():
                    raise ValueError('Refusing to overwrite existing modified file ' + str(path))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def main() -> None:
    pairs = [('source_parts', 'source_data_results.tar.xz'), ('native_parts', 'all_native_checkpoints.tar.xz')]
    for sub, name in pairs:
        archive_path = ROOT / 'artifacts' / name
        restore_archive(ROOT / 'artifacts' / sub, archive_path)
        extract_safe(archive_path)
    sha_manifest = json.loads((ROOT / 'metrics/PACKAGED_SOURCE_FILE_HASHES.json').read_text())
    for rel, item in sha_manifest['files'].items():
        path = ROOT / rel
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError('Restored source/data hash mismatch ' + rel)
    for cp in sorted((ROOT / 'checkpoints').glob('step_*')):
        x = json.loads((cp / 'MANIFEST.json').read_text())
        assert digest(cp / 'adapter.safetensors') == x['weights_sha256']
        assert digest(cp / 'training_state.pt') == x['training_state_sha256']
    print(json.dumps({'status': 'verified_restored', 'source_data_files': len(sha_manifest['files']),
                      'native_checkpoint_steps': [int(x.name.split('_')[-1]) for x in sorted((ROOT / 'checkpoints').glob('step_*'))],
                      'parent_100M_restored_here': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()