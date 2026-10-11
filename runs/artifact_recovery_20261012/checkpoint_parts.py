#!/usr/bin/env python3
"""Lossless, bounded checkpoint parts. Standard library only; no pickle/code loading."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

LIMIT = 90_000_000
DEFAULT_PART_BYTES = 80_000_000
BLOCK = 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(BLOCK), b''):
            value.update(data)
    return value.hexdigest()


def safe_name(value: object) -> str:
    if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value or '\x00' in value or ':' in value:
        raise ValueError('Expected a plain filename')
    return value


def read_manifest(folder: Path) -> dict:
    path = folder / 'MANIFEST.json'
    if path.stat().st_size > 4_000_000:
        raise ValueError('Oversized manifest')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Manifest must be an object')
    if manifest.get('format') != 'flygraph.checkpoint-parts.v1':
        raise ValueError('Unsupported manifest format')
    safe_name(manifest['filename'])
    if type(manifest.get('size_bytes')) is not int or manifest['size_bytes'] <= 0:
        raise ValueError('Invalid total size')
    if not HEX.fullmatch(str(manifest.get('sha256', ''))):
        raise ValueError('Invalid total SHA256')
    parts = manifest.get('parts')
    if not isinstance(parts, list) or not parts or len(parts) > 10000:
        raise ValueError('Invalid parts')
    names, total = set(), 0
    for index, part in enumerate(parts):
        name = safe_name(part['name'])
        if name != f"{manifest['filename']}.part{index:04d}" or name in names:
            raise ValueError('Part order, duplicate or filename mismatch')
        names.add(name)
        if type(part.get('size_bytes')) is not int or not 0 < part['size_bytes'] < LIMIT:
            raise ValueError('Part must be positive and strictly smaller than 90 MB')
        if not HEX.fullmatch(str(part.get('sha256', ''))):
            raise ValueError('Invalid part SHA256')
        total += part['size_bytes']
    if total != manifest['size_bytes']:
        raise ValueError('Part sizes do not sum to checkpoint size')
    return manifest


def split(source: Path, folder: Path, part_bytes: int = DEFAULT_PART_BYTES,
          identity: str = '', expected_sha256: str | None = None) -> dict:
    source, folder = source.resolve(), folder.absolute()
    if type(part_bytes) is not int or not 0 < part_bytes < LIMIT:
        raise ValueError('part_bytes must be below 90,000,000')
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError('Missing or empty source')
    if expected_sha256 and not HEX.fullmatch(expected_sha256):
        raise ValueError('Invalid expected SHA256')
    if folder.exists():
        raise FileExistsError(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.split-', dir=folder.parent))
    try:
        complete, total, parts = hashlib.sha256(), 0, []
        with source.open('rb') as stream:
            index = 0
            while True:
                first = stream.read(min(BLOCK, part_bytes))
                if not first:
                    break
                name = f'{source.name}.part{index:04d}'
                part_hash, size = hashlib.sha256(), 0
                with (stage / name).open('xb') as output:
                    data = first
                    while data:
                        output.write(data)
                        part_hash.update(data); complete.update(data)
                        size += len(data); total += len(data)
                        if size == part_bytes:
                            break
                        data = stream.read(min(BLOCK, part_bytes - size))
                    output.flush(); os.fsync(output.fileno())
                parts.append({'name': name, 'size_bytes': size, 'sha256': part_hash.hexdigest()})
                index += 1
        if expected_sha256 and complete.hexdigest() != expected_sha256:
            raise ValueError('Source SHA256 does not match the expected checkpoint')
        manifest = {'format': 'flygraph.checkpoint-parts.v1', 'identity': identity,
                    'filename': source.name, 'size_bytes': total,
                    'sha256': complete.hexdigest(), 'max_part_bytes': part_bytes,
                    'parts': parts}
        (stage / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        verify(stage)
        stage.rename(folder)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _consume(folder: Path, manifest: dict, output=None) -> None:
    complete, size = hashlib.sha256(), 0
    root = folder.resolve()
    for part in manifest['parts']:
        path = folder / part['name']
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError('Missing or unsafe part: ' + part['name'])
        if path.stat().st_size != part['size_bytes']:
            raise ValueError('Part size mismatch: ' + part['name'])
        part_hash = hashlib.sha256()
        with path.open('rb') as stream:
            for data in iter(lambda: stream.read(BLOCK), b''):
                part_hash.update(data); complete.update(data); size += len(data)
                if output is not None:
                    output.write(data)
        if part_hash.hexdigest() != part['sha256']:
            raise ValueError('Part SHA256 mismatch: ' + part['name'])
    if size != manifest['size_bytes'] or complete.hexdigest() != manifest['sha256']:
        raise ValueError('Reconstructed checkpoint SHA256 or size mismatch')


def verify(folder: Path) -> dict:
    manifest = read_manifest(folder)
    _consume(folder, manifest)
    return manifest


def restore(folder: Path, output: Path) -> dict:
    manifest = read_manifest(folder)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.restore-', dir=output.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            _consume(folder, manifest, stream)
            stream.flush(); os.fsync(stream.fileno())
        # Atomic publication without clobbering any existing destination.
        os.link(temp, output)
        return {'output': str(output), 'size_bytes': output.stat().st_size, 'sha256': digest(output)}
    finally:
        temp.unlink(missing_ok=True)


def git(*args: str, cwd: Path | None = None) -> str:
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    result = subprocess.run(['git', *args], cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900)
    if result.returncode:
        raise RuntimeError(f'git {args[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def publish(folder: Path, repo_url: str, branch: str, destination: str,
            author_name: str, author_email: str) -> dict:
    """Push binary parts, then independently clone and re-read all remote bytes."""
    manifest = verify(folder)
    if not re.fullmatch(r'[A-Za-z0-9._/-]+', destination) or any(p in ('', '.', '..', '.git') for p in destination.split('/')) or destination.startswith('/'):
        raise ValueError('Unsafe repository destination')
    if branch.startswith('-') or repo_url.startswith('-'):
        raise ValueError('Invalid branch or repository')
    with tempfile.TemporaryDirectory(prefix='flygraph-publish-') as temporary:
        root = Path(temporary); checkout = root / 'checkout'
        git('clone', '--depth', '1', '--single-branch', '--branch', branch, '--', repo_url, str(checkout))
        target = checkout / destination
        if target.is_symlink() or not target.resolve().is_relative_to(checkout.resolve()):
            raise ValueError('Repository destination escapes the checkout')
        if target.exists():
            existing = read_manifest(target)
            if existing != manifest:
                raise FileExistsError('Different checkpoint already occupies destination')
        else:
            target.mkdir(parents=True)
        for part in manifest['parts']:
            path = target / part['name']
            if path.is_symlink():
                raise ValueError('Symlink at destination')
            if path.exists():
                if path.stat().st_size != part['size_bytes'] or digest(path) != part['sha256']:
                    raise FileExistsError('Different bytes at destination: ' + part['name'])
            else:
                shutil.copyfile(folder / part['name'], path)
        shutil.copyfile(folder / 'MANIFEST.json', target / 'MANIFEST.json')
        # Prevent inherited LFS/text filters from converting or replacing actual parts.
        (target / '.gitattributes').write_text('*.part* -text -filter -diff -merge\n', encoding='utf-8')
        (target / '.gitignore').write_text(manifest['filename'] + '\n', encoding='utf-8')
        git('add', '-f', '--', destination, cwd=checkout)
        changed = git('diff', '--cached', '--name-only', cwd=checkout)
        if changed:
            git('-c', f'user.name={author_name}', '-c', f'user.email={author_email}',
                'commit', '-m', f"Store verified {manifest['identity']} weights in parts below 90 MB", cwd=checkout)
            git('push', 'origin', f'HEAD:refs/heads/{branch}', cwd=checkout)  # never force
        commit = git('rev-parse', 'HEAD', cwd=checkout)
        remote = root / 'readback'
        git('clone', '--depth', '1', '--single-branch', '--branch', branch, '--', repo_url, str(remote))
        fetched = verify(remote / destination)
        if fetched != manifest:
            raise ValueError('Remote checkpoint does not match local checkpoint')
        restored = restore(remote / destination, root / manifest['filename'])
        return {'status': 'remote_bytes_verified', 'repo': repo_url, 'branch': branch,
                'commit': commit, 'destination': destination, 'parts': len(manifest['parts']),
                'size_bytes': restored['size_bytes'], 'sha256': restored['sha256']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('split'); p.add_argument('source', type=Path); p.add_argument('folder', type=Path)
    p.add_argument('--part-bytes', type=int, default=DEFAULT_PART_BYTES); p.add_argument('--identity', default='')
    p.add_argument('--expected-sha256')
    p = commands.add_parser('verify'); p.add_argument('folder', type=Path)
    p = commands.add_parser('restore'); p.add_argument('folder', type=Path); p.add_argument('output', type=Path)
    p = commands.add_parser('publish'); p.add_argument('folder', type=Path)
    p.add_argument('--repo', default='https://github.com/binouv/model.git'); p.add_argument('--branch', default='main')
    p.add_argument('--destination', required=True); p.add_argument('--author-name', required=True); p.add_argument('--author-email', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'split':
            result = split(args.source, args.folder, args.part_bytes, args.identity, args.expected_sha256)
        elif args.command == 'restore':
            result = restore(args.folder, args.output)
        elif args.command == 'verify':
            result = verify(args.folder)
        else:
            result = publish(args.folder, args.repo, args.branch, args.destination, args.author_name, args.author_email)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')

if __name__ == '__main__':
    main()
