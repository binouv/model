"""One-shot encrypted artifact handoff. Never store bearer URLs or private keys in Git."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request

CHUNK = 320
PREFIX = 'runs/artifact_transfer_20261012'
DEST = 'models/l100ec_curriculum_bf16_07e33eb2'
MODEL_SHA = '07e33eb2c6aab5727fbb72c0b403bc83ca8e5695d379106af089b27baf995b25'
OPTIONS = ['-pkeyopt', 'rsa_padding_mode:oaep', '-pkeyopt', 'rsa_oaep_md:sha256', '-pkeyopt', 'rsa_mgf1_md:sha256']


def openssl(args: list[str], data: bytes | None = None) -> bytes:
    p = subprocess.run(['openssl', *args], input=data, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    if p.returncode:
        raise ValueError('OpenSSL operation failed')
    return p.stdout


def make_key(private: Path) -> dict:
    openssl(['genpkey', '-algorithm', 'RSA', '-pkeyopt', 'rsa_keygen_bits:4096', '-out', str(private)])
    private.chmod(0o600)
    public = openssl(['pkey', '-in', str(private), '-pubout'])
    return {'transfer_id': secrets.token_hex(16), 'public_key_pem': public.decode('ascii'),
            'public_key_sha256': hashlib.sha256(public).hexdigest(), 'checkpoint_sha256': MODEL_SHA}


def encrypt(public: dict, payload: dict) -> dict:
    raw = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    if len(raw) > 16000 or payload.get('transfer_id') != public['transfer_id']:
        raise ValueError('Invalid payload size or transfer identity')
    public_bytes = public['public_key_pem'].encode('ascii')
    if hashlib.sha256(public_bytes).hexdigest() != public['public_key_sha256']:
        raise ValueError('Public key fingerprint mismatch')
    nonce = bytes.fromhex(public['transfer_id'])
    if len(nonce) != 16:
        raise ValueError('Invalid transfer identity')
    count = (len(raw) + CHUNK - 1) // CHUNK
    full = hashlib.sha256(raw).digest()
    blocks = []
    with tempfile.TemporaryDirectory() as folder:
        key = Path(folder) / 'public.pem'; key.write_bytes(public_bytes)
        for index in range(count):
            block = nonce + struct.pack('>II', index, count) + full + raw[index * CHUNK:(index + 1) * CHUNK]
            cipher = openssl(['pkeyutl', '-encrypt', '-pubin', '-inkey', str(key), *OPTIONS], block)
            if len(cipher) != 512:
                raise ValueError('Expected a 4096-bit RSA key')
            blocks.append(base64.b64encode(cipher).decode('ascii'))
    return {'format': 'flygraph.rsa-oaep-sha256.v1', 'transfer_id': public['transfer_id'],
            'public_key_sha256': public['public_key_sha256'], 'blocks': blocks}


def decrypt(private: Path, public: dict, envelope: dict) -> dict:
    if (envelope.get('format') != 'flygraph.rsa-oaep-sha256.v1'
            or envelope.get('transfer_id') != public['transfer_id']
            or envelope.get('public_key_sha256') != public['public_key_sha256']):
        raise ValueError('Envelope identity mismatch')
    blocks = envelope.get('blocks')
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 50:
        raise ValueError('Invalid block count')
    data = bytearray(); expected_hash = None
    for index, encoded in enumerate(blocks):
        if not isinstance(encoded, str) or len(encoded) > 700:
            raise ValueError('Invalid ciphertext')
        cipher = base64.b64decode(encoded, validate=True)
        if len(cipher) != 512:
            raise ValueError('Invalid ciphertext size')
        block = openssl(['pkeyutl', '-decrypt', '-inkey', str(private), *OPTIONS], cipher)
        if len(block) <= 56 or block[:16] != bytes.fromhex(public['transfer_id']):
            raise ValueError('Invalid block identity')
        if struct.unpack('>II', block[16:24]) != (index, len(blocks)):
            raise ValueError('Block order mismatch')
        if expected_hash is None:
            expected_hash = block[24:56]
        if expected_hash != block[24:56]:
            raise ValueError('Mixed payload blocks')
        data.extend(block[56:])
    if len(data) > 16000 or hashlib.sha256(data).digest() != expected_hash:
        raise ValueError('Payload integrity mismatch')
    value = json.loads(data)
    if not isinstance(value, dict) or value.get('transfer_id') != public['transfer_id']:
        raise ValueError('Decrypted identity mismatch')
    return value


def allowed_url(url: str) -> bool:
    try:
        p = urllib.parse.urlsplit(url)
        return (p.scheme == 'https' and bool(p.hostname) and p.hostname.endswith('.oaiusercontent.com')
                and p.port in (None, 443) and p.username is None and p.password is None)
    except (ValueError, AttributeError):
        return False


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError('Redirect outside the artifact host')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download(url: str, path: Path, expected: dict) -> None:
    if not allowed_url(url) or not 0 < expected['size_bytes'] < 90_000_000:
        raise ValueError('Invalid download destination or size')
    if path.exists():
        raise FileExistsError('Download destination exists')
    try:
        opener = urllib.request.build_opener(SafeRedirect())
        size, digest = 0, hashlib.sha256()
        with opener.open(url, timeout=60) as response, path.open('xb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > expected['size_bytes']:
                    raise ValueError('Download exceeds expected size')
                digest.update(chunk); output.write(chunk)
        if size != expected['size_bytes'] or digest.hexdigest() != expected['sha256']:
            raise ValueError('Downloaded bytes mismatch')
    except Exception:
        path.unlink(missing_ok=True)
        raise RuntimeError('Artifact download or checksum validation failed') from None


def run() -> None:
    if os.environ.get('GITHUB_REPOSITORY') != 'binouv/model':
        raise ValueError('This transfer is restricted to binouv/model')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'runs/artifact_recovery_20261012'))
    import checkpoint_parts as cp
    os.chdir(root)
    token = os.environ.get('GH_TOKEN')
    if not token:
        raise ValueError('Missing workflow token')
    os.environ['GIT_CONFIG_COUNT'] = '1'
    os.environ['GIT_CONFIG_KEY_0'] = 'http.https://github.com/.extraheader'
    os.environ['GIT_CONFIG_VALUE_0'] = 'AUTHORIZATION: basic ' + base64.b64encode(('x-access-token:' + token).encode()).decode()
    cp.git('config', 'user.name', 'github-actions[bot]')
    cp.git('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    request = json.loads((root / PREFIX / 'REQUEST.json').read_text())
    if request != {'checkpoint_sha256': MODEL_SHA, 'operation': 'publish_existing_bytes_only', 'new_optimizer_steps': 0}:
        raise ValueError('Unexpected request')
    expected = cp.read_manifest(root / DEST)
    if expected['sha256'] != MODEL_SHA:
        raise ValueError('Unexpected checkpoint')
    if (root / DEST / 'GITHUB_BINARY_RECEIPT.json').exists():
        cp.verify(root / DEST)
        print('Previously published checkpoint is present; no transfer repeated.')
        return

    def commit(paths: list[str], message: str) -> str:
        cp.git('add', '-f', '--', *paths)
        if cp.git('diff', '--cached', '--name-only'):
            cp.git('commit', '-m', message + ' [skip ci]')
            for attempt in range(3):
                try:
                    cp.git('push', 'origin', 'HEAD:refs/heads/main')
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    cp.git('pull', '--rebase', 'origin', 'main')
        return cp.git('rev-parse', 'HEAD')

    with tempfile.TemporaryDirectory(prefix='flygraph-private-transfer-') as temporary:
        temp = Path(temporary)
        private = temp / 'private.pem'
        public = make_key(private)
        public.update(workflow_run_id=os.environ.get('GITHUB_RUN_ID'), request_commit=os.environ.get('GITHUB_SHA'))
        cp.git('pull', '--ff-only', 'origin', 'main')
        (root / PREFIX / 'PUBLIC_KEY.json').write_text(json.dumps(public, indent=2) + '\n')
        commit([PREFIX + '/PUBLIC_KEY.json'], 'Publish one-shot artifact transfer public key')
        print('Public key published; awaiting encrypted handoff.', flush=True)
        deadline = time.monotonic() + 600
        envelope = None
        while time.monotonic() < deadline:
            cp.git('fetch', '--depth', '1', 'origin', 'main')
            try:
                cp.git('cat-file', '-e', 'FETCH_HEAD:' + PREFIX + '/ABORT.json')
            except RuntimeError:
                pass
            else:
                raise RuntimeError('Transfer aborted')
            try:
                text = cp.git('show', 'FETCH_HEAD:' + PREFIX + '/ENCRYPTED_URLS.json')
                if len(text) > 50000:
                    raise ValueError('Oversized encrypted envelope')
                candidate = json.loads(text)
                if candidate.get('transfer_id') == public['transfer_id']:
                    envelope = candidate; break
            except RuntimeError:
                pass
            time.sleep(5)
        if envelope is None:
            raise RuntimeError('No encrypted handoff received within this finite run')
        payload = decrypt(private, public, envelope)
        if payload.get('checkpoint_sha256') != MODEL_SHA:
            raise ValueError('Payload checkpoint mismatch')
        files = payload.get('files')
        if not isinstance(files, list) or len(files) != len(expected['parts']):
            raise ValueError('Unexpected files')
        by_name = {row['name']: row for row in files}
        if set(by_name) != {row['name'] for row in expected['parts']}:
            raise ValueError('Unexpected file names')
        staged = temp / 'parts'; staged.mkdir()
        (staged / 'MANIFEST.json').write_text(json.dumps(expected, indent=2) + '\n')
        for part in expected['parts']:
            row = by_name[part['name']]
            if row['sha256'] != part['sha256'] or row['size_bytes'] != part['size_bytes']:
                raise ValueError('Part metadata mismatch')
            download(row['url'], staged / part['name'], part)
            print('Verified ' + part['name'], flush=True)
        del payload, by_name, files, row
        cp.verify(staged)
        cp.git('pull', '--ff-only', 'origin', 'main')
        target = root / DEST
        for part in expected['parts']:
            path = target / part['name']
            if path.exists():
                if cp.digest(path) != part['sha256']:
                    raise ValueError('Refusing to replace different existing weights')
            else:
                shutil.copyfile(staged / part['name'], path)
        (target / '.gitattributes').write_text('*.part* -text -filter -diff -merge\n')
        binary_commit = commit([DEST + '/' + p['name'] for p in expected['parts']] + [DEST + '/.gitattributes'],
                               'Store real L100EC BF16 checkpoint as 80/80/40MB Git blobs')
        readback = temp / 'fresh-clone'
        cp.git('clone', '--depth', '1', '--single-branch', '--branch', 'main',
               'https://github.com/binouv/model.git', str(readback))
        cp.verify(readback / DEST)
        restored = cp.restore(readback / DEST, temp / 'readback.safetensors')
        receipt = {'status': 'github_binary_roundtrip_verified', 'binary_commit': binary_commit,
                   'checkpoint_identity': 'L100EC_curriculum_100M', 'size_bytes': restored['size_bytes'],
                   'sha256': restored['sha256'], 'parts': expected['parts'],
                   'fresh_remote_clone_verified': True, 'new_optimizer_steps': 0,
                   'l100ql_checkpoint_recovered': False, 'workflow_run_id': os.environ.get('GITHUB_RUN_ID')}
        (target / 'GITHUB_BINARY_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (target / 'README.md').write_text(
            '# L100EC curriculum BF16 — binary weights published\n\n'
            'Three real binary Git blobs are present: 80,000,000 + 80,000,000 + 40,069,384 bytes. '
            'An independent HTTPS clone and complete reconstruction matched SHA256 `' + MODEL_SHA + '`.\n\n'
            'This is the inherited L100EC checkpoint, not L100QL; no new training was performed. '
            'Original optimizer/RNG were not recovered. See GITHUB_BINARY_RECEIPT.json for the binary commit and verification.\n\n'
            'From repository root:\n\n```sh\npython runs/artifact_recovery_20261012/checkpoint_parts.py restore '
            + DEST + ' model.safetensors\n```\n\n'
            'The recovery toolkit and inference runtime are also available in the conversation/private backup. '
            'Earlier pending-publication notes are historical, superseded by this receipt.\n')
        final_commit = commit([DEST + '/GITHUB_BINARY_RECEIPT.json', DEST + '/README.md'],
                              'Verify GitHub binary checkpoint by fresh clone and exact SHA256 reconstruction')
        print(json.dumps({'status': receipt['status'], 'commit': final_commit, 'sha256': MODEL_SHA}), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        # Do not let HTTP exception contexts leak scoped artifact bearer URLs.
        print('Transfer failed: ' + type(error).__name__ + '. No success is claimed.', file=sys.stderr)
        raise SystemExit(1) from None
