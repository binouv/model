"""Apply three fixed corrections, requiring exact known source hashes."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
PATCHES = {
    'transport.py': (
        '806e47d897bca9d224697b054137e851ff70587a73abac25784662c68ec24325',
        '4ce47c68bece9bab40d2409af7e6143ffa289c0a7d2541c01a74240271695b8e',
        [("commit([PREFIX + '/PUBLIC_KEY.json'], 'Publish one-shot artifact transfer public key')", "commit([PREFIX + '/PUBLIC_KEY.json', PREFIX + '/transport.py', PREFIX + '/test_transport.py'], 'Publish one-shot public key and byte-verified expanded transport source')"),
         ("cp.git('fetch', '--depth', '1', 'origin', 'main')", "cp.git('fetch', 'origin', 'main')")]),
    'test_transport.py': (
        'd3a406b5ef7d744e7f07e463c4bdf658090df575ccb339b2d83a83ab1503eb0c',
        '8f0ede553753a1415466495c68042eb55e65a8ef50e74bea2cb32c13b5176fee',
        [("c['blocks'][0]='A'+c['blocks'][0][1:]", "c['blocks'][0]=('B' if c['blocks'][0][0]=='A' else 'A')+c['blocks'][0][1:]")])
}

def main():
    outputs = []
    for name, (before, after, replacements) in PATCHES.items():
        path = ROOT / name
        raw = path.read_bytes()
        current = hashlib.sha256(raw).hexdigest()
        if current == after:
            continue
        if current != before:
            raise ValueError('Unexpected source bytes: ' + name)
        text = raw.decode('utf-8')
        for old, new in replacements:
            if text.count(old) != 1:
                raise ValueError('Non-unique correction: ' + name)
            text = text.replace(old, new)
        raw = text.encode('utf-8')
        if hashlib.sha256(raw).hexdigest() != after:
            raise ValueError('Corrected source hash mismatch: ' + name)
        outputs.append((path, raw))
    for path, raw in outputs:
        path.write_bytes(raw)
    print('Transport sources match the locally tested SHA256 hashes.')

if __name__ == '__main__':
    main()
