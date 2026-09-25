"""Offline exact data reconstruction. Complete SourceData bundles need no sibling."""
import hashlib,json,subprocess,sys,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    expect=json.loads((ROOT/'configs/expected_data.json').read_text())
    for n,h in expect['jsonl'].items():
        if sha(ROOT/'data'/n)!=h:raise ValueError('data hash mismatch: '+n)
    if sha(ROOT/'data/tokenizer.json')!=expect['tokenizer_sha256']:raise ValueError('tokenizer mismatch')
    return expect
def main():
    if all((ROOT/'data'/n).exists() for n in json.loads((ROOT/'configs/expected_data.json').read_text())['jsonl']):
        print(json.dumps({'status':'verified_existing','hashes':verify()},indent=2));return
    sibling=ROOT.parent/'g2r_310m_sft'
    if not (sibling/'src/setup_data.py').exists():raise FileNotFoundError('Need complete SourceData bundle or the repo g2r_310m_sft sibling')
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable,str(sibling/'src/setup_data.py'),'--output',td],check=True)
        dest=ROOT/'data';dest.mkdir(parents=True,exist_ok=True)
        for p in Path(td).glob('test_*.jsonl'):shutil.copy2(p,dest/p.name)
        shutil.copy2(Path(td)/'tokenizer.json',dest/'tokenizer.json')
        subprocess.run([sys.executable,str(ROOT/'src/build_data.py')],check=True)
    print(json.dumps({'status':'regenerated_exactly','hashes':verify()},indent=2))
if __name__=='__main__':main()
