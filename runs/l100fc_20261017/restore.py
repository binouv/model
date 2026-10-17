"""Recover registered L100FC source/tests/train/held data; byte-check compressed Git blob."""
from __future__ import annotations
import hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'archives/L100FC_src_data_pretraining.tar.xz'
EXPECTED='ba760aa62b4478f8d85c6848cb0b0a7599642cdd20e7f22ad695b34395981a61'
def main():
    assert ARCHIVE.is_file(),'Missing archived source bytes'
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==EXPECTED,'Source checksum failed'
    with tarfile.open(ARCHIVE,'r:xz') as tar:
        for m in tar.getmembers():
            n=Path(m.name)
            if n.is_absolute() or '..' in n.parts or not (m.isfile() or m.isdir()) or m.issym() or m.islnk():
                raise RuntimeError('Unsafe archived path '+m.name)
        tar.extractall(ROOT,filter='data')
    print('Restored verified registration, source, tests, train, held and schedule into',ROOT)
if __name__=='__main__':main()
