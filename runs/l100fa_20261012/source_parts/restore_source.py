#!/usr/bin/env python3
"""Restore byte-exact L100FA source/data/tests from normal binary Git parts."""
from pathlib import Path
import hashlib, json, tarfile, tempfile, shutil

def main():
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/"SOURCE_MANIFEST.json").read_text())
    whole=hashlib.sha256()
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(tmp)/manifest["filename"]
        with archive.open("wb") as out:
            for p in manifest["parts"]:
                data=(here/p["name"]).read_bytes()
                if len(data)!=p["bytes"] or hashlib.sha256(data).hexdigest()!=p["sha256"]:
                    raise ValueError("part corrupted "+p["name"])
                out.write(data);whole.update(data)
        if archive.stat().st_size!=manifest["bytes"] or whole.hexdigest()!=manifest["sha256"]:
            raise ValueError("whole archive corrupt")
        with tarfile.open(archive,"r:gz") as z:
            for m in z.getmembers():
                path=Path(m.name)
                if path.is_absolute() or ".." in path.parts or not m.isfile():
                    raise ValueError("untrusted tar entry")
            z.extractall(here,filter="data")
    print("Verified and restored",manifest["bytes"],"source bytes")

if __name__=="__main__": main()
