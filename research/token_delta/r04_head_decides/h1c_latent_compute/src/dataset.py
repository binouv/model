"""Fresh locked R04-H1c dataset with no tuple overlap with H1/H1b."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"h1"/"src"))
sys.path.insert(0,str(ROOT/"h1b_diagnosis"/"src"))
from h1_dataset import build_dataset,question,solve,DOMAINS,LANGUAGES,OPERATIONS
from h1b_dataset import new_cases

SEED=20261014
PER_STRATUM={"train":300,"val":50,"test":75}
DATASET_COUNT=1700

def canonical_sha(records:list[dict])->str:
    payload=json.dumps(records,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def build(seed:int=SEED)->list[dict]:
    old={x["group"] for x in build_dataset(20261012)}
    h1b={x["group"] for x in new_cases(20261013)}
    exclude=old|h1b
    keys=[(op,a,b,c) for op in OPERATIONS for a,b,c in itertools.product(range(20),repeat=3)
          if f"{op}:{a}:{b}:{c}" not in exclude]
    random.Random(seed).shuffle(keys)
    assert len(keys)>=DATASET_COUNT
    result=[]
    for domain in DOMAINS:
        for language in LANGUAGES:
            for split,n in PER_STRATUM.items():
                for _ in range(n):
                    op,a,b,c=keys[len(result)]
                    p,q=solve(op,a,b,c)
                    result.append({
                        "id":f"h1c-{len(result):04d}","split":split,
                        "domain":domain,"language":language,
                        "operator":op,"operands":[a,b,c],
                        "group":f"{op}:{a}:{b}:{c}",
                        "question":question(domain,language,op,a,b,c),
                        "digits":[p,q],"answer":f"{p} {q}"})
    assert len(result)==DATASET_COUNT
    assert len({r["group"] for r in result})==DATASET_COUNT
    assert not {r["group"] for r in result}&exclude
    return result

def save(out:Path,seed:int=SEED)->dict:
    out.mkdir(parents=True,exist_ok=True)
    data=build(seed)
    path=out/"h1c_dataset.jsonl"
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in data:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    manifest={"status":"oracle_data_only_no_model_results","seed":seed,
              "count":len(data),
              "splits":{s:sum(r["split"]==s for r in data) for s in PER_STRATUM},
              "dataset_sha256_canonical":canonical_sha(data),
              "file_sha256":file_sha(path),
              "excluded_splits":["all H1 1700","all H1b 128"],
              "teacher_weights_revision":"c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97"}
    (out/"dataset_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return manifest

def load(out:Path)->tuple[list[dict],dict]:
    mf=json.loads((out/"dataset_manifest.json").read_text(encoding="utf-8"))
    p=out/"h1c_dataset.jsonl"
    if file_sha(p)!=mf["file_sha256"]:
        raise RuntimeError("H1c dataset bytes have changed")
    items=[json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
    if len(items)!=DATASET_COUNT or canonical_sha(items)!=mf["dataset_sha256_canonical"]:
        raise RuntimeError("H1c canonical data do not match manifest")
    if any(sum(item["split"]==s for item in items)!=n*4 for s,n in PER_STRATUM.items()):
        raise RuntimeError("H1c train/validation/test split changed")
    forbidden={r["group"] for r in build_dataset(20261012)}|{r["group"] for r in new_cases(20261013)}
    if {r["group"] for r in items}&forbidden:
        raise RuntimeError("H1c leakage from H1 or H1b")
    return items,mf

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    args=parser.parse_args()
    print(json.dumps(save(args.out),indent=2))

if __name__=="__main__":main()
