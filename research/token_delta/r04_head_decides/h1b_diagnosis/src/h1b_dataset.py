"""Fixed fresh H1b diagnostic cases, disjoint from ALL original H1 examples.

This does NOT re-open or replace the preregistered 300 H1 test outcomes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

H1_SRC = Path(__file__).resolve().parents[2] / "h1" / "src"
sys.path.insert(0, str(H1_SRC))
from h1_dataset import (  # noqa: E402
    build_dataset, question, solve, python_code, verify_python,
    dataset_sha256 as h1_sha, OPERATIONS, DOMAINS, LANGUAGES
)

SEED = 20261013
COUNT_PER_BUCKET = 32
BASE_SEED = 20261012


def new_cases(seed: int = SEED) -> list[dict]:
    old = build_dataset(BASE_SEED)
    old_keys = {r["group"] for r in old}
    all_keys = [(op,a,b,c) for op in OPERATIONS for a in range(20)
                for b in range(20) for c in range(20)
                if f"{op}:{a}:{b}:{c}" not in old_keys]
    rng = random.Random(seed)
    rng.shuffle(all_keys)
    n = COUNT_PER_BUCKET * len(DOMAINS) * len(LANGUAGES)
    result = []
    for domain in DOMAINS:
        for language in LANGUAGES:
            for _ in range(COUNT_PER_BUCKET):
                op,a,b,c = all_keys[len(result)]
                p,q = solve(op,a,b,c)
                if verify_python(op,a,b,c) != (p,q):
                    raise AssertionError("Oracle / interpreter mismatch")
                result.append({
                    "id":f"h1b-{len(result):03d}",
                    "domain":domain,"language":language,
                    "operator":op,"operands":[a,b,c],
                    "group":f"{op}:{a}:{b}:{c}",
                    "question":question(domain,language,op,a,b,c),
                    "answer":f"{p} {q}","digits":[p,q],
                })
    assert len(result)==n==128
    assert len({r["group"] for r in result})==128
    assert not {r["group"] for r in result} & old_keys
    return result


def one_step_question(case:dict) -> str:
    a,b,c = case["operands"]
    op=case["operator"]
    if case["domain"]=="arithmetic":
        expr = f"({a} + {b}) % 10" if op=="add" else f"(2 * {a} + {b}) % 10"
        if case["language"]=="en":
            return f"Calculate p = {expr}. Output only the single decimal digit p, no explanation."
        return f"Вычисли p = {expr}. Выведи только одну цифру p, без объяснения."
    script = python_code(op,a,b,c)
    lines=script.splitlines()
    code="\n".join(lines[:4] + ["print(x)"])
    if case["language"]=="en":
        return ("What single decimal digit does this Python 3 program print? "
                "Answer using only the digit, no explanation.\n```python\n"+code+"\n```")
    return ("Какую одну цифру печатает эта программа Python 3? Только цифра, без пояснений.\n"
            "```python\n"+code+"\n```")


def sha_json(data:list[dict])->str:
    source=json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def save_dataset(out:Path, seed:int=SEED)->dict:
    examples=new_cases(seed)
    out.mkdir(parents=True,exist_ok=True)
    path=out/"h1b_cases.jsonl"
    with path.open("w",encoding="utf-8",newline="\n") as writer:
        for e in examples:writer.write(json.dumps(e,sort_keys=True,ensure_ascii=False)+"\n")
    manifest={
        "status":"new_diagnostic_data_not_model_results",
        "seed":seed,"n":len(examples),"per_bucket":COUNT_PER_BUCKET,
        "old_dataset_sha256":h1_sha(build_dataset(BASE_SEED)),
        "new_dataset_sha256":sha_json(examples),
        "file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "not_part_of_original_h1_scientific_test":True,
    }
    (out/"h1b_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return manifest


def load_dataset(out:Path)->tuple[list[dict],dict]:
    manifest=json.loads((out/"h1b_manifest.json").read_text(encoding="utf-8"))
    path=out/"h1b_cases.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["file_sha256"]:
        raise RuntimeError("H1b dataset file hash mismatch")
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    if sha_json(rows)!=manifest["new_dataset_sha256"] or len(rows)!=128:
        raise RuntimeError("H1b dataset content changed")
    assert not {r["group"] for r in rows} & {r["group"] for r in build_dataset(BASE_SEED)}
    return rows,manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1]/"data")
    args=parser.parse_args()
    print(json.dumps(save_dataset(args.out),indent=2,ensure_ascii=False))

if __name__=="__main__":main()