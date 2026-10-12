"""R04-H2: fresh leak-resistant two-digit arithmetic and code-output benchmark.

Everything is generated deterministically. No old R04 H1/H1b/H1c heldout
problem key may reappear in any H2 split. Frozen source seeds are preregistered.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

SEED = 20261016
OPERATIONS = ('add', 'weighted')
DOMAINS = ('arithmetic', 'code')
LANGUAGES = ('en', 'ru')
PER_BUCKET = {'train': 300, 'val': 50, 'test': 75}
COUNT = 1700


def all_keys():
    return list(itertools.product(OPERATIONS, range(20), range(20), range(20)))


def excluded_keys():
    keys = all_keys()
    h1 = keys.copy()
    random.Random(20261012).shuffle(h1)
    old = set(h1[:1700])

    h1b = [x for x in keys if x not in old]
    random.Random(20261013).shuffle(h1b)
    new_128 = set(h1b[:128])

    h1c = [x for x in keys if x not in old and x not in new_128]
    random.Random(20261014).shuffle(h1c)
    new_1700 = set(h1c[:1700])
    assert len(old) == 1700 and len(new_128) == 128 and len(new_1700) == 1700
    assert not (old & new_128 or old & new_1700 or new_128 & new_1700)
    return old | new_128 | new_1700


def oracle(op: str, a: int, b: int, c: int):
    if op == 'add':
        p = (a + b) % 10
        q = (p + c) % 10
    elif op == 'weighted':
        p = (2*a + b) % 10
        q = (p + 2*c) % 10
    else:
        raise ValueError(op)
    return p, q


def source_code(op: str, a: int, b: int, c: int):
    if op not in OPERATIONS:
        raise ValueError(op)
    xline = 'x = (x + y) % 10' if op == 'add' else 'x = (2 * x + y) % 10'
    zline = 'z = (x + z) % 10' if op == 'add' else 'z = (x + 2 * z) % 10'
    return f'x = {a}\ny = {b}\nz = {c}\n{xline}\n{zline}\nprint(x, z)'


def question(domain: str, language: str, op: str, a: int, b: int, c: int):
    pexpr = f'({a} + {b}) % 10' if op == 'add' else f'(2 * {a} + {b}) % 10'
    qexpr = f'(p + {c}) % 10' if op == 'add' else f'(p + 2 * {c}) % 10'
    if domain == 'arithmetic':
        if language == 'en':
            return ('Solve the two-stage arithmetic problem. % is remainder and q uses computed p.\n'
                    f'p = {pexpr}\nq = {qexpr}\n'
                    'Output only the two final decimal DIGITS p and q separated by one space. No explanation.')
        if language == 'ru':
            return ('Выполни двухшаговый расчёт. % — остаток от деления, q использует вычисленное p.\n'
                    f'p = {pexpr}\nq = {qexpr}\n'
                    'Выведи только две итоговые ЦИФРЫ p и q через один пробел. Без пояснений.')
    if domain == 'code':
        code = source_code(op, a, b, c)
        if language == 'en':
            return ('What does this Python 3 program print? Output only two digits separated by '
                    'one space and no explanation.\n```python\n'+code+'\n```')
        if language == 'ru':
            return ('Что напечатает программа Python 3? Выведи только две цифры через '
                    'один пробел без пояснений.\n```python\n'+code+'\n```')
    raise ValueError((domain, language))


def build(seed: int = SEED):
    prior = excluded_keys()
    candidates = [x for x in all_keys() if x not in prior]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < COUNT:
        raise RuntimeError('Exclusion pools exhausted')
    rows = []
    cursor = 0
    for domain in DOMAINS:
        for language in LANGUAGES:
            for split, n in PER_BUCKET.items():
                for _ in range(n):
                    op, a, b, c = candidates[cursor]
                    cursor += 1
                    p, q = oracle(op, a, b, c)
                    rows.append({
                        'id': f'h2-{len(rows):04d}', 'split': split,
                        'domain': domain, 'language': language,
                        'operator': op, 'operands': [a,b,c],
                        'group': f'{op}:{a}:{b}:{c}',
                        'digits': [p, q], 'answer': f'{p} {q}',
                        'question': question(domain, language, op, a, b, c),
                    })
    assert len(rows) == COUNT
    assert len({x['group'] for x in rows}) == COUNT
    assert all(tuple([r['operator'],*r['operands']]) not in prior for r in rows)
    return rows


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for blk in iter(lambda:stream.read(2**20),b''):
            h.update(blk)
    return h.hexdigest()


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True,
             separators=(',',':')).encode('utf-8')).hexdigest()


def save(out: Path, seed: int = SEED):
    out.mkdir(parents=True, exist_ok=True)
    data = build(seed)
    path=out/'h2_dataset.jsonl'
    with path.open('w',encoding='utf-8',newline='\n') as stream:
        for item in data:
            stream.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n')
    manifest={
        'status':'dataset_generated_only_no_scientific_results',
        'seed':seed,'count':len(data),
        'splits':{s:sum(x['split']==s for x in data) for s in PER_BUCKET},
        'canonical_sha256':canonical_sha(data),
        'jsonl_sha256':sha256_file(path),
        'all_previous_groups_excluded':True,
        'excluded_from':['H1 1700','H1b 128','H1c 1700'],
        'teacher_revision':'c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97',
        'writer_revision':'23eda19ab0239bc12350efba411911b8eb89f486',
    }
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def load(out:Path):
    manifest=json.loads((out/'dataset_manifest.json').read_text(encoding='utf-8'))
    fp=out/'h2_dataset.jsonl'
    if sha256_file(fp)!=manifest['jsonl_sha256']:
        raise RuntimeError('H2 raw dataset SHA mismatch')
    rows=[json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines()]
    if len(rows)!=COUNT or canonical_sha(rows)!=manifest['canonical_sha256']:
        raise RuntimeError('H2 canonical dataset mismatch')
    if {s:sum(x['split']==s for x in rows) for s in PER_BUCKET}!=manifest['splits']:
        raise RuntimeError('H2 split mismatch')
    prior=excluded_keys()
    if any(tuple([r['operator'],*r['operands']]) in prior for r in rows):
        raise RuntimeError('H2 leaked prior H1/H1b/H1c examples')
    return rows,manifest


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    a=p.parse_args()
    print(json.dumps(save(a.out),indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
