"""Untrained next-curriculum prototype: verifiable RU/EN mutable-memory traces.

Only prompt is an inference input. case/answer/trace are dataset supervision.
Not an external benchmark, not broad language pretraining, not used by G1.
No external teacher is called. Supplied teacher answers can be checked by replay.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, random
from pathlib import Path


def interleave_preserving_key_order(writes, rng):
    groups = collections.defaultdict(list)
    for row in writes:
        groups[(row['subject'], row['relation'])].append(dict(row))
    out = []
    while groups:
        key = rng.choice(list(groups))
        out.append(groups[key].pop(0))
        if not groups[key]:
            del groups[key]
    return out


def replay(case):
    memory = {}
    for row in case['writes']:
        memory[(row['subject'], row['relation'])] = row['object']
    entity = case['start']; trace = []
    for relation in case['program']:
        nxt = memory.get((entity, relation))
        trace.append({'subject': entity, 'relation': relation, 'object': nxt})
        if nxt is None:
            return 'UNKNOWN', trace
        entity = nxt
    return entity, trace


def validate_prediction(case, prediction):
    """Exact verifier; no trust in teacher self-evaluation or executed code."""
    answer, trace = replay(case)
    return prediction.get('answer') == answer and prediction.get('trace') == trace


def make_example(seed, family='lookup', hops=4):
    rng = random.Random(seed)
    lang = rng.choice(['ru', 'en'])  # independent of task and example parity
    tag = hashlib.sha256(str(seed).encode()).hexdigest()[:9]
    names = [f'n{tag}_{i}' for i in range(max(16, hops+8))]
    values = [f'v{tag}_{i}' for i in range(50)]
    writes = []
    if family == 'lookup':
        keys = rng.sample(names, 8)
        for i, key in enumerate(keys):
            writes.append({'subject': key, 'relation': 'value', 'object': values[i]})
        for i in range(8, rng.randrange(16, 33)):
            writes.append({'subject': rng.choice(keys), 'relation': 'value', 'object': values[i]})
        start = rng.choice(keys + [f'absent_{tag}'])
        program = ['value']
    elif family == 'chain':
        path = rng.sample(names, hops+1)
        program = [rng.choice(['r0','r1','r2']) for _ in range(hops)]
        protected = {(path[j], program[j]) for j in range(hops)}
        for _ in range(16):
            s, rel = rng.choice(names), rng.choice(['r0','r1','r2'])
            if (s, rel) not in protected:
                writes.append({'subject': s, 'relation': rel, 'object': rng.choice(names)})
        for j, rel in enumerate(program):
            if rng.random() < .6:
                writes.append({'subject': path[j], 'relation': rel, 'object': rng.choice(names)})
            writes.append({'subject': path[j], 'relation': rel, 'object': path[j+1]})
        start = path[0]
    else:
        raise ValueError('family must be lookup or chain')
    writes = interleave_preserving_key_order(writes, rng)
    case = {'writes': writes, 'start': start, 'program': program}
    answer, trace = replay(case)
    facts = '\n'.join(f"{i}: {w['subject']}.{w['relation']} = {w['object']}" for i,w in enumerate(writes))
    if lang == 'ru':
        prompt = f'Записи идут по времени. Используй последнюю запись каждого ключа.\n{facts}\nНачало: {start}. Программа отношений: {", ".join(program)}.\nВерни JSON с answer и trace; при отсутствии записи answer=UNKNOWN.\n'
    else:
        prompt = f'Records are chronological. Use the latest write for each key.\n{facts}\nStart: {start}. Relation program: {", ".join(program)}.\nReturn JSON with answer and trace; use answer=UNKNOWN for a missing record.\n'
    return {'id': hashlib.sha256(prompt.encode()).hexdigest(), 'seed': seed, 'family': family, 'lang': lang, 'case': case, 'prompt': prompt, 'answer': answer, 'trace': trace,
            'completion': json.dumps({'answer': answer, 'trace': trace},ensure_ascii=False)}


def audit(n=1000):
    rows=[]; invariants=0; rejection=0; last_hits=0; lookup_n=0
    for i in range(n):
        z=make_example(720000+i,'lookup' if i%2==0 else 'chain',2+(i%7))
        assert validate_prediction(z['case'], z)
        altered={'writes':interleave_preserving_key_order(z['case']['writes'],random.Random(99+i)), 'start':z['case']['start'],'program':z['case']['program']}
        assert replay(altered)==replay(z['case']);invariants+=1
        assert not validate_prediction(z['case'],{'answer':'INTENTIONALLY_WRONG','trace':z['trace']});rejection+=1
        if z['family']=='lookup':
            lookup_n+=1;last_hits+=int(z['answer']==z['case']['writes'][-1]['object'])
        rows.append(z)
    assert len({z['id'] for z in rows})==n
    return {'status':'completed_engineering_data_audit','n':n,'semantic_replay_valid':n,'storage_interleaving_invariant':invariants,'wrong_answer_rejected':rejection,'lookup_last_global_value_shortcut_accuracy':last_hits/lookup_n,'language_counts':dict(collections.Counter((z['family']+':'+z['lang']) for z in rows)), 'used_for_G1_training':False,'external_teacher_called':False,'not_a_model_benchmark':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit-out',required=True);args=ap.parse_args()
    result=audit();Path(args.audit_out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
