"""CPU audit of original H1 test outputs, without re-scoring registered H1 gates.

Reads the user's authentic 300 teacher output lines and the H1 SHA-verified
original dataset. Reports strict valid/invalid and per-position correctness;
never updates the H1 metrics nor changes test predictions.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import sys

H1SRC=Path(__file__).resolve().parents[2]/'h1'/'src'
sys.path.insert(0,str(H1SRC))
from h1_dataset import load_dataset,strict_parse_answer  # noqa: E402


def audit(predictions:list[dict],dataset:list[dict])->dict:
    test=[x for x in dataset if x['split']=='test']
    if len(test)!=300 or len(predictions)!=300:
        raise ValueError('H1 audit requires all 300 original teacher predictions')
    by_id={x['id']:x for x in test}
    if len({r['id'] for r in predictions})!=300 or set(by_id)!={r['id'] for r in predictions}:
        raise ValueError('H1 prediction IDs do not match original test')
    records=[]
    for record in predictions:
        reference=by_id[record['id']]
        strict=strict_parse_answer(record['raw'])
        if record['oracle']!=reference['digits'] or record['exact']!=(strict==reference['digits']):
            raise ValueError('Original H1 exact/label inconsistent with dataset or parser')
        p_correct=bool(strict is not None and strict[0]==reference['digits'][0])
        q_correct=bool(strict is not None and strict[1]==reference['digits'][1])
        records.append({'id':record['id'],'domain':reference['domain'],
              'language':reference['language'],'operator':reference['operator'],
              'exact':record['exact'],'valid':strict is not None,
              'p_correct':p_correct,'q_correct':q_correct,
              'raw':record['raw']})
    def group(rows):
        return {
          'n':len(rows), 'strict_exact_count':sum(x['exact'] for x in rows),
          'strict_accuracy':sum(x['exact'] for x in rows)/len(rows),
          'strict_valid_rate':sum(x['valid'] for x in rows)/len(rows),
          'first_digit_accuracy_among_all':sum(x['p_correct'] for x in rows)/len(rows),
          'second_digit_accuracy_among_all':sum(x['q_correct'] for x in rows)/len(rows),
          'valid_but_wrong':sum(x['valid'] and not x['exact'] for x in rows),
          'invalid_format':sum(not x['valid'] for x in rows),
        }
    summary={'status':'audit_only_original_h1_conclusion_unchanged',
       'original_h1_gates_not_recomputed':True,
       'overall':group(records),'by_domain':{},'by_language':{},'by_operation':{},
       'raw_invalid_samples':[{'id':x['id'],'raw':x['raw']}
           for x in records if not x['valid']][:20]}
    for field,section in [('domain','by_domain'),('language','by_language'),('operator','by_operation')]:
        for value in sorted(set(x[field] for x in records)):
            summary[section][value]=group([x for x in records if x[field]==value])
    return summary


def main():
    ap=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--original-data',type=Path,default=root.parent/'h1'/'data')
    ap.add_argument('--original-predictions',type=Path,default=root.parent/'h1'/'cache'/'head_test_predictions.jsonl')
    ap.add_argument('--out',type=Path,default=root/'results'/'h1_original_audit.json')
    args=ap.parse_args()
    data,manifest=load_dataset(args.original_data)
    predictions=[json.loads(x) for x in args.original_predictions.read_text(encoding='utf-8').splitlines()]
    result=audit(predictions,data)
    result['original_h1_dataset_sha256']=manifest['dataset_sha256_canonical']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result['overall'],indent=2))

if __name__=='__main__':main()