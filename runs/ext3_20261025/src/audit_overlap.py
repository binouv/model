import json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'ext2_20261022';RAW=ROOT.parent/'ext_20261018'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).lower()).strip()
def main():
 prior=rows(OLD/'data/train.jsonl');raw=json.loads((RAW/'data/alpaca_raw.json').read_text());ids={int(str(r['id']).split(':')[-1]) for r in rows(RAW/'data/train.jsonl')};texts=[(r['id'],norm(r['user_text']+' '+r['completion'])) for r in prior if r['task']=='alpaca']+[(f'alpaca:{i}',norm(raw[i]['instruction']+' '+raw[i]['input']+' '+raw[i]['output'])) for i in ids]
 full=[];long=[];held=[]
 for split in ['dev','test']:
  for r in rows(ROOT/f'data/{split}_mcq.jsonl'):
   held.append(r)
   for id,t in texts:
    q=norm(r['question'])
    if q not in t:continue
    if len(q)>=20:long.append(dict(split=split,held_id=r['id'],prior_train_id=id))
    if all(norm(c) in t for c in r['choices']):full.append(dict(split=split,held_id=r['id'],prior_train_id=id))
 previous=rows(OLD/'data/train.jsonl')+rows(OLD/'data/dev_mcq.jsonl')+rows(OLD/'data/test_mcq.jsonl');prevq={norm(r['question']) for r in previous if 'question' in r};assert not prevq&{norm(r['question']) for r in held}
 train=rows(ROOT/'data/train.jsonl');qtrain={norm(r['question']) for r in train if 'question' in r};qheld={norm(r['question']) for r in held};assert not qtrain&qheld
 d={norm(r['question']) for r in rows(ROOT/'data/dev_mcq.jsonl')};t={norm(r['question']) for r in rows(ROOT/'data/test_mcq.jsonl')};assert not d&t
 result=dict(new_mcq_train_vs_dev_test_exact_normalized_stem=0,new_mcq_dev_vs_test_exact_normalized_stem=0,new_mcq_held_vs_prior_ext2_mcq_train_dev_test=0,known_prior_alpaca_training_texts_checked=len(texts),prior_alpaca_full_stem_and_all_choices_hits=full,prior_alpaca_long_stem_only_substring_hits=long,limitations='Generic short stems are not entire items. Exact screens do not exclude semantic/paraphrase overlap or unknown historical pretraining contamination.')
 (ROOT/'metrics/overlap_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
