"""Verify/score ALL stored primary predictions, without running a neural model.
Lossless dictionary coding only saves repeated text/tokens; not model compression.
Reconstructed inputs must match the four exact dataset SHA256 values.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,sys,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'base_src'))
from tokenizer import Tokenizer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--verify-raw',action='store_true');a=ap.parse_args()
 p=ROOT/'metrics/PREDICTIONS_COMPACT.json.gz'
 raw=gzip.decompress(p.read_bytes()) if p.exists() else (ROOT/'metrics/PREDICTIONS_COMPACT.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!='e0b8293dd9c5001d787df8d490d5f95cc470a94a596ec61ca7e4e3f697674d9f':raise ValueError('prediction archive SHA256 mismatch')
 archive=json.loads(raw)
 if archive['format']!='G2R_lossless_prediction_dictionary_v1':raise ValueError('unsupported format')
 tok=Tokenizer.load(ROOT/'data/tokenizer.json');d=archive['dictionary']
 for text,ids,eos in d:
  if tok.decode(ids)!=text:raise ValueError('token IDs do not decode to stored text')
  if not isinstance(eos,bool) or len(ids)>8:raise ValueError('bad generation metadata')
 result={};total=0
 for arm,splits in archive['arms'].items():
  result[arm]={}
  for split,indices in splits.items():
   p=ROOT/'data'/f'{split}.jsonl'
   if hashlib.sha256(p.read_bytes()).hexdigest()!=archive['dataset_sha256'][split]:raise ValueError('dataset SHA256 mismatch')
   cases=[json.loads(l) for l in p.read_text().splitlines()]
   if len(cases)!=len(indices):raise ValueError('prediction count mismatch')
   ref=None
   if a.verify_raw:
    rows=json.loads((ROOT/'metrics'/f'{arm}_{split}.json').read_text())['rows'];ref={z['id']:z for z in rows}
    if len(ref)!=len(rows) or set(ref)!={z['id'] for z in cases}:raise ValueError('original IDs mismatch')
   correct=0;pairs=collections.defaultdict(list)
   for c,i in zip(cases,indices):
    text,ids,eos=d[i];ok=text.strip()==c['answer'];correct+=ok;total+=1
    if 'pair_id' in c:pairs[c['pair_id']].append(ok)
    if ref is not None:
     z=ref[c['id']]
     if not (text==z['generated'] and ids==z['token_ids'] and eos==z['emitted_eos'] and ok==z['exact']):raise ValueError('lossy prediction archive')
   r={'n':len(cases),'correct':correct,'accuracy':correct/len(cases)}
   if pairs:
    if not all(len(x)==2 for x in pairs.values()):raise ValueError('partial counterfactual pair')
    r.update(pairs=len(pairs),both_correct=sum(all(x) for x in pairs.values()))
   result[arm][split]=r
 out={'status':'passed','records':total,'unique_dictionary_entries':len(d),'full_original_raw_records_compared':a.verify_raw,
  'dataset_hashes_verified':True,'all_token_sequences_decode_exactly':True,'results':result,'neural_inference_executed':False}
 (ROOT/'metrics/PREDICTION_ARCHIVE_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
