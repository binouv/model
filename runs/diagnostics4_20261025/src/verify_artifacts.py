"""Independent data/feature/probe checks; supports a clean remote readback."""
import argparse,datetime,json,re,unicodedata
from pathlib import Path
import torch
from safetensors.torch import load_file
from common import ROOT,sha,rows,atomic_json,setup
from run_diagnostics import fit_ridge,predict
from tokenizer import Tokenizer

def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s).lower()).strip()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'metrics/artifact_verification.json');ap.add_argument('--historical',action='store_true');args=ap.parse_args();setup()
 cfg=json.loads((ROOT/'config.json').read_text());res=json.loads((ROOT/'metrics/results.json').read_text());assert res['completed'];dm=json.loads((ROOT/'data/manifest.json').read_text());sets={n:rows(n+'.jsonl') for n in ['probe_train','seen_control','tiny_train','dev_mcq','test_mcq']}
 for n,v in dm['files'].items():assert sha(ROOT/'data'/n)==v['sha256'];assert len(rows(n))==v['n']
 for name,h in json.loads((ROOT/'source_manifest.json').read_text()).items():assert sha(ROOT/name)==h,name
 distinct={k:{norm(r['question']) for r in v} for k,v in sets.items()};assert not distinct['probe_train']&distinct['dev_mcq'];assert not distinct['probe_train']&distinct['test_mcq'];assert not distinct['dev_mcq']&distinct['test_mcq'];assert distinct['tiny_train']<=distinct['probe_train'];assert distinct['seen_control']<=distinct['probe_train']
 tok=Tokenizer.load(ROOT/'data/tokenizer.json');maxctx=0
 for rr in sets.values():
  for r in rr:
   assert r['completion']==' '+chr(65+r['gold_index'])
   for i in range(len(r['choices'])):assert len(tok.encode(' '+chr(65+i)))==1
   for c in r['choices']:
    for prefix in ['User: Answer the question briefly.\nQuestion: '+r['question']+'\nAssistant:','User: Answer the question briefly.\nAssistant:']:
     assert tok.encode(prefix+' '+c,bos=True)==tok.encode(prefix,bos=True)+tok.encode(' '+c)
     maxctx=max(maxctx,len(tok.encode(prefix+' '+c,bos=True)))
 assert maxctx<=2048
 historical=None
 if args.historical:
  excluded=set()
  for parent in ['ext2_20261022','ext3_20261025']:
   for file in ['train.jsonl','dev_mcq.jsonl','test_mcq.jsonl']:
    for line in (ROOT.parent/parent/'data'/file).read_text().splitlines():
     r=json.loads(line)
     if 'question' in r:excluded.add(norm(r['question']))
  for line in (ROOT.parent/'ext_20261018/data/arc_test128.jsonl').read_text().splitlines():excluded.add(norm(json.loads(line)['question']))
  assert not (distinct['dev_mcq']|distinct['test_mcq'])&excluded
  historical=dict(previous_mcq_unique_stems=len(excluded),new_holdout_overlap=0,prior_unrecorded_pretraining='unknown')
 feats={k:load_file(str(ROOT/f'artifacts/{k}_features.safetensors'))['last_hidden_state'] for k in ['probe_train','dev_mcq','test_mcq']};stored=load_file(str(ROOT/'artifacts/probe_weights.native_float64.safetensors'));selection=json.loads((ROOT/'metrics/probe_selection.json').read_text());checks=[];idx={r['id']:i for i,r in enumerate(sets['probe_train'])}
 for task,nc in [('openbookqa',4),('piqa',2)]:
  indices=[i for i,r in enumerate(sets['probe_train']) if r['task']==task];tiny=[r for r in sets['tiny_train'] if r['task']==task]
  for prefix,ii,lam in [(task,indices,selection[task]['selected_regularization']),('tiny_'+task,[idx[r['id']] for r in tiny],cfg['tiny_ridge_lambda'])]:
   y=torch.tensor([sets['probe_train'][i]['gold_index'] for i in ii]);head=fit_ridge(feats['probe_train'][ii],y,nc,lam)
   for k,v in head.items():
    original=stored[prefix+'_'+k];assert original.dtype==torch.float64;assert torch.equal(v,original),(prefix,k,float((v-original).abs().max()))
   checks.append(dict(head=prefix,refit_bitwise_equal=True,n=len(ii),regularization=lam))
 pp=[json.loads(s) for s in (ROOT/'metrics/probe_predictions.jsonl').read_text().splitlines()];predby={(r['split'],r['id']):r for r in pp};total=0
 for split in ['dev_mcq','test_mcq']:
  for i,r in enumerate(sets[split]):
   source='piqa' if r['task']=='piqa' else 'openbookqa';h={k:stored[source+'_'+k] for k in ['mean','std','weight']};p=int(predict(h,feats[split][i:i+1])[0]);assert p==predby[split,r['id']]['predicted'];total+=1
 info=dict(status='passed',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest_unchanged=True,data_checksums_passed=True,local_train_dev_test_stems_disjoint=True,tiny_and_seen_are_intentional_training_subsets=True,all_candidate_token_boundaries_exact=True,max_candidate_context_tokens=maxctx,historical_overlap_check=historical,probe_refit=checks,stored_probe_prediction_matches=total,features={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=sha(ROOT/f'artifacts/{k}_features.safetensors')) for k,v in feats.items()},probe_weights_sha256=sha(ROOT/'artifacts/probe_weights.native_float64.safetensors'),scope='Saved frozen features and closed-form coefficients replay; no optimizer state exists or is claimed. Backbone checkpoint bytes preserved by EXT2/EXT3 immutable references.')
 atomic_json(args.output,info);print(json.dumps(info,indent=2))
if __name__=='__main__':main()
