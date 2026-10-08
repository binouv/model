"""Prompt-only free generation. Candidate ranking is reported separately."""
from __future__ import annotations
import argparse,collections,json,re,sys,time
from pathlib import Path
import torch
from common import *
from checkpoint_io import load
from train import pair_list,make_batch,loss_components
SPLITS=('test_iid','test_surface','test_extrapolation')

def parse(text):
 if text.count('F=')!=1:return None
 m=re.fullmatch(r'(?:T=[^;\n]+;)?F=(-?\d+)',text.strip());return int(m.group(1)) if m else None

@torch.inference_mode()
def predict(m,tok,prompts):
 tt=[tok.encode(p,bos=True) for p in prompts];assert len({len(z) for z in tt})==1
 with torch.autocast('cpu',dtype=torch.bfloat16):out=m.generate(torch.tensor(tt),max_new_tokens=64)
 ans=[]
 for z in out.tolist():
  eos=tok.EOS in z;z=z[:z.index(tok.EOS)] if eos else z;ans.append({'text':tok.decode(z),'tokens':z,'eos':eos})
 return ans

def summarize(rows):
 pairs=collections.defaultdict(list)
 for z in rows:
  if z.get('pair_id'):pairs[z['pair_id']].append(z)
 assert all(len(p)==2 for p in pairs.values())
 return {'n':len(rows),'correct':sum(z['exact'] for z in rows),'accuracy':sum(z['exact'] for z in rows)/len(rows),'pairs':len(pairs),'both_correct_pairs':sum(all(z['exact'] for z in p) for p in pairs.values()),'query_changes':sum(p[0]['text']!=p[1]['text'] for p in pairs.values()),'truncated':sum(not z['eos'] for z in rows),'format_valid':sum(parse(z['text']) is not None for z in rows)}

@torch.inference_mode()
def evaluation(m,tok,rr):
 m.eval();groups=collections.defaultdict(list)
 for r in rr:groups[len(tok.encode(r['prompt'],bos=True))].append(r)
 rows=[];started=time.perf_counter()
 for _,pp in sorted(groups.items()):
  for i in range(0,len(pp),8):
   bb=pp[i:i+8];pred=predict(m,tok,[z['prompt'] for z in bb])
   for z,a in zip(bb,pred):
    p=parse(a['text']);rows.append({**a,'id':z['id'],'family':z['family'],'lang':z['lang'],'pair_id':z.get('pair_id'),'prompt':z['prompt'],'gold':z['answer'],'exact':a['eos'] and p is not None and p==int(z['answer']),'legacy_exact_trace':a['eos'] and 'canonical_trace' in z and a['text'].strip()==z['completion']})
 rows.sort(key=lambda r:r['id']);out={'status':'completed',**summarize(rows),'rows':rows,'seconds':time.perf_counter()-started,'generated_tokens':sum(len(z['tokens']) for z in rows),'unconstrained_vocabulary':8192,'max_new_tokens':64,'tools_used':False,'solver_in_generation':False}
 out['by_family']={f:summarize([z for z in rows if z['family']==f]) for f in sorted({z['family'] for z in rows})}
 out['by_language']={l:summarize([z for z in rows if z['lang']==l]) for l in ['ru','en']}
 return out

@torch.inference_mode()
def rank_eval(m,tok,split):
 m.eval();out=[]
 for pp in pair_list(split):
  x,y,_=make_batch([pp],tok)
  with torch.autocast('cpu',dtype=torch.bfloat16):_,_,_,scores,_=loss_components(m,x,y,.5,.2)
  out.append({'pair_id':pp[0]['pair_id'],'scores':scores.tolist(),'wins':int((scores[::2]>scores[1::2]).sum()),'ties':int((scores[::2]==scores[1::2]).sum())})
 return {'status':'completed','n':len(out)*2,'accuracy':sum(z['wins'] for z in out)/(len(out)*2),'rows':out,'candidate_answers_known_to_evaluator_not_generation':True,'not_free_generation_accuracy':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--checkpoint',required=True);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 m,mf=load(args.checkpoint);m=m.float().eval();tok=Tokenizer.load(ROOT/'data/tokenizer.json')
 if args.arm!='parent':
  st=json.loads((ROOT/'metrics'/f'{args.arm}_TRAINING_COMPLETE.json').read_text());assert st['status']=='completed' and mf['step']==2048
 for sp in SPLITS:
  r=evaluation(m,tok,read(ROOT/'data'/f'{sp}.jsonl'));put(ROOT/'metrics'/f'{args.arm}_{sp}.json',r);print('EVAL',args.arm,sp,r['correct'],r['n'],r['both_correct_pairs'],flush=True)
 r=evaluation(m,tok,read(ROOT/'data/parent_regression.jsonl'));put(ROOT/'metrics'/f'{args.arm}_parent_regression.json',r);print('REGRESSION',args.arm,r['correct'],flush=True)
 r=rank_eval(m,tok,'test_iid');put(ROOT/'metrics'/f'{args.arm}_diagnostic_ranking.json',r);print('RANKING',args.arm,r['accuracy'],flush=True)
 put(ROOT/'metrics'/f'{args.arm}_EVAL_COMPLETE.json',{'status':'completed','checkpoint_step':mf['step'],'scored_questions':920,'not_Qwen_run':True})
if __name__=='__main__':main()
