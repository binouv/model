import copy,json,math
import torch
import torch.nn.functional as F
from common import ROOT,rows,atomic_json

def configure(m):
 prefixes=('blocks.12.','blocks.13.','norm.')
 for name,p in m.named_parameters():p.requires_grad_(name.startswith(prefixes))
 return [p for p in m.parameters() if p.requires_grad]
def permute(r,shift):
 q=copy.deepcopy(r);n=len(r['choices']);order=[(i+shift)%n for i in range(n)];q['choices']=[r['choices'][i] for i in order];q['gold_index']=order.index(r['gold_index']);letters=[chr(65+i) for i in range(n)]
 u='Answer the question. Reply with only the letter '+', '.join(letters)+'.\nQuestion: '+r['question']+'\n'+'\n'.join(l+'. '+c for l,c in zip(letters,q['choices']))
 q.update(prompt='User: '+u+'\nAssistant:',completion=' '+letters[q['gold_index']],permutation=order)
 return q
def label_logits(m,tok,r):
 ids=torch.tensor([tok.encode(r['prompt'],bos=True)]);lg,_=m(ids,last_only=True);indices=[tok.encode(' '+chr(65+i))[0] for i in range(len(r['choices']))]
 return lg[0,-1,indices].float()
@torch.inference_mode()
def balanced_dev(m,tok,arm):
 m.eval();out=[]
 for r in rows('dev_mcq.jsonl'):
  if r['task']=='arc_challenge':continue
  for shift in range(len(r['choices'])):
   q=permute(r,shift)
   with torch.autocast('cpu',dtype=torch.bfloat16):lg=label_logits(m,tok,q)
   pred=int(lg.argmax());lp=lg.log_softmax(-1)
   out.append(dict(id=r['id'],task=r['task'],permutation=q['permutation'],gold=q['gold_index'],predicted=pred,predicted_source_option=q['permutation'][pred],gold_source_option=r['gold_index'],normalized_log_probs=lp.tolist(),nll=-float(lp[q['gold_index']]),correct=pred==q['gold_index']))
 tasks={}
 for task in ['openbookqa','piqa']:
  rr=[r for r in out if r['task']==task];tasks[task]={'n':len(rr),'correct':sum(r['correct'] for r in rr),'nll':sum(r['nll'] for r in rr)/len(rr),'prediction_counts':{str(i):sum(r['predicted']==i for r in rr) for i in range(4 if task=='openbookqa' else 2)}}
 consistency=[]
 for id in dict.fromkeys(r['id'] for r in out):consistency.append(len({r['predicted_source_option'] for r in out if r['id']==id})==1)
 summary=dict(arm=arm,selection_objective=sum(v['nll'] for v in tasks.values())/2,by_task=tasks,source_option_invariant_questions=sum(consistency),n_unique_questions=len(consistency))
 atomic_json(ROOT/f'metrics/{arm}_balanced_dev_summary.json',summary)
 with (ROOT/f'metrics/{arm}_balanced_dev_predictions.jsonl').open('x') as f:
  for r in out:f.write(json.dumps(r)+'\n')
 return summary
