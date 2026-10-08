"""Complete-only offline re-score of the fixed paired local100M experiment.
No checkpoint selection, no candidate-restricted solving, no hidden partial grid.
"""
from __future__ import annotations
import collections,csv,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=('parent','contrastive','nll_control')
SPLITS=('test_iid','test_surface','test_extrapolation','parent_regression')
def read(p):return [json.loads(z) for z in Path(p).read_text().splitlines()]
def load(p):return json.loads(Path(p).read_text())
def put(p,z):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def parse(s):
 if s.count('F=')!=1:return None
 m=re.fullmatch(r'(?:T=[^;\n]+;)?F=(-?\d+)',s.strip());return int(m.group(1)) if m else None
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,mid-half),min(1,mid+half)]
def pairs(rows):
 out=collections.defaultdict(list)
 for r in rows:
  if r.get('pair_id'):out[r['pair_id']].append(r)
 if any(len(v)!=2 for v in out.values()):raise ValueError('partial counterfactual pair')
 return out
def stats(rows):
 n=len(rows);k=sum(z['exact'] for z in rows);pp=pairs(rows);pk=sum(all(z['exact'] for z in vv) for vv in pp.values())
 return {'n':n,'correct':k,'accuracy':k/n,'wilson95':wilson(k,n),'n_pairs':len(pp),'both_correct_pairs':pk,'pair_accuracy':pk/len(pp) if pp else None,'pair_wilson95':wilson(pk,len(pp)),
 'query_changed_raw_text':sum(v[0]['text']!=v[1]['text'] for v in pp.values()),'query_changed_parsed_answer':sum(parse(v[0]['text'])!=parse(v[1]['text']) for v in pp.values()),
 'truncated':sum(not r['eos'] for r in rows),'format_valid':sum(parse(r['text']) is not None for r in rows),'legacy_exact_trace':sum(r['legacy_exact_trace'] for r in rows)}
def paired(a,b,pair_unit=False):
 if pair_unit:
  a=[{'id':k,'exact':all(x['exact'] for x in v)} for k,v in pairs(a).items()]
  b=[{'id':k,'exact':all(x['exact'] for x in v)} for k,v in pairs(b).items()]
 aa={x['id']:x['exact'] for x in a};bb={x['id']:x['exact'] for x in b}
 if set(aa)!=set(bb) or len(aa)!=len(a) or len(bb)!=len(b):raise ValueError('mismatched pairs')
 w=sum(aa[k] and not bb[k] for k in aa);l=sum(bb[k] and not aa[k] for k in aa);d=w+l
 p=min(1.,2*sum(math.comb(d,i) for i in range(min(w,l)+1))/2**d) if d else 1.
 return {'unit':'both-correct counterfactual pair' if pair_unit else 'question (within-pair dependence; descriptive p only)', 'n':len(aa),'wins':w,'losses':l,'ties':len(aa)-d,'delta_pp':100*(w-l)/len(aa),'exact_two_sided_p':p}
def main():
 cfg=load(ROOT/'configs/preregistered.json');met=ROOT/'metrics'
 for a in ARMS:
  p=met/f'{a}_EVAL_COMPLETE.json'
  if not p.is_file() or load(p)['status']!='completed':raise RuntimeError('Incomplete primary experiment; refusing final aggregate')
 trains={a:load(met/f'{a}_TRAINING_COMPLETE.json') for a in ARMS[1:]}
 for a,t in trains.items():
  assert t['status']=='completed' and t['additional_updates']==cfg['updates_per_arm'] and t['parameters']==cfg['parameters']
 logs={a:read(met/f'{a}_steps.jsonl') for a in ARMS[1:]}
 for a,ll in logs.items():
  assert [x['local_step'] for x in ll]==list(range(1,cfg['updates_per_arm']+1))
  assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in ll)
 fields=['batch_ids','lr','scored_input_tokens','scored_supervised_tokens','positive_supervised_tokens','padded_tokens']
 assert all(all(a[k]==b[k] for k in fields) for a,b in zip(logs['contrastive'],logs['nll_control']))
 init={a:load(met/f'{a}_initial_state.json')['sha256'] for a in ARMS[1:]};assert len(set(init.values()))==1
 raw={};res={};flat=[];rows_csv=[]
 for a in ARMS:
  raw[a]={};res[a]={}
  for s in SPLITS:
   z=load(met/f'{a}_{s}.json');rr=z['rows'];gold={g['id']:g for g in read(ROOT/'data'/f'{s}.jsonl')}
   assert z['status']=='completed' and len(rr)==len(gold) and {r['id'] for r in rr}==set(gold)
   for r in rr:
    g=gold[r['id']];assert r['gold']==g['answer'] and r['prompt']==g['prompt']
    ok=r['eos'] and parse(r['text']) is not None and parse(r['text'])==int(g['answer'])
    assert r['exact']==ok
    flat.append({'arm':a,'split':s,**r})
   raw[a][s]=rr;st=stats(rr);st['generation_seconds']=z['seconds'];st['generated_tokens']=z['generated_tokens']
   st['by_family']={f:stats([r for r in rr if r['family']==f]) for f in sorted({r['family'] for r in rr})}
   st['by_language']={l:stats([r for r in rr if r['lang']==l]) for l in ['ru','en']}
   res[a][s]=st
   rows_csv.append({'arm':a,'split':s,**{k:st[k] for k in ['n','correct','accuracy','n_pairs','both_correct_pairs','pair_accuracy','query_changed_parsed_answer','truncated','format_valid','generation_seconds','generated_tokens']}})
 comp={ref:{s:{'answer':paired(raw['contrastive'][s],raw[ref][s]),'pair':paired(raw['contrastive'][s],raw[ref][s],True) if s!='parent_regression' else None} for s in SPLITS} for ref in ['parent','nll_control']}
 g=cfg['success'];diff=comp['nll_control'];checks={
 'iid_pair_gain_at_least5pp':diff['test_iid']['pair']['delta_pp']>=g['iid_both_correct_pair_gain_pp'],
 'iid_answer_gain_at_least5pp':diff['test_iid']['answer']['delta_pp']>=g['iid_exact_answer_gain_pp'],
 'surface_nonnegative':diff['test_surface']['answer']['delta_pp']>=0,
 'extrapolation_regression_no_more2pp':diff['test_extrapolation']['answer']['delta_pp']>=-g['extrapolation_regression_max_pp'],
 'parent_regression_no_more2pp_vs_parent':comp['parent']['parent_regression']['answer']['delta_pp']>=-g['parent_regression_max_pp']}
 integrity={'same_initial_weight_state':True,'same_batches_learning_rate_shapes_tokens':True,'completed_updates_each':cfg['updates_per_arm'],'completed_models':2,'new_random_initialization_count':0,'inherited_seed_count':1,'free_generation_records_rescored':len(flat),'unique_fresh_test_questions':800,'unique_fresh_pairs':400,'regression_questions':120,'same_questions_reused_across_models':True,
 'training_examples_presentations_each':trains['contrastive']['counts']['positive_presentations'],'unique_train_queries_seen_each':len({k for z in logs['contrastive'] for k in z['batch_ids']}),'true_hardware_flops_measured':False,'local_cpu_training':True,'no_Qwen_or_teacher_in_training':True,'no_test_based_checkpoint_selection':True}
 rank={a:load(met/f'{a}_diagnostic_ranking.json')['accuracy'] for a in ARMS}
 out={'run':cfg['run'],'status':'completed','hypothesis':cfg['primary_hypothesis'],'training':trains,'results':res,'comparisons_main_vs':comp,'registered_gates':checks,'verdict':'SUPPORTED_ON_THIS_NARROW_SINGLE_SEED_SUITE' if all(checks.values()) else 'NOT_CONFIRMED_IN_REGISTERED_PROTOCOL','ranking_diagnostic_not_free_generation':rank,'integrity':integrity,'limitations':cfg['limitations']+['These are code execution traces, not real code-writing benchmarks.','No exact latent reasoning trace is observable for final-answer-only generation.','Question-level p values ignore pair dependence; pair-level both-correct p uses independent held groups.','All operations active, no extra parameter padding; throughput and full hardware FLOPs are different measurements.']}
 put(met/'FINAL.json',out);put(met/'RUN_INTEGRITY.json',integrity)
 with (met/'FINAL.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows_csv[0]));w.writeheader();w.writerows(rows_csv)
 with (met/'RAW_PREDICTIONS.jsonl').open('w') as f:
  for r in flat:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 lines=['# L100C — завершённое парное локальное обучение100M','',f'**{out["verdict"]}**','',cfg['primary_hypothesis'],'','| Split | N | Parent | Contrastive | Control |','|---|---:|---:|---:|---:|']
 for s in SPLITS:lines.append(f'| {s} | {res["parent"][s]["n"]} | '+ ' | '.join(f'{res[a][s]["correct"]} ({100*res[a][s]["accuracy"]:.2f}%)' for a in ARMS)+' |')
 lines+=['','## Критерии','',json.dumps(checks,ensure_ascii=False,indent=2),'','## Целостность','',json.dumps(integrity,ensure_ascii=False,indent=2),'','## Ограничения','',*['- '+x for x in out['limitations']],'','Сырые результаты сохранены до подготовки итогового текста. Каждая из двух моделей обучена1024новыхшага от одинаковогоL100MBF16с новымoptimizer,не exact resume предшественника. Доступные новыеFP32checkpoints включаютoptimizer/RNG. Qwen-reference отдельный; незавершённый результат не является итоговойоценкой.']
 (ROOT/'reports/REPORT_RU.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':out['status'],'verdict':out['verdict'],'gates':checks,'results':{a:{s:r['accuracy'] for s,r in v.items()} for a,v in res.items()}},indent=2))
if __name__=='__main__':main()
