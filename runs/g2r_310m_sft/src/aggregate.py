"""Aggregate only a fully completed, paired G2R experiment; no silent subsets."""
from __future__ import annotations
import collections,csv,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=('baseline','answer_only','full_sequence')
SPLITS=('test_iid','test_surface','test_extrapolation','test_counterfactual')
def load(path):return json.loads(Path(path).read_text())
def wilson(k,n):
 if n==0:return [None,None]
 z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,center-half),min(1,center+half)]
def counts(rows):
 n=len(rows);k=sum(bool(z['exact']) for z in rows)
 return {'n':n,'correct':k,'accuracy':k/n if n else None,'wilson95':wilson(k,n)}
def paired(a,b):
 aa={z['id']:bool(z['exact']) for z in a};bb={z['id']:bool(z['exact']) for z in b}
 if len(aa)!=len(a) or len(bb)!=len(b) or set(aa)!=set(bb):raise ValueError('unpaired or duplicated prediction IDs')
 win=sum(aa[k] and not bb[k] for k in aa);lose=sum(bb[k] and not aa[k] for k in aa);n=win+lose
 p=min(1.,2.*sum(math.comb(n,k) for k in range(min(win,lose)+1))/2**n) if n else 1.
 return {'n':len(a),'wins':win,'losses':lose,'ties':len(a)-n,'delta_pp':100*(sum(aa.values())-sum(bb.values()))/len(a),'two_sided_exact_sign_p':p}
def pair_rows(rows):
 g=collections.defaultdict(list)
 for z in rows:g[z['pair_id']].append(z)
 if not all(len(rr)==2 for rr in g.values()):raise ValueError('incomplete counterfactual pairs')
 return [{'id':k,'exact':all(z['exact'] for z in rr)} for k,rr in sorted(g.items())]
def main():
 metrics=ROOT/'metrics';data=ROOT/'data'
 if not all((metrics/f'{a}_DONE.json').is_file() for a in ARMS):raise RuntimeError('full protocol unfinished; refusing aggregate')
 result={};raw={};rows_csv=[];raw_join={}
 for arm in ARMS:
  if load(metrics/f'{arm}_DONE.json')['status']!='completed':raise ValueError('incomplete arm')
  result[arm]={};raw[arm]={};raw_join[arm]=[]
  for split in SPLITS:
   z=load(metrics/f'{arm}_{split}.json');gold=[json.loads(l) for l in (data/f'{split}.jsonl').read_text().splitlines()]
   expected={x['id']:x for x in gold}
   if z['status']!='completed' or len(z['rows'])!=len(gold) or {x['id'] for x in z['rows']}!=set(expected):raise ValueError('partial evaluation')
   for x in z['rows']:
    if x['exact']!=(x['generated'].strip()==expected[x['id']]['answer']):raise ValueError('incorrect answer scoring')
   raw[arm][split]=z['rows'];raw_join[arm].extend(z['rows']);m=counts(z['rows'])
   m['by_family']={f:counts([x for x in z['rows'] if x['family']==f]) for f in sorted({x['family'] for x in z['rows']})}
   m['by_language']={l:counts([x for x in z['rows'] if x['lang']==l]) for l in ['ru','en']}
   m['eos_fraction']=sum(x['emitted_eos'] for x in z['rows'])/len(z['rows'])
   completions=collections.Counter(x['generated'].strip() for x in z['rows'])
   m['format_diagnostic']={'valid_integer_fraction':sum(re.fullmatch(r'-?\d+',x['generated'].strip()) is not None for x in z['rows'])/len(z['rows']),'unique_completions':len(completions),'most_common':completions.most_common(5),'not_a_success_criterion':True}
   if split=='test_counterfactual':m['both_correct_pairs']=counts(pair_rows(z['rows']))
   result[arm][split]=m;rows_csv.append({'arm':arm,'split':split,'n':m['n'],'correct':m['correct'],'accuracy':m['accuracy'],'ci_low':m['wilson95'][0],'ci_high':m['wilson95'][1],'pair_accuracy':m.get('both_correct_pairs',{}).get('accuracy')})
  result[arm]['all_cases']=counts(raw_join[arm])
 comparison={}
 for ref in ['baseline','full_sequence']:
  cc={s:paired(raw['answer_only'][s],raw[ref][s]) for s in SPLITS}
  cc['surface_extrapolation_pooled']=paired(raw['answer_only']['test_surface']+raw['answer_only']['test_extrapolation'],raw[ref]['test_surface']+raw[ref]['test_extrapolation'])
  cc['both_correct_counterfactual_pairs']=paired(pair_rows(raw['answer_only']['test_counterfactual']),pair_rows(raw[ref]['test_counterfactual']))
  comparison[ref]=cc
 logs={a:[json.loads(l) for l in (metrics/f'{a}_steps.jsonl').read_text().splitlines()] for a in ['answer_only','full_sequence']}
 for a,ll in logs.items():
  assert [z['step'] for z in ll]==list(range(1,257))
  assert all(math.isfinite(z['train_loss']) and math.isfinite(z['grad_norm']) for z in ll)
  cp=ROOT/'checkpoints'/a/'step_0256'
  assert (cp/'COMPLETE').exists() and load(cp/'index.json')['step']==256
 assert all(x['batch_ids']==y['batch_ids'] and x['lr']==y['lr'] and x['padded_tokens']==y['padded_tokens'] for x,y in zip(logs['answer_only'],logs['full_sequence']))
 training={a:load(metrics/f'{a}_training.json') for a in logs}
 integrity={'matched_batches_all_256':True,'matched_learning_rate_all_256':True,'same_input_and_padded_tokens_all_steps':True,'finite_losses_and_gradnorms_all_steps':True,'completed_optimizer_updates_per_arm':256,'sampled_examples_per_arm':1024,'unique_train_examples_per_arm':len({k for z in logs['answer_only'] for k in z['batch_ids']}),'test_cases_per_arm':400,'training_model_seed_count':1,'qwen_inference_executed':False,'external_teacher_used':False,'full_rollouts_no_early_stop_oracle':True,'test_results_selected_by_checkpoint':False}
 exposure=load(ROOT/'reports/G1_SEMANTIC_EXPOSURE_AUDIT.json')
 canonical=load(ROOT/'reports/CANONICAL_SFT_EXPOSURE_AUDIT.json')
 novel={}
 for arm in ARMS:
  novel[arm]={s:counts([z for z in raw[arm][s] if z['id'] not in (set(exposure['held_overlap'][s]['ids'])|set(canonical['overlap_with_sft_train'][s]['ids']))]) for s in SPLITS}
 gates={
  'iid_gain_over_G1_at_least5pp':comparison['baseline']['test_iid']['delta_pp']>=5,
  'iid_gain_over_full_sequence_at_least5pp':comparison['full_sequence']['test_iid']['delta_pp']>=5,
  'pooled_surface_extrapolation_gain_positive':comparison['full_sequence']['surface_extrapolation_pooled']['delta_pp']>0,
  'both_correct_counterfactual_pair_gain_positive':comparison['full_sequence']['both_correct_counterfactual_pairs']['delta_pp']>0,
 }
 effect=all(gates.values());stat=comparison['full_sequence']['test_iid']['two_sided_exact_sign_p']<.05
 verdict={'effect_gates':gates,'effect_criterion_met':effect,'primary_paired_test_p_below_005':stat,'status':'SUPPORTED_ON_THIS_SUITE' if effect and stat else 'EFFECT_THRESHOLD_ONLY_NOT_STATISTICALLY_CONFIRMED' if effect else 'NOT_CONFIRMED_IN_THIS_PROTOCOL','no_broad_intelligence_claim':True}
 out={'run':'G2R-310M','status':'completed','hypothesis':'Completion-only verified-answer loss improves free generation over full-sequence training at equal examples, initial weights and optimizer updates.','parameters_per_model':309993253,'training':training,'results':result,'paired_comparisons_main_vs':comparison,'verdict':verdict,'integrity':integrity,'diagnostic_excluding_known_semantic_exposure':novel,'limitations':['Only one paired training seed; no across-seed stability claim.','4096 local synthetic cases; 1024 sampled presentations per arm; not broad pretraining.','Some held arithmetic cases also occur in earlier G1 CORPUS; actual exposure not established.','Defined semantic groups are disjoint, but some algebraically equivalent (+/* commuted operands) held cases overlap SFT train: IID4,surface3,extrapolation0,counterfactual3.','No Qwen process was run in this local primary experiment. A separately verified external40case Qwen run is supplemental only.','Attention graph is within-token readout, not inter-token persistent memory.','400 cases include80 dependent counterfactual members: use40 pairs for paired counterfactual inference.','No tool/executor given to model; correctness computed only after generation.']}
 (metrics/'G2R_METRICS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 with (metrics/'G2R_METRICS.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows_csv[0]));w.writeheader();w.writerows(rows_csv)
 (metrics/'RUN_INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
 (metrics/'RUN_STATUS.json').write_text(json.dumps({'status':'completed','trained_arms':2,'steps_per_arm':256,'evaluation_cases_per_arm':400,'verdict':verdict['status']},indent=2))
 lines=['# FlyGraph G2R-310M — завершённое парное обучение','',f'**Вердикт: {verdict["status"]}**','',out['hypothesis'],'','Две модели по309993253параметра,256optimizerupdatesкаждая,одинаковые minibatches и исходные G1 BF16→FP32weights. Новый optimizer, не exact resume прежнего G2. Одна основная гипотеза и один control: response-only против full-sequence loss.','', '| Проверка | N | G1 | Response-only | Full-sequence |','|---|---:|---:|---:|---:|']
 for s in SPLITS:
  n=result['baseline'][s]['n']; vals=[f"{result[a][s]['correct']}/{n} ({100*result[a][s]['accuracy']:.2f}%)" for a in ARMS]
  lines.append(f"| {s} | {n} | {' | '.join(vals)} |")
 vals=[f"{result[a]['test_counterfactual']['both_correct_pairs']['correct']}/40" for a in ARMS];lines.append(f"| Both-correct counterfactual pairs | 40 | {' | '.join(vals)} |")
 lines+=['','## Парная статистика','',json.dumps(comparison,ensure_ascii=False,indent=2),'','## Зафиксированные критерии','',json.dumps(verdict,ensure_ascii=False,indent=2),'','## Обучение и ограничения','',json.dumps(training,ensure_ascii=False,indent=2),'',*['- '+x for x in out['limitations']],'','Сырые prompts, gold, completions, token IDs доступны в metrics/*test*.json. Ни gold, ни case не входят в generation interface. Снижение NLL и освоение формата не считаются самостоятельным reasoning-успехом.','', '## Заимствования','', 'Прочитаны Qwen3.5 implementation, DeepSeek-V3/R1/Engram и MobileLLM-R1/TRM. Решения и первоисточники: research/BORROWING_DECISIONS_RU.md. Эти архитектуры не были тайно добавлены в G2R.','', '## Публикация','', 'Код и сводки публикуются через GitHub connector. Локальные weights не считаются опубликованными, пока нет подтверждённой загрузки bytes. См. PUBLICATION_RECEIPT.json.']
 (ROOT/'reports/G2R_COMPLETED_REPORT_RU.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'completed','verdict':verdict,'results':{a:{s:z['accuracy'] for s,z in zz.items()} for a,zz in result.items()}},indent=2))
if __name__=='__main__':main()
