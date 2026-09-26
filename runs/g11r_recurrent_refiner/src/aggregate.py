import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ARMS=('recur4','recur1_control');SEEDS=(8401,8402,8403);SPLITS=('test_iid','test_surface','test_extrapolation','test_counterfactual')
def sign_p(w,l):
 n=w+l
 if n==0:return 1.0
 k=min(w,l);return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def main():
 rr={(a,s):json.loads((ROOT/'metrics'/f'{a}_s{s}.json').read_text()) for a in ARMS for s in SEEDS};assert all(x['status']=='completed' and x['steps']==2000 and x['reload_mismatches']==0 and x['resume_next_update_verified'] for x in rr.values());assert len({rr[k]['batch_sequence_sha256'] for k in rr})==1
 res={}
 for a in ARMS:
  res[a]={}
  for sp in SPLITS:
   vals=[rr[(a,s)]['tests'][sp] for s in SEEDS];res[a][sp]={'mean_accuracy':sum(v['accuracy'] for v in vals)/3,'correct_by_seed':[v['correct'] for v in vals],'n_unique':vals[0]['n'],'family_correct_by_seed':{f:[rr[(a,s)]['tests'][sp]['family'][f]['correct'] for s in SEEDS] for f in vals[0]['family']},'family_n':{f:vals[0]['family'][f]['n'] for f in vals[0]['family']},'both_correct_pairs_by_seed':[v['both_correct_pairs'] for v in vals],'pair_n':vals[0]['pair_count'],'generation_seconds_by_seed':[v['seconds'] for v in vals],'generated_bytes_by_seed':[v['generated_bytes'] for v in vals]}
  res[a]['train']={'train_seconds_by_seed':[rr[(a,s)]['train_seconds'] for s in SEEDS],'profiler_supported_flops_by_seed':[rr[(a,s)]['profiler']['supported_forward_backward_flops'] for s in SEEDS],'total_parameters':rr[(a,SEEDS[0])]['total_parameters'],'shared_refiner_parameters':rr[(a,SEEDS[0])]['shared_refiner_parameters']}
 def fam_acc(a,f):
  z=res[a]['test_iid'];return sum(z['family_correct_by_seed'][f])/(3*z['family_n'][f])
 reason={a:(fam_acc(a,'arithmetic')+fam_acc(a,'code_trace'))/2 for a in ARMS};memlist={a:(fam_acc(a,'memory_update')+fam_acc(a,'list_reasoning'))/2 for a in ARMS}
 gains={sp:100*(res['recur4'][sp]['mean_accuracy']-res['recur1_control'][sp]['mean_accuracy']) for sp in SPLITS};rg=100*(reason['recur4']-reason['recur1_control']);mg=100*(memlist['recur4']-memlist['recur1_control'])
 paired={}
 for s in SEEDS:
  A={x['id']:x for x in rr[('recur4',s)]['tests']['test_iid']['rows']};B={x['id']:x for x in rr[('recur1_control',s)]['tests']['test_iid']['rows']};ids=[k for k in A if A[k]['family'] in ('arithmetic','code_trace')];w=sum(A[k]['exact'] and not B[k]['exact'] for k in ids);l=sum(B[k]['exact'] and not A[k]['exact'] for k in ids);paired[str(s)]={'arith_code_wins':w,'losses':l,'delta_pp':100*(sum(A[k]['exact'] for k in ids)-sum(B[k]['exact'] for k in ids))/len(ids),'exact_two_sided_p':sign_p(w,l)}
 gates={'arith_code_gain_ge3pp':rg>=3,'positive_arith_code_at_least2seeds':sum(x['delta_pp']>0 for x in paired.values())>=2,'iid_overall_nonnegative':gains['test_iid']>=0,'memory_list_regression_le2pp':mg>=-2,'surface_plus_extrapolation_nonnegative':gains['test_surface']+gains['test_extrapolation']>=0}
 out={'run':'G11R','status':'completed','training_models':6,'seeds':list(SEEDS),'results':res,'arith_code_accuracy':reason,'memory_list_accuracy':memlist,'gains_pp':gains,'arith_code_gain_pp':rg,'memory_list_gain_pp':mg,'paired_primary':paired,'gates':gates,'verdict':'CONFIRMED_IN_REGISTERED_PROTOCOL' if all(gates.values()) else 'NOT_CONFIRMED_IN_REGISTERED_PROTOCOL','parameter_matched':True,'compute_matched':False,'same_batch_sequence_across_all_models':True,'scope':'small fresh multi-family recurrent-refinement screen; not300-800M and not Qwen parity'}
 (ROOT/'metrics/FINAL.json').write_text(json.dumps(out,indent=2));
 with (ROOT/'metrics/FINAL.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['metric','recur4','recur1','gain_pp']);w.writerow(['arith_code',reason['recur4'],reason['recur1_control'],rg]);w.writerow(['memory_list',memlist['recur4'],memlist['recur1_control'],mg]);[w.writerow([sp,res['recur4'][sp]['mean_accuracy'],res['recur1_control'][sp]['mean_accuracy'],gains[sp]]) for sp in SPLITS]
 print(json.dumps({'verdict':out['verdict'],'arith_code_gain_pp':rg,'gains_pp':gains,'gates':gates}))
if __name__=='__main__':main()
