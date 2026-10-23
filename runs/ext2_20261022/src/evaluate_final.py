"""One final held-out comparison after model selection is frozen."""
import collections,datetime,json,math,time
import numpy as np
import torch
from common import ROOT,setup,load,rows,sha,atomic_json,evaluate
def wilson(k,n):
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]
def main():
    setup();cfg=json.loads((ROOT/'config.json').read_text());training=json.loads((ROOT/'metrics/training_summary.json').read_text());selection=json.loads((ROOT/'metrics/selection.json').read_text())
    if (ROOT/'metrics/final_results.json').exists():raise FileExistsError('Final evaluation already completed')
    frozen={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selection_sha256':sha(ROOT/'metrics/selection.json'),'training_summary_sha256':sha(ROOT/'metrics/training_summary.json'),'test_mcq_sha256':sha(ROOT/'data/test_mcq.jsonl'),'test_alpaca_sha256':sha(ROOT/'data/test_alpaca.jsonl'),'selected_step':selection['best_step'],'criterion':selection['criterion']}
    atomic_json(ROOT/'metrics/final_evaluation_lock.json',frozen)
    parent=(ROOT/cfg['parent_path']).resolve();selected=parent if selection['best_step']==0 else ROOT/'checkpoints'/f"step_{selection['best_step']:04d}"/'model.safetensors'
    summaries={};raw={};generations=[]
    for arm,weights in [('parent_EXT256',parent),('selected_EXT2',selected)]:
        m,tok=load(weights);summaries[arm],raw[arm],_=evaluate(m,tok,'test',arm);summaries[arm]['weight_sha256']=sha(weights)
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            for r in rows('test_alpaca.jsonl')[:8]:
                seq=m.generate(torch.tensor([tok.encode(r['prompt'],bos=True)]),max_new_tokens=32)[0].tolist();body=seq[:seq.index(2)] if 2 in seq else seq
                generations.append({'arm':arm,'id':r['id'],'prompt':r['prompt'],'gold_external_target':r['completion'],'generation':tok.decode(body),'token_ids':seq})
        print('TEST',json.dumps(summaries[arm]),flush=True);del m
    assert [p['id'] for p in raw['parent_EXT256']]==[p['id'] for p in raw['selected_EXT2']]
    rng=np.random.default_rng(20261023);comparisons={};task_diffs=[]
    for task in ['openbookqa','piqa','arc_challenge']:
        old=[p for p in raw['parent_EXT256'] if p['task']==task];new=[p for p in raw['selected_EXT2'] if p['task']==task];n=len(old);idx=rng.integers(n,size=(10000,n));c={}
        for metric,key in [('label','predicted_label'),('generation','parsed_generation')]:
            b=np.array([p[key]==p['gold'] for p in old],dtype=int);a=np.array([p[key]==p['gold'] for p in new],dtype=int);diff=a-b;boots=diff[idx].mean(1)
            wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum());discord=wins+losses
            p=min(1.,2*sum(math.comb(discord,k) for k in range(min(wins,losses)+1))/2**discord) if discord else 1.
            c[metric]={'before_correct':int(b.sum()),'after_correct':int(a.sum()),'n':n,'difference':float(diff.mean()),'paired_bootstrap95':np.quantile(boots,[.025,.975]).tolist(),'before_wilson95':wilson(int(b.sum()),n),'after_wilson95':wilson(int(a.sum()),n),'mcnemar_exact_p':p}
            if metric=='label':task_diffs.append((float(diff.mean()),boots))
        comparisons[task]=c
    # Diagnostic control, derived from train labels only: distinguish useful
    # input-dependent predictions from a newly learned constant letter prior.
    training_rows=rows('train.jsonl');prior={}
    for task in ['openbookqa','piqa']:
        counts=collections.Counter(r['gold_index'] for r in training_rows if r['task']==task)
        prior[task]=min(counts,key=lambda k:(-counts[k],k))
    prior['arc_challenge']=prior['openbookqa'];prior_diagnostics={}
    for task in ['openbookqa','piqa','arc_challenge']:
        pp=[p for p in raw['selected_EXT2'] if p['task']==task];freq=collections.Counter(p['predicted_label'] for p in pp)
        prior_diagnostics[task]={'constant_prediction_from_train':prior[task],'constant_correct':sum(p['gold']==prior[task] for p in pp),'n':len(pp),'selected_prediction_counts':dict(freq),'selected_most_common_prediction_fraction':max(freq.values())/len(pp)}
    mean_diff=float(np.mean([d[0] for d in task_diffs]));ci=np.quantile(np.stack([d[1] for d in task_diffs]).mean(0),[.025,.975]).tolist()
    language_no_worse=summaries['selected_EXT2']['alpaca_nll']<=summaries['parent_EXT256']['alpaca_nll']
    criterion=ci[0]>0 and sum(d[0]>0 for d in task_diffs)>=2 and language_no_worse
    output={'run_id':'ext2_20261022','evaluation_completed':True,'training':training,'selected_step':selection['best_step'],'summaries':summaries,'paired_comparisons':comparisons,'mean_label_accuracy_difference':mean_diff,'stratified_paired_bootstrap95':ci,'bootstrap_repeats':10000,'bootstrap_seed':20261023,'held_alpaca_nll_not_worse':language_no_worse,'registered_improvement_criterion_met':bool(criterion),'general_superiority_to_Qwen_and_MiniCPM_established':False,'references':json.loads((ROOT/'metrics/reference_status.json').read_text()),'scope':'English external multiple-choice pilot and instruction-language NLL; PIQA test is a reserved subset of its official labeled validation split; historical contamination unknown.'}
    output['training_label_prior_diagnostic']=prior_diagnostics
    atomic_json(ROOT/'metrics/final_results.json',output)
    with (ROOT/'metrics/test_alpaca_generations.jsonl').open('x') as f:
        for g in generations:f.write(json.dumps(g,ensure_ascii=False)+'\n')
    import csv
    with (ROOT/'metrics/final_results.csv').open('w',newline='') as f:
        fields=['model','dataset','status','n','label_correct','label_accuracy','generation_correct','generation_accuracy','valid_generation','alpaca_nll'];w=csv.DictWriter(f,fields);w.writeheader()
        for arm,s in summaries.items():
            for task,t in s['by_task'].items():w.writerow(dict(model=arm,dataset=task,status='measured',alpaca_nll=s['alpaca_nll'],**{k:t[k] for k in fields if k in t}))
        for model in cfg['reference_models']:w.writerow({'model':model,'status':'blocked_proxy_403_not_measured'})
    print('FINAL_CRITERION',criterion,'mean_difference',mean_diff,'CI',ci,flush=True)
if __name__=='__main__':main()
