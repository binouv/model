"""Complete-only, paired raw-output scoring. Fixed gates are not relaxed."""
from __future__ import annotations
import json,csv,math,collections,hashlib
from pathlib import Path
from pointer_model import ROOT,atomic_json,sha
from data import read
from train import SPLITS,parse
def score(rr):
    n=len(rr);k=sum(z['exact'] for z in rr)
    return {'n':n,'correct':k,'accuracy':k/n if n else None}
def paired(a,b):
    aa={z['id']:bool(z['exact']) for z in a};bb={z['id']:bool(z['exact']) for z in b}
    assert len(aa)==len(a) and len(bb)==len(b) and set(aa)==set(bb)
    w=sum(aa[k] and not bb[k] for k in aa);l=sum(bb[k] and not aa[k] for k in aa);n=w+l
    p=min(1,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1.
    return {'n':len(a),'wins':w,'losses':l,'delta_pp':100*(w-l)/len(a),'two_sided_exact_sign_p':p}
def cf_pair_rows(rr):
    g=collections.defaultdict(list)
    for z in rr:
        if z['pair_id'] is not None:g[z['pair_id']].append(z)
    assert all(len(x)==2 for x in g.values())
    return [{'id':k,'exact':all(z['exact'] for z in v)} for k,v in sorted(g.items())]
def main():
    arms=['baseline','pointer','no_copy']
    cfg=json.loads((ROOT/'configs/preregistered.json').read_text())
    for arm in arms:
        if not (ROOT/f'metrics/{arm}_DONE.json').exists():raise RuntimeError('incomplete experiment: '+arm)
    allraw={};res={};csvrows=[]
    for arm in arms:
        res[arm]={};allraw[arm]={}
        for split in SPLITS:
            z=json.loads((ROOT/f'metrics/{arm}_{split}.json').read_text())
            g={x['id']:x for x in read(ROOT/f'data/{split}.jsonl')}
            assert z['status']=='completed' and len(z['rows'])==len(g)
            assert {x['id'] for x in z['rows']}==set(g)
            for x in z['rows']:
                val=parse(x['text'])
                assert x['gold']==g[x['id']]['answer']
                assert x['exact']==(x['eos'] and val is not None and val==int(x['gold']))
            allraw[arm][split]=z['rows'];res[arm][split]={k:v for k,v in z.items() if k!='rows'}
            csvrows.append({'arm':arm,'split':split,'n':z['n'],'correct':z['correct'],'accuracy':z['accuracy'],
              'cf_both_correct':z['both_correct_pairs'],'cf_pairs':z['pairs'],'truncated':z['truncated'],
              'seconds':z['seconds'],'tokens_generated':z['generated_tokens']})
    pair={s:paired(allraw['pointer'][s],allraw['no_copy'][s]) for s in SPLITS}
    subset=lambda rr:[z for z in rr if z['family'] in ['memory_update','list_reasoning']]
    pair['iid_memory_list']=paired(subset(allraw['pointer']['iid']),subset(allraw['no_copy']['iid']))
    pair['both_correct_cf']=paired(cf_pair_rows(allraw['pointer']['counterfactual']),cf_pair_rows(allraw['no_copy']['counterfactual']))
    gates={
       'iid_memory_list_gain_ge5pp':pair['iid_memory_list']['delta_pp']>=5,
       'both_correct_counterfactual_gain_ge5pp':pair['both_correct_cf']['delta_pp']>=5,
       'overall_iid_regression_le2pp':pair['iid']['delta_pp']>=-2,
       'surface_regression_le2pp':pair['surface']['delta_pp']>=-2}
    logs={a:read(ROOT/f'metrics/{a}_TRAIN.jsonl') for a in ['pointer','no_copy']}
    for a,rr in logs.items():assert [r['step'] for r in rr]==list(range(1,cfg['updates_per_arm']+1))
    assert all(a['batch_ids']==b['batch_ids'] and a['lr']==b['lr'] and a['padded_tokens']==b['padded_tokens']
       for a,b in zip(logs['pointer'],logs['no_copy']))
    out={'status':'completed','run':cfg['run'],'parameters_stored_per_arm':100127401,
      'functionally_active_parameters':cfg['functionally_active_parameters'],
      'trained_arms':2,'new_ancestor_seeds':0,'ancestral_initializations':1,
      'additional_optimizer_steps_per_arm':cfg['updates_per_arm'],
      'results':res,'paired_main_vs_control':pair,'gates':gates,
      'verdict':'EFFECT_GATES_MET_ON_THIS_SYNTHETIC_SUITE' if all(gates.values()) else 'NOT_CONFIRMED_IN_REGISTERED_PROTOCOL',
      'matched_minibatches_LR_shapes_all_steps':True,
      'training':{a:json.loads((ROOT/f'metrics/{a}_TRAINING.json').read_text()) for a in ['pointer','no_copy']},
      'scope':'Limited synthetic RU/EN mechanism test, not general pretraining or code-writing benchmark; Qwen was not run.',
      'Qwen_superiority_demonstrated':False,
      'data_case_dependence':'Counterfactual members are paired, baseline reused across models; not independent multiplied-N.',
      'full_training_compute':'Same executed graph and tensor shapes; zero copy gradients in control; hardware total FLOPs not fully measured.'}
    atomic_json(ROOT/'metrics/FINAL.json',out)
    with (ROOT/'metrics/FINAL.csv').open('w',newline='') as f:
        ww=csv.DictWriter(f,fieldnames=list(csvrows[0]));ww.writeheader();ww.writerows(csvrows)
    atomic_json(ROOT/'reports/NEXT_STATE.json',{'status':'completed','run':'L100P','result':out['verdict'],
       'parameters_stored':100127401,'ancestor':'L100Mstep1024BF16','new_updates_each':cfg['updates_per_arm'],
       'checkpoints':{a:'checkpoints/'+a+'/step_'+f"{cfg['updates_per_arm']:04d}" for a in ['pointer','no_copy']},
       'both_optimizer_RNG_saved':True,'do_not_duplicate':True,'new_weight_bytes_uploaded':False,
       'Qwen_goal_not_achieved':True,'L100C_parallel_not_touched':True})
    print(json.dumps({'status':'completed','verdict':out['verdict'],'results':{a:{s:[r['correct'],r['n']] for s,r in z.items()} for a,z in res.items()}},indent=2))
if __name__=='__main__':main()
