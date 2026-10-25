"""Check new label likelihood/target alignment on published dev records."""
import json
import torch
from common import ROOT,setup,load,rows,encode,score_labels,atomic_json,sha
def main():
    setup();cfg=json.loads((ROOT/'config.json').read_text());sel=json.loads((ROOT/'metrics/selection.json').read_text())
    weights=(ROOT/cfg['parent_path']).resolve() if sel['best_step']==0 else ROOT/'checkpoints'/f"step_{sel['best_step']:04d}"/'model.safetensors'
    m,tok=load(weights);m.eval();checks=[]
    with torch.inference_mode():
        for task in ['openbookqa','piqa','arc_challenge']:
            r=next(r for r in rows('dev_mcq.jsonl') if r['task']==task);x,y,n=encode(tok,r)
            full=float(m(x,targets=y));scores,eos=score_labels(m,tok,r);cached=-eos[r['gold_index']]/n
            assert abs(full-cached)<1e-4,(task,full,cached)
            ids=torch.tensor([tok.encode(r['prompt'],bos=True)]);lg,cache=m(ids,last_only=True);diffs=[];argmax=[]
            for _ in range(4):
                ref,_=m(ids,last_only=True);diffs.append(float((ref-lg).abs().max()));argmax.append(int(ref[0,-1].argmax())==int(lg[0,-1].argmax()))
                nxt=int(lg[0,-1].argmax())
                if nxt==2:break
                ids=torch.cat((ids,torch.tensor([[nxt]])),1);lg,cache=m(torch.tensor([[nxt]]),cache=cache,last_only=True)
            assert max(diffs)<.002 and all(argmax)
            checks.append({'id':r['id'],'task':task,'full_teacher_forced_nll':full,'cached_label_and_eos_nll':cached,'loss_abs_difference':abs(full-cached),'cache_full_max_abs_logit_differences':diffs,'greedy_argmax_equal':argmax})
    atomic_json(ROOT/'metrics/protocol_verification.json',{'status':'passed','selected_model_sha256':sha(weights),'checks':checks,'input_and_target_alignment_passed':True,'test_used':False})
    print('Protocol alignment and cache checks passed',flush=True)
if __name__=='__main__':main()
