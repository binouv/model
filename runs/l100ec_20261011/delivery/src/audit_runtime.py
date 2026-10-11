"""Actual restored-weight audit. Engineering replay, not new held benchmark."""
from __future__ import annotations
import argparse,gc,json,time
import torch
from runtime import ROOT,load_model,generate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['curriculum','shuffled'],required=True);args=ap.parse_args();arm=args.arm
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    prompts=['Calculate 5 + 8.\nAnswer:','Writes in time order: a=5; b=7; a=9. Latest value of a?\nAnswer:','Find the minimum of [7, 3, 9].\nAnswer:']
    m,t,s=load_model(ROOT,arm);start=time.perf_counter()
    finite=all(bool(torch.isfinite(p).all()) for p in m.parameters());assert finite
    with torch.inference_mode():
        ids=torch.tensor([t.encode('Memory: a=5; b=7. Read a.',bos=True)])
        logits,_=m(ids);_,cache=m(ids[:,:-1]);cached,_=m(ids[:,-1:],cache=cache)
        diff=float((logits[:,-1:]-cached).abs().max());assert diff<1e-3
    first=[generate(m,t,p,48) for p in prompts]
    del m,logits,cached,cache;gc.collect()
    m,t,_=load_model(ROOT,arm);second=[generate(m,t,p,48) for p in prompts];assert first==second
    report={'status':'completed','scope':'BF16 artifact reload and generation correctness, not quality evaluation','torch':torch.__version__,'arm':arm,'new_optimizer_updates':0,'parameters':sum(p.numel() for p in m.parameters()),'all_finite':finite,'strict_state_dict_load':True,'cache_max_abs_difference_fp32':diff,'reload_queries':len(prompts),'reload_mismatches':0,'eos_count':sum(x['eos'] for x in first),'seconds':time.perf_counter()-start,'rows':first,'weight_sha256':s['models'][arm]['sha256'],'original_FP32_and_optimizer_recovered':False}
    (ROOT/f'reports/{arm}_RUNTIME_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('AUDITED',arm,'reload matches',len(prompts),flush=True)
if __name__=='__main__':main()
