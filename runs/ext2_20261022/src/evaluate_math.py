"""Additional frozen GSM8K transfer check; never used for model selection."""
import json,re,time,csv
from decimal import Decimal
import torch
from common import ROOT,setup,load,rows,atomic_json,sha
from evaluate_final import wilson
def number(s):
    s=s.strip().replace(',','')
    return str(Decimal(s).normalize()) if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)',s) else None
def main():
    setup();cfg=json.loads((ROOT/'config.json').read_text());result=json.loads((ROOT/'metrics/final_results.json').read_text());assert result['evaluation_completed'];reg=json.loads((ROOT/'metrics/math_registration.json').read_text());assert sha(ROOT/'data/test_gsm8k.jsonl')==reg['selected_sha256']
    data=rows('test_gsm8k.jsonl');parent=(ROOT/cfg['parent_path']).resolve();step=result['selected_step'];selected=parent if step==0 else ROOT/'checkpoints'/f'step_{step:04d}'/'model.safetensors';summary={}
    for arm,weights in [('parent_EXT256',parent),('selected_EXT2',selected)]:
        m,tok=load(weights);m.eval();out=[];start=time.perf_counter()
        with torch.inference_mode(),(ROOT/f'metrics/{arm}_gsm8k_predictions.jsonl').open('x') as f:
            for r in data:
                with torch.autocast('cpu',dtype=torch.bfloat16):seq=m.generate(torch.tensor([tok.encode(r['prompt'],bos=True)]),max_new_tokens=64)[0].tolist()
                body=seq[:seq.index(2)] if 2 in seq else seq;text=tok.decode(body);parsed=number(text);matches=re.findall(r'[+-]?\d[\d,]*(?:\.\d+)?',text);last=number(matches[-1]) if matches else None;gold=number(r['gold_numeric']);assert gold is not None
                row={'id':r['id'],'prompt':r['prompt'],'gold_numeric':gold,'completion':text,'new_token_ids':seq,'strict_numeric':parsed,'last_number_diagnostic':last,'strict_correct':parsed==gold,'last_number_correct':last==gold,'eos':2 in seq,'budget_exhausted':len(seq)==64 and 2 not in seq};out.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
        k=sum(r['strict_correct'] for r in out);summary[arm]={'n':len(out),'strict_correct':k,'strict_accuracy':k/len(out),'strict_wilson95':wilson(k,len(out)),'strict_valid':sum(r['strict_numeric'] is not None for r in out),'last_number_correct_diagnostic':sum(r['last_number_correct'] for r in out),'budget_exhausted':sum(r['budget_exhausted'] for r in out),'seconds':time.perf_counter()-start,'weight_sha256':sha(weights)}
        print('MATH',arm,json.dumps(summary[arm]),flush=True);del m
    atomic_json(ROOT/'metrics/math_results.json',{'completed':True,'registration':reg,'selected_step':step,'summary':summary,'scope':'direct numeric answers under64-token budget; diagnostic last-number score is reported separately; no GSM8K training or selection'})
    with (ROOT/'metrics/math_results.csv').open('w',newline='') as f:
        fields=['model','n','strict_correct','strict_accuracy','strict_valid','last_number_correct_diagnostic','budget_exhausted','seconds'];w=csv.DictWriter(f,fields);w.writeheader()
        for arm,s in summary.items():w.writerow(dict(model=arm,**{k:s[k] for k in fields if k in s}))
if __name__=='__main__':main()
