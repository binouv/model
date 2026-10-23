"""Reproduce the pre-evaluation frozen GSM8K selection from public source bytes."""
import argparse,hashlib,json,random,re,zipfile
from pathlib import Path
from common import ROOT,sha
from tokenizer import Tokenizer
from prepare_data import norm
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT.parent/'ext_20261018/data/OpenCompassData-core-20240207.zip');p.add_argument('--alpaca',type=Path,default=ROOT.parent/'ext_20261018/data/alpaca_raw.json');a=p.parse_args()
    reg=json.loads((ROOT/'metrics/math_registration.json').read_text());assert sha(a.archive)=='c259fa1b2f8f5c9ae2df58bb6cc3200e705e2945b24e896988006c546a572607';assert sha(a.alpaca)=='2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419'
    raw=zipfile.ZipFile(a.archive).read('data/gsm8k/test.jsonl');assert hashlib.sha256(raw).hexdigest()==reg['source_sha256'];source=[json.loads(s) for s in raw.splitlines()]
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');indices=list(range(len(source)));random.Random(20261024).shuffle(indices)
    train=[json.loads(s) for s in (ROOT/'data/train.jsonl').read_text().splitlines()];texts=[norm(r['prompt']+' '+r['completion']) for r in train]
    alpaca=json.loads(a.alpaca.read_text());prior=json.loads((ROOT/'data/prior_ext_split_ids.json').read_text())
    for id in prior['train']:
        r=alpaca[int(id.split(':')[1])];user=r['instruction']+('\n'+r['input'] if r['input'] else '');texts.append(norm('User: '+user+'\nAssistant:  '+r['output']))
    chosen=[]
    for i in indices:
        r=source[i];q=norm(r['question'])
        if any(q in t for t in texts):continue
        prompt='User: Solve the math problem. Reply with only the final numeric answer.\n'+r['question']+'\nAssistant:'
        if len(tok.encode(prompt,bos=True))+64>2048:continue
        answer=r['answer'].split('####')[-1].strip();assert re.fullmatch(r'-?[\d,]+(?:\.\d+)?',answer)
        chosen.append({'id':'gsm8k:test:'+str(i),'source_index':i,'question':r['question'],'gold_numeric':answer,'prompt':prompt,'original':r})
        if len(chosen)==64:break
    encoded=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen).encode();assert hashlib.sha256(encoded).hexdigest()==reg['selected_sha256']
    out=ROOT/'data/test_gsm8k.jsonl'
    if out.exists():assert out.read_bytes()==encoded
    else:out.write_bytes(encoded)
    print('Exact GSM8K64 selection independently reproduced:',reg['selected_sha256'])
if __name__=='__main__':main()
