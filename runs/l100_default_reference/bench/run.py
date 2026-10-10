"""Fixed32 non-thinking companion; reference inference only, not student training."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE='https://raw.githubusercontent.com/binouv/model/'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=90) as response:return response.read()
raw=get('b09de746a12f882e9249b4d5018937f23646fbb4/runs/l100p_20261009/bench/qwen08_reference.py')
assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='994cffab4c137bf5e5bb4b129241f30dd88e12a3'
mem=get('1567186b34dc52f595ee8fff327de9d2b347b07c/runs/l100s_20261011/bench/PROMPTS.jsonl')
old=get('b09de746a12f882e9249b4d5018937f23646fbb4/runs/l100p_20261009/bench/PROMPTS.jsonl')
assert hashlib.sha256(mem).hexdigest()=='ba2f624c62eea62fe7487f33b8461f0fe3c7bedb60d4f80c59bcbfe1c1d2046f'
assert hashlib.sha256(old).hexdigest()=='74bfffeb6cf3014b58d14cc1402fb8e9fca28818a0962d5deeb9140e792de26c'
rows=[json.loads(line) for line in mem.decode().splitlines()+old.decode().splitlines()]
assert len(rows)==32 and len({r['id'] for r in rows})==32
fixture=''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in rows).encode()
(ROOT/'PROMPTS.jsonl').write_bytes(fixture)
source=raw.decode()
amendments={
 "assert len(rows)==20":"assert len(rows)==32",
 "'planned_requests':20":"'planned_requests':32",
 "'seed':92201+i":"'seed':1101400+i",
 "'thinking':True":"'thinking':False",
 "'native_open_think_prefill':True":"'native_open_think_prefill':False",
 "'top_p':.95":"'top_p':1.0",
 "'presence_penalty':1.5":"'presence_penalty':2.0",
 "'<|im_start|>user\\n'+PREFIX+z['prompt']+'<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'":"'<|im_start|>user\\n'+PREFIX+z['prompt'].strip()+'<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'",
 "assert '</think>' not in prompt":"assert prompt.endswith('<think>\\n\\n</think>\\n\\n')",
 "text=response['content'];closed='</think>' in text":"text=response['content'];closed=None",
 "'final_text':text.split('</think>',1)[1].strip() if closed else None":"'final_text':text",
 "'nonempty_thinking':bool(text.split('</think>',1)[0].strip())":"'nonempty_thinking':False",
 " or not x['thinking_closed']":"",
 "L100P: save actual Qwen0.8 BF16 thinking reference outputs; no training":"L100 default-mode reference: save actual fixed32 BF16 responses; no training"
}
for a,b in amendments.items():
    assert source.count(a)==1,(a,source.count(a))
    source=source.replace(a,b)
ns={'__name__':'reference_module','__file__':str(ROOT/'verified_base.py')}
exec(compile(source,'verified_base.py','exec'),ns)
ns.update(ROOT=ROOT,BRANCH='flygraph/l100-default-reference-20261011',FIXTURE=hashlib.sha256(fixture).hexdigest(),PREFIX='')
(ROOT/'outputs').mkdir(exist_ok=True)
(ROOT/'outputs/RUNNER_AUDIT.json').write_text(json.dumps({'mode':'official default non-thinking','upstream_git_blob':'994cffab4c137bf5e5bb4b129241f30dd88e12a3','patched_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'amendments':amendments,'fixture_sha256':hashlib.sha256(fixture).hexdigest(),'own_model_training':False,'scope':'12 fixed memory tasks plus20 historical multitype held tasks; companion not a replacement for thinking reference'},indent=2))
ns['main']()
