"""Reuse the byte-verified previous BF16 runner with explicit fixture amendments."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/binouv/model/b09de746a12f882e9249b4d5018937f23646fbb4/runs/l100p_20261009/bench/qwen08_reference.py'
with urllib.request.urlopen(URL,timeout=60) as response:
    raw=response.read()
assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='994cffab4c137bf5e5bb4b129241f30dd88e12a3'
source=raw.decode()
amendments={"assert len(rows)==20":"assert len(rows)==12","'planned_requests':20":"'planned_requests':12","'seed':92201+i":"'seed':1101210+i","L100P: save actual Qwen0.8 BF16 thinking reference outputs; no training":"L100S: save actual fixed12 Qwen0.8 BF16 thinking outputs; no training"}
for old,new in amendments.items():
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
ns={'__name__':'reference_module','__file__':str(ROOT/'verified_base.py')}
exec(compile(source,'verified_base.py','exec'),ns)
ns['ROOT']=ROOT
ns['BRANCH']='flygraph/l100s-qwen-reference-20261011'
ns['FIXTURE']='ba2f624c62eea62fe7487f33b8461f0fe3c7bedb60d4f80c59bcbfe1c1d2046f'
ns['PREFIX']=''
(ROOT/'outputs').mkdir(exist_ok=True)
(ROOT/'outputs/RUNNER_AUDIT.json').write_text(json.dumps({'upstream_git_blob':'994cffab4c137bf5e5bb4b129241f30dd88e12a3','upstream_source_sha256':hashlib.sha256(raw).hexdigest(),'patched_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'amendments':amendments,'extra_instruction_prefix':'','own_model_training':False},indent=2))
ns['main']()
