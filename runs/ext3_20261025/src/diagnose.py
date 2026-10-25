import datetime,json,torch
from common import ROOT,setup,load,sha,atomic_json
from objective import configure,balanced_dev
setup();parent=ROOT.parent/'ext2_20261022/checkpoints/step_3584/model.safetensors';assert sha(parent)=='b79907ef591f16f73ac98c9c4774f02d26cfc404b12272587cb0e4900530effe'
m,tok=load(parent);params=configure(m);s=balanced_dev(m,tok,'parent_EXT2')
atomic_json(ROOT/'metrics/diagnosis.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_sha256=sha(parent),strict_load_135_tensors_passed=True,parameters=sum(p.numel() for p in m.parameters()),proposed_trainable_parameters=sum(p.numel() for p in params),proposed_trainable_prefixes=['blocks.12.','blocks.13.','norm.'],balanced_dev=s))
print(json.dumps(s,indent=2),flush=True)
