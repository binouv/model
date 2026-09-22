import numpy as np
import r41_pairwise_disagreement as m

def test_seed_disjoint():
 c={"train":{4111,4112,4113},"cal":{4121,4122},"held":{4141,4142,4143}}
 assert c["train"].isdisjoint(c["cal"]|c["held"]) and c["cal"].isdisjoint(c["held"])

def test_inference_feature_no_oracle_args():
 import inspect
 s=inspect.signature(m.pair_feature)
 assert "ep" not in s.parameters and "idx" not in s.parameters

def test_horizon_feature_superset():
 a={"x":np.zeros(len(m.q.PREFIX_FEATURE_NAMES)),"base":0.,"combined":0.,"rsum":0.,"corr":0.}
 b={"x":np.ones(len(m.q.PREFIX_FEATURE_NAMES)),"base":1.,"combined":1.,"rsum":0.,"corr":0.}
 x0=m.pair_feature(a,b,.1,.2,1,64,False); x1=m.pair_feature(a,b,.1,.2,1,64,True)
 assert len(x1)>len(x0)
