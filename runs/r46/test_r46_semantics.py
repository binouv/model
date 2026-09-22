import r46_distributional_address_belief as m
def test_seeds_disjoint(): assert {4621,4622}.isdisjoint({4641,4642,4643})
def test_overlap_identity(): assert abs(m.overlap({1:.7,2:.3},{1:.7,2:.3})-1)<1e-12
def test_overlap_disjoint(): assert m.overlap({1:1.},{2:1.})==0.
def test_topk(): assert m.TOPK==4
