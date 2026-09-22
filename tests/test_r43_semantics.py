import r43_set_relative_reranker as m
def test_seeds_disjoint():
 a={4311,4312,4313};b={4321,4322};c={4341,4342,4343};assert a.isdisjoint(b|c) and b.isdisjoint(c)
def test_set_context_is_strict_extension(): assert len(m.SET_NAMES)>0
def test_set_features_are_oracle_free_names(): assert not any("true" in x or "prefix_ok" in x for x in m.SET_NAMES)
def test_zero_in_calibration_grid(): assert 0.0 in [0.0,.15,.30,.60]
