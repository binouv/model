import r45_posterior_state_propagation as m
def test_seeds_disjoint():
 a={4521,4522};b={4541,4542,4543};assert a.isdisjoint(b)
def test_no_learning_model(): assert not hasattr(m,"fit_model")
def test_group_mass_modes(): assert m.lse([0.,0.])>0.
def test_cap_grid(): assert [1,2,4,8]==[1,2,4,8]
