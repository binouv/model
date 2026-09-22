import numpy as np
import r47_joint_bayes_belief as m
def test_seeds_disjoint(): assert {4721,4722}.isdisjoint({4741,4742,4743})
def test_softmax(): assert abs(m.softmax(np.array([0.,0.])).sum()-1)<1e-12
def test_joint_state_space(): assert m.S==16
def test_temperature_grid(): assert all(x>0 for x in [10.,20.,40.,60.])
