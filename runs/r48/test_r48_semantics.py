import numpy as np
import r48_sparse_joint_belief as m
def test_seeds_disjoint(): assert {4821,4822}.isdisjoint({4841,4842,4843})
def test_truncate_mass():
 a=np.array([[.6,.3,.1]]);b,k=m.truncate(a,2);assert abs(b.sum()-1)<1e-12 and abs(k-.9)<1e-12
def test_k_grid(): assert [16,32,64,128,256]==[16,32,64,128,256]
