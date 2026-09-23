import r50_backward_consistency as m
def test_seeds_disjoint():
 a={5021,5022};b={5041,5042,5043};assert a.isdisjoint(b)
def test_no_soft_forward_filter(): assert m.WIDTH==32 and m.EXPAND==8
def test_backward_is_reverse_program(): assert m.REV_WIDTH>0 and m.REV_EXPAND>0
def test_modes_exactly_main_and_one_ablation(): assert {'anchored','address_only'}=={'anchored','address_only'}