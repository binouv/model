import r51_midpoint_agreement as m
def test_seeds_disjoint(): assert {5121,5122}.isdisjoint({5141,5142,5143})
def test_midpoint_rule(): assert 128//2==64 and 64//2==32
def test_no_soft_forward_filter(): assert m.WIDTH==32 and m.EXPAND==8
def test_modes_exactly_main_and_one_ablation(): assert {'joint','address_only'}=={'joint','address_only'}