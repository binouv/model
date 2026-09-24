import inspect
import r60_selective_fallback as m

def test_trigger_oracle_free():
    s=inspect.getsource(m.select_step)
    assert "['target']" not in s and "['idx']" not in s

def test_seed_split():
    assert set(m.CAL_SEEDS).isdisjoint(m.HELD_SEEDS)

def test_ecc256_unique_address_capacity():
    assert m.EXT_CONDS['ood256ecc'][1] <= 512
