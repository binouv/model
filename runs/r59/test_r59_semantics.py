import inspect, json
import r59_value_distillation as m

def test_seed_splits():
    assert set(m.TRAIN_SEEDS).isdisjoint(m.CAL_SEEDS)
    assert set(m.TRAIN_SEEDS).isdisjoint(m.HELD_SEEDS)
    assert set(m.CAL_SEEDS).isdisjoint(m.HELD_SEEDS)

def test_oracle_free_feature():
    s=inspect.getsource(m.candidate_feature)
    assert "['target']" not in s and "['idx']" not in s

def test_model_shape_contract():
    assert m.TRAIN_SEEDS==(5911,5912,5913)
