from eval.sweep.sweep_utils import make_grid, subsample


def test_make_grid_is_full_cartesian_with_stable_ids():
    grid = make_grid(lrs=[5e-5, 1e-4, 2e-4], ranks=[8, 16, 32], seed=13)
    assert len(grid) == 9
    # alpha is always 2 * rank
    assert all(c["alpha"] == 2 * c["rank"] for c in grid)
    # every config carries the seed and a unique, stable id
    ids = [c["config_id"] for c in grid]
    assert len(set(ids)) == 9
    assert all(c["seed"] == 13 for c in grid)
    assert "lr0.0001_r16" in ids


def test_subsample_is_deterministic_and_bounded():
    records = [{"i": i} for i in range(200)]
    a = subsample(records, n=70, seed=13)
    b = subsample(records, n=70, seed=13)
    assert len(a) == 70
    assert a == b                      # deterministic for a fixed seed
    assert a != subsample(records, n=70, seed=99)  # seed actually matters


def test_subsample_returns_all_when_n_none_or_larger():
    records = [{"i": i} for i in range(10)]
    assert subsample(records, n=None, seed=1) == records
    assert subsample(records, n=50, seed=1) == records
