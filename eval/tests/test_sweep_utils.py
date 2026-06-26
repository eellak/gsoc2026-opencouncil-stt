from eval.sweep.sweep_utils import (
    build_leaderboard,
    make_grid,
    pick_best,
    render_markdown,
    subsample,
)


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


def test_build_leaderboard_sorts_and_adds_regression_delta():
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0}
    rows = [
        {"config_id": "a", "lr": 1e-4, "rank": 16, "alpha": 32, "epoch": 2,
         "val_corr_wer_norm": 26.0, "val_reg_wer": 17.0, "val_corr_cer": 10.0,
         "train_loss": 0.4, "wall_s": 600},
        {"config_id": "b", "lr": 2e-4, "rank": 32, "alpha": 64, "epoch": 4,
         "val_corr_wer_norm": 24.0, "val_reg_wer": 30.0, "val_corr_cer": 9.0,
         "train_loss": 0.2, "wall_s": 900},
    ]
    lb = build_leaderboard(rows, baseline)
    # sorted ascending by val_corr_wer_norm -> 'b' (24.0) first
    assert [r["config_id"] for r in lb] == ["b", "a"]
    # reg_delta = val_reg_wer - baseline val_reg_wer
    assert lb[0]["reg_delta"] == 3.0    # b regressed: 30 - 27
    assert lb[1]["reg_delta"] == -10.0  # a improved: 17 - 27
    # original rows are not mutated
    assert "reg_delta" not in rows[0]


def _sample_lb():
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0}
    rows = [
        {"config_id": "b", "lr": 2e-4, "rank": 32, "alpha": 64, "epoch": 4,
         "val_corr_wer_norm": 24.0, "val_reg_wer": 30.0, "val_corr_cer": 9.0,
         "train_loss": 0.2, "wall_s": 900},
        {"config_id": "a", "lr": 1e-4, "rank": 16, "alpha": 32, "epoch": 2,
         "val_corr_wer_norm": 26.0, "val_reg_wer": 17.0, "val_corr_cer": 10.0,
         "train_loss": 0.4, "wall_s": 600},
    ]
    return build_leaderboard(rows, baseline)


def test_pick_best_skips_configs_that_regress_val_reg():
    lb = _sample_lb()
    # 'b' has the lowest val_corr_wer_norm but reg_delta +3.0 > 1.0 -> skipped
    best = pick_best(lb, max_reg_delta=1.0)
    assert best["config_id"] == "a"


def test_pick_best_returns_none_when_all_regress():
    lb = _sample_lb()
    assert pick_best(lb, max_reg_delta=-100.0) is None


def test_render_markdown_contains_table_and_best_line():
    lb = _sample_lb()
    md = render_markdown(lb, pick_best(lb, max_reg_delta=1.0))
    assert "| config_id |" in md
    assert "lr1e-04_r16" not in md   # ids in fixture are 'a'/'b'
    assert "**Best (regression-guarded):** a" in md
    assert "24.0" in md              # values rendered
