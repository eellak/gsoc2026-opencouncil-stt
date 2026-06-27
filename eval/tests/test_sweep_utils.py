from eval.sweep.sweep_utils import (
    _disp,
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


def test_reg_delta_is_none_and_best_passes_when_val_reg_missing():
    # No val_reg set (NaN) -> reg_delta unavailable; the guard must not reject everything.
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": float("nan")}
    rows = [{"config_id": "x", "lr": 1e-4, "rank": 16, "alpha": 32, "epoch": 1,
             "val_corr_wer_norm": 25.0, "val_reg_wer": float("nan"),
             "val_corr_cer": 9.0, "eval_loss": 0.3, "wall_s": 100}]
    lb = build_leaderboard(rows, baseline)
    assert lb[0]["reg_delta"] is None
    assert pick_best(lb, max_reg_delta=1.0)["config_id"] == "x"


def test_render_markdown_contains_table_and_best_line():
    lb = _sample_lb()
    md = render_markdown(lb, pick_best(lb, max_reg_delta=1.0))
    assert "| config_id |" in md
    assert "lr1e-04_r16" not in md   # ids in fixture are 'a'/'b'
    assert "**Best (regression-guarded):** a" in md
    assert "24.0" in md              # values rendered


def test_disp_keeps_small_learning_rates_readable():
    # Bug: round(2e-4, 3) collapses to 0.0 -> the lr column showed 0.0 for every config.
    assert _disp(2e-4) == 0.0002
    assert _disp(1e-4) == 0.0001
    assert _disp(5e-5) == 5e-05
    # Normal-magnitude values are still rounded to 3dp, and 0.0 stays 0.0.
    assert _disp(20.4823) == 20.482
    assert _disp(0.0) == 0.0
    assert _disp(8) == 8             # ints pass through unchanged


def test_render_markdown_renders_real_learning_rates_not_zero():
    grid_rows = build_leaderboard(
        [{"config_id": "lr0.0002_r32", "lr": 2e-4, "rank": 32, "alpha": 64,
          "epoch": 3, "val_corr_wer_norm": 20.482, "val_reg_wer": 25.0,
          "val_corr_cer": 14.8, "eval_loss": 0.9, "wall_s": 161}],
        {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0},
    )
    md = render_markdown(grid_rows, pick_best(grid_rows, 1.0))
    assert "| 0.0002 | 32 |" in md   # the lr cell renders the real lr, not 0.0


def test_render_markdown_includes_baseline_row_and_eval_set_sizes():
    lb = _sample_lb()
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0}
    counts = {"n_corr_clips": 70, "n_corr_words": 1234,
              "n_reg_clips": 16, "n_reg_words": 512}
    md = render_markdown(lb, pick_best(lb, 1.0), baseline=baseline, counts=counts)
    assert "BASELINE" in md          # baseline anchor row present
    assert "33.0" in md              # baseline corr WER shown
    assert "70 clips" in md and "1234 words" in md   # corr eval-set size
    assert "16 clips" in md and "512 words" in md     # reg eval-set size


def test_disp_handles_nan_none_and_negative_small_floats():
    import math
    assert math.isnan(_disp(float("nan")))   # NaN survives (no formatting crash)
    assert _disp(None) is None               # non-floats pass through
    assert _disp(-2e-4) == -0.0002           # negative small lrs keep sig-figs


def test_render_markdown_baseline_row_renders_zero_reg_delta():
    lb = _sample_lb()
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0}
    md = render_markdown(lb, pick_best(lb, 1.0), baseline=baseline)
    assert "| BASELINE |" in md      # baseline row rendered
    assert "| 0.0 |" in md           # its reg_delta cell is 0.0 and renders fine
