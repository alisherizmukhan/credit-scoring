import numpy as np
import pytest

from credit_scoring.evaluate import (
    business_costs,
    business_summary,
    classification_metrics,
    comparison_table,
    expected_cost,
    find_optimal_threshold,
    gini,
    ks_statistic,
    plot_calibration,
    plot_cost_curve,
    plot_roc,
)

BUSINESS = {"loan_amount": 1000, "lgd": 0.5, "profit_margin": 0.1}


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    return y, p


def test_ks_and_gini_on_perfect_separation():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert ks_statistic(y, p) == pytest.approx(1.0)
    assert gini(y, p) == pytest.approx(1.0)


def test_metrics_on_random_scores_are_near_chance(calibrated):
    y, _ = calibrated
    rng = np.random.default_rng(1)
    m = classification_metrics(y, rng.uniform(size=y.size))
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.02)
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert set(m) == {"roc_auc", "gini", "pr_auc", "ks", "brier", "log_loss"}


def test_business_costs():
    assert business_costs(BUSINESS) == (500.0, 100.0)


def test_expected_cost_counts_errors():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.9, 0.1])  # with t=0.5: FN at idx 0, FP at idx 1
    assert expected_cost(y, p, 0.5, cost_fn=10, cost_fp=1) == pytest.approx((10 + 1) / 4)


def test_optimal_threshold_matches_theory_for_calibrated_scores(calibrated):
    y, p = calibrated
    cost_fn, cost_fp = 5.0, 1.0
    t, _ = find_optimal_threshold(y, p, cost_fn, cost_fp)
    # Bayes-optimal decision: approve if PD < cost_fp / (cost_fp + cost_fn)
    assert t == pytest.approx(cost_fp / (cost_fp + cost_fn), abs=0.03)


def test_business_summary(calibrated):
    y, p = calibrated
    s = business_summary(y, p, 0.2, BUSINESS)
    assert s["approval_rate"] == pytest.approx(np.mean(p < 0.2))
    assert 0 <= s["bad_rate_approved"] <= 0.2
    assert s["expected_cost_per_1000_apps"] == pytest.approx(
        s["credit_loss_per_1000_apps"] + s["lost_profit_per_1000_apps"]
    )
    reject_all = business_summary(y, p, 0.0, BUSINESS)
    assert reject_all["approval_rate"] == 0 and reject_all["bad_rate_approved"] == 0


def test_plots_are_written(tmp_path, calibrated):
    y, p = calibrated
    curves = {"model": (y[:2000], p[:2000])}
    plot_calibration(curves, tmp_path / "cal.png")
    plot_roc(curves, tmp_path / "roc.png")
    plot_cost_curve(y[:2000], p[:2000], 5, 1, 0.17, tmp_path / "cost.png")
    for name in ["cal.png", "roc.png", "cost.png"]:
        assert (tmp_path / name).stat().st_size > 0


def test_comparison_table():
    row = {
        "model": "lightgbm",
        "cv_roc_auc_mean": 0.86,
        "cv_roc_auc_std": 0.004,
        "test": {"roc_auc": 0.87, "gini": 0.74, "pr_auc": 0.41, "ks": 0.59, "brier": 0.048},
    }
    table = comparison_table([row])
    assert table.startswith("| Model |")
    assert "| lightgbm | 0.8600 ± 0.0040 | 0.8700 |" in table
