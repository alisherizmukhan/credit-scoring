import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import TARGET
from credit_scoring.drift import psi, psi_report, psi_status, simulate_drift, to_markdown


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 1, 20_000))


def test_psi_is_zero_for_same_distribution(reference):
    rng = np.random.default_rng(1)
    assert psi(reference, pd.Series(rng.normal(0, 1, 20_000))) < 0.01


def test_psi_detects_shift(reference):
    rng = np.random.default_rng(2)
    small = psi(reference, pd.Series(rng.normal(0.2, 1, 20_000)))
    large = psi(reference, pd.Series(rng.normal(1.0, 1, 20_000)))
    assert small < large
    assert large > 0.25


def test_psi_counts_missing_values_as_a_bin(reference):
    cur = reference.copy()
    cur.iloc[: len(cur) // 3] = np.nan
    assert psi(reference, cur) > 0.25


def test_psi_edge_cases():
    assert psi(pd.Series([np.nan, np.nan]), pd.Series([1.0, 2.0])) == 0.0
    assert psi(pd.Series([5.0] * 100), pd.Series([5.0] * 100)) == pytest.approx(0.0)


def test_psi_status_thresholds():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "moderate"
    assert psi_status(0.40) == "significant"


def test_report_and_simulation(gmsc_df):
    X = gmsc_df.drop(columns=TARGET)
    features = ["age", "monthly_income", "revolving_utilization", "missing_feature"]
    stable = psi_report(X.iloc[:1000], X.iloc[1000:], features)
    assert set(stable["feature"]) == {"age", "monthly_income", "revolving_utilization"}
    assert (stable["status"] == "stable").all()

    shifted = psi_report(X, simulate_drift(X, "gmsc"), features)
    assert shifted["psi"].is_monotonic_decreasing
    assert (shifted["status"] != "stable").any()
    assert to_markdown(shifted).startswith("| Feature | PSI | Status |")


def test_simulation_home_credit(home_credit_tables):
    app, _ = home_credit_tables
    shifted = simulate_drift(app, "home_credit")
    assert (shifted["AMT_CREDIT"] > app["AMT_CREDIT"]).all()
