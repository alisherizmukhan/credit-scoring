"""Shared fixtures. Tests never need the real dataset: data is synthetic,
so they run in CI without Kaggle/OpenML access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scoring.config import TARGET, load_config

ROOT = Path(__file__).resolve().parents[1]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_gmsc(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the Give Me Some Credit schema and a learnable signal."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 85, n)
    util = rng.beta(1.2, 3.0, n) * 1.2
    income = rng.lognormal(8.6, 0.5, n)
    income[rng.random(n) < 0.15] = np.nan
    debt_ratio = rng.gamma(2.0, 0.2, n)
    pd30, pd60, pd90 = rng.poisson(0.3, n), rng.poisson(0.1, n), rng.poisson(0.1, n)
    codes = rng.random(n) < 0.01
    pd30[codes], pd60[codes], pd90[codes] = 98, 98, 98
    dependents = rng.poisson(0.8, n).astype(float)
    dependents[rng.random(n) < 0.03] = np.nan
    logit = (
        -3.2
        + 2.5 * util
        + 0.8 * np.minimum(pd30, 5)
        + 1.0 * np.minimum(pd90, 5)
        - 0.03 * (age - 45)
        + 0.5 * codes
    )
    y = (rng.random(n) < _sigmoid(logit)).astype(int)
    return pd.DataFrame(
        {
            "revolving_utilization": util,
            "age": age,
            "past_due_30_59": pd30,
            "debt_ratio": debt_ratio,
            "monthly_income": income,
            "open_credit_lines": rng.poisson(8, n),
            "times_90_days_late": pd90,
            "real_estate_loans": rng.poisson(1, n),
            "past_due_60_89": pd60,
            "dependents": dependents,
            TARGET: y,
        }
    )


def make_home_credit(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic application + bureau tables with the Home Credit schema."""
    rng = np.random.default_rng(seed)
    ids = np.arange(100000, 100000 + n)
    income = rng.lognormal(12, 0.4, n)
    credit = income * rng.uniform(1, 5, n)
    employed = -rng.integers(100, 8000, n)
    employed[rng.random(n) < 0.1] = 365243
    ext2 = rng.uniform(0, 1, n)
    ext3 = rng.uniform(0, 1, n)
    ext3[rng.random(n) < 0.2] = np.nan
    logit = -2.5 - 2.0 * (ext2 - 0.5) + 0.2 * (credit / income - 3)
    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": (rng.random(n) < _sigmoid(logit)).astype(int),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": credit / rng.uniform(10, 30, n),
            "AMT_GOODS_PRICE": credit * 0.9,
            "DAYS_BIRTH": -rng.integers(21 * 365, 68 * 365, n),
            "DAYS_EMPLOYED": employed,
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
        }
    )
    # every second client has 1..4 previous credits in the bureau
    rows = []
    for cid in ids[::2]:
        for _ in range(rng.integers(1, 5)):
            rows.append(
                {
                    "SK_ID_CURR": cid,
                    "CREDIT_ACTIVE": rng.choice(["Active", "Closed"]),
                    "DAYS_CREDIT": -int(rng.integers(1, 2900)),
                    "DAYS_CREDIT_ENDDATE": float(rng.integers(-2000, 2000)),
                    "CREDIT_DAY_OVERDUE": int(rng.integers(0, 3)),
                    "AMT_CREDIT_SUM": float(rng.lognormal(12, 0.8)),
                    "AMT_CREDIT_SUM_DEBT": float(rng.lognormal(10, 1.0)),
                    "AMT_CREDIT_SUM_OVERDUE": 0.0,
                    "CNT_CREDIT_PROLONG": int(rng.integers(0, 2)),
                }
            )
    return app, pd.DataFrame(rows)


@pytest.fixture
def gmsc_df() -> pd.DataFrame:
    return make_gmsc()


@pytest.fixture
def home_credit_tables():
    return make_home_credit()


def fast_config(tmp_path: Path) -> dict:
    """Project config with tiny CV/tuning and every output redirected to tmp_path."""
    db = (tmp_path / "mlflow.db").as_posix()
    return load_config(
        ROOT / "configs" / "config.yaml",
        {
            "paths": {
                "raw": str(tmp_path / "raw"),
                "processed": str(tmp_path / "processed"),
                "models": str(tmp_path / "models"),
                "reports": str(tmp_path / "reports"),
            },
            "cv": {"n_splits": 2},
            "tuning": {
                "cv_splits": 2,
                "sample_size": None,
                "timeout_sec": 60,
                "n_trials": {"lightgbm": 2, "catboost": 1},
            },
            "explain": {"sample_size": 150},
            "mlflow": {
                "tracking_uri": f"sqlite:///{db}",
                "artifact_location": (tmp_path / "mlruns").as_uri(),
                "experiment": "test-credit-scoring",
            },
        },
    )


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Run the real training routine once on synthetic data (session-wide)."""
    from credit_scoring.train import run_training

    tmp = tmp_path_factory.mktemp("train")
    cfg = fast_config(tmp)
    summary = run_training(cfg, df=make_gmsc(1500, seed=1), dataset="gmsc")
    return {"cfg": cfg, "summary": summary, "model_dir": Path(cfg["paths"]["models"])}
