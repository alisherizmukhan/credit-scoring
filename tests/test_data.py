import pandas as pd
import pytest

from credit_scoring import data
from credit_scoring.config import TARGET, deep_update, load_config
from tests.conftest import ROOT


def test_prepare_gmsc_from_kaggle_layout(tmp_path):
    raw = pd.DataFrame(
        {
            "SeriousDlqin2yrs": [0, 1],
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.9],
            "age": [45, 30],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 2],
            "DebtRatio": [0.3, 0.8],
            "MonthlyIncome": [5000, None],
            "NumberOfOpenCreditLinesAndLoans": [5, 3],
            "NumberOfTimes90DaysLate": [0, 1],
            "NumberRealEstateLoansOrLines": [1, 0],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 0],
            "NumberOfDependents": [2, None],
        }
    )
    path = tmp_path / "cs.csv"
    raw.to_csv(path, index=False)
    df = data.prepare_gmsc(path)
    assert list(df.columns) == data.GMSC_FEATURES + [TARGET]
    assert df[TARGET].tolist() == [0, 1]


def test_prepare_home_credit(tmp_path, home_credit_tables):
    app, bureau = home_credit_tables
    app.to_csv(tmp_path / "application_train.csv", index=False)
    bureau.to_csv(tmp_path / "bureau.csv", index=False)
    df = data.prepare_home_credit(tmp_path)
    assert len(df) == len(app)
    assert TARGET in df.columns and "BUREAU_LOAN_COUNT" in df.columns


def test_save_and_load_processed(tmp_path, gmsc_df):
    data.save_processed(gmsc_df, "gmsc", tmp_path)
    df, meta = data.load_processed(tmp_path)
    pd.testing.assert_frame_equal(df, gmsc_df)
    assert meta["dataset"] == "gmsc" and meta["id_column"] is None
    assert meta["default_rate"] == pytest.approx(gmsc_df[TARGET].mean())


def test_load_processed_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_processed(tmp_path)


def test_dataset_resolution(monkeypatch, tmp_path):
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    assert not data.kaggle_credentials_available()
    assert data.resolve_dataset("auto") == "gmsc"

    monkeypatch.setenv("KAGGLE_USERNAME", "user")
    monkeypatch.setenv("KAGGLE_KEY", "key")
    assert data.kaggle_credentials_available()
    assert data.resolve_dataset("auto") == "home_credit"
    assert data.resolve_dataset("gmsc") == "gmsc"
    with pytest.raises(ValueError):
        data.resolve_dataset("mnist")


def test_auto_falls_back_to_gmsc_when_home_credit_fails(monkeypatch, tmp_path, gmsc_df):
    def fail(_):
        raise data.DataUnavailableError("no access")

    csv = tmp_path / "gmsc.csv"
    gmsc_df.to_csv(csv, index=False)
    monkeypatch.setattr(data, "resolve_dataset", lambda _: "home_credit")
    monkeypatch.setattr(data, "download_home_credit", fail)
    monkeypatch.setattr(data, "download_gmsc", lambda _: csv)
    cfg = {"dataset": "auto", "paths": {"raw": str(tmp_path / "raw")}}
    df, dataset = data.prepare_dataset(cfg)
    assert dataset == "gmsc" and len(df) == len(gmsc_df)


def test_config_helpers(monkeypatch):
    merged = deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    cfg = load_config(ROOT / "configs" / "config.yaml")
    assert cfg["mlflow"]["tracking_uri"] == "http://mlflow:5000"
    assert cfg["dataset"] in {"auto", "gmsc", "home_credit"}
