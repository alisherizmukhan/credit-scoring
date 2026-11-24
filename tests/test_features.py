import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_scoring.config import TARGET
from credit_scoring.features import (
    FeatureEngineer,
    QuantileClipper,
    aggregate_bureau,
    build_model_pipeline,
    build_preprocessor,
    categorical_columns,
    merge_bureau,
    safe_div,
)


def test_safe_div_returns_nan_on_zero_and_missing():
    out = safe_div(pd.Series([10, 5, 3]), pd.Series([2, 0, np.nan]))
    assert out.iloc[0] == 5
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])
    assert not np.isinf(out).any()


def test_gmsc_engineering(gmsc_df):
    X = gmsc_df.drop(columns=TARGET)
    out = FeatureEngineer("gmsc").fit_transform(X)
    for col in [
        "total_past_due",
        "monthly_debt",
        "income_per_person",
        "non_real_estate_lines",
        "income_missing",
        "delinquency_code",
    ]:
        assert col in out.columns
    # reporting codes 96/98 become NaN, and the flag keeps the information
    coded = X["past_due_30_59"] >= 90
    assert out.loc[coded, "past_due_30_59"].isna().all()
    assert (out.loc[coded, "delinquency_code"] == 1).all()
    assert (out["income_missing"] == X["monthly_income"].isna()).all()
    # does not modify the input frame
    assert "total_past_due" not in X.columns


def test_home_credit_engineering(home_credit_tables):
    app, _ = home_credit_tables
    X = app.drop(columns=["TARGET", "SK_ID_CURR"])
    out = FeatureEngineer("home_credit").fit_transform(X)
    anomaly = X["DAYS_EMPLOYED"] == 365243
    assert out.loc[anomaly, "DAYS_EMPLOYED"].isna().all()
    assert (out.loc[anomaly, "DAYS_EMPLOYED_ANOM"] == 1).all()
    np.testing.assert_allclose(out["CREDIT_INCOME_RATIO"], X["AMT_CREDIT"] / X["AMT_INCOME_TOTAL"])
    np.testing.assert_allclose(
        out["ANNUITY_INCOME_RATIO"], X["AMT_ANNUITY"] / X["AMT_INCOME_TOTAL"]
    )
    assert {"EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "GOODS_CREDIT_RATIO"} <= set(out.columns)


def test_feature_engineer_unknown_dataset():
    with pytest.raises(ValueError):
        FeatureEngineer("unknown").fit(pd.DataFrame({"a": [1]}))


def test_bureau_aggregation(home_credit_tables):
    _, bureau = home_credit_tables
    agg = aggregate_bureau(bureau)
    assert agg["SK_ID_CURR"].is_unique
    cid = bureau["SK_ID_CURR"].iloc[0]
    rows = bureau[bureau["SK_ID_CURR"] == cid]
    rec = agg.set_index("SK_ID_CURR").loc[cid]
    assert rec["BUREAU_LOAN_COUNT"] == len(rows)
    assert rec["BUREAU_ACTIVE_COUNT"] == (rows["CREDIT_ACTIVE"] == "Active").sum()
    assert rec["BUREAU_AMT_CREDIT_SUM_MAX"] == pytest.approx(rows["AMT_CREDIT_SUM"].max())
    assert rec["BUREAU_DAYS_CREDIT_MEAN"] == pytest.approx(rows["DAYS_CREDIT"].mean())


def test_merge_bureau_fills_counts_for_clients_without_history(home_credit_tables):
    app, bureau = home_credit_tables
    merged = merge_bureau(app, aggregate_bureau(bureau))
    assert len(merged) == len(app)
    no_history = ~merged["SK_ID_CURR"].isin(bureau["SK_ID_CURR"])
    assert no_history.any()
    assert (merged.loc[no_history, "BUREAU_LOAN_COUNT"] == 0).all()
    # amounts stay missing: "no history" is not the same as "zero debt"
    assert merged.loc[no_history, "BUREAU_AMT_CREDIT_SUM_MEAN"].isna().all()


def test_quantile_clipper():
    X = pd.DataFrame({"a": np.arange(100, dtype=float)})
    clipper = QuantileClipper(0.05, 0.95).fit(X)
    out = clipper.transform(pd.DataFrame({"a": [-1000.0, 50.0, 1000.0]}))
    assert out["a"].min() >= clipper.lower_bounds_[0]
    assert out["a"].max() <= clipper.upper_bounds_[0]
    assert out["a"].iloc[1] == 50.0
    assert list(clipper.get_feature_names_out(["a"])) == ["a"]
    assert isinstance(clipper.transform(X.to_numpy()), np.ndarray)


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_preprocessor_output_is_numeric_and_complete(gmsc_df, kind):
    X = gmsc_df.drop(columns=TARGET)
    Xt = build_preprocessor("gmsc", kind).fit_transform(X)
    assert len(Xt) == len(X)
    assert all(np.issubdtype(t, np.number) for t in Xt.dtypes)
    if kind == "linear":
        assert not Xt.isna().any().any()
        assert any(c.startswith("missingindicator_") for c in Xt.columns)


def test_preprocessor_with_categorical_columns(home_credit_tables):
    app, bureau = home_credit_tables
    df = merge_bureau(app, aggregate_bureau(bureau))
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    assert categorical_columns(X) == ["NAME_CONTRACT_TYPE", "CODE_GENDER"]
    linear = build_preprocessor("home_credit", "linear").fit_transform(X)
    assert any(c.startswith("CODE_GENDER_") for c in linear.columns)
    tree = build_preprocessor("home_credit", "tree").fit_transform(X)
    assert set(tree["CODE_GENDER"].unique()) <= {0.0, 1.0}


def test_full_pipeline_fits_and_predicts(gmsc_df):
    X, y = gmsc_df.drop(columns=TARGET), gmsc_df[TARGET]
    pipe = build_model_pipeline("gmsc", "linear", LogisticRegression(max_iter=2000))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba > 0) & (proba < 1)).all()


def test_unknown_preprocessor_kind():
    with pytest.raises(ValueError):
        build_preprocessor("gmsc", "deep")
