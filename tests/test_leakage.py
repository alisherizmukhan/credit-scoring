"""Leakage guards: anything learned from data must be learned on the training part only."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from credit_scoring.config import TARGET
from credit_scoring.data import split_features_target
from credit_scoring.features import (
    FeatureEngineer,
    aggregate_bureau,
    build_preprocessor,
)
from credit_scoring.train import cross_validate_oof


def test_target_and_id_are_not_features(home_credit_tables):
    app, _ = home_credit_tables
    df = app.rename(columns={"TARGET": TARGET})
    X, y = split_features_target(df, id_column="SK_ID_CURR")
    assert TARGET not in X.columns
    assert "SK_ID_CURR" not in X.columns
    assert len(X) == len(y)


def test_imputer_and_clipper_statistics_come_from_train_only(gmsc_df):
    X = gmsc_df.drop(columns=TARGET)
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
    X_test = X_test.copy()
    X_test["monthly_income"] = 10_000_000.0  # absurd values in test only

    prep = build_preprocessor("gmsc", "linear").fit(X_train)
    num = prep.named_steps["columns"].named_transformers_["num"]
    imputer, clipper = num.named_steps["impute"], num.named_steps["clip"]

    income_pos = list(imputer.feature_names_in_).index("monthly_income")
    assert imputer.statistics_[income_pos] == np.nanmedian(X_train["monthly_income"])
    # clipping bounds were learned on train: test outliers are capped, bounds do not move
    assert clipper.upper_bounds_[income_pos] < 10_000_000.0
    transformed = prep.transform(X_test)
    assert transformed["monthly_income"].nunique() == 1


def test_refit_on_different_test_data_does_not_change_train_transform(gmsc_df):
    """Transforming train rows gives the same result no matter what the test rows are."""
    X = gmsc_df.drop(columns=TARGET)
    X_train, X_test = X.iloc[:1500], X.iloc[1500:]
    prep = build_preprocessor("gmsc", "linear").fit(X_train)
    before = prep.transform(X_train)
    prep.transform(X_test * 100)  # transform must be side-effect free
    after = prep.transform(X_train)
    pd.testing.assert_frame_equal(before, after)


def test_feature_engineering_is_row_wise(gmsc_df):
    """Stateless features of a row do not depend on other rows -> safe before the split."""
    X = gmsc_df.drop(columns=TARGET)
    fe = FeatureEngineer("gmsc").fit(X)
    full = fe.transform(X).iloc[:10].reset_index(drop=True)
    alone = fe.transform(X.iloc[:10]).reset_index(drop=True)
    pd.testing.assert_frame_equal(full, alone)


def test_bureau_aggregates_do_not_depend_on_other_clients(home_credit_tables):
    _, bureau = home_credit_tables
    ids = bureau["SK_ID_CURR"].unique()
    full = aggregate_bureau(bureau).set_index("SK_ID_CURR").loc[ids[:5]]
    subset = aggregate_bureau(bureau[bureau["SK_ID_CURR"].isin(ids[:5])]).set_index("SK_ID_CURR")
    pd.testing.assert_frame_equal(full, subset.loc[ids[:5]])
    assert "TARGET" not in full.columns


class _MemorizingClassifier(ClassifierMixin, BaseEstimator):
    """Predicts 1 for rows it has seen during fit, 0 otherwise."""

    def fit(self, X, y):
        self.seen_ = set(X["row_id"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        seen = X["row_id"].isin(self.seen_).to_numpy().astype(float)
        return np.column_stack([1 - seen, seen])


def test_oof_predictions_come_from_models_that_did_not_see_the_row(gmsc_df):
    X = pd.DataFrame({"row_id": np.arange(len(gmsc_df))})
    y = gmsc_df[TARGET]
    oof, aucs = cross_validate_oof(
        lambda: Pipeline([("model", _MemorizingClassifier())]), X, y, n_splits=4, seed=0
    )
    assert oof.sum() == 0  # no row was predicted by a model trained on it
    assert len(aucs) == 4
