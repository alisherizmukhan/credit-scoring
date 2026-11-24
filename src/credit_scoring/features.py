"""Feature engineering and preprocessing pipelines.

Two kinds of transformations live here:

* **Stateless** (row-wise) — ratios, anomaly fixes, bureau aggregates per client.
  They use only the values of the same client, so they cannot leak information
  between train and test and may be applied before the split.
* **Stateful** — imputation medians, clipping quantiles, scaling, category encoding.
  They *learn* from data, so they are only ever fitted inside a sklearn ``Pipeline``
  on the training fold (``build_model_pipeline``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PreprocessorKind = Literal["linear", "tree"]

# --------------------------------------------------------------------------------------
# Bureau aggregates (Home Credit): one row per previous credit -> one row per client
# --------------------------------------------------------------------------------------
BUREAU_NUM_AGGS: dict[str, list[str]] = {
    "DAYS_CREDIT": ["mean", "min", "max"],
    "DAYS_CREDIT_ENDDATE": ["mean", "max"],
    "CREDIT_DAY_OVERDUE": ["mean", "max"],
    "AMT_CREDIT_SUM": ["mean", "max", "sum"],
    "AMT_CREDIT_SUM_DEBT": ["mean", "max", "sum"],
    "AMT_CREDIT_SUM_OVERDUE": ["mean", "max"],
    "CNT_CREDIT_PROLONG": ["sum"],
}
BUREAU_COUNT_COLUMNS = ["BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_COUNT"]


def aggregate_bureau(bureau: pd.DataFrame, key: str = "SK_ID_CURR") -> pd.DataFrame:
    """Aggregate the credit-bureau history to client level (count / mean / max / sum).

    Uses no target information and no other clients' data -> safe before the split.
    """
    aggs = {col: funcs for col, funcs in BUREAU_NUM_AGGS.items() if col in bureau.columns}
    grouped = bureau.groupby(key)
    agg = grouped.agg(aggs)
    agg.columns = [f"BUREAU_{col}_{func.upper()}" for col, func in agg.columns]
    agg["BUREAU_LOAN_COUNT"] = grouped.size()
    if "CREDIT_ACTIVE" in bureau.columns:
        active = (bureau["CREDIT_ACTIVE"] == "Active").groupby(bureau[key]).sum()
        agg["BUREAU_ACTIVE_COUNT"] = active
        agg["BUREAU_ACTIVE_SHARE"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_LOAN_COUNT"]
    return agg.reset_index()


def merge_bureau(app: pd.DataFrame, bureau_agg: pd.DataFrame, key: str = "SK_ID_CURR"):
    """Left-join bureau aggregates; a client absent from the bureau has 0 previous loans."""
    merged = app.merge(bureau_agg, on=key, how="left")
    for col in BUREAU_COUNT_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    return merged


# --------------------------------------------------------------------------------------
# Row-wise feature engineering
# --------------------------------------------------------------------------------------
def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division that returns NaN instead of inf when the denominator is 0 or missing."""
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float).replace(0, np.nan)
    return num / den


GMSC_DELINQUENCY = ["past_due_30_59", "past_due_60_89", "times_90_days_late"]


def engineer_gmsc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 96/98 are reporting codes, not real counts: keep the signal as a flag,
    # and treat the count itself as missing.
    out["delinquency_code"] = (out[GMSC_DELINQUENCY] >= 90).any(axis=1).astype(np.int8)
    for col in GMSC_DELINQUENCY:
        out[col] = out[col].astype(float).where(out[col] < 90)
    out["total_past_due"] = out[GMSC_DELINQUENCY].sum(axis=1, min_count=1)
    out["monthly_debt"] = out["debt_ratio"] * out["monthly_income"]
    out["income_per_person"] = safe_div(out["monthly_income"], out["dependents"].fillna(0) + 1)
    out["non_real_estate_lines"] = out["open_credit_lines"] - out["real_estate_loans"]
    out["income_missing"] = out["monthly_income"].isna().astype(np.int8)
    return out


def engineer_home_credit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DAYS_EMPLOYED" in out:
        # 365243 days (~1000 years) is a placeholder for pensioners / unemployed.
        anomaly = out["DAYS_EMPLOYED"] == 365243
        out["DAYS_EMPLOYED_ANOM"] = anomaly.astype(np.int8)
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].astype(float).mask(anomaly)
    ratios = {
        "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "ANNUITY_CREDIT_RATIO": ("AMT_ANNUITY", "AMT_CREDIT"),
        "GOODS_CREDIT_RATIO": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        "EMPLOYED_AGE_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
        "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        "BUREAU_DEBT_CREDIT_RATIO": ("BUREAU_AMT_CREDIT_SUM_DEBT_SUM", "BUREAU_AMT_CREDIT_SUM_SUM"),
        "BUREAU_DEBT_INCOME_RATIO": ("BUREAU_AMT_CREDIT_SUM_DEBT_SUM", "AMT_INCOME_TOTAL"),
    }
    for name, (num, den) in ratios.items():
        if num in out and den in out:
            out[name] = safe_div(out[num], out[den])
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in out]
    if ext:
        out["EXT_SOURCE_MEAN"] = out[ext].mean(axis=1)
        out["EXT_SOURCE_MIN"] = out[ext].min(axis=1)
    return out


ENGINEERS = {"gmsc": engineer_gmsc, "home_credit": engineer_home_credit}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, dataset-specific feature construction (ratios, anomaly fixes)."""

    def __init__(self, dataset: str = "gmsc"):
        self.dataset = dataset

    def fit(self, X: pd.DataFrame, y=None):
        if self.dataset not in ENGINEERS:
            raise ValueError(f"Unknown dataset '{self.dataset}'. Options: {list(ENGINEERS)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        # Learns nothing from the values: only records the output column names.
        self.feature_names_out_ = np.asarray(self._engineer(X.head(5)).columns, dtype=object)
        return self

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        out = ENGINEERS[self.dataset](X)
        bool_cols = out.select_dtypes(include="bool").columns
        out[bool_cols] = out[bool_cols].astype(np.int8)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._engineer(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns to [q_low, q_high] quantiles learned on the training data."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        clipped = np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(clipped, columns=X.columns, index=X.index)
        return clipped

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", None)


def build_preprocessor(dataset: str, kind: PreprocessorKind) -> Pipeline:
    """Feature engineering + column-wise preprocessing.

    * ``linear`` — median imputation with missing-flags, winsorizing, scaling, one-hot.
    * ``tree``   — numeric passthrough (LightGBM/CatBoost handle NaN natively),
      ordinal encoding of categories.
    """
    numeric = make_column_selector(dtype_include="number")
    categorical = make_column_selector(dtype_exclude="number")

    if kind == "linear":
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("clip", QuantileClipper(0.01, 0.99)),
                ("scale", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=0.01,
                        sparse_output=False,
                    ),
                ),
            ]
        )
    elif kind == "tree":
        num_pipe = "passthrough"
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind}")

    columns = ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([("engineer", FeatureEngineer(dataset)), ("columns", columns)])
    return pipe.set_output(transform="pandas")


def build_model_pipeline(dataset: str, kind: PreprocessorKind, estimator) -> Pipeline:
    """Full pipeline: everything learned from data is fitted together with the model."""
    return Pipeline([("prep", build_preprocessor(dataset, kind)), ("model", estimator)])


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return list(X.select_dtypes(exclude="number").columns)
