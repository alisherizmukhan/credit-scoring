"""End-to-end: the real training routine on synthetic data, then batch inference."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_scoring.config import TARGET
from credit_scoring.features import aggregate_bureau, build_model_pipeline, merge_bureau
from credit_scoring.predict import CreditScorer, ScoreCard
from credit_scoring.train import make_estimator, stratified_sample, suggest_params
from tests.conftest import make_gmsc


def test_training_produces_artifacts(trained):
    summary, cfg = trained["summary"], trained["cfg"]
    assert summary["best_model"] in {"logreg", "lightgbm", "catboost"}
    assert set(summary["models"]) == {"logreg", "lightgbm", "catboost"}
    for model in summary["models"].values():
        assert 0.5 < model["test"]["roc_auc"] <= 1.0
    assert 0 < summary["threshold"] < 1

    reports = trained["model_dir"].parent / "reports"
    for name in [
        "calibration.png",
        "roc.png",
        "cost_curve.png",
        "shap_summary.png",
        "shap_client.png",
        "model_comparison.md",
        "metrics.json",
        "feature_importance.csv",
    ]:
        assert (reports / name).exists(), name

    meta = json.loads((trained["model_dir"] / "metadata.json").read_text())
    assert meta["dataset"] == "gmsc"
    assert meta["model_version"] == "1"  # registered in the MLflow model registry
    assert meta["threshold"] == summary["threshold"]
    assert cfg["mlflow"]["experiment"] == "test-credit-scoring"


def test_business_cutoff_beats_naive_policies(trained):
    biz = trained["summary"]["business"]
    chosen = biz["chosen_cutoff"]["expected_cost_per_1000_apps"]
    assert chosen <= biz["approve_all"]["expected_cost_per_1000_apps"]
    assert biz["approve_all"]["approval_rate"] == 1.0


def test_model_registered_with_champion_alias(trained):
    import mlflow

    cfg = trained["cfg"]
    client = mlflow.MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    version = client.get_model_version_by_alias(cfg["mlflow"]["registered_model"], "champion")
    assert str(version.version) == "1"


def test_scorer_batch_prediction(trained):
    scorer = CreditScorer.load(trained["model_dir"])
    df = make_gmsc(50, seed=7).drop(columns=TARGET)
    results = scorer.score_records(df)
    assert len(results) == 50
    for r in results:
        assert 0 <= r["probability_default"] <= 1
        assert 300 <= r["score"] <= 850
        assert r["decision"] == (
            "approve" if r["probability_default"] < scorer.threshold else "decline"
        )


def test_scorer_imputes_missing_fields(trained):
    scorer = CreditScorer.load(trained["model_dir"])
    record = make_gmsc(1).drop(columns=[TARGET, "monthly_income", "dependents"]).iloc[0].to_dict()
    result = scorer.score_records([record])[0]
    assert 0 <= result["probability_default"] <= 1


def test_scorecard_properties():
    card = ScoreCard(base_score=650, base_odds=50, pdo=40)
    assert card.score(1 / 51)[()] == 650  # odds 50:1 -> base score
    assert card.score(1 / 101) - card.score(1 / 51) == pytest.approx(40, abs=1)  # PDO
    scores = card.score(np.linspace(0.001, 0.999, 200))
    assert (np.diff(scores) <= 0).all()  # higher risk -> lower score
    assert scores.min() >= 300 and scores.max() <= 850


def test_scorer_with_categorical_features(home_credit_tables):
    app, bureau = home_credit_tables
    df = merge_bureau(app, aggregate_bureau(bureau))
    X, y = df.drop(columns=["TARGET", "SK_ID_CURR"]), df["TARGET"]
    pipe = build_model_pipeline("home_credit", "tree", make_estimator("lightgbm", {}, 0))
    pipe.fit(X, y)
    scorer = CreditScorer(
        pipe,
        {
            "dataset": "home_credit",
            "threshold": 0.2,
            "scorecard": {"base_score": 650, "base_odds": 50, "pdo": 40},
            "feature_columns": list(X.columns),
            "categorical_columns": ["NAME_CONTRACT_TYPE", "CODE_GENDER"],
        },
    )
    # partial record with an unseen category and most fields missing
    out = scorer.score_records(
        [{"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 600000, "CODE_GENDER": "XNA"}]
    )
    assert 0 <= out[0]["probability_default"] <= 1
    assert scorer.dataset == "home_credit" and scorer.version == "local"


def test_make_estimator_and_search_space():
    import optuna

    assert isinstance(make_estimator("logreg", {"C": 0.5}, 0), LogisticRegression)
    with pytest.raises(ValueError):
        make_estimator("svm", {}, 0)
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        }
    )
    assert suggest_params(trial, "lightgbm")["num_leaves"] == 31
    with pytest.raises(ValueError):
        suggest_params(trial, "logreg")


def test_stratified_sample_keeps_class_balance(gmsc_df):
    X, y = gmsc_df.drop(columns=TARGET), gmsc_df[TARGET]
    Xs, ys = stratified_sample(X, y, 500, seed=0)
    assert len(Xs) == 500
    assert ys.mean() == pytest.approx(y.mean(), abs=0.01)
    X_same, _ = stratified_sample(X, y, None, seed=0)
    assert X_same is X
    assert isinstance(Xs, pd.DataFrame)
