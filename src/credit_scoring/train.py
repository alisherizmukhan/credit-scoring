"""Training: hold-out split, Stratified K-Fold CV, Optuna tuning, MLflow tracking.

Protocol
--------
1. Stratified 80/20 split. The 20% test set is used only once, for the final report.
2. For each model (logreg -> LightGBM -> CatBoost):
   Optuna tunes hyper-parameters on CV inside the training part, then a 5-fold
   CV produces out-of-fold (OOF) predictions and the model is refitted on the whole train.
3. The best model is chosen by mean CV ROC-AUC (not by test!).
4. The business cut-off is chosen on the OOF predictions of the best model.
5. Test metrics, plots, SHAP, the model file and the MLflow registry entry are produced.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from credit_scoring import __version__
from credit_scoring.config import (
    TARGET,
    ensure_dir,
    load_config,
    logger,
    save_json,
    setup_logging,
)
from credit_scoring.data import load_processed, split_features_target
from credit_scoring.evaluate import (
    business_costs,
    business_summary,
    classification_metrics,
    comparison_table,
    find_optimal_threshold,
    plot_calibration,
    plot_cost_curve,
    plot_roc,
)
from credit_scoring.features import build_model_pipeline, categorical_columns
from credit_scoring.predict import METADATA_FILE, MODEL_FILE, scorecard_dict, scorecard_from_config

MODEL_KIND = {"logreg": "linear", "lightgbm": "tree", "catboost": "tree"}

DEFAULT_PARAMS: dict[str, dict] = {
    "logreg": {"C": 1.0},
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "catboost": {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
}

FAST_OVERRIDES = {
    "cv": {"n_splits": 3},
    "tuning": {"n_trials": {"lightgbm": 3, "catboost": 3}, "sample_size": 20000},
    "explain": {"sample_size": 500},
}


# --------------------------------------------------------------------------------------
# Model factory & search spaces
# --------------------------------------------------------------------------------------
def make_estimator(name: str, params: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, random_state=seed, **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if name == "catboost":
        return CatBoostClassifier(
            random_seed=seed, verbose=0, thread_count=-1, allow_writing_files=False, **params
        )
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(name: str, params: dict, dataset: str, seed: int) -> Pipeline:
    return build_model_pipeline(dataset, MODEL_KIND[name], make_estimator(name, params, seed))


def suggest_params(trial, name: str) -> dict:
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }
    raise ValueError(f"No search space for {name}")


# --------------------------------------------------------------------------------------
# Cross-validation & tuning
# --------------------------------------------------------------------------------------
def cross_validate_oof(
    factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seed: int,
    on_fold: Callable[[int, float], None] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions. A fresh pipeline is fitted on every training fold,
    so imputers/encoders never see the validation fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        pipe = factory()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y.iloc[va], oof[va])
        fold_aucs.append(float(auc))
        if on_fold is not None:
            on_fold(fold, float(np.mean(fold_aucs)))
    return oof, fold_aucs


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int | None, seed: int):
    if not n or len(y) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


def tune(name: str, X, y, dataset: str, cfg: dict) -> tuple[dict, float, int]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = cfg["seed"]
    tcfg = cfg["tuning"]
    Xs, ys = stratified_sample(X, y, tcfg.get("sample_size"), seed)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, name)

        def report(fold: int, running_auc: float) -> None:
            trial.report(running_auc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, aucs = cross_validate_oof(
            lambda: make_pipeline(name, params, dataset, seed),
            Xs,
            ys,
            tcfg["cv_splits"],
            seed,
            on_fold=report,
        )
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=f"{name}-tuning",
    )
    study.optimize(
        objective, n_trials=tcfg["n_trials"][name], timeout=tcfg.get("timeout_sec"), n_jobs=1
    )
    best = dict(study.best_params)
    if name == "lightgbm":
        best["subsample_freq"] = 1
    return best, float(study.best_value), len(study.trials)


# --------------------------------------------------------------------------------------
# Main training routine
# --------------------------------------------------------------------------------------
def run_training(
    cfg: dict,
    df: pd.DataFrame | None = None,
    dataset: str | None = None,
    id_column: str | None = None,
) -> dict:
    import mlflow
    from mlflow.models import infer_signature

    seed = cfg["seed"]
    if df is None:
        df, meta = load_processed(cfg["paths"]["processed"])
        dataset, id_column = meta["dataset"], meta.get("id_column")
    if dataset is None:
        raise ValueError("dataset must be given together with df")

    X, y = split_features_target(df, id_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y, random_state=seed
    )
    logger.info(
        "Dataset %s: train %d / test %d rows, %d features, default rate %.2f%%",
        dataset,
        len(X_train),
        len(X_test),
        X.shape[1],
        100 * y.mean(),
    )

    processed_dir = ensure_dir(cfg["paths"]["processed"])
    reports_dir = ensure_dir(cfg["paths"]["reports"])
    models_dir = ensure_dir(cfg["paths"]["models"])
    # Reference (train) and hold-out samples for drift monitoring.
    X_train.to_parquet(processed_dir / "train_reference.parquet", index=False)
    X_test.assign(**{TARGET: y_test.values}).to_parquet(processed_dir / "test.parquet", index=False)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, cfg["mlflow"].get("artifact_location"))
    mlflow.set_experiment(experiment)

    results: dict[str, dict] = {}
    with mlflow.start_run(run_name=f"training-{dataset}") as parent_run:
        mlflow.log_params(
            {
                "dataset": dataset,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X.shape[1],
                "default_rate": round(float(y.mean()), 5),
                "cv_splits": cfg["cv"]["n_splits"],
                "seed": seed,
            }
        )
        for name in cfg["models"]:
            results[name] = _train_one(name, cfg, dataset, X_train, y_train, X_test, y_test)

        best_name = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
        best = results[best_name]
        logger.info("Best model by CV ROC-AUC: %s", best_name)

        # ---- business cut-off on OOF predictions (no test data involved) ----
        cost_fn, cost_fp = business_costs(cfg["business"])
        threshold, _ = find_optimal_threshold(y_train, best["oof"], cost_fn, cost_fp)
        business = {
            "chosen_cutoff": business_summary(
                y_test, best["test_prob"], threshold, cfg["business"]
            ),
            "cutoff_0.5": business_summary(y_test, best["test_prob"], 0.5, cfg["business"]),
            "approve_all": business_summary(y_test, best["test_prob"], 1.01, cfg["business"]),
            "theoretical_cutoff": cost_fp / (cost_fp + cost_fn),
        }

        # ---- plots ----
        curves = {n: (y_test.values, r["test_prob"]) for n, r in results.items()}
        plot_calibration(curves, reports_dir / "calibration.png")
        plot_roc(curves, reports_dir / "roc.png")
        plot_cost_curve(
            y_test, best["test_prob"], cost_fn, cost_fp, threshold, reports_dir / "cost_curve.png"
        )

        # ---- SHAP ----
        from credit_scoring.explain import explain_model

        n_explain = min(cfg["explain"]["sample_size"], len(X_test))
        X_explain = X_test.sample(n=n_explain, random_state=seed)
        explanation = explain_model(
            best["pipeline"],
            X_explain,
            best["pipeline"].predict_proba(X_explain)[:, 1],
            reports_dir,
        )

        # ---- reports ----
        rows = [
            {
                "model": n,
                "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                "cv_roc_auc_std": r["cv_roc_auc_std"],
                "test": r["test"],
            }
            for n, r in results.items()
        ]
        table = comparison_table(rows)
        (reports_dir / "model_comparison.md").write_text(table, encoding="utf-8")
        summary = {
            "dataset": dataset,
            "best_model": best_name,
            "threshold": threshold,
            "models": {
                n: {
                    "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                    "cv_roc_auc_std": r["cv_roc_auc_std"],
                    "cv": r["cv"],
                    "test": r["test"],
                    "params": r["params"],
                    "tuning": r["tuning"],
                    "train_seconds": r["train_seconds"],
                }
                for n, r in results.items()
            },
            "business": business,
            "explain": explanation,
        }
        save_json(summary, reports_dir / "metrics.json")

        mlflow.log_params({"best_model": best_name, "threshold": threshold})
        mlflow.log_metrics({f"best_test_{k}": v for k, v in best["test"].items()})
        mlflow.log_metrics(
            {f"business_{k}": v for k, v in business["chosen_cutoff"].items() if k != "threshold"}
        )
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")

        # ---- model registry ----
        example = X_test.head(5)
        numeric = example.select_dtypes("number").columns
        # integers declared as doubles: a missing value at inference must not break the schema
        example = example.astype(dict.fromkeys(numeric, "float64"))
        signature = infer_signature(example, best["pipeline"].predict_proba(example)[:, 1])
        model_info = mlflow.sklearn.log_model(
            sk_model=best["pipeline"],
            name="model",
            signature=signature,
            input_example=example,
            registered_model_name=cfg["mlflow"]["registered_model"],
            skops_trusted_types=_own_types(best["pipeline"]),
        )
        version = getattr(model_info, "registered_model_version", None)
        if version is not None:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                cfg["mlflow"]["registered_model"], "champion", version
            )
            client.set_model_version_tag(
                cfg["mlflow"]["registered_model"], version, "algorithm", best_name
            )

        # ---- local artifact for the API ----
        joblib.dump(best["pipeline"], models_dir / MODEL_FILE)
        metadata = {
            "dataset": dataset,
            "model_name": best_name,
            "model_version": str(version) if version is not None else "local",
            "package_version": __version__,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mlflow_run_id": parent_run.info.run_id,
            "threshold": threshold,
            "scorecard": scorecard_dict(scorecard_from_config(cfg)),
            "feature_columns": list(X_train.columns),
            "categorical_columns": categorical_columns(X_train),
            "test_metrics": best["test"],
        }
        save_json(metadata, models_dir / METADATA_FILE)

    _log_summary(summary, table)
    return summary


def _train_one(name, cfg, dataset, X_train, y_train, X_test, y_test) -> dict:
    import mlflow

    seed = cfg["seed"]
    start = time.perf_counter()
    with mlflow.start_run(run_name=name, nested=True) as run:
        params = dict(DEFAULT_PARAMS[name])
        tuning_info = None
        if cfg["tuning"]["enabled"] and name in cfg["tuning"]["n_trials"]:
            logger.info("[%s] Optuna: %d trials", name, cfg["tuning"]["n_trials"][name])
            best_params, best_value, n_done = tune(name, X_train, y_train, dataset, cfg)
            params.update(best_params)
            tuning_info = {"best_cv_auc": best_value, "n_trials": n_done}
            mlflow.log_metrics({"tuning_best_cv_auc": best_value, "tuning_trials": n_done})

        def factory() -> Pipeline:
            return make_pipeline(name, params, dataset, seed)

        oof, fold_aucs = cross_validate_oof(factory, X_train, y_train, cfg["cv"]["n_splits"], seed)
        cv = classification_metrics(y_train, oof)

        pipeline = factory()
        pipeline.fit(X_train, y_train)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test = classification_metrics(y_test, test_prob)
        elapsed = time.perf_counter() - start

        mlflow.log_params({"model": name, **params})
        mlflow.log_metrics(
            {
                "cv_roc_auc_mean": float(np.mean(fold_aucs)),
                "cv_roc_auc_std": float(np.std(fold_aucs)),
                **{f"cv_{k}": v for k, v in cv.items()},
                **{f"test_{k}": v for k, v in test.items()},
                "train_seconds": elapsed,
            }
        )
        logger.info(
            "[%s] CV AUC %.4f ± %.4f | test AUC %.4f | Gini %.4f | KS %.4f | %.0fs",
            name,
            np.mean(fold_aucs),
            np.std(fold_aucs),
            test["roc_auc"],
            test["gini"],
            test["ks"],
            elapsed,
        )
    return {
        "pipeline": pipeline,
        "params": params,
        "tuning": tuning_info,
        "oof": oof,
        "test_prob": test_prob,
        "cv": cv,
        "cv_roc_auc_mean": float(np.mean(fold_aucs)),
        "cv_roc_auc_std": float(np.std(fold_aucs)),
        "test": test,
        "run_id": run.info.run_id,
        "train_seconds": elapsed,
    }


def _own_types(pipeline: Pipeline) -> list[str]:
    """MLflow serialises sklearn models with skops, which refuses unknown classes on load.
    The pipeline was built by us in this process, so its own types are trusted."""
    import skops.io

    return skops.io.get_untrusted_types(data=skops.io.dumps(pipeline))


def _log_summary(summary: dict, table: str) -> None:
    biz = summary["business"]["chosen_cutoff"]
    logger.info("Model comparison:\n%s", table)
    logger.info(
        "Cut-off %.3f -> approval %.1f%%, bad rate among approved %.2f%%, "
        "expected cost %.0f KZT per 1000 applications",
        biz["threshold"],
        100 * biz["approval_rate"],
        100 * biz["bad_rate_approved"],
        biz["expected_cost_per_1000_apps"],
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Train credit scoring models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fast", action="store_true", help="few trials, for smoke runs")
    args = parser.parse_args()
    cfg = load_config(args.config, FAST_OVERRIDES if args.fast else None)
    Path(cfg["paths"]["models"]).mkdir(parents=True, exist_ok=True)
    run_training(cfg)


if __name__ == "__main__":
    main()
