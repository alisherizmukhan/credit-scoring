"""SHAP explanations: global (summary plot) and local (one client)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def compute_shap(pipeline: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds scale) of the final model on preprocessed features."""
    prep, model = pipeline[:-1], pipeline[-1]
    Xt = prep.transform(X)
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, Xt)
    else:
        explainer = shap.TreeExplainer(model)
    explanation = explainer(Xt)
    if explanation.values.ndim == 3:  # some models return one slice per class
        explanation = explanation[..., 1]
    return explanation


def save_summary_plot(explanation: shap.Explanation, path: str | Path, max_display: int = 15):
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("SHAP summary: impact on default log-odds")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def save_client_plot(explanation: shap.Explanation, index: int, path: str | Path, title: str):
    plt.figure()
    shap.plots.waterfall(explanation[index], max_display=12, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")


def feature_importance(explanation: shap.Explanation) -> pd.DataFrame:
    imp = np.abs(explanation.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_reasons(explanation: shap.Explanation, index: int, k: int = 5) -> list[dict]:
    """Largest risk-increasing contributions for one client (adverse-action style)."""
    values = explanation.values[index]
    data = explanation.data[index]
    order = np.argsort(values)[::-1][:k]
    return [
        {
            "feature": explanation.feature_names[i],
            "value": None if pd.isna(data[i]) else float(data[i]),
            "shap": float(values[i]),
        }
        for i in order
    ]


def explain_model(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    y_prob: np.ndarray,
    out_dir: str | Path,
) -> dict:
    """Create the summary plot and a local explanation for a high-risk client."""
    out_dir = Path(out_dir)
    explanation = compute_shap(pipeline, X_sample)
    save_summary_plot(explanation, out_dir / "shap_summary.png")
    # client at the 95th percentile of predicted risk: a typical rejected applicant
    client_idx = int(np.argsort(y_prob)[int(0.95 * (len(y_prob) - 1))])
    save_client_plot(
        explanation,
        client_idx,
        out_dir / "shap_client.png",
        title=f"Why this client is risky (PD = {y_prob[client_idx]:.1%})",
    )
    importance = feature_importance(explanation)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    return {
        "client_index": client_idx,
        "client_pd": float(y_prob[client_idx]),
        "client_top_reasons": top_reasons(explanation, client_idx),
        "top_features": importance.head(10)["feature"].tolist(),
    }
