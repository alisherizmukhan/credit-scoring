"""Model quality metrics, calibration and the cost-based business cut-off."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


# --------------------------------------------------------------------------------------
# Ranking & probability metrics
# --------------------------------------------------------------------------------------
def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: max distance between the score CDFs of bads and goods."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini(y_true, y_prob) -> float:
    return float(2 * roc_auc_score(y_true, y_prob) - 1)


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


# --------------------------------------------------------------------------------------
# Cost-based cut-off
# --------------------------------------------------------------------------------------
def business_costs(cfg_business: dict) -> tuple[float, float]:
    """(cost of a missed default, cost of rejecting a good client) in currency units."""
    loan = cfg_business["loan_amount"]
    return loan * cfg_business["lgd"], loan * cfg_business["profit_margin"]


def expected_cost(y_true, y_prob, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Average cost per application when approving everyone with PD < threshold."""
    y_true = np.asarray(y_true)
    approve = np.asarray(y_prob) < threshold
    fn = np.sum(approve & (y_true == 1))  # approved, then defaulted
    fp = np.sum(~approve & (y_true == 0))  # rejected a good client
    return float((fn * cost_fn + fp * cost_fp) / len(y_true))


def find_optimal_threshold(
    y_true, y_prob, cost_fn: float, cost_fp: float, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Threshold minimising expected cost. Must be fitted on OOF/validation predictions."""
    if grid is None:
        grid = np.round(np.arange(0.01, 0.99, 0.005), 4)
    costs = np.array([expected_cost(y_true, y_prob, t, cost_fn, cost_fp) for t in grid])
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


def business_summary(y_true, y_prob, threshold: float, cfg_business: dict) -> dict[str, float]:
    y_true = np.asarray(y_true)
    approve = np.asarray(y_prob) < threshold
    cost_fn, cost_fp = business_costs(cfg_business)
    n = len(y_true)
    n_approved = int(approve.sum())
    defaults_approved = int(np.sum(approve & (y_true == 1)))
    good_rejected = int(np.sum(~approve & (y_true == 0)))
    credit_loss = defaults_approved * cost_fn
    lost_profit = good_rejected * cost_fp
    return {
        "threshold": float(threshold),
        "approval_rate": n_approved / n,
        "bad_rate_approved": defaults_approved / n_approved if n_approved else 0.0,
        "defaults_caught_share": float(np.sum(~approve & (y_true == 1)) / max(y_true.sum(), 1)),
        "credit_loss_per_1000_apps": credit_loss / n * 1000,
        "lost_profit_per_1000_apps": lost_profit / n * 1000,
        "expected_cost_per_1000_apps": (credit_loss + lost_profit) / n * 1000,
    }


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_calibration(curves: dict[str, tuple], path: str | Path, n_bins: int = 15) -> None:
    """Reliability diagram for several models: {name: (y_true, y_prob)}."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    max_p = 0.0
    for name, (y, p) in curves.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y, p)
        ax.plot(mean_pred, frac_pos, marker="o", ms=3, label=f"{name} (Brier {brier:.4f})")
        max_p = max(max_p, mean_pred.max(), frac_pos.max())
    lim = min(1.0, max_p * 1.1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Mean predicted PD", ylabel="Observed default rate")
    ax.set_title("Calibration (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cost_curve(
    y_true, y_prob, cost_fn: float, cost_fp: float, threshold: float, path: str | Path
) -> None:
    grid = np.round(np.arange(0.01, 0.6, 0.005), 4)
    y_true = np.asarray(y_true)
    costs = [expected_cost(y_true, y_prob, t, cost_fn, cost_fp) / 1000 for t in grid]
    approval = [np.mean(np.asarray(y_prob) < t) for t in grid]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(grid, costs, color="tab:red", label="expected cost per application, k KZT")
    ax1.axvline(threshold, color="gray", ls="--", label=f"chosen cut-off = {threshold:.3f}")
    ax1.set_xlabel("PD cut-off (approve if PD < cut-off)")
    ax1.set_ylabel("Expected cost, k KZT / application", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(grid, approval, color="tab:blue", label="approval rate")
    ax2.set_ylabel("Approval rate", color="tab:blue")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, fontsize=8, loc="center right")
    ax1.set_title("Business cut-off: cost vs. approval rate")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def comparison_table(rows: list[dict]) -> str:
    """Markdown table: one row per model with CV and test metrics."""
    header = (
        "| Model | CV ROC-AUC | Test ROC-AUC | Test Gini | Test PR-AUC | Test KS | Test Brier |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['model']} | {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} "
        f"| {r['test']['roc_auc']:.4f} | {r['test']['gini']:.4f} | {r['test']['pr_auc']:.4f} "
        f"| {r['test']['ks']:.4f} | {r['test']['brier']:.4f} |"
        for r in rows
    ]
    return header + "\n".join(lines) + "\n"
