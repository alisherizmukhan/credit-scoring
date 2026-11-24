"""Data drift monitoring with the Population Stability Index (PSI).

PSI = sum((cur% - ref%) * ln(cur% / ref%)) over bins built on the reference (train) data.
Rule of thumb used by banks: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant.

Usage:
    python -m credit_scoring.drift --current path/to/new_sample.parquet
    python -m credit_scoring.drift --simulate     # demo: hold-out set with injected shift
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.config import ensure_dir, load_config, load_json, logger, setup_logging

EPS = 1e-4


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of one numeric feature. Bin edges are reference quantiles; NaN is its own bin."""
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    ref_valid = ref[~np.isnan(ref)]
    if ref_valid.size == 0:
        return 0.0
    edges = np.unique(np.quantile(ref_valid, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if edges.size < 2:  # constant feature
        edges = np.array([-np.inf, np.inf])

    def distribution(values: np.ndarray) -> np.ndarray:
        valid = values[~np.isnan(values)]
        counts = np.histogram(valid, bins=edges)[0].astype(float)
        counts = np.append(counts, np.isnan(values).sum())  # missing-value bin
        return np.clip(counts / max(len(values), 1), EPS, None)

    ref_dist, cur_dist = distribution(ref), distribution(cur)
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def psi_status(value: float) -> str:
    if value < 0.10:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def psi_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], bins: int = 10
) -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        value = psi(reference[feat], current[feat], bins)
        rows.append({"feature": feat, "psi": round(value, 4), "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, dataset: str, seed: int = 0) -> pd.DataFrame:
    """Inject a plausible shift: e.g. a new marketing channel brings younger,
    lower-income, more leveraged applicants."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    if dataset == "gmsc":
        out["monthly_income"] = out["monthly_income"] * rng.normal(0.75, 0.05, len(out))
        out["age"] = (out["age"] - 7).clip(lower=21)
        out["revolving_utilization"] = out["revolving_utilization"] * 1.4
    else:
        out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"] * rng.normal(0.75, 0.05, len(out))
        out["DAYS_BIRTH"] = out["DAYS_BIRTH"] + 7 * 365
        out["AMT_CREDIT"] = out["AMT_CREDIT"] * 1.3
    return out


def to_markdown(report: pd.DataFrame) -> str:
    lines = ["| Feature | PSI | Status |", "|---|---|---|"]
    lines += [f"| {r.feature} | {r.psi:.4f} | {r.status} |" for r in report.itertuples()]
    return "\n".join(lines) + "\n"


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="PSI drift report")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--current", help="new sample (csv/parquet) to compare with train")
    parser.add_argument("--simulate", action="store_true", help="use hold-out set + fake shift")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed"])
    dataset = load_json(processed / "meta.json")["dataset"]
    reference = pd.read_parquet(processed / "train_reference.parquet")
    features = cfg["drift"]["features"][dataset]

    reports_dir = ensure_dir(cfg["paths"]["reports"])
    if args.current:
        path = Path(args.current)
        current = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        report = psi_report(reference, current, features, cfg["drift"]["bins"])
    else:
        holdout = pd.read_parquet(processed / "test.parquet")
        no_drift = psi_report(reference, holdout, features, cfg["drift"]["bins"])
        logger.info("Train vs hold-out (no drift expected):\n%s", no_drift.to_string(index=False))
        if not args.simulate:
            report = no_drift
        else:
            shifted = simulate_drift(holdout, dataset, cfg["seed"])
            report = psi_report(reference, shifted, features, cfg["drift"]["bins"])
            (reports_dir / "drift_baseline.md").write_text(to_markdown(no_drift), encoding="utf-8")

    (reports_dir / "drift_report.md").write_text(to_markdown(report), encoding="utf-8")
    logger.info("PSI report:\n%s", report.to_string(index=False))
    n_alerts = int((report["status"] == "significant").sum())
    if n_alerts:
        logger.warning("%d feature(s) with significant drift -> consider retraining", n_alerts)


if __name__ == "__main__":
    main()
