"""Dataset download and preparation.

``python -m credit_scoring.data`` downloads the raw files and writes a single
model-ready table to ``data/processed/dataset.parquet`` (features + ``target``).

Source priority (``dataset: auto``):
1. Kaggle "Home Credit Default Risk" (application_train + bureau) — needs Kaggle token;
2. Kaggle "Give Me Some Credit" — needs Kaggle token;
3. "Give Me Some Credit" mirror on OpenML (id 46929) — no credentials needed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from credit_scoring.config import (
    TARGET,
    ensure_dir,
    load_config,
    load_json,
    logger,
    save_json,
    setup_logging,
)
from credit_scoring.features import aggregate_bureau, merge_bureau

DATASETS = ("home_credit", "gmsc")
PROCESSED_FILE = "dataset.parquet"
META_FILE = "meta.json"

ID_COLUMNS = {"home_credit": "SK_ID_CURR", "gmsc": None}

# Original GMSC names -> readable snake_case (same names are used by the API schema).
GMSC_RENAME = {
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "past_due_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_days_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "past_due_60_89",
    "NumberOfDependents": "dependents",
}
GMSC_FEATURES = list(GMSC_RENAME.values())
GMSC_OPENML_ID = 46929


class DataUnavailableError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def kaggle_credentials_available() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").exists()


def _kaggle_download(competition: str, files: list[str], dest: Path) -> None:
    exe = shutil.which("kaggle")
    if exe is None:
        raise DataUnavailableError("kaggle CLI not installed (pip install kaggle)")
    for name in files:
        if (dest / name).exists():
            continue
        cmd = [exe, "competitions", "download", "-c", competition, "-f", name, "-p", str(dest)]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise DataUnavailableError(f"kaggle download failed: {result.stderr.strip()}")
    for archive in dest.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()


def download_home_credit(raw_dir: Path) -> Path:
    dest = ensure_dir(raw_dir / "home_credit")
    _kaggle_download("home-credit-default-risk", ["application_train.csv", "bureau.csv"], dest)
    return dest


def download_gmsc(raw_dir: Path) -> Path:
    """Kaggle first, OpenML mirror as a credential-free fallback."""
    dest = ensure_dir(raw_dir / "gmsc")
    target_file = dest / "gmsc.csv"
    if target_file.exists():
        return target_file
    if kaggle_credentials_available():
        try:
            _kaggle_download("GiveMeSomeCredit", ["cs-training.csv"], dest)
            raw = pd.read_csv(dest / "cs-training.csv", index_col=0)
            raw = raw.rename(columns={"SeriousDlqin2yrs": TARGET})
            raw.to_csv(target_file, index=False)
            return target_file
        except DataUnavailableError as exc:
            logger.warning("Kaggle GMSC unavailable (%s); falling back to OpenML", exc)

    from sklearn.datasets import fetch_openml

    logger.info("Downloading Give Me Some Credit from OpenML (id=%s)", GMSC_OPENML_ID)
    bunch = fetch_openml(data_id=GMSC_OPENML_ID, as_frame=True, data_home=str(dest / "openml"))
    frame = bunch.frame.copy()
    target_col = bunch.target_names[0]
    frame[TARGET] = (frame.pop(target_col).astype(str) == "Yes").astype(int)
    frame.to_csv(target_file, index=False)
    return target_file


def resolve_dataset(requested: str) -> str:
    if requested == "auto":
        return "home_credit" if kaggle_credentials_available() else "gmsc"
    if requested not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS} or 'auto', got {requested!r}")
    return requested


# --------------------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------------------
def prepare_gmsc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=GMSC_RENAME)
    if "SeriousDlqin2yrs" in df.columns:
        df = df.rename(columns={"SeriousDlqin2yrs": TARGET})
    df = df[GMSC_FEATURES + [TARGET]]
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def prepare_home_credit(raw_dir: Path) -> pd.DataFrame:
    app = pd.read_csv(raw_dir / "application_train.csv")
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    df = merge_bureau(app, aggregate_bureau(bureau))
    df = df.rename(columns={"TARGET": TARGET})
    df[TARGET] = df[TARGET].astype(int)
    return df


def prepare_dataset(cfg: dict) -> tuple[pd.DataFrame, str]:
    raw_dir = ensure_dir(cfg["paths"]["raw"])
    dataset = resolve_dataset(cfg["dataset"])
    if dataset == "home_credit":
        try:
            df = prepare_home_credit(download_home_credit(raw_dir))
        except DataUnavailableError as exc:
            if cfg["dataset"] != "auto":
                raise
            logger.warning("Home Credit unavailable (%s); using Give Me Some Credit", exc)
            dataset = "gmsc"
    if dataset == "gmsc":
        df = prepare_gmsc(download_gmsc(raw_dir))
    return df, dataset


def save_processed(df: pd.DataFrame, dataset: str, processed_dir: str | Path) -> Path:
    out_dir = ensure_dir(processed_dir)
    path = out_dir / PROCESSED_FILE
    df.to_parquet(path, index=False)
    save_json(
        {
            "dataset": dataset,
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1] - 1),
            "default_rate": float(df[TARGET].mean()),
            "id_column": ID_COLUMNS[dataset],
        },
        out_dir / META_FILE,
    )
    return path


def load_processed(processed_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    processed_dir = Path(processed_dir)
    path = processed_dir / PROCESSED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return pd.read_parquet(path), load_json(processed_dir / META_FILE)


def split_features_target(df: pd.DataFrame, id_column: str | None = None):
    drop = [TARGET] + ([id_column] if id_column and id_column in df.columns else [])
    return df.drop(columns=drop), df[TARGET].astype(int)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Download and prepare the credit dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["auto", *DATASETS], default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, {"dataset": args.dataset} if args.dataset else None)
    df, dataset = prepare_dataset(cfg)
    path = save_processed(df, dataset, cfg["paths"]["processed"])
    logger.info(
        "Saved %s: %d rows, %d columns, default rate %.2f%%",
        path,
        len(df),
        df.shape[1] - 1,
        100 * df[TARGET].mean(),
    )


if __name__ == "__main__":
    main()
