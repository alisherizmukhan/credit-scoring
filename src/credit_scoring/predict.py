"""Inference: load the saved pipeline, compute PD, map it to a score and a decision."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_scoring.config import load_json

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class ScoreCard:
    """PD -> score with "points to double the odds" scaling.

    score = offset + factor * ln(odds_good),  odds_good = (1 - PD) / PD,
    factor = pdo / ln 2,  offset = base_score - factor * ln(base_odds).
    Higher score = lower risk, as in credit-bureau scores.
    """

    base_score: float = 650
    base_odds: float = 50
    pdo: float = 40
    min_score: int = 300
    max_score: int = 850

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score(self, pd_: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
        raw = self.offset + self.factor * np.log((1 - p) / p)
        return np.clip(np.round(raw), self.min_score, self.max_score).astype(int)


class CreditScorer:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold = float(metadata["threshold"])
        self.scorecard = ScoreCard(**metadata["scorecard"])
        self.feature_columns: list[str] = metadata["feature_columns"]
        self.categorical_columns: list[str] = metadata.get("categorical_columns", [])

    @classmethod
    def load(cls, model_dir: str | Path) -> CreditScorer:
        model_dir = Path(model_dir)
        pipeline = joblib.load(model_dir / MODEL_FILE)
        metadata = load_json(model_dir / METADATA_FILE)
        return cls(pipeline, metadata)

    @property
    def dataset(self) -> str:
        return self.metadata["dataset"]

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "local"))

    def _to_frame(self, records: pd.DataFrame | list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
        # Columns the client did not send become NaN and are imputed by the pipeline.
        df = df.reindex(columns=self.feature_columns)
        for col in self.feature_columns:
            if col in self.categorical_columns:
                df[col] = df[col].astype(object)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def predict_proba(self, records: pd.DataFrame | list[dict]) -> np.ndarray:
        return self.pipeline.predict_proba(self._to_frame(records))[:, 1]

    def decide(self, pd_: np.ndarray) -> list[str]:
        return ["approve" if p < self.threshold else "decline" for p in np.asarray(pd_)]

    def score_records(self, records: pd.DataFrame | list[dict]) -> list[dict]:
        proba = self.predict_proba(records)
        scores = self.scorecard.score(proba)
        decisions = self.decide(proba)
        return [
            {"probability_default": float(p), "score": int(s), "decision": d}
            for p, s, d in zip(proba, scores, decisions, strict=True)
        ]


def scorecard_from_config(cfg: dict) -> ScoreCard:
    return ScoreCard(**cfg["scorecard"])


def scorecard_dict(card: ScoreCard) -> dict:
    return asdict(card)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scoring of a CSV/parquet file")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    scorer = CreditScorer.load(args.model_dir)
    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    result = pd.DataFrame(scorer.score_records(df))
    result.to_csv(args.output, index=False)
    print(result["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
