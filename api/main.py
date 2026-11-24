"""FastAPI service: POST /score and GET /health.

The model directory is taken from the MODEL_DIR env variable (default: ./models).
The request schema is chosen from the dataset the model was trained on.
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import REQUEST_SCHEMAS, GMSCApplication, HealthResponse, ScoreResponse
from credit_scoring import __version__
from credit_scoring.predict import CreditScorer

logger = logging.getLogger("credit_scoring.api")


def _load_scorer(model_dir: Path) -> CreditScorer | None:
    try:
        return CreditScorer.load(model_dir)
    except FileNotFoundError:
        logger.warning("No model found in %s; run `make train` first", model_dir)
        return None


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    model_dir = Path(model_dir or os.getenv("MODEL_DIR", "models"))
    scorer = _load_scorer(model_dir)
    request_model = REQUEST_SCHEMAS[scorer.dataset] if scorer else GMSCApplication

    app = FastAPI(
        title="Credit Scoring API",
        version=__version__,
        description="Probability of default, credit score (300-850) and approve/decline decision.",
    )
    app.state.scorer = scorer

    @app.get("/health", response_model=HealthResponse)
    def health():
        if scorer is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(status="model_not_loaded", model_loaded=False).model_dump(),
            )
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=scorer.metadata.get("model_name"),
            model_version=scorer.version,
            dataset=scorer.dataset,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(application: request_model) -> ScoreResponse:  # type: ignore[valid-type]
        if scorer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        result = scorer.score_records([application.model_dump()])[0]
        return ScoreResponse(**result, threshold=scorer.threshold, model_version=scorer.version)

    return app


app = create_app()
