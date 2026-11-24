FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime required by LightGBM; curl: container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer), then the code.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY api ./api

RUN useradd --create-home --uid 1000 appuser
USER appuser

# The trained model is mounted at runtime (see docker-compose.yml)
ENV MODEL_DIR=/app/models
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
