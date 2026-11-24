# Uses the project virtualenv automatically if it exists (Windows or Unix layout).
PY := $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))

.PHONY: help install data train train-fast test lint format drift notebook api mlflow-ui docker-up docker-down clean

help:
	@echo "install     - install package with train/dev extras"
	@echo "data        - download and prepare the dataset (data/processed)"
	@echo "train       - CV + Optuna + MLflow, saves models/ and reports/"
	@echo "train-fast  - quick run (few trials) for smoke checks"
	@echo "test        - pytest with coverage"
	@echo "lint        - ruff check + format check"
	@echo "drift       - PSI report: train vs. a simulated new sample"
	@echo "notebook    - execute notebooks/01_eda.ipynb in place"
	@echo "api         - run FastAPI locally on :8000"
	@echo "mlflow-ui   - MLflow UI for the local tracking DB on :5000"
	@echo "docker-up   - docker compose up (api + mlflow)"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[train,dev]"

data:
	$(PY) -m credit_scoring.data

train:
	$(PY) -m credit_scoring.train

train-fast:
	$(PY) -m credit_scoring.train --fast

test:
	$(PY) -m pytest --cov --cov-report=term-missing --cov-fail-under=70

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

drift:
	$(PY) -m credit_scoring.drift --simulate

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov catboost_info
