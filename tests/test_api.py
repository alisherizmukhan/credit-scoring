import pytest
from fastapi.testclient import TestClient

from api.main import create_app

VALID = {
    "revolving_utilization": 0.35,
    "age": 42,
    "past_due_30_59": 0,
    "debt_ratio": 0.28,
    "monthly_income": 6500,
    "open_credit_lines": 7,
    "times_90_days_late": 0,
    "real_estate_loans": 1,
    "past_due_60_89": 0,
    "dependents": 2,
}


@pytest.fixture
def client(trained):
    return TestClient(create_app(trained["model_dir"]))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["dataset"] == "gmsc"


def test_score_valid_request(client):
    r = client.post("/score", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_default"] <= 1
    assert 300 <= body["score"] <= 850
    assert body["decision"] in {"approve", "decline"}
    assert body["decision"] == (
        "approve" if body["probability_default"] < body["threshold"] else "decline"
    )


def test_riskier_client_gets_lower_score(client):
    safe = client.post("/score", json=VALID).json()
    risky = client.post(
        "/score",
        json={**VALID, "revolving_utilization": 1.0, "past_due_30_59": 4, "times_90_days_late": 3},
    ).json()
    assert risky["probability_default"] > safe["probability_default"]
    assert risky["score"] < safe["score"]


def test_optional_fields_may_be_null(client):
    r = client.post("/score", json={**VALID, "monthly_income": None, "dependents": None})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "patch",
    [
        {"age": 10},  # below 18
        {"debt_ratio": -1},  # negative
        {"age": "forty"},  # wrong type
        {"unexpected_field": 1},  # extra fields are forbidden
    ],
)
def test_invalid_requests_are_rejected(client, patch):
    assert client.post("/score", json={**VALID, **patch}).status_code == 422


def test_missing_required_field(client):
    body = {k: v for k, v in VALID.items() if k != "age"}
    assert client.post("/score", json=body).status_code == 422


def test_service_without_model(tmp_path):
    client = TestClient(create_app(tmp_path / "empty"))
    assert client.get("/health").status_code == 503
    assert client.post("/score", json=VALID).status_code == 503
