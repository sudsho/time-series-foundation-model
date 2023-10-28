from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint_when_no_ckpt():
    # in tests we don't have a real checkpoint -> health is still ok,
    # but model_loaded should be False
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "model_loaded" in body


def test_forecast_returns_503_without_ckpt():
    with TestClient(app) as client:
        body = {"series": [0.0] * 256, "context_length": 256}
        r = client.post("/forecast", json=body)
        # since startup couldn't load a real ckpt, expect 503
        assert r.status_code == 503


def test_forecast_validates_short_series():
    with TestClient(app) as client:
        # below pydantic min_items=8
        r = client.post("/forecast", json={"series": [1.0, 2.0]})
        assert r.status_code == 422
