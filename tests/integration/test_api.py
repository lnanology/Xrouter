"""Full-stack tests against the real FastAPI app (lifespan included), same
as what `curl http://localhost:20128/...` exercises. No real Ollama/cloud
provider is required — with none reachable, the app must still start and
serve /health, and /v1/chat/completions must return a structured 503
rather than hanging or crashing (this is the graceful-degradation
requirement from section 三十七)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["name"] == "XRouter"


def test_health_returns_200_even_with_no_providers_reachable(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "models_available" in body


def test_health_providers(client):
    resp = client.get("/health/providers")
    assert resp.status_code == 200
    assert "providers" in resp.json()


def test_v1_models_returns_list_shape(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


def test_chat_completions_structured_error_when_no_providers(client):
    resp = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    body = resp.json()
    assert "detail" in body
    assert body["detail"]["type"] == "xrouter_error"


def test_streaming_structured_error_when_no_providers(client):
    with client.stream(
        "POST", "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert '"type": "xrouter_error"' in body


def test_admin_requires_auth(client):
    resp = client.get("/admin/providers")
    assert resp.status_code == 401


def test_admin_with_valid_token(client):
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    resp = client.get("/admin/providers", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "providers" in resp.json()


def test_admin_metrics_reflects_recorded_requests(client):
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    resp = client.get("/admin/metrics", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["metrics"]["counters"]["requests_total"] >= 1


def test_request_body_too_large_rejected(client):
    from app.core.config import get_settings

    settings = get_settings()
    original_limit = settings.server.max_request_body_bytes
    settings.server.max_request_body_bytes = 10  # anything above this triggers 413
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "this body is well over ten bytes"}]},
        )
        assert resp.status_code == 413
    finally:
        settings.server.max_request_body_bytes = original_limit


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
