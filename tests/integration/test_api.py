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


def _disable_all_providers(client):
    """Force a genuine 'no providers available' state. We cannot rely on the
    test machine simply lacking Ollama/cloud keys (e.g. a dev's real Mac may
    have Ollama installed and running), so explicitly disable every
    configured provider on this test's own AppContext instead."""
    ctx = app.state.context
    for pid in ctx.providers.all():
        ctx.providers.set_enabled(pid, False)


def test_chat_completions_structured_error_when_no_providers(client):
    _disable_all_providers(client)
    resp = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 503
    body = resp.json()
    assert "detail" in body
    assert body["detail"]["type"] == "xrouter_error"


def test_streaming_structured_error_when_no_providers(client):
    _disable_all_providers(client)
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


def test_admin_benchmark_history_requires_auth(client):
    resp = client.get("/admin/benchmark/history")
    assert resp.status_code == 401


def test_admin_benchmark_and_history_are_graceful_with_no_providers_reachable(client):
    # This suite runs against the real, persistent data/xrouter.sqlite3 (not
    # a fresh per-test DB), so the benchmarks table may already hold rows
    # from earlier real usage -- asserting the history comes back empty is
    # never safe here. Instead assert the real invariant: a benchmark run
    # with every provider disabled adds no new row, whatever was there
    # before.
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    _disable_all_providers(client)

    before = client.get("/admin/benchmark/history", headers={"Authorization": f"Bearer {token}"}).json()["benchmarks"]

    resp = client.post("/admin/benchmark", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []  # no enabled/reachable provider in this environment

    resp = client.get("/admin/benchmark/history", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["benchmarks"] == before  # no new row persisted by a no-op run


def test_admin_ab_routing_summary_requires_auth(client):
    resp = client.get("/admin/ab_routing/summary")
    assert resp.status_code == 401


def test_admin_ab_routing_summary_reports_the_real_running_config(client):
    # ab_routing is off by default in config/config.yaml. Mirrors the lesson
    # from the Automated Benchmark piece's own two follow-up fixes: this
    # suite runs against the real, persistent data/xrouter.sqlite3, so
    # ab_results may already hold rows from an earlier run (possibly with a
    # different config) -- never assert `results` comes back empty. Instead
    # assert the shape and that `enabled`/`variants` reflect the actual
    # running Settings.
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    resp = client.get("/admin/ab_routing/summary", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] == ctx.settings.ab_routing.enabled
    assert body["variants"] == ctx.settings.ab_routing.variants
    assert isinstance(body["results"], dict)


def test_admin_policy_learning_weights_requires_auth(client):
    resp = client.get("/admin/policy_learning/weights")
    assert resp.status_code == 401


def test_admin_policy_learning_relearn_requires_auth(client):
    resp = client.post("/admin/policy_learning/relearn")
    assert resp.status_code == 401


def test_admin_policy_learning_weights_reports_the_real_running_config(client):
    # policy_learning is off by default. Same lesson as the ab_routing
    # summary test above: this suite runs against the real, persistent
    # data/xrouter.sqlite3, so learned_overrides may already hold entries
    # from an earlier run -- never assert it's empty, only that the shape
    # and the enabled/variants fields match the real running Settings.
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    resp = client.get("/admin/policy_learning/weights", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] == ctx.settings.policy_learning.enabled
    assert body["variants"] == ctx.settings.ab_routing.variants
    assert isinstance(body["learned_overrides"], dict)


def test_admin_policy_learning_relearn_is_graceful_regardless_of_data(client):
    # A manual relearn must always return 200 with an "applied" key,
    # whether or not there's enough real ab_results data to act on -- never
    # erroring, and never assuming a specific outcome either way.
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    resp = client.post("/admin/policy_learning/relearn", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert "applied" in resp.json()


def test_admin_evolution_status_requires_auth(client):
    resp = client.get("/admin/evolution/status")
    assert resp.status_code == 401


def test_admin_evolution_run_once_requires_auth(client):
    resp = client.post("/admin/evolution/run_once")
    assert resp.status_code == 401


def test_admin_evolution_status_reports_pending_on_a_fresh_engine(client):
    # Unlike ab_results/learned_overrides (persisted in the real DB across
    # runs), EvolutionEngine.pending is pure in-process state scoped to one
    # AppContext -- this test's own TestClient triggers a fresh startup(),
    # so a freshly-built engine has never recorded a nudge and asserting
    # emptiness here is safe (unlike the persisted-DB reads above).
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    resp = client.get("/admin/evolution/status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] == ctx.settings.evolution.enabled
    assert body["interval_seconds"] == ctx.settings.evolution.interval_seconds
    assert body["pending"] == {}


def test_admin_evolution_run_once_is_graceful_regardless_of_data(client):
    # A manual cycle must always return 200 with the top-level shape,
    # whether or not there's enough real ab_results/pending-nudge data to
    # act on -- never a specific outcome asserted, same reasoning as the
    # policy_learning relearn smoke test above.
    ctx = app.state.context
    token = ctx.settings.server.admin_token
    resp = client.post("/admin/evolution/run_once", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"evaluated", "learn", "pending"}


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


def test_dag_run_empty_nodes_returns_400(client):
    resp = client.post("/v1/dag/run", json={"nodes": []})
    assert resp.status_code == 400
    assert resp.json()["detail"]["type"] == "xrouter_error"


def test_dag_run_cycle_returns_400(client):
    resp = client.post("/v1/dag/run", json={"nodes": [
        {"id": "a", "depends_on": ["b"], "messages": [{"role": "user", "content": "hi"}]},
        {"id": "b", "depends_on": ["a"], "messages": [{"role": "user", "content": "hi"}]},
    ]})
    assert resp.status_code == 400
    assert "cycle" in resp.json()["detail"]["message"]


def test_dag_run_well_formed_but_no_providers_returns_200_with_failed_node(client):
    # A valid DAG still gets a 200 with per-node status — unlike the plain
    # /v1/chat/completions 503, a DAG run's "something failed" is reported
    # inside the body since a run can be partially successful. Explicitly
    # disable every provider rather than relying on the host having none
    # reachable (a dev's real Mac may have Ollama installed and running).
    # The prompt text must also be unique: the response cache is keyed on
    # model+messages and backed by the real on-disk data/xrouter.sqlite3,
    # so a plain "hi"/"auto" request -- also used by
    # test_admin_metrics_reflects_recorded_requests above, which runs with
    # providers enabled -- can still be served from a same-session cache
    # hit even with every provider disabled, since the cache is checked
    # before providers are.
    _disable_all_providers(client)
    resp = client.post(
        "/v1/dag/run",
        json={"nodes": [{"id": "a", "messages": [{"role": "user", "content": "dag no-providers probe, do not cache-collide"}]}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["nodes"][0]["status"] == "failed"


def test_plan_run_no_providers_returns_503(client):
    # Mirrors test_chat_completions_structured_error_when_no_providers:
    # the planning call itself is a normal handle_chat() call, so with no
    # provider able to even answer it, NoAvailableModelError propagates as
    # a structured 503 -- there's no plan yet to report a partial result
    # for. Unique task text avoids the cache-collision trap documented on
    # the DAG no-providers test above.
    _disable_all_providers(client)
    resp = client.post(
        "/v1/plan/run",
        json={"task": "plan no-providers probe, do not cache-collide"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["type"] == "xrouter_error"


def test_agents_run_no_providers_returns_503(client):
    # Same shape as test_plan_run_no_providers_returns_503: with no
    # provider able to answer even the classifier-driven first call
    # (solver for a trivial/simple task, or the Planner for anything
    # harder), NoAvailableModelError propagates as a structured 503.
    _disable_all_providers(client)
    resp = client.post(
        "/v1/agents/run",
        json={"task": "agents no-providers probe, do not cache-collide"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["type"] == "xrouter_error"


def test_research_no_providers_returns_503(client):
    _disable_all_providers(client)
    resp = client.post(
        "/v1/research",
        json={"query": "research no-providers probe, do not cache-collide"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["type"] == "xrouter_error"
