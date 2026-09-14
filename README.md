# XRouter

An OpenAI-compatible AI Intelligence Gateway / Router. Point any OpenAI SDK
at `http://localhost:20128/v1` — XRouter decides which provider/model
actually serves each request (local Ollama, Groq, OpenRouter, Gemini, or any
other OpenAI-compatible endpoint), with health checking, circuit breaking,
bounded retries, quota-aware fallback, streaming, and caching.

This is **Phase 1**: fast, stable, low-cost, recoverable. No agents, no
Kubernetes/Kafka/Redis, no quota/CAPTCHA/IP bypass of any kind.

## Quick start

```bash
cd xrouter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # optional: add GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY

bash scripts/start.sh          # starts in background, logs -> data/xrouter.log
bash scripts/health.sh         # or: curl http://localhost:20128/health
```

Or via the CLI:

```bash
python3 -m app.cli start
python3 -m app.cli status
python3 -m app.cli models
python3 -m app.cli providers
python3 -m app.cli benchmark
python3 -m app.cli logs
python3 -m app.cli stop
```

Ollama is the default local/free provider (`config/providers.yaml`,
`base_url: http://localhost:11434`). If Ollama isn't installed or running,
XRouter still starts fine — it just reports that provider as offline and
serves a structured 503 until at least one provider is healthy.

## Configuring providers

Everything is config-driven — no code changes needed to add/remove a
provider:

- `config/providers.yaml` — which providers exist, their `type` (adapter),
  `base_url`, and which environment variable holds their API key
  (`api_key_env`). A provider with no key configured disables itself
  gracefully.
- `.env` (copy from `.env.example`) — the actual secret values. Never
  committed (see `.gitignore`).
- `config/routing.yaml` — default routing policy and fallback depth.
- `config/models.yaml` — optional score overrides per model, applied on top
  of whatever each adapter self-reports.

Adding a brand-new *type* of provider (not just a new OpenAI-compatible
endpoint) means writing one adapter class implementing
`app/providers/base.py`'s `Provider` interface and registering it in
`app/providers/factory.py` — core/routing/reliability code never needs to
change.

## API usage

```bash
curl http://localhost:20128/v1/models

curl http://localhost:20128/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'

# streaming
curl -N http://localhost:20128/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

`model: "auto"` lets the router pick the best available model under the
default policy. Passing a specific known id (e.g. `"ollama/llama3.1"`) pins
a preference but XRouter still falls back automatically if that
provider/model is unhealthy. An optional `routing_policy` field selects
`fastest | cheapest | reliable | quota_aware | quality | balanced`.

Admin endpoints require a bearer token (`XROUTER_ADMIN_TOKEN` in `.env`; if
unset, a random token is generated at startup and printed once to
`data/xrouter.log`):

```bash
curl http://localhost:20128/health/providers
curl http://localhost:20128/admin/providers   -H "Authorization: Bearer $TOKEN"
curl http://localhost:20128/admin/metrics     -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:20128/admin/benchmark -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:20128/admin/reload    -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:20128/admin/providers/groq/enable   -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:20128/admin/providers/groq/disable  -H "Authorization: Bearer $TOKEN"
curl -X POST "http://localhost:20128/admin/providers/groq/cooldown?seconds=60" -H "Authorization: Bearer $TOKEN"
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

68 tests across `tests/unit`, `tests/reliability`, `tests/routing`,
`tests/providers`, `tests/integration` — circuit breaker state machine,
bounded retry/backoff, quota risk escalation, cache TTL/volatility rules,
router scoring/exclusion rules, fallback chains (timeout/429/500/mid-stream
failure), provider adapters against mocked HTTP (Ollama + generic
OpenAI-compatible), and full-stack FastAPI integration tests (including the
"no provider reachable" degraded-but-alive path and graceful shutdown).

## What's implemented (Phase 1)

FastAPI gateway · OpenAI-compatible `/v1/models` + `/v1/chat/completions`
(incl. `stream=true`) · Provider adapter interface + Ollama / generic
OpenAI-compatible / Groq / OpenRouter / Gemini adapters · Provider + model
registry · Adaptive router with 6 policies and score-based exclusion ·
Background health monitor · Circuit breaker (CLOSED/OPEN/HALF_OPEN) ·
Bounded retry with backoff + jitter (honors `Retry-After`) · Quota
tracker/limiter (reactive risk levels, no assumptions about free-tier
limits) · Ordered fallback chain (chat + streaming, with a first-chunk
fallback boundary for streams) · Per-provider/global concurrency limiter ·
L1 (memory) + L2 (SQLite) cache with volatile-keyword exclusion · SQLite
telemetry (requests, attempts, provider health, benchmarks, events) ·
Metrics with P50/P95/P99, not just averages · Admin API (token-protected) ·
CLI · Structured errors everywhere, no infinite retries, no crash on a
missing API key or missing Ollama install.

## What's not implemented yet (by design — see Phase 2-5 in the spec)

Task-complexity-driven multi-agent orchestration (Planner/Researcher/
Critic/Verifier), DAG executor, race mode, confidence/quality-gate engine,
RAG/memory, plugin loader, browser/web-AI adapter, network failover/VPN
layer, performance controller that auto-tunes routing weights from
telemetry, PostgreSQL migration. Their directories exist as reserved,
empty packages (`app/agents`, `app/execution/{race,dag,cancellation}.py`,
`app/intelligence`, `app/network`, `app/plugins`) so Phase 2+ work has a
home without restructuring Phase 1.

## Project layout

See the full tree below (or run `find app config scripts tests -type f`).
