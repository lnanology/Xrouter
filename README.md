# XRouter

An OpenAI-compatible AI Intelligence Gateway / Router. Point any OpenAI SDK
at `http://localhost:20128/v1` — XRouter decides which provider/model
actually serves each request (local Ollama, Groq, OpenRouter, Gemini, or any
other OpenAI-compatible endpoint), with health checking, circuit breaking,
bounded retries, quota-aware fallback, streaming, and caching.

This is **Phase 1** (fast, stable, low-cost, recoverable — no agents, no
Kubernetes/Kafka/Redis, no quota/CAPTCHA/IP bypass of any kind) plus four
pieces of **Phase 2**: a request-level task classifier, a telemetry-driven
performance controller that auto-tunes routing weights, an opt-in race
mode that dispatches the top candidates concurrently, and a confidence/
quality-gate engine that catches degenerate responses and retries them
with a different candidate.

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
- `config/routing.yaml` — default routing policy, fallback depth,
  `task_aware_policy` (default `true`) — when on, each request is
  classified (chat/code/research/reasoning/creative/tool_use +
  complexity) and routed under the policy that classification suggests,
  unless the client passes an explicit `routing_policy` — plus
  `race_mode_enabled` / `race_candidate_count` (see Race mode below) and
  `quality_gate_enabled` / `quality_gate_min_score` / `max_quality_retries`
  (see Quality gate below).
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

`model: "auto"` lets the router pick the best available model. With no
explicit `routing_policy`, XRouter classifies the request (task type +
complexity) and picks a policy for you — e.g. a trivial chat message
routes `fastest`, a multi-step refactor or a tool-calling request routes
`quality`/`reliable`. Passing a specific known id (e.g.
`"ollama/llama3.1"`) pins a preference but XRouter still falls back
automatically if that provider/model is unhealthy. An optional
`routing_policy` field always overrides the classifier and selects
`fastest | cheapest | reliable | quota_aware | quality | balanced`
directly.

Admin endpoints require a bearer token (`XROUTER_ADMIN_TOKEN` in `.env`; if
unset, a random token is generated at startup and printed once to
`data/xrouter.log`). `/admin/metrics` includes a `performance` section —
per-provider EMA latency/success rate and the resulting routing weight
multiplier (neutral `1.0` until a provider has at least 10 samples):

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

### Race mode

Off by default (`routing.race_mode_enabled: false` in `config/routing.yaml`)
so a misconfigured server never starts silently burning quota on multiple
providers per request. Turn it on server-wide, then opt individual
requests in with `"race": true`:

```bash
curl http://localhost:20128/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}], "race": true}'
```

XRouter dispatches the top `routing.race_candidate_count` ranked
candidates (default 2) concurrently, returns whichever answers first, and
cancels whatever's still in flight — the response's `xrouter` metadata
carries `"race": true` and `"race_candidates": N` so you can see it
happened. If every raced candidate fails, XRouter falls back to the
remaining candidates one at a time, same as the non-race path. Race mode
only applies to non-streaming requests in this phase — `stream: true`
silently ignores the `race` flag (see `app/execution/race.py` for why).

### Quality gate

Also off by default (`routing.quality_gate_enabled: false`) — a failed
gate costs an extra provider call, so it doesn't turn on silently either.
When enabled, every non-streaming response is scored by
`app/intelligence/quality_gate.py`: an empty reply, output the provider
itself cut short (`finish_reason: "length"`), a small local model stuck
repeating one word or character, or (when the request forced a specific
tool via `tool_choice`) a response with no tool call at all. None of this
is a judgment of whether the *content* is factually right — that would
need paying for a second model call to grade the first, which XRouter
doesn't do by default. A response scoring below `quality_gate_min_score`
(default `0.5`) gets retried with the next untried candidate from the same
routing decision, up to `max_quality_retries` times (default `1`); if
every candidate is exhausted, XRouter returns the best response it found
rather than erroring — the assessment is always attached so you can see
what happened:

```json
"xrouter": {
  "provider": "ollama", "model": "qwen3:latest", "...": "...",
  "quality": {"score": 1.0, "passed": true, "reasons": []},
  "quality_retries": 1
}
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

110 tests across `tests/unit`, `tests/reliability`, `tests/routing`,
`tests/execution`, `tests/providers`, `tests/integration` — circuit
breaker state machine, bounded retry/backoff, quota risk escalation,
cache TTL/volatility rules, router scoring/exclusion rules, fallback
chains (timeout/429/500/mid-stream failure), provider adapters against
mocked HTTP (Ollama + generic OpenAI-compatible), full-stack FastAPI
integration tests (including the "no provider reachable" degraded-but-alive
path and graceful shutdown), the task classifier (task-type detection,
complexity scoring, suggested policy, tool-use override), the performance
controller (neutral below `MIN_SAMPLES`, EMA latency/success weighting,
graceful start/stop of its background snapshot loop), race mode
(fastest-candidate-wins, loser cancellation, all-raced-failed fallback,
`race.started`/`race.completed` events, candidate-count clamping, and the
server-switch/per-request-opt-in/no-streaming gating logic), and the
quality gate (empty/truncated/degenerate-repetition/missing-forced-
tool-call detection, retry-to-next-untried-candidate, best-effort return
when every candidate still fails, `quality.failed` events, and the
`max_quality_retries: 0` no-retry-but-still-assess case).

Verified end-to-end against real hardware: real Ollama (non-streaming and
`stream=true`, both producing correctly formatted chunks/`[DONE]`), and a
real macOS + Python 3.14 run of the full suite (the "no providers
reachable" tests now explicitly disable every provider rather than relying
on the host having none installed, and the cache fixture uses
`asyncio.run()` instead of the now-removed implicit-event-loop fallback).

## What's implemented (Phase 1 + partial Phase 2)

**Phase 1:** FastAPI gateway · OpenAI-compatible `/v1/models` +
`/v1/chat/completions` (incl. `stream=true`) · Provider adapter interface +
Ollama / generic OpenAI-compatible / Groq / OpenRouter / Gemini adapters ·
Provider + model registry · Adaptive router with 6 policies and
score-based exclusion · Background health monitor · Circuit breaker
(CLOSED/OPEN/HALF_OPEN) · Bounded retry with backoff + jitter (honors
`Retry-After`) · Quota tracker/limiter (reactive risk levels, no
assumptions about free-tier limits) · Ordered fallback chain (chat +
streaming, with a first-chunk fallback boundary for streams) ·
Per-provider/global concurrency limiter · L1 (memory) + L2 (SQLite) cache
with volatile-keyword exclusion · SQLite telemetry (requests, attempts,
provider health, benchmarks, events) · Metrics with P50/P95/P99, not just
averages · Admin API (token-protected) · CLI · Structured errors
everywhere, no infinite retries, no crash on a missing API key or missing
Ollama install.

**Phase 2 (so far):** Task Classifier (`app/intelligence/task_classifier.py`)
— keyword-heuristic task-type detection (chat/code/research/reasoning/
creative/tool_use) and complexity scoring, feeding a suggested routing
policy that's always overridable by an explicit client `routing_policy`,
gated by `routing.task_aware_policy`. Performance Controller
(`app/routing/performance_controller.py`) — EMA-based per-provider
latency/success tracking (neutral until `MIN_SAMPLES=10`, to avoid
oscillation from small sample sizes) feeding a weight multiplier into the
scorer, with a background loop persisting snapshots to `routing_metrics`
and emitting `routing.changed` events on significant weight shifts;
visible live via `/admin/metrics`. Race Mode (`app/execution/race.py`) —
opt-in (`routing.race_mode_enabled` + per-request `"race": true`)
concurrent dispatch of the top `race_candidate_count` ranked candidates
for non-streaming chat completions, returning whichever answers first and
cancelling the rest; reuses the exact same breaker/quota/retry/performance
bookkeeping as the sequential fallback chain via a shared
`try_candidate()` helper, and falls back to the remaining candidates
sequentially if every raced one fails. Confidence/Quality-Gate Engine
(`app/intelligence/quality_gate.py`) — opt-in (`routing.quality_gate_enabled`)
deterministic scoring of a non-streaming response (empty, truncated,
degenerate word/character repetition, missing forced tool call) that
retries a below-threshold response with the next untried candidate, up to
`max_quality_retries` times, always attaching the assessment to
`xrouter.quality` and never erroring out even if nothing better is found.

## What's not implemented yet (by design — see Phase 2-5 in the spec)

Task-complexity-driven multi-agent orchestration (Planner/Researcher/
Critic/Verifier), DAG executor, RAG/memory, plugin loader, browser/web-AI
adapter, network failover/VPN layer, PostgreSQL migration, LLM-graded (as
opposed to structural) quality assessment, and streaming versions of race
mode and the quality gate (both above only cover non-streaming requests —
a streamed response has already reached the client chunk by chunk by the
time either could act on it). Their directories exist as reserved, empty
packages (`app/agents`, `app/execution/{dag,cancellation}.py`,
`app/network`, `app/plugins`) so the rest of Phase 2+ has a home without
restructuring what's already built.

## Project layout

See the full tree below (or run `find app config scripts tests -type f`).
