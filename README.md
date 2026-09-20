# XRouter

An OpenAI-compatible AI Intelligence Gateway / Router. Point any OpenAI SDK
at `http://localhost:20128/v1` — XRouter decides which provider/model
actually serves each request (local Ollama, Groq, OpenRouter, Gemini, or any
other OpenAI-compatible endpoint), with health checking, circuit breaking,
bounded retries, quota-aware fallback, streaming, and caching.

This is **Phase 1** (fast, stable, low-cost, recoverable — no agents, no
Kubernetes/Kafka/Redis, no quota/CAPTCHA/IP bypass of any kind), five
pieces of **Phase 2**: a request-level task classifier, a telemetry-driven
performance controller that auto-tunes routing weights, an opt-in race
mode that dispatches the top candidates concurrently, a quality gate
that catches degenerate responses and retries them with a different
candidate, and a DAG executor that runs a client-supplied graph of chat-
completion nodes with wave-based concurrency and `{{node_id}}` output
substitution — plus all of **Phase 3** (multi-agent orchestration): a
Planner that turns a single free-form task into an explicit DAG (an
actual LLM call, not a fixed template) and runs it through that same DAG
executor, a Verifier that closes the plan → execute → verify loop, a
real, config-driven Web Search + Web Fetch tools (both Tavily-backed)
that a DAG/Planner node can invoke through an XRouter-executed call →
tool → call loop —
not a fake placeholder that just echoes the query back — a Critic that
reviews a single node's own output against its own instruction and
reruns that node if unsatisfied, distinct from the Verifier's once-at-
the-end whole-run check, a Dynamic Agent Team (`POST /v1/agents/run`)
that assembles whichever of the above pieces a task's own complexity
actually calls for (solver-only up through planner + specialists +
critic + research + verifier + synthesizer), a standalone Researcher
(`POST /v1/research`), and Memory/RAG — a real retrieve → augment-context
→ generate loop backed by a persistent, scope-isolated memory store, with
a swappable `Retriever`: always-on keyword search by default, or a
genuine opt-in embedding + cosine-similarity `Retriever` once
`retrieval.enabled` is turned on. All five pieces of **Phase 4**
are in too: an Evidence Graph that traces a finished answer's own claims
back to whichever DAG step actually produced each one, a Debate stage —
automatic at tier 4 — where an Advocate and a Skeptic argue for and
against the draft answer and a Judge reconciles both into a final,
strengthened one, a Counterfactual analysis that identifies an answer's
own load-bearing assumptions and how the answer would change if each one
didn't hold, Simulation, which actually re-answers the task under a
caller-supplied changed premise instead of just guessing what would
happen, and a Confidence Engine that rolls up whichever of those signals
a run actually produced into one free, automatic confidence score on
every response.

## Quick start

```bash
cd xrouter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # optional: add GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY / TAVILY_API_KEY

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
  `race_mode_enabled` / `race_candidate_count` (see Race mode below),
  `quality_gate_enabled` / `quality_gate_min_score` / `max_quality_retries`
  (see Quality gate below), `max_dag_nodes` (see DAG Executor below),
  `planner_routing_policy` / `max_plan_retries` / `max_verify_retries`
  (see Planner / Verifier below), `max_tool_iterations` (see Tools below),
  and `max_critique_retries` (see Critic below).
- `config/models.yaml` — optional score overrides per model, applied on top
  of whatever each adapter self-reports.
- `config/tools.yaml` — which XRouter-executed tools exist (currently
  `web_search`), their `enabled` flag, and which environment variable
  holds their API key (`api_key_env`) — same config-driven,
  graceful-degradation pattern as `providers.yaml`. A tool with no key
  configured stays registered but simply never shows up as "available".

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
curl "http://localhost:20128/admin/benchmark/history?limit=50" -H "Authorization: Bearer $TOKEN"
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
remaining candidates one at a time, same as the non-race path.

Streaming requests (`"stream": true`) are raced too, the same opt-in way —
`app/execution/race.py`'s `run_stream_race` races on *first-chunk
arrival* instead of a full response (`app/routing/fallback.py`'s
`try_candidate_stream`/`run_stream_chat` already commit to a candidate at
that same boundary for the non-race streaming path), then streams the
winner's remaining chunks through as they arrive. One thing streaming
racing has to do that non-streaming racing never needed to: a losing
candidate that already received its first chunk is holding an open
provider-side stream nothing will ever read further — `run_stream_race`
explicitly closes (`aclose()`s) every such generator so it can't leak a
connection, both for candidates it cancels outright and for the rarer case
where two candidates' first chunks land in the same instant and only one
can win.

### Quality gate

Off by default (`routing.quality_gate_enabled: false`) — a failed gate
costs an extra provider call, so it doesn't turn on silently either.
When enabled, every **non-streaming** response is scored by
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

**LLM-graded judgment** is the second, opt-in half of the gate —
`routing.quality_gate_llm_grading_enabled` (default `false`), nested
*inside* `quality_gate_enabled`, since it only ever runs on top of an
already-opt-in feature. Once both are on, a response that already passed
the free structural check above gets a real second opinion: XRouter asks
another model — via a forced `submit_quality_judgment` tool call, the same
pattern the Critic and Verifier (below) already use — whether the response
is actually accurate, complete, and on-topic, not just well-formed. A
"not satisfied" verdict is treated exactly like a structural failure and
retries the next untried candidate through the same `max_quality_retries`
budget (no separate cap); the feedback comes back on
`xrouter.quality.llm_feedback`:

```json
"quality": {"score": 1.0, "passed": false, "reasons": ["llm_graded_unsatisfactory"], "llm_feedback": "Cites the wrong year for the treaty."}
```

The judge call runs under its own routing policy,
`routing.quality_gate_judge_policy` (default `"quality"`, independent of
whatever policy the original request used — same reasoning as the
Planner's own dedicated `planner_routing_policy`), and prefers a provider
different from the one that actually answered when the ranked candidate
list offers one, to avoid a model grading its own homework. A structurally
broken response never reaches this step at all — no point paying for a
second opinion on an empty or truncated answer — and the judge call itself
fails open (a missing/malformed tool call, or no available judge candidate
at all, counts as satisfied) rather than ever blocking a response that
already passed the cheap check. **Cost note**: with this on, a
structural-pass-but-LLM-fail retry costs *two* extra provider calls per
attempt (the retried candidate, plus its own judge call) — `quality_gate_enabled`
alone already costs one extra call per failed structural retry, so this is
a genuine multiplier, which is exactly why it's a separate, nested,
off-by-default switch rather than bundled into `quality_gate_enabled`
itself.

One architectural note for anyone reading `app/core/engine.py`: unlike the
Critic/Verifier judge calls (which route through the normal
`ChatEngine.handle_chat`), this judge call deliberately bypasses
`handle_chat` and calls the router + `run_chat` directly. It has to —
this call happens *from inside* the quality gate itself, so routing it
back through `handle_chat` would re-enter the gate for the judge's own
response and, with LLM grading on, try to grade the judge's grading,
unboundedly.

**Streaming requests never run the quality gate** (structural or
LLM-graded), and this is a
deliberate, permanent scope decision rather than a gap waiting to be
filled — see `app/intelligence/quality_gate.py`'s module docstring for the
full reasoning. In short: a streamed response is already being sent to
the client chunk by chunk as it's produced, so by the time enough of it
exists to assess, there's nothing left to retry — the client has already
seen it. The alternatives all have real costs that weren't worth taking on
for this phase: gating on the first chunk alone can only ever check 1 of
the gate's 5 real signals (a forced tool call missing from that chunk) and
would be a misleadingly thin "quality gate"; buffering the whole response
server-side before forwarding anything would give the full 5-signal gate
but defeats the entire point of streaming (the client would wait for the
complete response either way, just receive it pre-chunked afterward).
Streaming race mode (above) was extended because it only ever needed a
first-chunk-arrival signal to begin with — the quality gate's signals
mostly aren't available that early.

### DAG Executor

Runs a client-supplied *explicit* graph of chat-completion nodes — the
graph itself is given, not generated (see Planner below for the piece
that generates one from a single free-form task and hands it to this
same executor). Each node is a normal chat-completion request plus an
`id` and an optional `depends_on` list; a node's message content can
reference an upstream
node's output with `{{node_id}}`, substituted with that node's extracted
response text before the node runs:

```bash
curl http://localhost:20128/v1/dag/run \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "capital", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
      {"id": "translate", "depends_on": ["capital"],
       "messages": [{"role": "user", "content": "Translate {{capital}} to Spanish"}]}
    ]
  }'
```

Nodes are layered into concurrent "waves" by topological order (Kahn's
algorithm) — independent nodes in the same wave run concurrently via
`asyncio.gather`, and every node runs through the real `ChatEngine`, so it
gets the exact same caching/routing/circuit-breaking/race/quality-gate
behavior as `/v1/chat/completions`. If a node fails, everything that
transitively depends on it is marked `"skipped"` without running, while
unrelated nodes still execute normally; the overall run is reported as
`"success"` (all nodes succeeded), `"partial"` (a mix), or `"failed"`
(none succeeded) — always `200 OK`, since a partially-successful DAG isn't
a server error. The request itself is validated before anything executes:
an empty node list, more than `routing.max_dag_nodes` nodes (default
`20`), duplicate ids, a self-dependency, an unknown dependency, or a cycle
all return `400` with a structured `xrouter_error` body.

### Tools (XRouter-executed, e.g. Web Search, Web Fetch)

A DAG/Planner node can opt into a real, XRouter-executed tool by name via
`"enable_tools": ["web_search"]` on that node — distinct from the plain
`tools`/`tool_choice` fields, which XRouter always passes through
unexecuted for the *client* to run, per the standard OpenAI contract. A
name listed in `enable_tools` is instead executed by XRouter itself:
`app/execution/tool_loop.py` drives the call → tool call → execute → feed
result back → call again cycle on top of the node's normal
`ChatEngine.handle_chat()` call, bounded by `routing.max_tool_iterations`
(default `3`, after which whatever the model last said — tool call or
not — is returned rather than looping forever):

```bash
curl http://localhost:20128/v1/dag/run \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "research", "enable_tools": ["web_search"],
       "messages": [{"role": "user", "content": "What is XRouter'"'"'s current stable release?"}]}
    ]
  }'
```

Only a name that is both *registered* (`app/tools/factory.py` has a
builder for it) and *configured* (its API key env var is actually set) is
ever executed; a node requesting anything else just falls back to a plain
call, and a tool call the model makes for a name that was never enabled is
left completely untouched in the response for the client to handle itself
— XRouter never silently "answers" a tool call it has no executor for. A
tool failure (bad arguments, HTTP error, timeout) is fed back to the model
as that tool call's own result (`"Tool error: ..."`) rather than failing
the node outright — the model can often recover by rephrasing or
answering without it.

Two tools ship today, both backed by [Tavily](https://tavily.com) — one
`TAVILY_API_KEY` covers both:

- **`web_search`** (`app/tools/web_search.py`) — Tavily's search API,
  returns a short list of relevant results (title, url, snippet).
- **`web_fetch`** (`app/tools/web_fetch.py`) — Tavily's *Extract* API,
  reads one specific URL in full and returns its extracted text. The
  natural complement to `web_search`: search finds candidate URLs,
  `web_fetch` reads one of them in full once a snippet isn't enough. Not
  a real browser — there's no JS execution or screenshots, Tavily's own
  extraction handles page rendering server-side. A node can enable both
  at once (`"enable_tools": ["web_search", "web_fetch"]`); `tool_loop.py`
  offers whichever are registered and configured, and the model picks.

```bash
curl http://localhost:20128/v1/dag/run \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "research", "enable_tools": ["web_search", "web_fetch"],
       "messages": [{"role": "user", "content": "Find XRouter'"'"'s GitHub repo and summarize its README."}]}
    ]
  }'
```

Both are on by default (`tools.web_search.enabled` / `tools.web_fetch.
enabled: true` in `config/tools.yaml`) — set `TAVILY_API_KEY` in `.env`
to actually turn them on; with no key, each stays gracefully absent
(`configured` is `False`, so `ToolRegistry.get()` returns `None`), the
same rule a provider with no API key follows. Registering a tool this
way costs nothing by itself: a node still has to list it in
`enable_tools` *and* the model still has to choose to call it before any
real request happens.

`web_fetch` was chosen deliberately over two other shapes for what
README used to call a bare "browser/web-AI adapter" bullet with no
further spec anywhere in the repo: a hand-rolled `httpx` GET + stdlib
HTML stripper (zero dependency, but no JS rendering and weaker
extraction) and real Playwright browser automation (a true "browser",
but a large new dependency plus browser binaries for a single optional
tool — disproportionate infrastructure). Tavily Extract reuses the exact
`TAVILY_API_KEY`/`httpx`-only integration `web_search` already
established — zero new dependency, zero new secret — while still doing
real server-side page extraction rather than a heuristic.

Adding a new tool type means writing one class implementing
`app/tools/base.py`'s `ExecutableTool` Protocol (`name`, `schema`,
`configured`, async `execute()`, async `close()`) and registering a
builder in `app/tools/factory.py` — no other code needs to change, same
pattern as adding a provider adapter; `web_fetch` itself needed no
changes to `ToolRegistry`, `tool_loop.py`, or `DagNodeRequest`/
`PlanNodeSpec`, confirming the extension point already generalizes.

### Critic (opt-in per node: `"critique": true`)

Per-node review — distinct from the Verifier below, which only ever
judges the *whole* run against the *original* task, once, after
everything has finished. A bad step three nodes deep can already have
poisoned everything downstream (via `{{node_id}}` substitution) long
before the Verifier ever gets a look, or if `verify` was never set at
all. Set `"critique": true` on a node and, right after it produces a
response, XRouter asks the Critic — another actual LLM call (forced
`submit_critique` tool call), not a heuristic — whether that node's own
output actually satisfies that node's own instruction. If not, the node
re-runs with the Critic's feedback appended, up to
`routing.max_critique_retries` times (default `1`), before accepting
whatever the last attempt produced:

```bash
curl http://localhost:20128/v1/dag/run \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {"id": "summary", "critique": true,
       "messages": [{"role": "user", "content": "Summarize this in exactly 3 bullet points: ..."}]}
    ]
  }'
```

The result's `nodes[].critique` reports the Critic's final judgment
(`{"satisfied": true, "feedback": null}` or similar) whenever a node had
`critique: true` — `null` for every node that didn't opt in. Off by
default, same reasoning as race mode/the quality gate/the Verifier: a
failed critique costs an extra call *and* re-runs that node, so it
shouldn't turn on silently. The Critic **fails open** exactly like the
Verifier — a missing/malformed tool call, or no provider able to answer
the critique call at all, counts as "satisfied" rather than blocking or
endlessly retrying a node that already produced *something*; and even
after every retry is exhausted with the Critic still unsatisfied, the
node's own status stays `"success"` — critique is a best-effort review,
never a pass/fail gate on whether the node ran. The retry loop itself
(`app/execution/critique_loop.py`) re-runs through the node's own
`responder` — a plain call, or `app/execution/tool_loop.py`'s tool loop
when the node also has `enable_tools` — so it never duplicates
provider-calling logic of its own.

The Planner can also set `critique: true` on a node it generates itself,
for a step it judges "genuinely matters and is easy to get subtly wrong
in one shot" (e.g. a final synthesis step) — see Planner below.

### Planner

The first piece of Phase 3 (multi-agent orchestration groundwork):
`POST /v1/plan/run` turns a single free-form task into an explicit DAG and
runs it through the exact same `DagExecutor` above — no separate
provider-calling logic. The plan itself is generated by an actual LLM
call (a forced `submit_plan` tool call, for structured output), not a
fixed template or keyword heuristic — consistent with XRouter's "no fake
placeholder functionality" principle:

```bash
curl http://localhost:20128/v1/plan/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Look up the capital of France, then translate it to Spanish"}'
```

```json
{
  "id": "plan_...",
  "plan": {"nodes": [
    {"id": "capital", "depends_on": [], "prompt": "..."},
    {"id": "translate", "depends_on": ["capital"], "prompt": "... {{capital}} ..."}
  ]},
  "plan_attempts": 1,
  "dag": { "...": "a full DagRunResponse, same shape as /v1/dag/run" },
  "latency_ms": 842.1
}
```

The planning call uses `routing.planner_routing_policy` (default
`quality`, since a bad plan wastes every node run under it) unless the
request sets its own `routing_policy`. If the model's `submit_plan` call
doesn't parse as JSON, doesn't match the expected node schema, or
produces a graph that wouldn't validate as a DAG (cycle, duplicate id,
dangling dependency, too many nodes) — reusing the DAG executor's own
validator, so a plan rejected here is guaranteed to be rejected for the
identical reason if it ever reached the executor directly — XRouter
re-prompts with the specific error, up to `routing.max_plan_retries`
times (default `2`), before giving up. Response codes: `503` if no
provider could even answer the planning call (nothing to report yet, same
as a plain `/v1/chat/completions` outage); `422` if every retry still
produced an unusable plan; `200` with the full `plan` + `dag` result
otherwise, `plan_attempts` telling you whether it took more than one try.

Each planned node can also carry three optional, richer fields the model
fills in itself when it judges a step needs them: `routing_policy`
overrides the routing policy for just that one step (e.g. `"quality"` for
a final synthesis step, while the rest stay on the default), `enable_tools`
lets a step use a real tool (see Tools above) if it genuinely needs
current or external information it can't answer from its own knowledge,
and `critique` (see Critic above) flags a step whose correctness genuinely
matters for independent per-node review. The `submit_plan` schema's
`enable_tools` is never a fixed list — it's built fresh per call from
`ToolRegistry.available_names()`, so a plan can never even syntactically
request a tool that isn't actually registered and configured right now;
`_validate_plan_shape()` also double-checks this itself after the call
returns, since not every provider strictly enforces a JSON-schema `enum`
on generated tool-call arguments, and a plan requesting an unknown tool is
rejected (and re-prompted, same as any other invalid plan) rather than
silently running that step with no tool at all.

#### Verifier (opt-in: `"verify": true`)

Closes the loop: pass `"verify": true` and, once the DAG finishes,
XRouter asks a Verifier — another actual LLM call (forced
`submit_verification` tool call), not a heuristic — whether the run
*genuinely* accomplished the original task. If not, the Verifier's own
feedback gets folded into the task's context and the whole plan+execute
cycle runs again, up to `routing.max_verify_retries` times (default `1`):

```bash
curl http://localhost:20128/v1/plan/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Draft a 3-bullet summary of the attached notes", "verify": true}'
```

```json
{
  "id": "plan_...",
  "plan": {"...": "the *winning* plan -- the last one generated"},
  "plan_attempts": 1,
  "dag": {"...": "the winning plan's DagRunResponse"},
  "verification": {"satisfied": true, "feedback": null},
  "replan_count": 1,
  "latency_ms": 1730.4
}
```

Off by default, same reasoning as race mode/the quality gate: a failed
verification costs an entire extra plan+execute cycle to retry, not just
one call, so it shouldn't turn on silently. The Verifier **fails open** —
if its own tool call is missing, malformed, or unanswerable, that counts
as "satisfied" rather than blocking the response or looping forever; a
best-effort second opinion should never hold a result hostage. This is
the full extent of Phase 3's first loop: Planner → DAG executor →
Verifier → (maybe) re-plan. Everything (the loop itself, the fail-open
behavior, the feedback actually reaching the re-plan's prompt) is
orchestrated by `app/execution/plan_runner.py`, independently of the
`POST /v1/plan/run` HTTP layer.

### Dynamic Agent Team (`POST /v1/agents/run`)

The piece the Task Classifier's own module docstring used to flag as "not
yet built": `app/agents/orchestrator.py` turns a single free-form task
straight into whichever team of the pieces above the task's own
complexity (`app/intelligence/task_classifier.py`, `0`..`4`) actually
calls for, and returns one final answer:

```bash
curl http://localhost:20128/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Design a caching strategy for a read-heavy API and justify the trade-offs"}'
```

```json
{
  "id": "orch_...",
  "team": ["planner", "specialists", "critic", "verifier", "debate"],
  "complexity": 4,
  "task_type": "reasoning",
  "answer": "...",
  "dag": { "...": "set whenever the team ran more than a bare single call" },
  "verification": { "...": "set only at tier 4" },
  "debate": { "...": "set only at tier 4, and only when the debate actually completed" },
  "evidence": { "...": "set only when trace_evidence was requested" },
  "counterfactual": { "...": "set only when trace_counterfactual was requested" },
  "simulations": [{ "...": "one entry per scenario in `simulate` that actually produced an answer" }],
  "confidence": { "score": 1.0, "label": "high", "reasons": [] },
  "latency_ms": 2140.7
}
```

`team` always reflects what actually ran, never what a tier "should" have
used. The tiers, straight from the routing table in the spec:

- **0-1 (trivial/simple):** a single fast model answers directly — the
  existing fast path, completely unchanged, zero extra calls, zero memory
  overhead. `team: ["solver"]`.
- **2 (medium):** one model answers, then the Critic reviews it and
  reruns it if unsatisfied (one `DagExecutor` node, `critique: true`).
  `team: ["solver", "critic"]`.
- **3 (hard):** the Planner designs a multi-step DAG ("specialists");
  every terminal node (nothing else depends on it) is forced to
  `critique: true` deterministically, rather than trusting the Planner to
  remember. Whenever more than one node ran, the Synthesizer composes the
  final answer from all of them. `team` gains `"synthesizer"` only when it
  was actually used, and `"research"` only when the plan itself used
  `enable_tools` on some step.
- **4 (very hard):** everything tier 3 does, plus a mandatory Verifier
  pass (reusing `run_plan_with_verification`'s own bounded
  re-plan-on-failure loop wholesale) and a stronger hint folded into the
  Planner's own context that a step may genuinely need Research — never
  forced, since forcing a tool call a task doesn't actually need would be
  exactly the "fake placeholder functionality" the project's rules forbid.

All five of Phase 4's pieces are wired in on top of these tiers — see
their own sections below for the full reasoning. Debate is the one
exception to "`team` only reflects what a tier actually ran, nothing
opt-in by default": it's a spec-named *standing* member of the tier-4
team (section 十九's own "very hard" table lists it right next to
Verification), so it runs automatically whenever complexity reaches tier
4, no flag required. Evidence Graph, Counterfactual, and Simulation are
all opt-in (`trace_evidence` / `trace_counterfactual` / `simulate`) since
none of the three is named in that table. Confidence Engine is neither
opt-in nor a `team`-list entry: it makes zero provider calls, so it just
runs automatically on every response at every tier, rolling up whichever
of `dag`/`verification`/`debate`/`evidence` a run actually produced into
`confidence`. "Coder" also isn't a separate stage: `task_type: code`
already gets a quality-biased routing policy from the Task Classifier,
which is the existing, real behavior this module reuses rather than
duplicating.
Response codes mirror the pieces it's built from: `503` if no provider
could even answer (nothing to report yet), `422` if the Planner exhausted
every retry, `503` (`OrchestrationError`) if a whole DAG's worth of steps
all failed and there's nothing honest to synthesize or fall back to.

### Researcher (`POST /v1/research`)

A focused, reusable research step, reachable directly or used internally
by the Orchestrator's higher tiers — not a new execution mechanism, it
reuses the exact same tool-execution loop a DAG node's own `enable_tools`
does:

```bash
curl http://localhost:20128/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What changed in the latest stable release of PostgreSQL?"}'
```

```json
{"answer": "...", "tool_available": true}
```

`tool_available` reports whether a real search tool (e.g. `web_search`)
was actually registered and configured for this call — not whether the
model chose to invoke it, which is always the model's own judgment, same
as any `enable_tools` node. Graceful degradation, not a fake answer: with
no search tool configured, the Researcher still answers from the model's
own knowledge, honestly — the system prompt tells the model plainly that
no search is available, so it can say when it isn't confident — rather
than fabricating results or refusing outright.

### Memory / RAG

Every Orchestrator run at tier ≥ 2 reads relevant memory before, and
writes a summary of its own answer after. Memory is a real SQLite-backed
table (`memory_entries`, `app/storage/repositories/memory.py`), grouped
by an opaque `scope` string (`OrchestrationRequest.scope`, `"global"` by
default — set it to a session/user id to keep recall from leaking across
unrelated callers) so retrieval never wanders outside where it should.

Retrieval itself (`app/retrieval/`) is a swappable `Retriever` Protocol —
the same "everything replaceable" pattern as `Provider`/`ExecutableTool`
— with **two** real implementations today. `KeywordRetriever` is the
always-on default: genuine substring/word-overlap search over stored
memory, zero extra cost, not a fixed template. `EmbeddingRetriever`
(`app/retrieval/embedding.py`) is a genuine embedding + cosine-similarity
search, **opt-in** via `retrieval.enabled` in `config/config.yaml`
(off by default — same "must not turn on silently" reasoning as
`race_mode_enabled`/`quality_gate_enabled`, since it spends a real
provider `embed()` call on every recall *and* every remember):

```yaml
retrieval:
  enabled: false                        # off by default
  embedding_provider: ollama            # resolved directly, no router
  embedding_model: nomic-embed-text     # `ollama pull nomic-embed-text` first
  max_candidates: 200                   # per-query scan cap, same shape as search()'s own LIMIT
```

`app/agents/orchestrator.py`'s `_build_retriever()` is the one place that
picks between them, from config — neither `_recall()` nor `_remember()`
needed to change shape when the second implementation was added, exactly
as `app/retrieval/base.py`'s own `Retriever` Protocol was designed to
allow. When enabled, `_remember()` also embeds the new summary at save
time (`MemoryRepository.save(..., embedding=...)`, stored as
`embedding_json`) so it's part of future queries' candidate pool.

Two deliberate design choices worth calling out. First, similarity is
computed in **pure Python** (`math.sqrt` + a hand-written dot-product
loop in `_cosine_similarity`), not with numpy: XRouter has no
numerical-computation dependency today, and per-scope memory volume is
small enough (`max_candidates`, the same shape as `search()`'s own
`LIMIT 200` scan cap) that a plain loop is genuinely fast enough —
adding numpy just to vectorize a loop over a few hundred short vectors
would itself be the "unnecessary infrastructure" the project's rules
warn against. Second, `EmbeddingRetriever` **fails open to an empty
result, never to `KeywordRetriever`**, on any failure — a missing
provider, an adapter that never declared `ProviderCapability.EMBEDDINGS`
(`embed()`'s base-class default raises
`ProviderCapabilityUnsupportedError` rather than inventing a fake
vector), or a real provider error. That's the same "nothing found" shape
a genuinely empty scope already produces, which keeps the two
retrievers' behavior simple to reason about independently rather than
silently cascading between two different notions of relevance.

`embed()` is implemented today for `OllamaAdapter` (`/api/embed`, the
only provider enabled by default) and `OpenAICompatibleAdapter`
(`/embeddings`, the generic adapter behind Groq/OpenRouter/self-hosted
servers) — both resolved directly by provider id + model name from
config, not through the router: there's exactly one caller and one fixed
use, so a router-based "pick a candidate for this capability" path would
be new routing infrastructure this single call site doesn't need.

Tiers 0-1 never touch memory at all, for the same "don't tax the fast
path" reason they skip every other piece of Phase 3.

### Evidence Graph (Phase 4, opt-in: `"trace_evidence": true`)

The first piece of Phase 4: once an Orchestrator run at tier ≥ 2
finishes, `app/intelligence/evidence.py` asks — another actual LLM call
(forced `submit_evidence_graph` tool call, not a heuristic) — which
concrete claims the final answer makes, and for each one, which DAG step
actually produced it:

```bash
curl http://localhost:20128/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Look up the capital of France, then translate it to Spanish", "trace_evidence": true}'
```

```json
{
  "...": "the rest of the OrchestrationResult",
  "evidence": {
    "claims": [
      {"claim": "Paris is the capital of France", "supported_by": ["capital"], "supported": true},
      {"claim": "\"París\" is Spanish for Paris", "supported_by": ["translate"], "supported": true}
    ]
  }
}
```

Despite the name, this ships as a flat claim → DAG-node-id mapping, not a
literal multi-hop graph — nothing in XRouter today produces or consumes
multi-hop evidence relationships, and building traversal machinery
nobody uses would be exactly the "unnecessary infrastructure" the
project's own rules forbid. It's also deliberately **DAG-node-level, not
URL-level**: the Researcher's `web_search` results reach the model as
prose its system prompt asks it to cite inline, but nothing in
`app/execution/tool_loop.py` captures those citations as structured,
addressable objects yet, so pretending to trace a claim to a specific URL
today would be faking data XRouter doesn't actually have. DAG node
outputs are the one thing this system genuinely has in structured form
today, so that's what v1 traces against — a URL-level layer can extend
this later without touching a single caller, the same "swappable, add
without breaking callers" shape `app/retrieval/`'s `Retriever` Protocol
already uses for the identical reason.

`supported_by` is never trusted blindly — its JSON-schema `enum` is built
fresh per call from the run's own real DAG node ids (plus
`"model_knowledge"` for an untraceable claim), and any reference that
slips past that enum anyway is dropped rather than propagated, the same
defense-in-depth `app/intelligence/planner.py`'s `_validate_plan_shape`
applies to a hallucinated `enable_tools` entry. Off by default (opt-in
`trace_evidence`, same reasoning as `verify`/`critique`/race mode — an
extra call shouldn't turn on silently) and fails open exactly like the
Critic/Verifier: a missing/malformed tool call, or no provider able to
answer at all, returns an empty evidence graph rather than losing the
answer that's already been produced — this is advisory metadata about a
response XRouter already committed to returning, never something that
should hold it hostage. A silent no-op at tier 0-1, since there's no DAG
there to trace any claim against.

### Debate (Phase 4, tier 4 only — no opt-in flag)

The second piece of Phase 4, and the only one so far that isn't opt-in:
spec section 十九's own "Very hard" team table lists `Planner + parallel
specialists + Research + Debate + Verification + Synthesizer` as the
tier-4 team, the same standing membership the Verifier already has there
— so Debate just runs, automatically, whenever complexity reaches tier 4.

Once tier 4's draft answer exists (the same draft `_synthesize_or_fallback`
already produces), three real calls stress-test it: an Advocate argues
it's correct and well-supported, a Skeptic argues the opposite — real
gaps or overstatements, not reflexive contrarianism — and a Judge is
shown both arguments plus the draft and writes ONE final, strengthened
answer. That Judge output *replaces* the draft as `OrchestrationResult.answer`
— Debate revises the answer, it doesn't just annotate it on the side:

```bash
curl http://localhost:20128/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Should we use microservices or a monolith for a 3-person startup building an MVP?"}'
```

```json
{
  "...": "the rest of the OrchestrationResult",
  "team": ["planner", "specialists", "critic", "verifier", "debate"],
  "answer": "... the judge's strengthened final answer ...",
  "debate": {
    "position": "... the pre-debate draft ...",
    "advocate": "... the case for the draft ...",
    "skeptic": "... the case against it ...",
    "resolution": "... same text as the top-level answer ..."
  }
}
```

Deliberately **not** wired in literally between DAG execution and
Verification the way spec section 十九's left-to-right list might
suggest — `app/execution/plan_runner.py`'s `run_plan_with_verification()`
is reused wholesale, the same reuse discipline the rest of Phase 3/4
already follows, so Debate instead runs on the DAG that already passed
verification (or was re-planned until it did). Debating a draft
verification was about to throw away would waste three calls on a losing
position; debating the winning one is what actually matters.

Fails open with a sharper edge than Evidence Graph, since there's no
"just return nothing" option for something that's supposed to revise the
answer: any `NoAvailableModelError` from the three calls, or an empty
Judge output, rolls the whole debate back and keeps the untouched
pre-debate draft — the same "an already-produced answer must never be
lost to a best-effort enrichment" discipline `_synthesize_or_fallback`
itself already applies to a failed `synthesize()` call. `"debate"` is
only added to `team` when it actually completed. Never runs at tier ≤ 3.

### Counterfactual (Phase 4, opt-in: `"trace_counterfactual": true`)

The third piece of Phase 4: `app/intelligence/counterfactual.py` asks —
another forced tool call (`submit_counterfactual_analysis`), not a
heuristic — which of the final answer's own premises are actually
load-bearing, and for each one, how the answer would meaningfully change
if it turned out to be false:

```bash
curl http://localhost:20128/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task": "What is the capital of France?", "trace_counterfactual": true}'
```

```json
{
  "...": "the rest of the OrchestrationResult",
  "counterfactual": {
    "points": [
      {
        "assumption": "the question refers to mainland France, not an overseas territory",
        "if_false": "the seat of government for an overseas territory would be a different city entirely"
      }
    ]
  }
}
```

Not named in spec section 十九's own "Very hard" team table (unlike
Debate), so it ships opt-in exactly like Evidence Graph, same reasoning:
an extra call shouldn't turn on silently. But it's a genuinely different
shape from Evidence Graph, not a copy of it — Evidence Graph traces
claims to specific DAG nodes, so it's a structural no-op below tier 2 (no
DAG exists yet at tier 0-1); Counterfactual's question is "what does this
answer's own reasoning depend on", which needs only the task and the
final answer, never a DAG. So `trace_counterfactual` genuinely runs at
**every** tier, including 0-1 — the one place `trace_evidence` can't
reach. When a DAG did run, its own step-by-step summary is folded in as
extra grounding, the fifth module to reuse `summarize_dag()` now (after
the Verifier, Synthesizer, Evidence Graph and Debate), but it's optional
context, never a requirement.

Deliberately scoped **away** from actually re-running anything: no input
gets perturbed and no DAG gets re-executed here, that's left entirely to
Simulation (Phase 4's next piece, see below) so the two pieces never
duplicate the same execution machinery. Pure advisory annotation, same
family as Evidence Graph, not Debate — it never revises
`OrchestrationResult.answer`, only adds optional metadata beside it.
Fails open exactly like Evidence Graph: a missing/malformed tool call, or
no provider able to answer at all, returns an empty analysis rather than
losing the answer that's already been produced. `"counterfactual"` is
added to `team` whenever `trace_counterfactual` was requested, whether or
not the analysis itself came back non-empty — same "team reflects the
attempt, not just the outcome" convention `trace_evidence` already uses.

### Simulation (Phase 4, opt-in: `"simulate": ["..."]`)

The fourth piece of Phase 4: where Counterfactual *guesses* how the
answer would change under a different premise, `app/agents/simulation.py`
*actually re-answers* the original task once per caller-supplied
scenario, told to genuinely work out the answer under that changed
premise rather than describe how it might change:

```bash
curl http://localhost:20128/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Should we use microservices or a monolith for a 3-person startup building an MVP?", "simulate": ["assume the team grows to 15 engineers within a year"]}'
```

```json
{
  "...": "the rest of the OrchestrationResult",
  "simulations": [
    {
      "scenario": "assume the team grows to 15 engineers within a year",
      "answer": "... a genuinely different answer, worked out under that premise ..."
    }
  ]
}
```

XRouter deliberately never invents the scenarios itself — Counterfactual
already owns "identify what's load-bearing"; if Simulation also guessed
its own scenarios, it would either duplicate that judgment or silently
second-guess it. So a caller who ran Counterfactual now has concrete
premises in hand to actually test here, and the two pieces stay honestly
divided: reasoning in Counterfactual, real execution in Simulation.

Each scenario is one plain prose call (same family as Synthesizer/
Debate — a generative answer, not a structured judgment, so no forced
tool call), never a recursive call back into the Orchestrator's own
`orchestrate()`. Re-running the whole Planner/Critic/Verifier/Debate
pipeline per scenario would make `simulate` an unbounded cost multiplier
on an already-expensive tier-4 request; one direct call per scenario
keeps this honestly bounded. `simulate` is capped at 3 scenarios — extras
are dropped with a logged warning rather than silently fanning out
further, and blank entries are skipped. Like Counterfactual, it needs
only the task and context, never a DAG, so it genuinely runs at every
tier including 0-1.

Fails open **per scenario**, inside the module, not propagated to the
caller: scenarios are independent, so one `NoAvailableModelError` only
drops that one scenario's result, not the whole batch — matches Evidence
Graph/Counterfactual's "fail open, keep going" discipline, not Debate's
"roll the whole thing back" (there's no single draft to protect here) or
Synthesizer's "propagate to a caller with a real fallback" (there's no
meaningful fallback content for one simulated scenario — it simply
doesn't appear in the results). Unlike `trace_evidence`/
`trace_counterfactual`, `"simulation"` is only added to `team` when at
least one scenario actually produced a result — there's no honest
"attempted but empty" state the way an LLM judgment can legitimately
return zero claims/points; an entirely-failed batch just didn't run.

### Confidence Engine (Phase 4, last piece — always on, no flag)

The fifth and last piece of Phase 4: `app/intelligence/confidence.py`
rolls up everything a finished Orchestrator run already observed about
its own answer — which layers of scrutiny actually ran, and whether each
one that ran was satisfied — into one deterministic confidence score.
Unlike every other Phase 4 piece, it isn't opt-in and it isn't a
`team`-list entry, because it isn't another LLM judgment: it never spends
a provider call and never re-reads response text, it only combines
structured results (`dag`, `verification`, `debate`, `evidence`) the run
already produced for other reasons. Since it costs nothing extra, gating
it behind a flag would be arbitrary, so `confidence` is a required field
on every `OrchestrationResult`, at every tier, including 0-1:

```json
{
  "...": "the rest of the OrchestrationResult",
  "confidence": {
    "score": 0.7,
    "label": "medium",
    "reasons": ["critique_unsatisfied"]
  }
}
```

At tier 0-1, nothing ever reviewed the answer (no DAG ran at all), so
`confidence` reports an honest neutral baseline — `score: 0.5`, `label:
"medium"`, `reasons: ["unreviewed"]` — rather than guessing. From tier ≥
2 on, it starts at a perfect `1.0` and subtracts a fixed penalty for each
concretely observed problem — the DAG finishing `"partial"` rather than
`"success"`, any node's Critic judgment coming back unsatisfied, a
`VerificationResult` coming back unsatisfied, Debate having been due
(the run reached tier 4) but failing open with no result, and — when
Evidence Graph ran — a share of claims judged unsupported, scaled by
their ratio of the total — clamping to `[0.0, 1.0]` and labeling `"high"`
(≥ 0.8), `"medium"` (≥ 0.5), or `"low"` otherwise. Same "start at 1.0,
subtract concrete penalties, clamp, round" shape the pre-existing Quality
Gate (`app/intelligence/quality_gate.py`, Phase 2) already uses for its
own, differently-scoped, single-response structural check — see that
module's own docstring for the full division between the two.

### Automated Benchmark (Phase 5, opt-in: `benchmark.enabled` in `config.yaml`)

Phase 5's first piece. The spec lists Phase 5 as Evolution Engine, A/B
Routing, Policy Learning, Self-healing, Automated Benchmark — but that
literal order doesn't survive contact with what those names actually
imply: Evolution Engine is, by construction, a loop that repeatedly runs
A/B Routing to compare policy variants and feeds the result into Policy
Learning to update policy weights. Built first, with neither of those two
existing yet, it would have nothing real to evolve and could only ever be
a stub — exactly the "fake placeholder functionality" this project's own
rules forbid. Phase 5 is therefore being built in dependency order
instead: **Automated Benchmark → A/B Routing → Policy Learning → Evolution
Engine → Self-healing**. Self-healing doesn't depend on the other four, so
it's simply placed last rather than forced earlier.

Two benchmark code paths already existed before this piece:
`scripts/benchmark.py` (a richer manual CLI tool — multiple runs,
p50/p95/p99 — that only ever prints, never persists) and `POST
/admin/benchmark` (a single-prompt probe that does persist to the
`benchmarks` table, but only when someone calls it — nothing scheduled it,
and `MetricsRepository.recent_benchmarks()` already existed to read that
table back but was never exposed anywhere). This piece doesn't add a third
path: it moves `/admin/benchmark`'s own probe logic, verbatim, into
`app/reliability/benchmark.py`'s `BenchmarkScheduler`, so the exact same
behavior now also runs on a timer, and exposes the read-back that already
existed as `GET /admin/benchmark/history`:

```bash
curl -X POST http://localhost:20128/admin/benchmark -H "Authorization: Bearer $TOKEN"
curl "http://localhost:20128/admin/benchmark/history?limit=20" -H "Authorization: Bearer $TOKEN"
```

```json
{ "benchmarks": [
  { "provider_id": "groq", "model_id": "llama-3.1-8b-instant", "ttft_ms": null,
    "total_latency_ms": 214.3, "tokens_per_sec": null, "success": true, "recorded_at": 1234567890.1 }
] }
```

`BenchmarkScheduler` reuses the exact same hand-rolled background-loop
shape `HealthMonitor` and `PerformanceController` already use
(`start()`/`_loop()`/`stop()`, an `asyncio.Event` stop signal, one unit of
work per interval) — no new generic scheduler abstraction, since two
existing instances of that shape is this codebase's own established
convention for a periodic task, not a gap needing to be filled. Off by
default (`benchmark.enabled: false`), the same "shouldn't turn on
silently" reasoning already applied to `routing.race_mode_enabled` and
`routing.quality_gate_enabled` — this fires a real chat completion against
every enabled provider on a timer with no per-request trigger, so it must
be opted into explicitly rather than starting automatically like the
always-on, assumed-cheap `HealthMonitor` polling does.

Deliberately narrow: this piece does not analyze trends, does not feed
results back into routing scores, and takes no automatic action on what it
finds. Policy Learning (below) ended up being the piece that acts on data
this codebase collects — though on A/B Routing's outcomes, not this
piece's benchmark history; Self-healing (also below) acts on provider
health/circuit-breaker state instead. This piece's own history stays a
read-only, human-consulted signal for now — a future extension point, not
built ahead of need.

### A/B Routing (Phase 5, opt-in: `ab_routing.enabled` in `config.yaml`)

Phase 5's second piece (see the Automated Benchmark section above for why
Phase 5 is being built Automated Benchmark → A/B Routing → Policy Learning
→ Evolution Engine → Self-healing rather than the spec's literal order).
Splits real, live traffic between two or more named routing-policy variants
(`ab_routing.variants`, default `["quality", "balanced"]`) and records each
request's real comparative outcome — success, latency, and quality score
when the Quality Gate is also on — tagged by variant. This is the first
real comparative dataset Policy Learning (the next piece) will have
anything to learn from.

Deliberately **not** the same thing as race mode (`app/execution/race.py`):
race mode hedges the top-N candidates from a *single* policy's own ranked
list concurrently and keeps whichever answers first — latency hedging
within one policy. A/B Routing compares *different* policies against each
other using real, separate traffic; it never dispatches more than one
provider call per request.

`ChatEngine._resolve_policy` is the single choke point every request's
policy already passed through, so that's where assignment plugs in: a
client-supplied `routing_policy` always wins and never participates (a
self-selected policy isn't part of the controlled comparison); otherwise,
when A/B Routing is enabled with ≥2 configured variants, `ABRouter.assign()`
picks one — sticky per `request.user` (hashed with md5, modulo the variant
count), or a fresh per-request UUID when the caller sends no `user` at all
(XRouter has no session concept today, so an anonymous caller gets
per-request rather than sticky assignment — a documented, honest
limitation, still statistically valid in aggregate). That assignment wins
over both the Task Classifier's suggestion and `default_policy`, since
turning A/B Routing on is a real, documented behavior change for all
non-pinned traffic.

A successful response produced under an assignment gets
`response.xrouter["ab_experiment"] = True` (mirroring how race mode already
sets `response.xrouter["race"] = True` alongside the existing `policy`
field, rather than duplicating the variant name under a second key). A
cache hit never records an A/B outcome — a cached response reflects a
possibly-different request/variant's prior real work, not this variant's
work this time — though assignment itself still happens first, for
stickiness. Outcomes are recorded at every existing telemetry point
(mirroring, not replacing, the pre-existing `metrics.record_request(...)`
calls): both `NoAvailableModelError` catches record a failure, and the
non-streaming/streaming success paths record a success, with a quality
score only when the (also opt-in) Quality Gate ran.

```bash
curl "http://localhost:20128/admin/ab_routing/summary" -H "Authorization: Bearer $TOKEN"
```

```json
{ "enabled": true, "variants": ["quality", "balanced"],
  "results": {
    "quality": { "count": 12, "success_rate": 0.917, "avg_latency_ms": 812.4, "avg_quality_score": 0.78 },
    "balanced": { "count": 9, "success_rate": 1.0, "avg_latency_ms": 340.1, "avg_quality_score": null }
  } }
```

Off by default (`ab_routing.enabled: false`) — the same "must not turn on
silently" reasoning already applied to `routing.race_mode_enabled`,
`routing.quality_gate_enabled`, and `benchmark.enabled`, since this changes
how every non-pinned request already being served gets routed, not just an
extra background call.

### Policy Learning (Phase 5, opt-in: `policy_learning.enabled` in `config.yaml`)

Phase 5's third piece. Reads A/B Routing's own `ab_results` data (A/B
variant names are literally routing-policy names, e.g. `"quality"` vs
`"balanced"`) and nudges the *losing* policy's weights a step toward the
*winning* policy's weights, so a policy demonstrably underperforming in
real traffic gradually converges toward one that's winning — real
learning from real outcomes, scoped honestly: `PolicyLearner.run_once()`
computes one deterministic adjustment step, it is **not** a scheduled
background loop and **not** a real ML model. The repeated scheduling is
Evolution Engine's job (the next piece, described in this same README as
"a loop that repeatedly runs A/B Routing ... and feeds the result into
Policy Learning") — building a second competing scheduler here would be
exactly the unneeded infrastructure this project's own rules forbid.

(`run_once()` also records, per nudge, the loser's pre-nudge fitness and
its previous weights snapshot — not used by anything in this piece itself,
but exactly what Evolution Engine, described next, needs to later judge
whether that nudge actually helped.)

Each eligible variant (one of `ab_routing.variants` with at least
`policy_learning.min_samples` recorded outcomes) gets one transparent
fitness score:

```
fitness = success_rate + quality_weight * avg_quality_score - latency_weight_per_second * (avg_latency_ms / 1000)
```

`success_rate` (in `[0, 1]`) dominates by construction; latency and
quality are small, explicitly-weighted tie-breakers, not hidden
heuristics. The variant with the highest fitness is the winner; every
other eligible variant gets nudged toward it **only if** the fitness gap
clears `policy_learning.min_margin` — a gap that small is treated as
statistically indistinguishable, so weights aren't churned on noise. The
nudge moves every one of `PolicyWeights`' 6 dimensions by
`learning_rate * (winner - current)`, clamped to `[0.05, 5.0]` as a safety
net against runaway drift, and compounds: a second `run_once()` nudges
from the *already-learned* weights, not the original static constants, so
repeated calls converge toward whatever's currently winning.

`AdaptiveRouter.resolve_policy` — the one place every request's weights
were already being looked up — is where this plugs in: `PolicyLearner.
weights_for(key, base)` returns the learned override when enabled and one
exists, otherwise the static base weights, unchanged. Disabled is a
complete no-op for real routing (same precedent as `ABRouter.assign()`),
even though a manual relearn can still compute and store a preview while
disabled — it's the read path, not the compute path, that's gated.

```bash
curl -X POST http://localhost:20128/admin/policy_learning/relearn -H "Authorization: Bearer $TOKEN"
curl "http://localhost:20128/admin/policy_learning/weights" -H "Authorization: Bearer $TOKEN"
```

```json
{ "enabled": true, "variants": ["quality", "balanced"],
  "learned_overrides": {
    "balanced": { "quality": 1.225, "speed": 0.925, "reliability": 1.03, "cost": 0.895, "quota_risk": 0.97, "local_preference": 0.94 }
  } }
```

Off by default (`policy_learning.enabled: false`) — the same "must not
turn on silently" reasoning as everywhere else in this family, since this
changes how live traffic gets scored. It also has a real dependency on
A/B Routing already being enabled and trafficked: with no `ab_results`
data, `run_once()` just reports `insufficient_samples` every time — an
expected consequence of the dependency chain, not a bug in this piece.
Learned overrides live in memory only (process-lifetime, not persisted to
a table) in this first version — if Evolution Engine later needs them to
survive a restart, that's its own extension point, not built ahead of
need.

### Evolution Engine (Phase 5, opt-in: `evolution.enabled` in `config.yaml`)

Phase 5's fourth piece. Policy Learning (above) computes one honest
adjustment step but deliberately doesn't decide *when* to run, or check
whether a nudge it already made actually helped once real traffic ran
under the new weights — that's this piece. `EvolutionEngine` is a
background loop (same `start()`/`_loop()`/`stop()` shape as
`BenchmarkScheduler`) that each cycle: (1) evaluates every nudge still
pending from a prior cycle against fresh post-nudge evidence, rolling it
back via `PolicyLearner.revert()` if real-world performance came in worse
than before the nudge; (2) asks `PolicyLearner.run_once()` for a fresh
nudge, excluding any policy still awaiting evaluation from (1) so the same
policy can never be nudged twice before anyone's checked whether the
first nudge even helped; (3) records any brand-new adjustment into its
own pending set for the next cycle to evaluate.

Evaluation reuses `MetricsRepository.ab_summary(since=...)` — a pure
additive time filter on the existing `ab_results` table, no new
schema — to compute fitness (via `PolicyLearner`'s own public `fitness()`
method, so the nudge decision and the rollback evaluation can never drift
out of formula-sync with each other) over only the outcomes recorded
*after* the nudge was applied, compared against the fitness at the moment
of the nudge:

```
new_fitness < baseline_fitness - evolution.rollback_tolerance  ->  revert
otherwise                                                       ->  confirm (leave as nudged)
```

`evolution.rollback_tolerance` (default `0.02`) is deliberately smaller
than `policy_learning.min_margin` (default `0.05`) — confirming a nudge
helped should be *easier* to fail than the original nudge was to trigger,
since a bad nudge is actively hurting live routing right now, while a
skipped nudge just leaves things unchanged.

This is genuine "variation, evaluation, retention-or-reversion," not a
fake wrapper that calls `run_once()` on a timer with no real evaluation
step. It's also an honest limitation, not a rigorous causal test: this is
a pre/post comparison on the same policy across time, and real-world
traffic can drift for unrelated reasons too — the same way A/B Routing's
anonymous-caller limitation is documented above rather than glossed over.

```bash
curl "http://localhost:20128/admin/evolution/status" -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:20128/admin/evolution/run_once -H "Authorization: Bearer $TOKEN"
```

```json
{ "enabled": true, "interval_seconds": 3600,
  "pending": { "balanced": { "applied_at": 1234567890.1, "previous_weights": null, "baseline_fitness": 0.42 } } }
```

Off by default (`evolution.enabled: false`) — it runs its own background
timer, same "must not turn on silently" reasoning as Automated Benchmark.
It also has a real dependency on Policy Learning already being enabled:
if `policy_learning.enabled` is false, `PolicyLearner.weights_for()` never
applies anything this engine computes, so cycles run against routing that
isn't actually using the result — an expected consequence of the
dependency chain, not a bug in this piece. Pending state lives in memory
only, scoped to one process, same as Policy Learning's learned overrides.

### Self-healing (Phase 5, last piece, opt-in: `self_healing.enabled` in `config.yaml`)

Phase 5's fifth and final piece. The rest of the reliability stack
already self-heals in its own scope — the circuit breaker auto-recovers
per provider (time+probe based, no admin involved), the router filters
out circuit-open/quota-critical candidates per request — but
`ProviderRegistry.set_enabled()`, the one lever that takes a provider out
of the routing pool entirely, had exactly 3 callers in the whole codebase
before this piece: its own definition, and the two admin routes. A
provider that's circuit-flapping or persistently unhealthy stayed in the
pool forever unless a person noticed and disabled it by hand.

This piece closes that gap by correlating two signals that already exist
for free — `ProviderRegistry.health_of()` (updated by `HealthMonitor`'s
own polling) and `CircuitBreakerRegistry`'s per-provider state (updated
by real request traffic) — rather than re-probing anything itself.
`SelfHealer` never calls `provider.health()`; it only reads state two
other pieces already computed:

```
unhealthy = health.status in (OFFLINE, DEGRADED) or breaker.state == OPEN
```

A provider that's unhealthy by this check for `self_healing.
confirm_cycles` consecutive background ticks (default 3, ~90s at the
default 30s interval) gets auto-disabled (`set_enabled(False)`,
`self_healing.disabled` event emitted); a single healthy tick resets an
in-progress bad streak — no leaky partial credit. A provider *this piece
itself* disabled gets auto re-enabled once it's looked healthy for
`confirm_cycles` consecutive ticks the same way. Deliberately one
threshold, not two independent disable/recover knobs — there's no
evidence-backed reason for them to differ, and an extra config field is
exactly the kind of unneeded surface area this project's rules warn
against.

The one thing this design had to get right: never fight an admin.
Recovery candidates are drawn *only* from providers `SelfHealer` itself
disabled — a provider an admin disables directly is never a candidate to
auto re-enable, full stop. And every provider-mutating admin route
(`enable`, `disable`, `cooldown`) now calls a new `SelfHealer.forget()`
right after it acts, clearing that provider's streaks and disabled-by-
self bookkeeping — so an admin disabling a self-disabled provider (to
keep it off deliberately) sticks, and an admin re-enabling a still-
unhealthy provider gets a full fresh `confirm_cycles` streak rather than
an instant re-disable from leftover history.

```bash
curl "http://localhost:20128/admin/self_healing/status" -H "Authorization: Bearer $TOKEN"
curl -X POST http://localhost:20128/admin/self_healing/run_once -H "Authorization: Bearer $TOKEN"
```

```json
{ "enabled": true, "interval_seconds": 30, "confirm_cycles": 3,
  "disabled_by_self_healing": { "groq": 1234567890.1 } }
```

Off by default (`self_healing.enabled: false`) — the same "must not turn
on silently" reasoning as every other opt-in Phase 5 piece, since this is
the one piece that actually takes provider enablement out of human hands
automatically. Deliberately excludes two other signals: `QuotaTracker.
is_paused()` is already consulted by the router's own request-time
scoring, a different (temporary, volume-driven) kind of risk than
sustained unavailability, and folding it in here would blur two pieces'
responsibilities; `BenchmarkScheduler`'s history is a third passive probe
covering the same ground `HealthMonitor.health()` already covers —
a future extension point, not built ahead of need. All state (streaks,
disabled-by-self bookkeeping) lives in memory only, scoped to one
process — nothing here touches the database.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

500 tests across `tests/unit`, `tests/reliability`, `tests/routing`,
`tests/execution`, `tests/providers`, `tests/tools`, `tests/agents`,
`tests/retrieval`, `tests/storage`, `tests/integration` — circuit
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
server-switch/per-request-opt-in gating logic for both the non-streaming
and streaming gates — plus, for streaming racing specifically, first-chunk
arrival deciding the winner, a losing generator that already got its first
chunk being closed rather than leaked, and skipped candidates being logged
by `run_stream_race` even though `run_stream_chat` deliberately still
doesn't log them), the
quality gate (empty/truncated/degenerate-repetition/missing-forced-
tool-call detection, retry-to-next-untried-candidate, best-effort return
when every candidate still fails, `quality.failed` events, the
`max_quality_retries: 0` no-retry-but-still-assess case, and its opt-in
LLM-graded layer: judge skipped entirely when grading is off, a
structurally-good response left alone when the judge is satisfied, a
retry to the next candidate sharing the same budget when it isn't,
`llm_graded_unsatisfactory`/`llm_feedback` in the final result, the
judge preferring a provider different from the one that answered,
and falling open when the judge has no available candidate or fails
outright), and the DAG
executor (wave/topological ordering, validation errors for empty/
too-many/duplicate/self-dependent/unknown-dependency/cyclic graphs,
placeholder substitution incl. unknown-placeholder passthrough and
repeated placeholders, and — end-to-end through a real `ChatEngine` with
fake providers via the new `tests/helpers.py:build_test_engine` harness —
linear-chain substitution actually reaching the provider call, parallel
fan-in combining two upstream outputs, cascading skip-on-failure that
leaves sibling nodes unaffected, and the `max_dag_nodes` cap enforced
end-to-end), and the Planner (forced-tool-call request building, tool-
argument extraction, JSON/schema-validation failures, `depends_on: null`
tolerance, DAG-shape validation reusing the executor's own validator, and
— end-to-end through `build_test_engine` — first-try success, the
error-message-fed retry loop actually recovering from an invalid
`submit_plan` call, giving up once retries are exhausted, rejecting a
response with no tool call at all, rejecting a plan over the node cap,
and a generated plan actually executing end-to-end with its
planner-authored `{{node_id}}` dependency wiring reaching the provider
call), and the Verifier + plan-runner loop (DAG-result summarization,
forced-tool-call request building, fail-open behavior on a missing tool
call / malformed arguments / no provider available, and — end-to-end
through `app/execution/plan_runner.py` — `verify: false` running exactly
once with no verification call at all, `verify: true` satisfied on the
first try skipping any re-plan, an unsatisfied-then-satisfied cycle
actually re-planning once with the verifier's feedback provably reaching
the second planning call's prompt, and giving up after exhausting
`max_verify_retries` rather than looping forever), the Web Search tool
(`tests/tools/test_web_search.py` — configured/unconfigured graceful
degradation, missing/non-string query rejection, result formatting,
empty-results text, `max_results` clamping and invalid-value fallback, and
non-200/timeout/connection-error handling, all against a mocked Tavily
endpoint via `respx` — never a real network call), the Web Fetch tool
(`tests/tools/test_web_fetch.py` — the same configured/unconfigured and
transport-failure coverage, plus a missing/non-string `url` rejected, the
`Authorization: Bearer` header and `urls` body actually sent, a long
page truncated to `_MAX_CONTENT_CHARS`, an empty `raw_content` returning
friendly text, and a per-URL `failed_results` entry surfaced as the
error even on an overall HTTP 200), the tool registry
(`tests/tools/test_registry.py` — unregistered vs. registered-but-
unconfigured both resolving to `None`, `available_names()` filtering, and
`close_all()` tolerating one tool's shutdown failing without blocking the
rest) and factory (`tests/tools/test_factory.py` — empty/disabled/missing-
API-key tools all gracefully absent rather than crashing, an unknown tool
id in config skipped, and one tool's construction failing without taking
the others down with it), the tool-execution loop
(`tests/execution/test_tool_loop.py` — no-enabled-tools and
requested-but-unregistered both falling through to a plain call, a single
call → tool → call round-trip actually feeding the tool's result back as
a `role: "tool"` message, multiple iterations before a final answer,
`max_iterations` respected without erroring, a `ToolError` fed back as the
tool's own result rather than failing the node, and a tool call for a name
that was never enabled left completely untouched), DAG-node `enable_tools`
end-to-end (a node with `enable_tools` actually invoking a fake tool and
its result reaching the final response, a node without `enable_tools`
never touching the registry at all, and a node requesting an unregistered
tool falling back to a plain call), and the Planner's richer per-node
schema (the `submit_plan` schema omitting `enable_tools` entirely when no
tools are available and otherwise enum-restricting it to exactly what's
registered and configured, the `routing_policy` enum always offered, both
fields carried through `to_dag_request()`, `_validate_plan_shape()`
rejecting a hallucinated/unavailable tool name, and — end-to-end through
`build_test_engine` — a plan actually offering only the registered
tool(s) in its schema and a plan that hallucinates an unavailable tool
being rejected and successfully retried), and the Critic
(`tests/execution/test_critic.py` — forced-tool-call request building,
satisfied/unsatisfied reporting, and the same fail-open behavior as the
Verifier on a missing tool call / malformed arguments / no provider
available, plus the retry loop itself: satisfied-first-try never
re-running the node, an unsatisfied-then-satisfied cycle actually
re-running through the node's own responder with the Critic's feedback
provably reaching the re-run request, giving up after exhausting
`max_retries` while still returning the last attempt, and
`max_retries=0` never re-running at all — and, end-to-end through
`DagExecutor` in `tests/execution/test_dag.py`, a node without
`critique` never triggering a review call, a node with `critique: true`
satisfied on the first try, an unsatisfied-then-satisfied node actually
rerunning itself and the corrected output reaching the final result, and
a node that stays unsatisfied through every retry still reporting overall
`"success"` since critique is a best-effort review, never a pass/fail
gate on the node itself).

Phase 3 completion coverage: `MemoryRepository`
(`tests/storage/test_memory_repository.py` — save/recent/scoping,
`search()`'s word-overlap matching actually excluding a query's own
common function words rather than false-positive-matching them as
substrings of unrelated content, `embedding_json` round-tripping through
`save()`/`recent()`, `embedded()` excluding entries with no embedding,
and a migration test proving `Database.init()` adds the `embedding_json`
column to a table created under the *old* schema without breaking),
`KeywordRetriever` (`tests/retrieval/test_keyword.py` — wraps matching
memory entries as `RetrievedChunk`s, respects `top_k`, empty-match and
cross-scope isolation), `EmbeddingRetriever` and `embed_for_memory`
(`tests/retrieval/test_embedding.py` — cosine-similarity ranking, `top_k`
and scope, and failing open to `[]`/`None` on a missing provider, an
adapter that doesn't support embeddings, or a real provider error;
`OllamaAdapter`/`OpenAICompatibleAdapter`'s own `embed()` HTTP behavior
is covered separately in `tests/providers/`), the Researcher (`tests/agents/test_researcher.py` — the
system prompt honestly reflecting tool availability, answering from the
model's own knowledge when no search tool is configured, and actually
driving a `web_search` tool call end-to-end when one is), the Synthesizer
(`tests/agents/test_synthesizer.py` — composing one answer from a
multi-node DAG's outputs and letting `NoAvailableModelError` propagate
for its caller to handle), and the Orchestrator
(`tests/agents/test_orchestrator.py` — `_force_terminal_critique` forcing
critique only onto nodes nothing else depends on across single-node and
multi-node chains, every complexity tier 0 through 4 assembling exactly
the team it should, `OrchestrationError` at both tier 2 and tier 3/4 when
every step fails, the `"research"`/`"synthesizer"`/`"verifier"` tags only
ever appearing when actually used, and memory recall/write-back actually
reaching a real `MemoryRepository`, scoped correctly). `plan_mutator`
(`tests/execution/test_plan_runner.py`) — applied to a freshly generated
plan before execution, left as a no-op when omitted, and reapplied on
every re-plan, not just the first.

Phase 4 coverage: Evidence Graph (`tests/execution/test_evidence.py`,
`tests/agents/test_orchestrator.py`) — the evidence tool schema's
`supported_by` enum only ever listing the run's own real DAG node ids
plus `"model_knowledge"`, a hallucinated node-id reference actually
getting dropped rather than trusted (both as a pure `_sanitize_claims`
unit test and end-to-end through the Orchestrator), fail-open on no
provider / no tool call / malformed arguments, `trace_evidence` being a
genuine no-op at tier 0-1, and the `"evidence"` tag only ever appearing
in `team` when a graph was actually built. Debate
(`tests/agents/test_debate.py`, `tests/agents/test_orchestrator.py`) —
the happy path actually returning the judge's resolution (not the
pre-debate draft) with all four `DebateResult` fields populated
correctly, three distinct fail-open cases (no provider at all, a later
call failing after an earlier one already succeeded, and the judge
returning nothing usable) all rolling back to the untouched draft rather
than a half-finished revision, the judge's own prompt actually
truncating an oversized advocate/skeptic argument instead of embedding it
whole, and — end-to-end through the Orchestrator — tier 4 both running
debate and replacing `answer` with its resolution, `"debate"` only
appearing in `team` when it actually completed, and tier 3 never
triggering it at all. Counterfactual
(`tests/execution/test_counterfactual.py`,
`tests/agents/test_orchestrator.py`) — pure request-building with and
without a dag and with/without context, `_sanitize_points` dropping
entries missing either field, fail-open on no provider / no tool call /
malformed arguments, an explicitly-empty points list round-tripping
correctly (never forced non-empty), and — end-to-end through the
Orchestrator — the key behavioral difference from Evidence Graph proven
directly: `trace_counterfactual` genuinely running (and calling the
provider) at tier 0-1 where `trace_evidence` is a no-op, plus tier 2 and
a tier 4 run confirming it runs after Debate's own revision, on the
Judge's resolution rather than the pre-debate draft. Simulation
(`tests/agents/test_simulation.py`, `tests/agents/test_orchestrator.py`)
— pure request-building with/without context, a happy path actually
running two independent scenarios and returning both, the `_MAX_SCENARIOS`
cap genuinely dropping a 4th scenario before it ever reaches the
provider, blank scenarios skipped, one scenario's `NoAvailableModelError`
(after exhausting its own internal retries) leaving the other scenario's
result intact rather than losing the whole batch, an entirely-failed
batch returning an empty list rather than raising, and — end-to-end
through the Orchestrator — `simulate` genuinely running at tier 0-1 (same
proof point as Counterfactual), a tier-2 run with two scenarios, and
`"simulation"` only appearing in `team` when at least one scenario
actually produced a result. Confidence Engine
(`tests/execution/test_confidence.py`, `tests/agents/test_orchestrator.py`)
— pure unit tests on `assess_confidence` itself: the tier-0-1 `dag=None`
baseline, a clean run scoring a perfect `1.0`/`"high"` with no reasons,
each of the five penalties triggering in isolation (dag partial, critique
unsatisfied, verification unsatisfied, debate failed open — and
confirming it's *not* penalized when no verifier ran, or when a debate
result is actually present — and unsupported claims scaled by their
ratio, including an empty/fully-supported claims list never penalizing),
multiple penalties stacking and clamping to `0.0` rather than going
negative, all three label boundaries, and — end-to-end through the
Orchestrator — the tier-0-1 unreviewed baseline, a tier-2 happy path
scoring `1.0`, and reusing the tier-4 debate-fails-open scenario to
confirm `confidence.reasons` actually includes `"debate_failed_open"`
there.

Phase 5 coverage: Automated Benchmark
(`tests/reliability/test_benchmark.py`, `tests/integration/test_api.py`)
— `run_once()` against `FakeProvider`s: one persisted result per enabled
provider, a failing provider recorded as `success: False` without
aborting the rest of the batch, disabled providers and providers with no
models both skipped, an empty provider set returning `[]` rather than
raising; `start()`/`stop()` lifecycle mirroring `HealthMonitor`'s own
tested shape (`stop()` before `start()` a safe no-op, `start()` itself
idempotent, a real background cycle actually persisting a row); and,
full-stack, `GET /admin/benchmark/history` requiring the same admin token
as every other `/admin/*` route, plus both the manual `POST
/admin/benchmark` trigger and its history read-back staying gracefully
empty rather than erroring when no provider is reachable. A/B Routing
(`tests/routing/test_ab_router.py`, `tests/unit/test_engine_ab_routing.py`,
`tests/integration/test_api.py`) — `ABRouter.assign()` in isolation:
`None` when disabled or with fewer than 2 configured variants, sticky
assignment for a repeated `request.user` across many calls, assignment
actually varying across distinct users, and a roughly even split across
many anonymous (no-`user`) requests; `record_outcome()` persisting and
`MetricsRepository.ab_summary()` aggregating count/success_rate/
avg_latency_ms/avg_quality_score correctly per variant, including a `NULL`
`quality_score` on some rows never poisoning the average of the rows that
have one; and, end-to-end through a real `ChatEngine.handle_chat()`, an
explicit `routing_policy` bypassing A/B entirely even when it happens to
match a configured variant name, a disabled config being a complete
no-op, an enabled config assigning a variant/tagging
`xrouter["ab_experiment"]`/persisting a matching outcome, a
`NoAvailableModelError` failure still recording a failed outcome, a cache
hit never recording a second outcome for the same sticky user, and, full
stack, `GET /admin/ab_routing/summary` requiring the same admin token as
every other `/admin/*` route and reporting the real running
`enabled`/`variants` config. Policy Learning
(`tests/routing/test_policy_learner.py`, `tests/routing/test_router.py`,
`tests/integration/test_api.py`) — `_nudge`/`_clamp` in isolation: moving
every `PolicyWeights` dimension by exactly `learning_rate * (target -
current)`, and clamping an intentionally extreme target to
`[0.05, 5.0]`; `weights_for()` returning the untouched base weights both
when disabled (even with a stored override present) and when no override
has been learned yet for that key; `run_once()` reporting
`insufficient_samples` (and touching no state) below `min_samples`,
nudging a clear loser toward the winner by the exact expected amount on
every dimension, skipping a loser whose fitness gap doesn't clear
`min_margin` (an exact tie), a 3-variant case nudging only the one
pairing that clears the margin and leaving the other untouched, and
repeated calls compounding from the previous call's already-learned
weights rather than the original constants; `snapshot()`'s shape before
and after learning; and, in `AdaptiveRouter` itself, `resolve_policy`
returning a stubbed learner's override for a matching policy key while
still falling back to the base constant for any key the learner hasn't
touched, and every pre-existing router test continuing to pass unchanged
with no `policy_learner` supplied at all; and, full stack, both new
`/admin/policy_learning/*` routes requiring the same admin token as every
other `/admin/*` route, `GET .../weights` reporting the real running
`enabled`/`variants` config, and `POST .../relearn` always returning 200
with an `applied` key present regardless of how much real data exists to
act on. Evolution Engine (`tests/routing/test_evolution_engine.py`,
`tests/routing/test_policy_learner.py`, `tests/integration/test_api.py`)
— `run_once()` recording a fresh nudge into `pending` with the correct
pre-nudge `baseline_fitness`/`previous_weights`; a pending entry staying
pending, untouched, across repeated calls when there's no fresh evidence
yet, with `exclude` threading through to `PolicyLearner.run_once()` so it
isn't nudged again in the meantime (and a separate direct test confirming
`PolicyLearner.run_once(exclude=...)` itself skips a qualifying loser
while still letting it serve as a winner reference for others);
`_evaluate_pending()` confirming (leaving weights untouched) when fresh
fitness holds up, and reverting via `PolicyLearner.revert()` when it drops
past `rollback_tolerance` — including the `previous_weights=None` case
(a first-ever nudge) correctly removing the override entirely rather than
restoring a bogus snapshot; `PolicyLearner.revert()` itself tested in
isolation for both the dict-restore and `None`-removal paths;
`start()`/`stop()` lifecycle mirroring `BenchmarkScheduler`'s own tested
shape; `snapshot()`'s shape; and, full stack, both new
`/admin/evolution/*` routes requiring the same admin token as every other
`/admin/*` route, `GET .../status` reporting `pending == {}` on a freshly
built engine (safe here since pending is pure in-process state, unlike
the persisted-DB reads above), and `POST .../run_once` always returning
200 with the `evaluated`/`learn`/`pending` keys present regardless of how
much real data exists to act on. Self-healing
(`tests/reliability/test_self_healer.py`, `tests/integration/test_api.py`)
— a healthy provider (default status, closed circuit) left untouched
across repeated `run_once()` calls; staying unhealthy for fewer than
`confirm_cycles` cycles leaving it enabled; reaching `confirm_cycles`
consecutive unhealthy cycles disabling it, emitting `self_healing.
disabled`, and appearing in `snapshot()`; a single healthy tick resetting
an in-progress bad streak rather than leaking partial credit; an
open circuit breaker *alone* (health left healthy) also triggering a
disable, proven via `force_open()` without touching `set_health` at all;
a self-disabled provider recovering after `confirm_cycles` consecutive
healthy+closed-circuit cycles and emitting `self_healing.recovered`; a
provider disabled by anything other than `SelfHealer` itself never
becoming a recovery candidate no matter how many healthy cycles pass;
`forget()` making a self-disabled provider stick disabled forever after
(the "admin wants it to stay off" case), and `forget()` after a manual
re-enable requiring a full fresh `confirm_cycles` streak rather than an
instant re-disable from leftover history; `snapshot()`'s shape; `start()`/
`stop()` lifecycle mirroring `BenchmarkScheduler`'s own tested shape; and,
full stack, both new `/admin/self_healing/*` routes requiring the same
admin token as every other `/admin/*` route, `GET .../status` reporting
`disabled_by_self_healing == {}` on a freshly built engine, and `POST
.../run_once` returning the exact no-op shape on a fresh app whose real
providers start healthy with a closed circuit.

Verified end-to-end against real hardware: real Ollama (non-streaming and
`stream=true`, both producing correctly formatted chunks/`[DONE]`), and a
real macOS + Python 3.14 run of the full suite (the "no providers
reachable" tests now explicitly disable every provider rather than relying
on the host having none installed, and the cache fixture uses
`asyncio.run()` instead of the now-removed implicit-event-loop fallback).

## What's implemented (Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5)

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
concurrent dispatch of the top `race_candidate_count` ranked candidates,
returning whichever answers first and cancelling the rest; reuses the
exact same breaker/quota/retry/performance bookkeeping as the sequential
fallback chain via a shared `try_candidate()` helper, and falls back to
the remaining candidates sequentially if every raced one fails. Covers
both non-streaming (`run_race`) and streaming (`run_stream_race`, which
races on first-chunk arrival via a shared `try_candidate_stream()` helper
and explicitly closes any losing candidate's already-open stream). Quality Gate
(`app/intelligence/quality_gate.py`) — opt-in (`routing.quality_gate_enabled`)
deterministic scoring of a non-streaming response (empty, truncated,
degenerate word/character repetition, missing forced tool call) that
retries a below-threshold response with the next untried candidate, up to
`max_quality_retries` times, always attaching the assessment to
`xrouter.quality` and never erroring out even if nothing better is found.
Optionally extended with LLM-graded judgment (nested opt-in:
`routing.quality_gate_llm_grading_enabled`) — a real second model call,
via a forced `submit_quality_judgment` tool call mirroring the Critic/
Verifier pattern, that actually judges correctness rather than just
structure, sharing the same retry budget and falling open on any error;
runs under its own `quality_gate_judge_policy` and prefers a provider
different from the one that answered when possible. DAG Executor (`app/execution/dag.py`, `app/contracts/dag.py`,
`POST /v1/dag/run`) — executes a client-supplied explicit graph of chat-
completion nodes in topologically-ordered concurrent waves (Kahn's
algorithm), substituting `{{node_id}}` placeholders with upstream node
output before each node runs; every node goes through the real
`ChatEngine.handle_chat()`, so it gets the same caching/routing/circuit-
breaking/race/quality-gate behavior as the plain chat endpoint. A failing
node cascades a `"skipped"` status to everything depending on it (directly
or transitively) without affecting unrelated nodes; the run reports
`"success"` / `"partial"` / `"failed"` overall. Full validation (node
count, duplicate/self/unknown dependencies, cycles) happens before any
node executes.

**Phase 3 (so far):** Planner (`app/intelligence/planner.py`,
`app/contracts/planner.py`, `POST /v1/plan/run`) — the first piece of
multi-agent orchestration groundwork: turns a single free-form task into
an explicit DAG via a forced `submit_plan` tool call (an actual LLM call,
never a fixed template), validated against both a pydantic schema and the
DAG executor's own structural validator (cycles, duplicate/dangling
dependencies, node cap), with a bounded, specific-error-fed retry loop
(`routing.max_plan_retries`) when the model's output doesn't parse or
doesn't form a runnable graph. The resulting plan runs through the exact
same `DagExecutor` a client-supplied DAG uses — the Planner never talks to
a provider itself outside its one planning call, so every node it
produces gets the identical caching/routing/circuit-breaking/race/
quality-gate behavior any other XRouter request gets. Verifier
(`app/intelligence/verifier.py`, `app/contracts/verifier.py`) — opt-in
(request `"verify": true`) second-opinion check, also a forced tool call
(`submit_verification`) rather than a heuristic, over what a completed DAG
run actually produced versus the original task; a `false` verdict feeds
its own feedback back into the task's context and triggers a fresh
plan+execute cycle, up to `routing.max_verify_retries` times, via
`app/execution/plan_runner.py` — the module that wires Planner +
DagExecutor + Verifier together (`POST /v1/plan/run` is a thin HTTP
wrapper around it). Fails open (missing/malformed tool call, or no
provider able to answer at all counts as "satisfied") so a best-effort
check can never hold an already-completed result hostage. Web Search +
Web Fetch tools + tool-execution loop (`app/tools/` — `base.py`'s
`ExecutableTool` Protocol, `web_search.py`'s and `web_fetch.py`'s
Tavily-backed implementations, `registry.py`, `factory.py`;
`app/execution/tool_loop.py`) — real, config-driven tools a DAG/Planner
node opts into via `enable_tools`, executed by XRouter itself in a
bounded call → tool → call loop (`routing.max_tool_iterations`) on top
of the existing `ChatEngine.handle_chat()`, never a fake placeholder
that echoes the query back; a tool call for anything not registered and
configured is left untouched for the client to handle. The Planner's
`submit_plan` schema now also offers a per-node `routing_policy` override
and a dynamically-built `enable_tools` enum that only ever lists tools the
registry actually has available, with a defense-in-depth runtime check
rejecting any hallucinated tool name that slipped past the JSON-schema
`enum`. Critic (`app/intelligence/critic.py`, `app/contracts/critic.py`,
`app/execution/critique_loop.py`) — opt-in per node
(`DagNodeRequest.critique` / `PlanNodeSpec.critique`) review, also a
forced tool call (`submit_critique`) rather than a heuristic, distinct
from the Verifier: it judges one node's own output against that node's
own instruction right after the node produces it, rather than the whole
run against the original task once everything has finished, so a weak
step can be caught and redone before it poisons anything downstream that
substitutes `{{node_id}}` from it. An unsatisfied critique re-runs the
node (through the node's own responder — a plain call, or the tool loop
when `enable_tools` is also set) with the feedback folded in, up to
`routing.max_critique_retries` times; fails open exactly like the
Verifier, and even after exhausting every retry the node's own status
stays `"success"` — critique is a best-effort review, never a pass/fail
gate on whether the node ran. Dynamic Agent Team
(`app/agents/orchestrator.py`, `app/contracts/orchestrator.py`,
`POST /v1/agents/run`) — the piece that actually closes Phase 3: turns a
single free-form task straight into whichever team of everything above a
task's own complexity calls for (see the Dynamic Agent Team section
above for the full tier table), reusing `run_plan_with_verification`'s
own bounded re-plan loop wholesale for tier 4 rather than duplicating it,
and a `plan_mutator` hook on that same function so the Orchestrator can
deterministically force `critique: true` onto a plan's terminal nodes on
every (re-)plan without forking the Planner → DAG → Verifier loop.
Researcher (`app/agents/researcher.py`, `app/contracts/researcher.py`,
`POST /v1/research`) — a focused research step reusing the existing tool-
execution loop, reachable standalone or via the Orchestrator. Synthesizer
(`app/agents/synthesizer.py`) — composes one direct final answer from a
completed DAG's own node outputs when more than one node ran, so a
multi-step Orchestrator run never returns a raw list of intermediate
results. Memory/RAG (`app/storage/repositories/memory.py`,
`app/retrieval/`) — a real, scope-isolated, SQLite-backed memory store
and a swappable `Retriever` Protocol, read before and written after every
Orchestrator run at tier ≥ 2, with two real implementations:
`KeywordRetriever` (keyword/substring, always on, free) and
`EmbeddingRetriever` (`app/retrieval/embedding.py` — genuine embedding +
cosine-similarity search via a provider's real `embed()` call, opt-in via
`retrieval.enabled`) — see the Memory/RAG section above for the full
design and why both are honest, non-faked implementations rather than
one dressing up as the other.

**Phase 4:** Evidence Graph (`app/intelligence/evidence.py`,
`app/contracts/evidence.py`) — opt-in (`OrchestrationRequest.trace_evidence`)
tracing of a finished Orchestrator answer's own claims back to whichever
DAG step actually produced each one, also a forced tool call
(`submit_evidence_graph`) rather than a heuristic; ships as a flat claim →
DAG-node-id mapping (not a literal multi-hop graph) and deliberately
DAG-node-level rather than URL-level, since XRouter doesn't yet capture
`web_search` citations as structured, addressable objects — see the
Evidence Graph section above for the full reasoning. Fails open exactly
like the Critic/Verifier, and a no-op at tier 0-1 where there's no DAG to
trace against. Debate (`app/agents/debate.py`, `app/contracts/debate.py`)
— a standing tier-4 team member per spec section 十九's own team table
(no opt-in flag, unlike Evidence Graph): an Advocate and a Skeptic argue
for and against the tier's draft answer and a Judge reconciles both into
a final, strengthened answer that replaces the draft as
`OrchestrationResult.answer`. Deliberately runs after
`run_plan_with_verification()` returns rather than literally between DAG
execution and Verification, so it debates only a plan that already
passed verification instead of burning calls on one about to be thrown
away — see the Debate section above. Fails open with a sharper edge than
Evidence Graph (any failed call, or an empty judge output, rolls back to
the untouched pre-debate draft, since there's no harmless "return
nothing" option for a step that's supposed to revise the answer).
Counterfactual (`app/intelligence/counterfactual.py`,
`app/contracts/counterfactual.py`) — opt-in
(`OrchestrationRequest.trace_counterfactual`) identification of a finished
answer's own load-bearing assumptions and how the answer would change if
each one didn't hold, also a forced tool call
(`submit_counterfactual_analysis`) rather than a heuristic; deliberately
scoped away from actually re-running anything (that's Simulation's job,
see below) so it stays pure analysis of an answer that already exists —
see the Counterfactual section above for the full reasoning. Unlike
Evidence Graph, genuinely runs at every tier including 0-1, since it
needs only the task and the final answer, never a DAG. Fails open exactly
like Evidence Graph, and never revises `OrchestrationResult.answer` —
pure advisory annotation, the same family as Evidence Graph rather than
Debate. Simulation (`app/agents/simulation.py`,
`app/contracts/simulation.py`) — opt-in (`OrchestrationRequest.simulate`,
a caller-supplied list of changed-premise scenarios) actual re-execution
of the task under each one, a plain prose call (no forced tool call,
same family as Synthesizer/Debate) rather than a heuristic; deliberately
takes scenarios as caller input rather than inventing its own, so it
never duplicates or second-guesses Counterfactual's own judgment about
what's load-bearing — see the Simulation section above for the full
reasoning. Capped at 3 scenarios per request, and like Counterfactual
runs at every tier including 0-1. Fails open **per scenario** rather than
as a whole batch, since scenarios are independent; `"simulation"` is only
added to `team` when at least one scenario actually produced a result.
Confidence Engine (`app/intelligence/confidence.py`,
`app/contracts/confidence.py`) — the fifth and last piece: a synchronous,
zero-cost roll-up of whichever of `dag`/`verification`/`debate`/`evidence`
a run actually produced into one deterministic `ConfidenceAssessment`,
never a second LLM judgment and never gated behind a flag since it spends
nothing extra — required on every `OrchestrationResult`, at every tier
including 0-1, where it reports a neutral `"unreviewed"` baseline rather
than guessing. Deliberately distinguished from the pre-existing Quality
Gate (`app/intelligence/quality_gate.py`, Phase 2), which judges a single
response's own text for structural defects and can trigger a retry — see
the Confidence Engine section above and that module's own docstring for
the full division.

**Phase 5 (complete):** Automated Benchmark (`app/reliability/benchmark.py`,
`app/api/admin.py`) — the first piece, built out of the spec's own listed
order (see the Automated Benchmark section above for why: Evolution
Engine, listed first, needs A/B Routing and Policy Learning to already
exist before it can genuinely evolve anything, so Phase 5 is being built
in dependency order instead — Automated Benchmark, A/B Routing, Policy
Learning, Evolution Engine, Self-healing). `BenchmarkScheduler` turns
`POST /admin/benchmark`'s own manual probe logic into a scheduled
background task (same hand-rolled loop shape `HealthMonitor` and
`PerformanceController` already use), off by default since it spends a
real provider call automatically, and exposes the previously-unreachable
`recent_benchmarks()` history over `GET /admin/benchmark/history`.
Deliberately does not analyze trends or act on what it finds — that's
later pieces' job. A/B Routing (`app/routing/ab_router.py`,
`app/core/engine.py`, `app/api/admin.py`) — the second piece. Splits real,
live traffic between named routing-policy variants (sticky per
`request.user`, off by default) and records each request's real
comparative outcome tagged by variant over `GET
/admin/ab_routing/summary`, producing the first real comparative dataset
Policy Learning (the next piece) will have anything to learn from. See
the A/B Routing section above for the full design, including how it
differs from race mode and why cache hits never record an outcome.
Policy Learning (`app/routing/policy_learner.py`, `app/routing/router.py`,
`app/api/admin.py`) — the third piece. Reads A/B Routing's own
`ab_results` and nudges a losing policy's weights a step toward the
winning policy's weights via one deterministic, documented fitness
formula — not a scheduled loop and not a real ML model; the repeated
scheduling is deliberately left to Evolution Engine (the next piece).
Plugs into the one existing weights-lookup point,
`AdaptiveRouter.resolve_policy`, off by default, over `GET/POST
/admin/policy_learning/{weights,relearn}`. See the Policy Learning section
above for the full fitness/nudge design and its honest dependency on A/B
Routing already being enabled and trafficked.
Evolution Engine (`app/routing/evolution_engine.py`, `app/api/admin.py`)
— the fourth piece. A background loop (same hand-rolled shape as
`BenchmarkScheduler`) that evaluates each prior cycle's nudge against
fresh post-nudge evidence — rolling it back via `PolicyLearner.revert()`
if it made things worse — before asking `PolicyLearner.run_once()` for a
fresh one, excluding any policy still awaiting evaluation. Genuine
"variation, evaluation, retention-or-reversion," off by default, over
`GET/POST /admin/evolution/{status,run_once}`. See the Evolution Engine
section above for the full design, its honest dependency on Policy
Learning already being enabled, and why it's a real (if imperfect)
evaluation rather than a fake scheduled wrapper.
Self-healing (`app/reliability/self_healer.py`, `app/api/admin.py`) — the
fifth and last piece. Correlates two signals that already existed for
free — `ProviderRegistry.health_of()` (from `HealthMonitor`'s own
polling) and `CircuitBreakerRegistry`'s per-provider state (from real
request traffic) — and closes the one real automation gap left in the
reliability stack: `set_enabled()` had exactly 3 callers in the whole
codebase (its own definition plus the two admin routes) before this
piece, meaning a persistently unhealthy or circuit-flapping provider
stayed in the routing pool forever unless a person disabled it by hand.
Auto-disables after `confirm_cycles` consecutive bad checks, auto
re-enables a self-disabled provider after `confirm_cycles` consecutive
good checks, off by default, over `GET/POST /admin/self_healing/
{status,run_once}` — and never fights an admin: every provider-mutating
admin route now calls a new `SelfHealer.forget()` so an explicit admin
decision always wins. See the Self-healing section above for the full
design and what it deliberately doesn't consider (quota risk, benchmark
history) and why.

Phase 5 is now complete: all five pieces (Automated Benchmark, A/B
Routing, Policy Learning, Evolution Engine, Self-healing) are shipped, in
the dependency order explained above, each with its own review checkpoint.

## What's not implemented yet

Also still open: multi-turn agent loops,
tool-using agents that act on a plan's own intermediate results mid-run
rather than a single forced-JSON planning call up front, plugin loader,
network failover/VPN layer, and PostgreSQL migration. Tools now cover
`web_search` *and* `web_fetch` (Tavily Extract-backed — see the Tools
section above for why that shape was chosen over a hand-rolled fetcher
or a Playwright-based browser), closing what used to be listed here as
"other tool types" and "browser/web-AI adapter". Memory/RAG now has a
real embeddings-backed `Retriever` (opt-in, off by default — see the
Memory/RAG section above) alongside the always-on keyword one. Race mode
covers streaming
too, and the Quality Gate now has an opt-in LLM-graded judgment layer on
top of its structural checks (see above for both) — the quality gate as a
whole remains **deliberately** non-streaming-only regardless, structural
or LLM-graded — not a gap, a permanent scope decision (a streamed response
has already reached the client chunk by chunk by the time it could be
assessed, so there's nothing left to retry; see
`app/intelligence/quality_gate.py`'s docstring and the Quality gate
section above for the full reasoning and the alternatives considered).
Their directories exist as reserved, empty packages
(`app/execution/cancellation.py`, `app/network`, `app/plugins`) so the
rest of Phase 5+ has a home without restructuring what's already
built.

## Project layout

See the full tree below (or run `find app config scripts tests -type f`).
