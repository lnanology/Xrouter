# BFCL benchmark harness

Runs a real subset of the [Berkeley Function-Calling Leaderboard
(BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) — part of UC
Berkeley's [Gorilla](https://github.com/ShishirPatil/gorilla) project —
against a live XRouter server's own OpenAI-compatible
`/v1/chat/completions` endpoint, and grades the results with BFCL's own
AST-matching checker (vendored under `vendor/`, unmodified except for
one documented simplification — see `vendor/NOTICE`).

This is a standalone tool, not part of the shipped server. It exists to
produce a real, reproducible, self-reported score for the "Agent
evaluation" section of the main [README](../../README.md#agent-evaluation-bfcl-score) —
not to submit to the official leaderboard.

## Why BFCL

Picked over GAIA (its dataset requires a Hugging Face account and
license agreement — friction that would make "just clone and run" false)
and tau-bench (needs a hand-written domain-tool harness and an LLM
playing "the user" — meaningfully more infrastructure for the same
credibility goal). BFCL's dataset and grader are openly licensed
(Apache-2.0), the grading is fully offline and deterministic (AST/value
matching against published ground truth — no LLM judge), and it measures
exactly what the README's "Agent evaluation" claim is about: whether a
model, routed through XRouter, can reliably call the right tool with the
right arguments — single calls, picking the right tool among several,
calling several tools in one turn, and correctly declining to call
anything at all.

## Categories run

All five are Python-language, "non-live" (the original hand-curated
academic set, not the crowd-sourced "live" split), and gradeable fully
offline with no code execution or external API calls:

| Category | Items | Tests |
|---|---|---|
| `simple_python` | 400 | One function offered, one correct call expected |
| `multiple` | 200 | Several functions offered, must pick the right one |
| `parallel` | 200 | One correct call turns into several independent calls in one turn |
| `parallel_multiple` | 200 | Both of the above at once |
| `irrelevance` | 240 | No offered function applies — correct answer is to call nothing |

1,240 items total. Data pinned at gorilla commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` (2026-03-23) — see
`vendor/NOTICE` for the full attribution.

## Running it

```bash
# start XRouter first (scripts/start.sh), then:
python3 benchmarks/bfcl/run_benchmark.py --model ollama/llama3.2:3b --limit 20   # smoke test
python3 benchmarks/bfcl/run_benchmark.py --model ollama/llama3.2:3b \
    --out benchmarks/bfcl/results.json                                          # the real, published run
```

`results.json` (once produced) is committed — it's the audit trail
behind the score in the main README, not a throwaway file; ad hoc runs
without `--out` land in a gitignored `results_<timestamp>.json` instead.

`--model` is any model id XRouter reports via `python3 -m app.cli
models` (an XRouter-pinned model, not a raw provider model — routing
resolves it exactly, see `AdaptiveRouter._requested_model_filter`). Any
model with real tool-calling support works, local or cloud — this isn't
Ollama-specific. Results are written as JSON (every item's request,
response, and verdict — full audit trail) and a markdown score table is
printed to stdout, ready to paste into the main README.

Local CPU inference is slow — a 3B model runs roughly one item every few
seconds, so a full 1,240-item run takes on the order of an hour or two.
Use `--limit` for a quick sanity check first.

## Honesty notes

- This is a **self-reported run**, not an official BFCL leaderboard
  submission — it isn't run through BFCL's own CLI/scoring pipeline or
  submitted anywhere. The main README says this explicitly next to the
  score.
- Only the AST-matching categories above are run — no code-execution,
  multi-turn, or "live" categories, and no claim is made about them.
- `--limit` runs are for local iteration only; the number that goes in
  the README is always a full run, and the results JSON committed
  alongside it is the receipt.
