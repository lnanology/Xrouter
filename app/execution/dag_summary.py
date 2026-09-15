"""Shared helper: renders a completed DagRunResponse as plain text,
one line per node (its status, and its output or error). Used by every
stage that needs to show a model "what a DAG run actually produced"
without re-deriving that itself -- the Verifier (app/intelligence/
verifier.py) and the Synthesizer (app/agents/synthesizer.py) both judge
or compose from this same rendering, so a change to how a node's result
reads never has to be kept in sync across two places."""
from __future__ import annotations

from app.contracts.dag import DagRunResponse
from app.contracts.response import extract_message_text

_MAX_NODE_OUTPUT_CHARS = 2000  # keep a single node's output from crowding out everything else in the reader's context


def summarize_dag(dag: DagRunResponse) -> str:
    lines: list[str] = []
    for node in dag.nodes:
        if node.status == "success" and node.response is not None:
            text = extract_message_text(node.response).strip()
            lines.append(f"- '{node.id}' (success): {text[:_MAX_NODE_OUTPUT_CHARS]}")
        elif node.status == "failed":
            lines.append(f"- '{node.id}' (failed): {node.error}")
        else:
            lines.append(f"- '{node.id}' (skipped): {node.error}")
    return "\n".join(lines) or "(no nodes ran)"
