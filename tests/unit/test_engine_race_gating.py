"""ChatEngine._use_race is a small pure gate (server switch AND per-request
opt-in AND not streaming) but it's the one place a misconfigured server
could start silently racing multiple providers against every request, or a
client could accidentally trigger racing on a streaming call that
app/execution/race.py doesn't support — so it gets its own focused test
rather than only being exercised indirectly through the full FastAPI app."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.core.engine import ChatEngine


def _engine(race_mode_enabled: bool) -> ChatEngine:
    ctx = SimpleNamespace(settings=SimpleNamespace(routing=SimpleNamespace(race_mode_enabled=race_mode_enabled)))
    return ChatEngine(ctx)


def _req(race: bool, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="hi")], race=race, stream=stream)


@pytest.mark.parametrize(
    "server_enabled,client_race,stream,expected",
    [
        (True, True, False, True),    # everything aligned -> race
        (False, True, False, False),  # server switch off -> never race, even if client asks
        (True, False, False, False),  # client didn't opt in -> no race
        (True, True, True, False),    # streaming always ignores race
    ],
)
def test_use_race_gating(server_enabled, client_race, stream, expected):
    engine = _engine(server_enabled)
    assert engine._use_race(_req(client_race, stream)) is expected
