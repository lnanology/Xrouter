"""ChatEngine._use_race/_use_stream_race are small pure gates (server
switch AND per-request opt-in, plus stream=False/True respectively) but
they're the one place a misconfigured server could start silently racing
multiple providers against every request — so they get their own focused
tests rather than only being exercised indirectly through the full FastAPI
app. The two gates are mutually exclusive by construction (one requires
`not request.stream`, the other requires `request.stream`), so exactly one
of them is ever true for a given request once race_mode_enabled and
request.race both hold."""
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
        (True, True, True, False),    # streaming requests use _use_stream_race instead
    ],
)
def test_use_race_gating(server_enabled, client_race, stream, expected):
    engine = _engine(server_enabled)
    assert engine._use_race(_req(client_race, stream)) is expected


@pytest.mark.parametrize(
    "server_enabled,client_race,stream,expected",
    [
        (True, True, True, True),      # everything aligned -> stream race
        (False, True, True, False),    # server switch off -> never race, even if client asks
        (True, False, True, False),    # client didn't opt in -> no race
        (True, True, False, False),    # non-streaming requests use _use_race instead
    ],
)
def test_use_stream_race_gating(server_enabled, client_race, stream, expected):
    engine = _engine(server_enabled)
    assert engine._use_stream_race(_req(client_race, stream)) is expected
