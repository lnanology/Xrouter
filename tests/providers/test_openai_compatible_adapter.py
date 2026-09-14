import os

import httpx
import pytest
import respx

from app.contracts.provider import ProviderConfig, ProviderStatus
from app.contracts.request import ChatCompletionRequest, ChatMessage
from app.core.errors import ProviderAuthError, ProviderRateLimitError, ProviderUnavailableError
from app.providers.openai_compatible.adapter import OpenAICompatibleAdapter


def _config(with_key=True):
    if with_key:
        os.environ["TEST_API_KEY"] = "sk-test-123"
    else:
        os.environ.pop("TEST_API_KEY", None)
    return ProviderConfig(
        id="testprov", name="testprov", type="openai_compatible",
        base_url="https://api.example.com/v1", api_key_env="TEST_API_KEY", timeout_seconds=5,
    )


@pytest.mark.asyncio
async def test_missing_api_key_disables_gracefully_no_crash():
    adapter = OpenAICompatibleAdapter(_config(with_key=False))
    models = await adapter.list_models()
    assert models == []
    health = await adapter.health()
    assert health.status == ProviderStatus.DISABLED
    with pytest.raises(ProviderUnavailableError):
        await adapter.chat("some-model", ChatCompletionRequest(model="x", messages=[ChatMessage(role="user", content="hi")]))
    await adapter.close()


@pytest.mark.asyncio
async def test_chat_success():
    adapter = OpenAICompatibleAdapter(_config())
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        )
        resp = await adapter.chat("gpt-test", ChatCompletionRequest(model="x", messages=[ChatMessage(role="user", content="hi")]))
    assert resp.choices[0].message["content"] == "hi"
    assert resp.usage.total_tokens == 5
    await adapter.close()


@pytest.mark.asyncio
async def test_401_raises_auth_error_never_retryable():
    adapter = OpenAICompatibleAdapter(_config())
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(401, text="invalid api key"))
        with pytest.raises(ProviderAuthError) as exc_info:
            await adapter.chat("gpt-test", ChatCompletionRequest(model="x", messages=[ChatMessage(role="user", content="hi")]))
    assert exc_info.value.retryable is False
    await adapter.close()


@pytest.mark.asyncio
async def test_429_raises_rate_limit_with_retry_after():
    adapter = OpenAICompatibleAdapter(_config())
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(429, headers={"retry-after": "2.5"}, text="slow down"))
        with pytest.raises(ProviderRateLimitError) as exc_info:
            await adapter.chat("gpt-test", ChatCompletionRequest(model="x", messages=[ChatMessage(role="user", content="hi")]))
    assert exc_info.value.retry_after == 2.5
    await adapter.close()


@pytest.mark.asyncio
async def test_500_raises_retryable_server_error():
    from app.core.errors import ProviderServerError

    adapter = OpenAICompatibleAdapter(_config())
    with respx.mock(base_url="https://api.example.com/v1") as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(500, text="oops"))
        with pytest.raises(ProviderServerError) as exc_info:
            await adapter.chat("gpt-test", ChatCompletionRequest(model="x", messages=[ChatMessage(role="user", content="hi")]))
    assert exc_info.value.retryable is True
    await adapter.close()
