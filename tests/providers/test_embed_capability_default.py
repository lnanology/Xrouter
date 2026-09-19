"""Provider.embed()'s base-class default (app/providers/base.py) is only
ever exercised in production by an adapter that never overrides it --
GeminiAdapter is exactly that case today, so it's the real regression
test for "an adapter that doesn't support embeddings fails closed with a
typed error, not a fake vector"."""
import pytest

from app.contracts.provider import ProviderCapability, ProviderConfig
from app.core.errors import ProviderCapabilityUnsupportedError
from app.providers.gemini.adapter import GeminiAdapter


def _config():
    return ProviderConfig(id="gemini", name="gemini", type="gemini", api_key_env="GEMINI_API_KEY")


@pytest.mark.asyncio
async def test_gemini_does_not_declare_embeddings_capability():
    adapter = GeminiAdapter(_config())
    assert ProviderCapability.EMBEDDINGS not in adapter.capabilities()
    await adapter.close()


@pytest.mark.asyncio
async def test_embed_on_an_unsupported_adapter_raises_typed_error_not_a_fake_vector():
    adapter = GeminiAdapter(_config())
    with pytest.raises(ProviderCapabilityUnsupportedError) as exc_info:
        await adapter.embed("some-model", ["hello"])
    assert exc_info.value.retryable is False
    assert exc_info.value.provider_id == "gemini"
    await adapter.close()
