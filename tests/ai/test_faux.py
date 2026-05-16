import pytest
from nuu.ai.api_registry import get_api_provider
from nuu.ai.providers.faux import register_faux_provider, faux_assistant_message
from nuu.ai.stream import complete
from nuu.ai.types import Context
from nuu.ai.env_api_keys import (
    get_api_key_env_vars,
    get_env_api_key,
    provider_supports_missing_api_key,
)
from nuu.ai.providers.openai_completions import (
    normalize_api_key,
    provider_supports_missing_api_key as openai_provider_supports_missing_api_key,
)
from nuu.ai.providers.ollama import resolve_ollama_base_url
from nuu.ai.providers.register_builtins import register_builtin_providers


@pytest.mark.asyncio
async def test_faux_provider():
    faux = register_faux_provider()
    model = faux.get_model()

    faux.set_responses(
        [
            faux_assistant_message("Hello, I am faux!"),
            faux_assistant_message("Second response"),
        ]
    )

    context = Context(messages=[])

    resp1 = await complete(model, context)
    assert resp1.content[0].text == "Hello, I am faux!"

    resp2 = await complete(model, context)
    assert resp2.content[0].text == "Second response"


def test_local_provider_aliases_resolve_without_keys(monkeypatch: pytest.MonkeyPatch):
    for key in [
        "OLLAMA_API_KEY",
        "OPENAI_COMPATIBLE_API_KEY",
        "LM_STUDIO_API_KEY",
        "LOCALAI_API_KEY",
        "VLLM_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)

    assert get_api_key_env_vars("openai_compatible") == [
        "OPENAI_COMPATIBLE_API_KEY"
    ]
    assert provider_supports_missing_api_key("openai_compatible")
    assert provider_supports_missing_api_key("OpenAI Compatible")
    assert provider_supports_missing_api_key("lmstudio")
    assert openai_provider_supports_missing_api_key("openai_compatible")
    assert openai_provider_supports_missing_api_key("OpenAI Compatible")
    assert normalize_api_key(get_env_api_key("openai_compatible")) == ""


def test_ollama_base_url_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    assert (
        resolve_ollama_base_url("http://localhost:11434/v1")
        == "http://localhost:11434/v1"
    )

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    assert resolve_ollama_base_url("http://localhost:11434/v1") == (
        "http://localhost:11434/v1"
    )

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    assert resolve_ollama_base_url("http://localhost:11434/v1") == (
        "http://localhost:11434/v1"
    )


def test_ollama_provider_is_registered() -> None:
    register_builtin_providers()
    provider = get_api_provider("ollama")
    assert provider is not None
    assert provider.stream is not None
    assert provider.stream_simple is not None
