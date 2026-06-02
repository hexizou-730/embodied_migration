"""Small OpenAI-compatible LLM client wrapper.

The project can call either OpenRouter or DeepSeek directly. Select the backend
with ``EM_LLM_PROVIDER``:

- ``openrouter`` uses ``OPENROUTER_API_KEY`` and OpenRouter model ids.
- ``deepseek`` uses ``DEEPSEEK_API_KEY`` and DeepSeek model ids.
"""

import os
from openai import OpenAI


PROVIDER_OPENROUTER = "openrouter"
PROVIDER_DEEPSEEK = "deepseek"

PROVIDER_CONFIG = {
    PROVIDER_OPENROUTER: {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-sonnet-4.5",
    },
    PROVIDER_DEEPSEEK: {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-pro",
    },
}


def current_provider() -> str:
    provider = os.environ.get("EM_LLM_PROVIDER", PROVIDER_OPENROUTER).strip().lower()
    if provider not in PROVIDER_CONFIG:
        allowed = ", ".join(sorted(PROVIDER_CONFIG))
        raise ValueError(f"Unknown EM_LLM_PROVIDER={provider!r}. Allowed: {allowed}")
    return provider


def default_model(provider: str | None = None) -> str:
    provider = provider or current_provider()
    return os.environ.get("EM_MODEL") or PROVIDER_CONFIG[provider]["default_model"]


DEFAULT_MODEL = default_model()
DEFAULT_MAX_TOKENS = 8192


def completion_token_limit() -> int:
    """Return a bounded output limit so providers do not reserve huge outputs."""

    value = int(os.environ.get("EM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    if value <= 0:
        raise ValueError("EM_MAX_TOKENS must be a positive integer.")
    return value


def api_key_env(provider: str | None = None) -> str:
    provider = provider or current_provider()
    return PROVIDER_CONFIG[provider]["api_key_env"]


def has_api_key(provider: str | None = None) -> bool:
    return bool(os.environ.get(api_key_env(provider)))


def make_client(provider: str | None = None) -> OpenAI:
    provider = provider or current_provider()
    key_env = api_key_env(provider)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(
            f"Please set {key_env} in your .env file or environment."
        )
    return OpenAI(
        base_url=PROVIDER_CONFIG[provider]["base_url"],
        api_key=api_key,
    )


def chat(
    client: OpenAI,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    resp = client.chat.completions.create(
        model=model or default_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=completion_token_limit(),
    )
    return resp.choices[0].message.content or ""
