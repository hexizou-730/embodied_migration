"""Small OpenAI-compatible LLM client wrapper.

The project can call either OpenRouter or DeepSeek directly. Select the backend
with ``EM_LLM_PROVIDER``:

- ``openrouter`` uses ``OPENROUTER_API_KEY`` and OpenRouter model ids.
- ``deepseek`` uses ``DEEPSEEK_API_KEY`` and DeepSeek model ids.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI


PROVIDER_OPENROUTER = "openrouter"
PROVIDER_DEEPSEEK = "deepseek"

# Every caller, including the runtime-contract recorder, must observe the same
# project-local LLM configuration before it selects a provider or model.
load_dotenv(Path(__file__).resolve().with_name(".env"))


def _experiment_llm_defaults() -> Dict[str, Any]:
    path = Path(__file__).resolve().with_name("experiment_config.json")
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("llm", {}))


EXPERIMENT_LLM = _experiment_llm_defaults()

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
    provider = os.environ.get(
        "EM_LLM_PROVIDER", EXPERIMENT_LLM.get("provider", PROVIDER_OPENROUTER)
    ).strip().lower()
    if provider not in PROVIDER_CONFIG:
        allowed = ", ".join(sorted(PROVIDER_CONFIG))
        raise ValueError(f"Unknown EM_LLM_PROVIDER={provider!r}. Allowed: {allowed}")
    return provider


def default_model(provider: str | None = None) -> str:
    provider = provider or current_provider()
    configured = EXPERIMENT_LLM.get("model") if provider == EXPERIMENT_LLM.get("provider") else None
    return os.environ.get("EM_MODEL") or configured or PROVIDER_CONFIG[provider]["default_model"]


DEFAULT_MODEL = default_model()
DEFAULT_MAX_TOKENS = int(EXPERIMENT_LLM.get("max_tokens", 8192))
DEEPSEEK_THINKING_MODES = {"enabled", "disabled"}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class ChatResponse:
    content: str
    model: str
    usage: Dict[str, Any]


def completion_token_limit() -> int:
    """Return a bounded output limit so providers do not reserve huge outputs."""

    value = int(os.environ.get("EM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    if value <= 0:
        raise ValueError("EM_MAX_TOKENS must be a positive integer.")
    return value


def generation_temperature() -> float:
    """Return the recorded sampling temperature for independent paper runs."""

    value = float(os.environ.get("EM_TEMPERATURE", EXPERIMENT_LLM.get("temperature", 0.0)))
    if value < 0.0 or value > 2.0:
        raise ValueError("EM_TEMPERATURE must be between 0 and 2.")
    return value


def deepseek_thinking_mode() -> str:
    """Use non-thinking output by default so code modules are not truncated."""

    value = os.environ.get("EM_DEEPSEEK_THINKING", "disabled").strip().lower()
    if value not in DEEPSEEK_THINKING_MODES:
        allowed = ", ".join(sorted(DEEPSEEK_THINKING_MODES))
        raise ValueError(f"Unknown EM_DEEPSEEK_THINKING={value!r}. Allowed: {allowed}")
    return value


def llm_seed() -> int:
    value = int(os.environ.get("EM_LLM_SEED", EXPERIMENT_LLM.get("seed", 0)))
    if value < 0:
        raise ValueError("EM_LLM_SEED must be non-negative.")
    return value


def openrouter_reasoning_effort() -> str:
    value = os.environ.get(
        "EM_REASONING_EFFORT", EXPERIMENT_LLM.get("reasoning_effort", "low")
    ).strip().lower()
    if value not in REASONING_EFFORTS:
        raise ValueError(f"EM_REASONING_EFFORT must be one of {sorted(REASONING_EFFORTS)}.")
    return value


def openrouter_exclude_reasoning() -> bool:
    raw = os.environ.get(
        "EM_EXCLUDE_REASONING",
        str(EXPERIMENT_LLM.get("exclude_reasoning_from_response", True)),
    ).strip().lower()
    if raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("EM_EXCLUDE_REASONING must be true or false.")
    return raw in {"true", "1", "yes"}


def openrouter_upstream_provider() -> str:
    return os.environ.get(
        "EM_OPENROUTER_PROVIDER", EXPERIMENT_LLM.get("upstream_provider", "deepseek")
    ).strip().lower()


def openrouter_allow_fallbacks() -> bool:
    raw = os.environ.get(
        "EM_OPENROUTER_ALLOW_FALLBACKS",
        str(EXPERIMENT_LLM.get("allow_provider_fallbacks", False)),
    ).strip().lower()
    if raw not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("EM_OPENROUTER_ALLOW_FALLBACKS must be true or false.")
    return raw in {"true", "1", "yes"}


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
    return chat_with_metadata(
        client=client,
        system=system,
        user=user,
        model=model,
        temperature=temperature,
    ).content


def chat_with_metadata(
    client: OpenAI,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatResponse:
    provider = current_provider()
    request = dict(
        model=model or default_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=completion_token_limit(),
        seed=llm_seed(),
    )
    if provider == PROVIDER_DEEPSEEK:
        request["extra_body"] = {"thinking": {"type": deepseek_thinking_mode()}}
    elif provider == PROVIDER_OPENROUTER:
        request["extra_body"] = {
            "reasoning": {
                "effort": openrouter_reasoning_effort(),
                "exclude": openrouter_exclude_reasoning(),
            },
            "provider": {
                "order": [openrouter_upstream_provider()],
                "allow_fallbacks": openrouter_allow_fallbacks(),
            },
        }
    resp = client.chat.completions.create(**request)
    usage_obj = getattr(resp, "usage", None)
    if usage_obj is None:
        usage: Dict[str, Any] = {}
    elif hasattr(usage_obj, "model_dump"):
        usage = dict(usage_obj.model_dump())
    else:
        usage = {
            key: getattr(usage_obj, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if getattr(usage_obj, key, None) is not None
        }
    return ChatResponse(
        content=resp.choices[0].message.content or "",
        model=str(getattr(resp, "model", None) or model or default_model()),
        usage=usage,
    )
