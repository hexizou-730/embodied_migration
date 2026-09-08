"""Validate paper-run LLM configuration without making an API request."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

from llm_client import (
    api_key_env,
    completion_token_limit,
    current_provider,
    deepseek_thinking_mode,
    default_model,
    generation_temperature,
)
from maniskill_backend.paper_benchmark import PAPER_METHODS


def run_llm_preflight(
    method_ids: Iterable[str],
    *,
    require_stochastic_repetitions: bool = False,
) -> dict[str, Any]:
    """Check provider settings and key shape while keeping the key secret."""

    selected = [str(item) for item in method_ids]
    llm_methods = [item for item in selected if PAPER_METHODS[item].uses_llm]
    required = bool(llm_methods)
    errors: list[str] = []
    provider = ""
    model = ""
    key_name = ""
    key_present = False
    max_tokens: int | None = None
    temperature: float | None = None
    thinking = "not_applicable"
    try:
        load_dotenv(Path.cwd() / ".env")
        provider = current_provider()
        model = default_model(provider)
        key_name = api_key_env(provider)
        raw_key = os.environ.get(key_name, "")
        key_present = bool(raw_key)
        if required and not raw_key:
            errors.append(f"missing {key_name}")
        if required and raw_key and raw_key != raw_key.strip():
            errors.append(f"{key_name} has leading or trailing whitespace")
        if required and raw_key and not raw_key.isascii():
            errors.append(f"{key_name} contains non-ASCII characters")
        max_tokens = completion_token_limit()
        temperature = generation_temperature()
        if required and require_stochastic_repetitions and temperature <= 0.0:
            errors.append(
                "paper tier has repeated LLM generations and requires EM_TEMPERATURE > 0"
            )
        if provider == "deepseek":
            thinking = deepseek_thinking_mode()
        if required and not str(model).strip():
            errors.append("EM_MODEL resolves to an empty model id")
    except Exception as exc:
        errors.append(repr(exc))
    return {
        "schema": "paper_llm_preflight.v1",
        "ready": not errors,
        "required": required,
        "selected_methods": selected,
        "llm_methods": llm_methods,
        "provider": provider,
        "model": model,
        "api_key_env": key_name,
        "api_key_present": key_present,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "require_stochastic_repetitions": require_stochastic_repetitions,
        "deepseek_thinking": thinking,
        "explicit_configuration": {
            "provider": "EM_LLM_PROVIDER" in os.environ,
            "model": "EM_MODEL" in os.environ,
            "max_tokens": "EM_MAX_TOKENS" in os.environ,
            "temperature": "EM_TEMPERATURE" in os.environ,
            "deepseek_thinking": (
                provider != "deepseek" or "EM_DEEPSEEK_THINKING" in os.environ
            ),
        },
        "errors": errors,
    }


def llm_preflight_markdown(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# LLM Configuration Preflight",
            "",
            f"- ready: `{payload.get('ready')}`",
            f"- required: `{payload.get('required')}`",
            f"- methods: `{payload.get('llm_methods')}`",
            f"- provider: `{payload.get('provider')}`",
            f"- model: `{payload.get('model')}`",
            f"- key variable: `{payload.get('api_key_env')}`",
            f"- key present: `{payload.get('api_key_present')}`",
            f"- max tokens: `{payload.get('max_tokens')}`",
            f"- temperature: `{payload.get('temperature')}`",
            f"- DeepSeek thinking: `{payload.get('deepseek_thinking')}`",
            f"- explicit configuration: `{payload.get('explicit_configuration')}`",
            f"- errors: `{payload.get('errors')}`",
            "",
        ]
    )


def write_llm_preflight(payload: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "llm_preflight.json"
    md_path = output_dir / "llm_preflight.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(llm_preflight_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
