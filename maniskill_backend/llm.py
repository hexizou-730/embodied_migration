"""LLM helper for early migration baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from llm_client import (
    api_key_env,
    chat_with_metadata,
    current_provider,
    default_model,
    generation_temperature,
    has_api_key,
    make_client,
)
from lmp.extractor import extract_code_or_text


@dataclass(frozen=True)
class LLMResult:
    code: str
    used_llm: bool
    model: str
    raw_text: str = ""
    reason: str = ""
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LLMTextResult:
    text: str
    used_llm: bool
    model: str
    raw_text: str = ""
    reason: str = ""
    usage: dict = field(default_factory=dict)


def has_llm_key() -> bool:
    load_dotenv(Path.cwd() / ".env")
    return has_api_key()


def gen_code(
    *,
    prompt: str,
    fallback_code: str,
    model: Optional[str] = None,
    dry_run: bool = False,
) -> LLMResult:
    """Generate adapted code, falling back to deterministic code without a key."""

    load_dotenv(Path.cwd() / ".env")
    provider = current_provider()
    chosen_model = model or default_model(provider)
    if dry_run:
        return LLMResult(
            code=fallback_code,
            used_llm=False,
            model=chosen_model,
            reason="dry_run",
        )

    if not has_api_key(provider):
        return LLMResult(
            code=fallback_code,
            used_llm=False,
            model=chosen_model,
            reason=f"missing_{api_key_env(provider).lower()}",
        )

    system = (
        "You adapt robot LMP Python code across robot embodiments. "
        "Return only executable Python code. Do not include Markdown."
    )
    response = chat_with_metadata(
        client=make_client(provider),
        system=system,
        user=prompt,
        model=chosen_model,
        temperature=generation_temperature(),
    )
    return LLMResult(
        code=extract_code_or_text(response.content),
        used_llm=True,
        model=response.model,
        raw_text=response.content,
        usage=response.usage,
    )


def gen_text(
    *,
    prompt: str,
    system: str,
    fallback_text: str = "",
    model: Optional[str] = None,
    dry_run: bool = False,
) -> LLMTextResult:
    """Generate free-form text for bounded repo-level repair workflows."""

    load_dotenv(Path.cwd() / ".env")
    provider = current_provider()
    chosen_model = model or default_model(provider)
    if dry_run:
        return LLMTextResult(
            text=fallback_text,
            used_llm=False,
            model=chosen_model,
            reason="dry_run",
        )
    if not has_api_key(provider):
        return LLMTextResult(
            text=fallback_text,
            used_llm=False,
            model=chosen_model,
            reason=f"missing_{api_key_env(provider).lower()}",
        )

    response = chat_with_metadata(
        client=make_client(provider),
        system=system,
        user=prompt,
        model=chosen_model,
        temperature=generation_temperature(),
    )
    return LLMTextResult(
        text=response.content.strip(),
        used_llm=True,
        model=response.model,
        raw_text=response.content,
        usage=response.usage,
    )
