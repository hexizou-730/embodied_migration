"""
LLM 客户端: 薄薄一层 OpenRouter 包装。
默认用 claude sonnet / opus, 也可以切 gpt-4o / deepseek。
"""
import os
from openai import OpenAI


DEFAULT_MODEL = os.environ.get("EM_MODEL", "anthropic/claude-sonnet-4.5")
DEFAULT_MAX_TOKENS = 8192


def completion_token_limit() -> int:
    """Return a bounded output limit so providers do not reserve huge outputs."""

    value = int(os.environ.get("EM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    if value <= 0:
        raise ValueError("EM_MAX_TOKENS must be a positive integer.")
    return value


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Please set OPENROUTER_API_KEY in your .env file or environment."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def chat(
    client: OpenAI,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=completion_token_limit(),
    )
    return resp.choices[0].message.content or ""
