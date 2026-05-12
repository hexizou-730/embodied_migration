"""
LLM 客户端: 薄薄一层 OpenRouter 包装。
默认用 claude sonnet / opus, 也可以切 gpt-4o / deepseek。
"""
import os
from openai import OpenAI


DEFAULT_MODEL = os.environ.get("EM_MODEL", "anthropic/claude-sonnet-4.5")


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
    )
    return resp.choices[0].message.content or ""
