"""
Disk cache for LLM responses used by benchmark runs.

The cache key is based on the complete system prompt, user prompt, model, and
temperature. This makes long benchmark runs resumable and protects against
paying twice for the same prompt during debugging.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class LLMResponseCache:
    def __init__(self, root: str = "results/llm_cache", enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def key(self, system: str, user: str, model: str, temperature: float) -> str:
        payload = {
            "system": system,
            "user": user,
            "model": model,
            "temperature": float(temperature),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(
        self,
        cache_key: str,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        response: str,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        payload = {
            "cache_key": cache_key,
            "model": model,
            "temperature": float(temperature),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "system_sha256": sha256_text(system),
            "user_sha256": sha256_text(user),
            "response": response,
        }
        self.path_for_key(cache_key).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def path_for_key(self, cache_key: str) -> Path:
        return self.root / f"{cache_key}.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cached_chat(
    *,
    client,
    system: str,
    user: str,
    model: str,
    temperature: float,
    cache: Optional[LLMResponseCache],
    chat_fn,
) -> Tuple[str, Dict[str, Any]]:
    cache_enabled = bool(cache and cache.enabled)
    cache_key = cache.key(system, user, model, temperature) if cache else ""
    if cache_enabled:
        cached = cache.get(cache_key)
        if cached is not None:
            return str(cached.get("response", "")), {
                "cache_enabled": True,
                "cache_hit": True,
                "cache_key": cache_key,
                "cache_path": str(cache.path_for_key(cache_key)),
                "model": model,
                "temperature": float(temperature),
            }

    if client is None:
        raise RuntimeError(
            "LLM cache miss and no live client is available. "
            "Disable --offline-cache-only or populate the cache first."
        )

    response = chat_fn(client, system, user, model=model, temperature=temperature)
    if cache_enabled:
        cache.put(
            cache_key,
            system=system,
            user=user,
            model=model,
            temperature=temperature,
            response=response,
        )
    return response, {
        "cache_enabled": cache_enabled,
        "cache_hit": False,
        "cache_key": cache_key,
        "cache_path": str(cache.path_for_key(cache_key)) if cache else "",
        "model": model,
        "temperature": float(temperature),
    }
