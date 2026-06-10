from __future__ import annotations

from typing import Any

import httpx


async def fetch_nim_catalog(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []


def suggest_models_for_task(
    catalog: list[dict[str, Any]],
    *,
    keywords: list[str],
    limit: int = 5,
) -> list[str]:
    matches: list[tuple[int, str]] = []
    for item in catalog:
        model_id = str(item.get("id", ""))
        if not model_id:
            continue
        haystack = model_id.lower()
        score = sum(1 for keyword in keywords if keyword.lower() in haystack)
        if score:
            matches.append((score, model_id))
    matches.sort(key=lambda pair: (-pair[0], pair[1]))
    return [model_id for _, model_id in matches[:limit]]
