from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from nim_model_router.config import Settings
from nim_model_router.types import RouteDecision


def _merge_extra_body(payload: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
    forwarded = dict(payload)
    forwarded["model"] = decision.model

    existing_extra = forwarded.pop("extra_body", None)
    merged_extra: dict[str, Any] = {}
    if isinstance(existing_extra, dict):
        merged_extra.update(existing_extra)
    merged_extra.update(decision.extra_body)
    if merged_extra:
        forwarded.update(merged_extra)
    return forwarded


class NimClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._headers = {
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json",
            "NVCF-POLL-SECONDS": settings.nvcf_poll_seconds,
        }

    @property
    def base_url(self) -> str:
        return self.settings.nim_base_url.rstrip("/")

    async def chat_completion(
        self,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[httpx.Response, dict[str, Any]]:
        body = _merge_extra_body(payload, decision)
        stream = bool(body.get("stream"))
        timeout = httpx.Timeout(300.0, connect=30.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            if stream:
                request = client.build_request(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=body,
                )
                response = await client.send(request, stream=True)
                return response, body

            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=body,
            )
            return response, body

    async def embedding(
        self,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> httpx.Response:
        body = dict(payload)
        body["model"] = decision.model
        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers,
                json=body,
            )

    async def list_upstream_models(self) -> list[dict[str, Any]]:
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self._headers,
            )
            if response.status_code != 200:
                return []
            data = response.json()
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                return data["data"]
            return []


async def iter_sse_lines(response: httpx.Response) -> AsyncIterator[bytes]:
    async for line in response.aiter_lines():
        if line:
            yield (line + "\n").encode("utf-8")
        else:
            yield b"\n"


def extract_usage(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def safe_json_loads(raw: bytes | str) -> dict[str, Any]:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}