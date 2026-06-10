from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from nim_model_router.config import Settings
from nim_model_router.types import RouteDecision

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


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
    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._external_client = http_client
        self._headers = {
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json",
            "NVCF-POLL-SECONDS": settings.nvcf_poll_seconds,
        }

    @property
    def base_url(self) -> str:
        return self.settings.nim_base_url.rstrip("/")

    def _client(self) -> httpx.AsyncClient:
        if self._external_client is not None:
            return self._external_client
        return httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        stream: bool = False,
        timeout: httpx.Timeout | None = None,
    ) -> httpx.Response:
        client = self._client()
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.settings.upstream_max_retries):
            try:
                if stream:
                    request = client.build_request(
                        method,
                        url,
                        headers=self._headers,
                        json=json_body,
                        timeout=timeout,
                    )
                    response = await client.send(request, stream=True)
                else:
                    response = await client.request(
                        method,
                        url,
                        headers=self._headers,
                        json=json_body,
                        timeout=timeout,
                    )

                if response.status_code not in RETRYABLE_STATUS:
                    return response

                if attempt < self.settings.upstream_max_retries - 1:
                    await response.aclose()
                    await asyncio.sleep(self.settings.upstream_retry_backoff_seconds * (2**attempt))
                    continue
                return response
            except httpx.TransportError as exc:
                last_error = exc
                if attempt < self.settings.upstream_max_retries - 1:
                    await asyncio.sleep(self.settings.upstream_retry_backoff_seconds * (2**attempt))
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("upstream request failed without response")

    async def chat_completion(
        self,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[httpx.Response, RouteDecision, bool]:
        body = _merge_extra_body(payload, decision)
        stream = bool(body.get("stream"))
        timeout = httpx.Timeout(300.0, connect=30.0)
        models_to_try = [decision.model, *decision.fallback_models]
        last_response: httpx.Response | None = None
        final_decision = decision

        for index, model in enumerate(models_to_try):
            attempt_body = dict(body)
            attempt_body["model"] = model
            response = await self._request_with_retry(
                "POST",
                "/chat/completions",
                json_body=attempt_body,
                stream=stream,
                timeout=timeout,
            )
            if response.status_code < 500 or index == len(models_to_try) - 1:
                fallback_used = index > 0 and response.status_code < 500
                if fallback_used:
                    final_decision = decision.model_copy(
                        update={
                            "model": model,
                            "reason": f"{decision.reason}; fallback model {model}",
                        }
                    )
                return response, final_decision, fallback_used

            last_response = response
            await response.aclose()

        if last_response is not None:
            return last_response, final_decision, len(models_to_try) > 1
        raise RuntimeError("chat completion produced no upstream response")

    async def embedding(
        self,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[httpx.Response, bool]:
        body = dict(payload)
        body["model"] = decision.model
        timeout = httpx.Timeout(120.0, connect=30.0)
        response = await self._request_with_retry(
            "POST",
            "/embeddings",
            json_body=body,
            timeout=timeout,
        )
        return response, False

    async def rerank(
        self,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[httpx.Response, bool]:
        body = dict(payload)
        body["model"] = decision.model
        path = decision.endpoint_path or "/ranking"
        timeout = httpx.Timeout(120.0, connect=30.0)
        response = await self._request_with_retry(
            "POST",
            path,
            json_body=body,
            timeout=timeout,
        )
        return response, False

    async def list_upstream_models(self) -> list[dict[str, Any]]:
        timeout = httpx.Timeout(30.0, connect=10.0)
        response = await self._request_with_retry(
            "GET",
            "/models",
            timeout=timeout,
        )
        if response.status_code != 200:
            return []
        data = response.json()
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    async def health_check(self) -> dict[str, object]:
        try:
            models = await self.list_upstream_models()
            return {"upstream": "ok", "model_count": len(models)}
        except Exception as exc:
            return {"upstream": "error", "detail": str(exc)}


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
