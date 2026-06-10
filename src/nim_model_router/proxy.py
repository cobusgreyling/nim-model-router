from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nim_model_router.classifier import _flatten_messages
from nim_model_router.client import NimClient, extract_usage, iter_sse_lines, safe_json_loads
from nim_model_router.config import Settings, load_registry
from nim_model_router.logging_store import RouteLogStore, Timer
from nim_model_router.router import ModelRouter
from nim_model_router.types import RouteLogEntry


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if not settings.nvidia_api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is required. Set it in .env or export NVIDIA_API_KEY=..."
        )

    registry = load_registry(settings.router_config)
    router = ModelRouter(registry)
    client = NimClient(settings)
    log_store = RouteLogStore(settings.router_log_path)

    app = FastAPI(
        title="NIM Model Router",
        description="OpenAI-compatible proxy that routes to the best NVIDIA NIM model by task.",
        version="0.1.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/router/stats")
    async def router_stats() -> dict[str, object]:
        return log_store.summary()

    @app.get("/v1/router/tasks")
    async def router_tasks() -> dict[str, object]:
        return {
            "aliases": registry.aliases,
            "tasks": {
                name: {
                    "model": cfg.model,
                    "description": cfg.description,
                    "extra_body": cfg.extra_body,
                }
                for name, cfg in registry.tasks.items()
            },
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        local_models = router.list_router_models()
        upstream = await client.list_upstream_models()
        merged = {model["id"]: model for model in local_models}
        for model in upstream:
            model_id = model.get("id")
            if model_id and model_id not in merged:
                merged[model_id] = {
                    "id": model_id,
                    "object": model.get("object", "model"),
                    "owned_by": model.get("owned_by", "nvidia-nim"),
                }
        return {"object": "list", "data": list(merged.values())}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_nim_task: str | None = Header(default=None, alias="X-NIM-Task"),
    ) -> Any:
        payload = await request.json()
        timer = Timer()
        decision = router.route_chat(payload, task_header=x_nim_task)
        streamed = bool(payload.get("stream"))

        try:
            upstream_response, forwarded_body = await client.chat_completion(payload, decision)
        except Exception as exc:  # pragma: no cover - network errors
            raise HTTPException(status_code=502, detail=f"upstream NIM error: {exc}") from exc

        routing_headers = {
            "X-NIM-Routed-Task": decision.task.value,
            "X-NIM-Routed-Model": decision.model,
            "X-NIM-Router-Reason": decision.reason,
        }

        if streamed:

            async def event_stream():
                upstream_timer = Timer()
                try:
                    async for chunk in iter_sse_lines(upstream_response):
                        yield chunk
                finally:
                    upstream_timer.stop()
                    timer.stop()
                    log_store.append(
                        RouteLogEntry(
                            task=decision.task.value,
                            model=decision.model,
                            reason=decision.reason,
                            latency_ms=timer.elapsed_ms,
                            prompt_chars=len(_flatten_messages(payload.get("messages"))),
                            has_tools=bool(payload.get("tools")),
                            streamed=True,
                            status_code=upstream_response.status_code,
                            upstream_latency_ms=upstream_timer.elapsed_ms,
                        )
                    )
                    await upstream_response.aclose()

            if upstream_response.status_code != 200:
                body = await upstream_response.aread()
                timer.stop()
                log_store.append(
                    RouteLogEntry(
                        task=decision.task.value,
                        model=decision.model,
                        reason=decision.reason,
                        latency_ms=timer.elapsed_ms,
                        prompt_chars=len(_flatten_messages(payload.get("messages"))),
                        has_tools=bool(payload.get("tools")),
                        streamed=True,
                        status_code=upstream_response.status_code,
                    )
                )
                return JSONResponse(
                    content=safe_json_loads(body),
                    status_code=upstream_response.status_code,
                    headers=routing_headers,
                )

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers=routing_headers,
            )

        timer.stop()
        content = upstream_response.json() if upstream_response.content else {}
        prompt_tokens, completion_tokens = extract_usage(content)
        log_store.append(
            RouteLogEntry(
                task=decision.task.value,
                model=decision.model,
                reason=decision.reason,
                latency_ms=timer.elapsed_ms,
                prompt_chars=len(_flatten_messages(payload.get("messages"))),
                has_tools=bool(payload.get("tools")),
                streamed=False,
                status_code=upstream_response.status_code,
                upstream_latency_ms=timer.elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        return JSONResponse(
            content=content,
            status_code=upstream_response.status_code,
            headers=routing_headers,
        )

    @app.post("/v1/embeddings")
    async def embeddings(
        request: Request,
        x_nim_task: str | None = Header(default=None, alias="X-NIM-Task"),
    ) -> Any:
        payload = await request.json()
        timer = Timer()
        decision = router.route_embedding(payload, task_header=x_nim_task)

        try:
            upstream_response = await client.embedding(payload, decision)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"upstream NIM error: {exc}") from exc

        timer.stop()
        routing_headers = {
            "X-NIM-Routed-Task": decision.task.value,
            "X-NIM-Routed-Model": decision.model,
            "X-NIM-Router-Reason": decision.reason,
        }
        content = upstream_response.json() if upstream_response.content else {}
        log_store.append(
            RouteLogEntry(
                task=decision.task.value,
                model=decision.model,
                reason=decision.reason,
                latency_ms=timer.elapsed_ms,
                prompt_chars=len(str(payload.get("input", ""))),
                has_tools=False,
                streamed=False,
                status_code=upstream_response.status_code,
                upstream_latency_ms=timer.elapsed_ms,
            )
        )
        return JSONResponse(
            content=content,
            status_code=upstream_response.status_code,
            headers=routing_headers,
        )

    @app.post("/v1/router/dry-run")
    async def dry_run(
        request: Request,
        x_nim_task: str | None = Header(default=None, alias="X-NIM-Task"),
    ) -> dict[str, object]:
        payload = await request.json()
        endpoint = payload.pop("endpoint", "chat")
        if endpoint == "embedding":
            decision = router.route_embedding(payload, task_header=x_nim_task)
        else:
            decision = router.route_chat(payload, task_header=x_nim_task)
        return {
            "task": decision.task.value,
            "model": decision.model,
            "reason": decision.reason,
            "extra_body": decision.extra_body,
            "requested_model": payload.get("model"),
        }

    return app
