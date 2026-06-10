from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from nim_model_router import __version__
from nim_model_router.auth import RouterAuthMiddleware
from nim_model_router.classifier import _flatten_messages, classify_with_llm
from nim_model_router.client import (
    NimClient,
    extract_usage,
    iter_sse_lines,
    safe_json_loads,
)
from nim_model_router.config import Settings, load_registry
from nim_model_router.cost import estimate_request_cost
from nim_model_router.logging_store import RouteLogStore, Timer
from nim_model_router.metrics import metrics_response, record_request
from nim_model_router.policies import validate_registry_policies
from nim_model_router.router import ModelRouter
from nim_model_router.types import RouteLogEntry


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = load_registry(settings.router_config)
        self.router = ModelRouter(self.registry)
        self.log_store = RouteLogStore(settings.router_log_path)
        self.http_client: httpx.AsyncClient | None = None
        self.client: NimClient | None = None

    def reload_registry(self) -> list[str]:
        self.registry = load_registry(self.settings.router_config)
        self.router = ModelRouter(self.registry)
        return validate_registry_policies(self.registry)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if not settings.nvidia_api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is required. Set it in .env or export NVIDIA_API_KEY=..."
        )

    state = AppState(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0))
        state.client = NimClient(settings, http_client=state.http_client)
        yield
        await state.http_client.aclose()

    app = FastAPI(
        title="NIM Model Router",
        description="OpenAI-compatible proxy that routes to the best NVIDIA NIM model by task.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.nim_state = state

    if settings.router_api_key:
        app.add_middleware(RouterAuthMiddleware, router_api_key=settings.router_api_key)

    if settings.router_cors_origins:
        origins = [
            origin.strip()
            for origin in settings.router_cors_origins.split(",")
            if origin.strip()
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _routing_headers(
        decision,
        *,
        fallback_used: bool = False,
        estimated_cost_usd: float | None = None,
    ) -> dict[str, str]:
        headers = {
            "X-NIM-Routed-Task": decision.task.value,
            "X-NIM-Routed-Model": decision.model,
            "X-NIM-Router-Reason": decision.reason,
            "X-NIM-Router-Confidence": f"{decision.confidence:.3f}",
            "X-NIM-Fallback-Used": "true" if fallback_used else "false",
        }
        if estimated_cost_usd is not None:
            headers["X-NIM-Estimated-Cost-USD"] = f"{estimated_cost_usd:.8f}"
        return headers

    async def _read_payload(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if len(raw) > settings.max_request_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        if not raw:
            return {}
        try:
            data = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        return data

    async def _maybe_llm_classify(payload: dict[str, Any]):
        if not state.registry.classifier.use_llm_classifier:
            return None
        return await classify_with_llm(
            messages=payload.get("messages"),
            input_text=payload.get("input") if "input" in payload else None,
            config=state.registry.classifier,
            api_key=settings.nvidia_api_key,
            base_url=settings.nim_base_url,
        )

    def _record(
        *,
        endpoint: str,
        decision,
        timer: Timer,
        upstream_timer: Timer | None,
        payload: dict[str, Any],
        streamed: bool,
        status_code: int,
        fallback_used: bool,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> None:
        upstream_ms = upstream_timer.elapsed_ms if upstream_timer else None
        if upstream_ms is not None:
            state.router.update_model_latency(decision.model, upstream_ms)

        if estimated_cost_usd is None:
            estimated_cost_usd = estimate_request_cost(
                state.registry,
                task=decision.task.value,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        state.log_store.append(
            RouteLogEntry(
                task=decision.task.value,
                model=decision.model,
                reason=decision.reason,
                confidence=decision.confidence,
                latency_ms=timer.elapsed_ms,
                prompt_chars=len(
                    _flatten_messages(payload.get("messages")) or str(payload.get("input", ""))
                ),
                has_tools=bool(payload.get("tools")),
                streamed=streamed,
                status_code=status_code,
                upstream_latency_ms=upstream_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=estimated_cost_usd,
                fallback_used=fallback_used,
            )
        )

        if settings.enable_prometheus:
            record_request(
                endpoint=endpoint,
                task=decision.task.value,
                model=decision.model,
                status_code=status_code,
                latency_seconds=timer.elapsed_ms / 1000,
                upstream_latency_seconds=upstream_ms / 1000 if upstream_ms else None,
                fallback_used=fallback_used,
                estimated_cost_usd=estimated_cost_usd,
            )

    @app.get("/health")
    async def health() -> dict[str, object]:
        result: dict[str, object] = {"status": "ok", "version": __version__}
        if settings.health_check_upstream and state.client is not None:
            result["upstream"] = await state.client.health_check()
        return result

    @app.get("/metrics")
    async def metrics() -> Response:
        if not settings.enable_prometheus:
            raise HTTPException(status_code=404, detail="prometheus metrics disabled")
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    @app.get("/v1/router/stats")
    async def router_stats() -> dict[str, object]:
        return state.log_store.summary()

    @app.get("/v1/router/tasks")
    async def router_tasks() -> dict[str, object]:
        return {
            "aliases": state.registry.aliases,
            "tasks": {
                name: {
                    "model": cfg.model,
                    "description": cfg.description,
                    "extra_body": cfg.extra_body,
                    "fallbacks": cfg.fallbacks,
                    "priority": cfg.priority,
                    "cost_per_1m_tokens": cfg.cost_per_1m_tokens,
                    "ab_test": cfg.ab_test.model_dump(),
                    "endpoint": cfg.endpoint,
                }
                for name, cfg in state.registry.tasks.items()
            },
            "policies": state.registry.policies.model_dump(),
            "cost_summary": state.router.cost_summary(),
        }

    @app.post("/v1/router/reload")
    async def router_reload() -> dict[str, object]:
        warnings = state.reload_registry()
        return {"status": "reloaded", "config": str(settings.router_config), "warnings": warnings}

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        local_models = state.router.list_router_models()
        merged = {model["id"]: model for model in local_models}
        if state.client is not None:
            upstream = await state.client.list_upstream_models()
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
        payload = await _read_payload(request)
        timer = Timer()
        classification = await _maybe_llm_classify(payload)
        decision = state.router.route_chat(
            payload,
            task_header=x_nim_task,
            classification=classification,
        )
        streamed = bool(payload.get("stream"))

        if state.client is None:
            raise HTTPException(status_code=503, detail="client not initialized")

        upstream_timer = Timer()
        try:
            upstream_response, decision, fallback_used = await state.client.chat_completion(
                payload, decision
            )
        except Exception as exc:  # pragma: no cover - network errors
            raise HTTPException(status_code=502, detail=f"upstream NIM error: {exc}") from exc
        finally:
            upstream_timer.stop()

        routing_headers = _routing_headers(decision, fallback_used=fallback_used)

        if streamed:

            async def event_stream():
                upstream_timer = Timer()
                try:
                    async for chunk in iter_sse_lines(upstream_response):
                        yield chunk
                finally:
                    upstream_timer.stop()
                    timer.stop()
                    _record(
                        endpoint="chat",
                        decision=decision,
                        timer=timer,
                        upstream_timer=upstream_timer,
                        payload=payload,
                        streamed=True,
                        status_code=upstream_response.status_code,
                        fallback_used=fallback_used,
                    )
                    await upstream_response.aclose()

            if upstream_response.status_code != 200:
                body = await upstream_response.aread()
                timer.stop()
                _record(
                    endpoint="chat",
                    decision=decision,
                    timer=timer,
                    upstream_timer=None,
                    payload=payload,
                    streamed=True,
                    status_code=upstream_response.status_code,
                    fallback_used=fallback_used,
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
        estimated_cost_usd = estimate_request_cost(
            state.registry,
            task=decision.task.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        _record(
            endpoint="chat",
            decision=decision,
            timer=timer,
            upstream_timer=upstream_timer,
            payload=payload,
            streamed=False,
            status_code=upstream_response.status_code,
            fallback_used=fallback_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        return JSONResponse(
            content=content,
            status_code=upstream_response.status_code,
            headers=_routing_headers(
                decision,
                fallback_used=fallback_used,
                estimated_cost_usd=estimated_cost_usd,
            ),
        )

    @app.post("/v1/embeddings")
    async def embeddings(
        request: Request,
        x_nim_task: str | None = Header(default=None, alias="X-NIM-Task"),
    ) -> Any:
        payload = await _read_payload(request)
        timer = Timer()
        classification = await _maybe_llm_classify(payload)
        decision = state.router.route_embedding(
            payload,
            task_header=x_nim_task,
            classification=classification,
        )

        if state.client is None:
            raise HTTPException(status_code=503, detail="client not initialized")

        try:
            upstream_response, fallback_used = await state.client.embedding(payload, decision)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"upstream NIM error: {exc}") from exc

        upstream_timer = Timer()
        upstream_timer.stop()
        timer.stop()
        routing_headers = _routing_headers(decision, fallback_used=fallback_used)
        content = upstream_response.json() if upstream_response.content else {}
        _record(
            endpoint="embeddings",
            decision=decision,
            timer=timer,
            upstream_timer=upstream_timer,
            payload=payload,
            streamed=False,
            status_code=upstream_response.status_code,
            fallback_used=fallback_used,
        )
        return JSONResponse(
            content=content,
            status_code=upstream_response.status_code,
            headers=routing_headers,
        )

    @app.post("/v1/rerank")
    @app.post("/v1/ranking")
    async def rerank(
        request: Request,
        x_nim_task: str | None = Header(default=None, alias="X-NIM-Task"),
    ) -> Any:
        payload = await _read_payload(request)
        timer = Timer()
        decision = state.router.route_rerank(payload, task_header=x_nim_task)

        if state.client is None:
            raise HTTPException(status_code=503, detail="client not initialized")

        try:
            upstream_response, fallback_used = await state.client.rerank(payload, decision)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"upstream NIM error: {exc}") from exc

        upstream_timer = Timer()
        upstream_timer.stop()
        timer.stop()
        routing_headers = _routing_headers(decision, fallback_used=fallback_used)
        content = upstream_response.json() if upstream_response.content else {}
        _record(
            endpoint="rerank",
            decision=decision,
            timer=timer,
            upstream_timer=upstream_timer,
            payload=payload,
            streamed=False,
            status_code=upstream_response.status_code,
            fallback_used=fallback_used,
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
        payload = await _read_payload(request)
        endpoint = payload.pop("endpoint", "chat")
        classification = await _maybe_llm_classify(payload)
        if endpoint == "embedding":
            decision = state.router.route_embedding(
                payload,
                task_header=x_nim_task,
                classification=classification,
            )
        elif endpoint in {"rerank", "ranking"}:
            decision = state.router.route_rerank(payload, task_header=x_nim_task)
        else:
            decision = state.router.route_chat(
                payload,
                task_header=x_nim_task,
                classification=classification,
            )
        return {
            "task": decision.task.value,
            "model": decision.model,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "extra_body": decision.extra_body,
            "fallback_models": decision.fallback_models,
            "endpoint_path": decision.endpoint_path,
            "requested_model": payload.get("model"),
        }

    return app
