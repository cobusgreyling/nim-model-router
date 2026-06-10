import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from nim_model_router.config import Settings
from nim_model_router.proxy import create_app


@pytest.fixture
def settings():
    return Settings(nvidia_api_key="test-key", enable_prometheus=True)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_router_tasks(client):
    response = client.get("/v1/router/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "fast" in data["tasks"]
    assert "nim-router/auto" in data["aliases"]
    assert "policies" in data


def test_dry_run_auto_fast(client):
    response = client.post(
        "/v1/router/dry-run",
        json={
            "model": "nim-router/auto",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "fast"
    assert "llama" in data["model"]
    assert "confidence" in data


def test_dry_run_tools_agentic(client):
    response = client.post(
        "/v1/router/dry-run",
        json={
            "model": "nim-router/auto",
            "messages": [{"role": "user", "content": "search docs"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "agentic"


def test_models_lists_aliases(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert "nim-router/auto" in ids
    assert "meta/llama-3.1-8b-instruct" in ids


def test_router_reload(client):
    response = client.post("/v1/router/reload")
    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"nim_router_requests_total" in response.content


@respx.mock
def test_chat_completion_proxy(settings):
    route = respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )
    )

    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nim-router/fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert route.called
    assert response.headers["X-NIM-Routed-Task"] == "fast"
    assert response.headers["X-NIM-Router-Confidence"]
    assert response.headers["X-NIM-Fallback-Used"] == "false"
    assert response.headers["X-NIM-Estimated-Cost-USD"]


@respx.mock
def test_chat_completion_fallback(settings):
    agentic_model = "nvidia/nemotron-3-super-120b-a12b"
    general_model = "nvidia/nemotron-3-nano-30b-a3b"

    respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(503, json={"error": "unavailable"}),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "fallback"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ),
        ]
    )

    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nim-router/agentic",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["X-NIM-Routed-Model"] in {agentic_model, general_model}
    assert response.headers["X-NIM-Fallback-Used"] == "true"


@respx.mock
def test_rerank_endpoint(settings):
    route = respx.post("https://integrate.api.nvidia.com/v1/ranking").mock(
        return_value=httpx.Response(200, json={"results": [{"index": 0, "score": 0.9}]})
    )

    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/rerank",
            json={
                "model": "nim-router/rerank",
                "query": "what is nim",
                "documents": ["NIM is ...", "other"],
            },
        )

    assert response.status_code == 200
    assert route.called
    assert response.headers["X-NIM-Routed-Task"] == "rerank"


@respx.mock
def test_streaming_chat(settings):
    respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text='data: {"choices":[]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )

    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nim-router/fast",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@respx.mock
def test_auth_middleware(settings):
    respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )
    )

    settings = settings.model_copy(update={"router_api_key": "secret"})
    app = create_app(settings)
    with TestClient(app) as test_client:
        denied = test_client.post(
            "/v1/chat/completions",
            json={"model": "nim-router/fast", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert denied.status_code == 401

        allowed = test_client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "secret"},
            json={"model": "nim-router/fast", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert allowed.status_code == 200


@respx.mock
def test_router_stats_includes_cost(settings):
    route = respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )
    )

    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nim-router/fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200
        assert route.called
        stats = test_client.get("/v1/router/stats").json()

    assert stats["total_requests"] >= 1
    assert stats["estimated_cost_usd"] > 0


def test_cors_headers(settings):
    app = create_app(settings.model_copy(update={"router_cors_origins": "http://localhost:3000"}))
    with TestClient(app) as test_client:
        response = test_client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_request_body_too_large(settings):
    app = create_app(settings.model_copy(update={"max_request_body_bytes": 32}))
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/router/dry-run",
            content=json.dumps({"messages": [{"role": "user", "content": "x" * 100}]}),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 413
