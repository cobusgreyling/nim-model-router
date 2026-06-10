import httpx
import pytest
import respx

from nim_model_router.client import NimClient, _merge_extra_body
from nim_model_router.config import Settings
from nim_model_router.types import RouteDecision, TaskType


def test_merge_extra_body():
    decision = RouteDecision(
        task=TaskType.AGENTIC,
        model="nvidia/test",
        reason="test",
        extra_body={"enable_thinking": True, "reasoning_budget": 1024},
    )
    merged = _merge_extra_body(
        {"model": "nim-router/agentic", "messages": [], "extra_body": {"temperature": 0.1}},
        decision,
    )
    assert merged["model"] == "nvidia/test"
    assert merged["enable_thinking"] is True
    assert merged["reasoning_budget"] == 1024
    assert merged["temperature"] == 0.1


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_429():
    route = respx.post("https://integrate.api.nvidia.com/v1/embeddings").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limit"}),
            httpx.Response(200, json={"data": []}),
        ]
    )

    settings = Settings(
        nvidia_api_key="k",
        upstream_max_retries=2,
        upstream_retry_backoff_seconds=0.01,
    )
    client = NimClient(settings)
    decision = RouteDecision(task=TaskType.EMBEDDING, model="nvidia/embed", reason="test")
    response, _ = await client.embedding({"input": "hello"}, decision)
    assert response.status_code == 200
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_chat_fallback_on_429():
    agentic_model = "nvidia/nemotron-3-super-120b-a12b"
    general_model = "nvidia/nemotron-3-nano-30b-a3b"

    respx.post("https://integrate.api.nvidia.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate limit"}),
            httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            ),
        ]
    )

    settings = Settings(
        nvidia_api_key="k",
        upstream_max_retries=1,
        upstream_retry_backoff_seconds=0.01,
    )
    client = NimClient(settings)
    decision = RouteDecision(
        task=TaskType.AGENTIC,
        model=agentic_model,
        reason="test",
        fallback_models=[general_model],
    )
    response, final_decision, fallback_used = await client.chat_completion(
        {"messages": [{"role": "user", "content": "hello"}]},
        decision,
    )
    assert response.status_code == 200
    assert fallback_used is True
    assert final_decision.model == general_model
