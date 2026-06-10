import httpx
import pytest
import respx

from nim_model_router.catalog import fetch_nim_catalog, suggest_models_for_task


@pytest.mark.asyncio
@respx.mock
async def test_fetch_nim_catalog():
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "nvidia/coder-model"}, {"id": "meta/llama-8b"}]},
        )
    )
    catalog = await fetch_nim_catalog(api_key="k", base_url="https://integrate.api.nvidia.com/v1")
    assert len(catalog) == 2


def test_suggest_models_for_task():
    catalog = [{"id": "nvidia/llama-coder-v1"}, {"id": "meta/llama-8b"}]
    suggestions = suggest_models_for_task(catalog, keywords=["coder", "llama"])
    assert "nvidia/llama-coder-v1" in suggestions