import pytest
from fastapi.testclient import TestClient

from nim_model_router.config import Settings
from nim_model_router.proxy import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    settings = Settings(nvidia_api_key="test-key")
    app = create_app(settings)
    return TestClient(app)


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