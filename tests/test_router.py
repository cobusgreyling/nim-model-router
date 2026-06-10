from nim_model_router.config import load_registry
from nim_model_router.router import ModelRouter
from nim_model_router.types import TaskType


def test_alias_auto_classifies():
    registry = load_registry()
    router = ModelRouter(registry)
    decision = router.route_chat(
        {
            "model": "nim-router/auto",
            "messages": [{"role": "user", "content": "Hi"}],
        }
    )
    assert decision.model == registry.tasks["fast"].model
    assert decision.task == TaskType.FAST


def test_alias_agentic():
    registry = load_registry()
    router = ModelRouter(registry)
    decision = router.route_chat(
        {
            "model": "nim-router/agentic",
            "messages": [{"role": "user", "content": "anything"}],
        }
    )
    assert decision.model == registry.tasks["agentic"].model
    assert decision.extra_body.get("enable_thinking") is True


def test_passthrough_concrete_model():
    registry = load_registry()
    router = ModelRouter(registry)
    decision = router.route_chat(
        {
            "model": "meta/llama-3.1-70b-instruct",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert decision.model == "meta/llama-3.1-70b-instruct"
    assert decision.reason == "passthrough to requested NIM model"


def test_task_header_override():
    registry = load_registry()
    router = ModelRouter(registry)
    decision = router.route_chat(
        {
            "model": "nim-router/auto",
            "messages": [{"role": "user", "content": "Hi"}],
        },
        task_header="reasoning",
    )
    assert decision.task == TaskType.REASONING
    assert decision.model == registry.tasks["reasoning"].model


def test_embedding_default_route():
    registry = load_registry()
    router = ModelRouter(registry)
    decision = router.route_embedding({"input": "hello world"})
    assert decision.task == TaskType.EMBEDDING
    assert decision.model == registry.tasks["embedding"].model
