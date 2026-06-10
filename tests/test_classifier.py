from nim_model_router.classifier import classify_request
from nim_model_router.config import load_registry
from nim_model_router.types import TaskType


def test_tools_route_to_agentic():
    registry = load_registry()
    task, reason = classify_request(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function"}],
        config=registry.classifier,
    )
    assert task == TaskType.AGENTIC
    assert "tool" in reason


def test_short_prompt_routes_fast():
    registry = load_registry()
    task, _ = classify_request(
        messages=[{"role": "user", "content": "Hi"}],
        config=registry.classifier,
    )
    assert task == TaskType.FAST


def test_coding_keyword_routes_coding():
    registry = load_registry()
    task, reason = classify_request(
        messages=[{"role": "user", "content": "Please refactor this Python function"}],
        config=registry.classifier,
    )
    assert task == TaskType.CODING
    assert "python" in reason.lower() or "refactor" in reason.lower()


def test_long_context_threshold():
    registry = load_registry()
    long_text = "word " * 50000
    task, reason = classify_request(
        messages=[{"role": "user", "content": long_text}],
        config=registry.classifier,
    )
    assert task == TaskType.LONG_CONTEXT
    assert "tokens" in reason
