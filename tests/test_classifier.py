from nim_model_router.classifier import classify_from_payload, classify_request
from nim_model_router.config import load_registry
from nim_model_router.types import ClassificationResult, TaskType


def test_tools_route_to_agentic():
    registry = load_registry()
    result = classify_request(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"type": "function"}],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task == TaskType.AGENTIC
    assert "tool" in result.reason


def test_short_prompt_routes_fast():
    registry = load_registry()
    result = classify_request(
        messages=[{"role": "user", "content": "Hi"}],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task == TaskType.FAST


def test_coding_keyword_routes_coding():
    registry = load_registry()
    result = classify_request(
        messages=[{"role": "user", "content": "Please refactor this Python function"}],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task == TaskType.CODING
    assert result.confidence >= 0.8


def test_long_context_threshold():
    registry = load_registry()
    long_text = "word " * 50000
    result = classify_request(
        messages=[{"role": "user", "content": long_text}],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task == TaskType.LONG_CONTEXT
    assert "tokens" in result.reason


def test_multimodal_image_block():
    registry = load_registry()
    result = classify_request(
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x"}}],
            }
        ],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task in {TaskType.FAST, TaskType.GENERAL}


def test_rerank_payload_detection():
    registry = load_registry()
    result = classify_from_payload(
        {
            "query": "what is nim?",
            "documents": ["NIM is great", "other"],
        },
        registry,
    )
    assert result.task == TaskType.RERANK


def test_system_prompt_coding_keyword():
    registry = load_registry()
    result = classify_request(
        messages=[
            {"role": "system", "content": "You are a Python coding assistant."},
            {"role": "user", "content": "Help me with this task"},
        ],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task == TaskType.CODING
    assert "system prompt" in result.reason


def test_ambiguous_defaults_general():
    registry = load_registry()
    result = classify_request(
        messages=[
            {
                "role": "user",
                "content": (
                    "Tell me about cloud architecture patterns for multi-region "
                    "deployments and how teams usually trade off latency versus cost."
                ),
            }
        ],
        config=registry.classifier,
        policies_registry=registry,
    )
    assert result.task == TaskType.GENERAL
    assert isinstance(result, ClassificationResult)
