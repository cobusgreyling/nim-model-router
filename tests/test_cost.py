import pytest

from nim_model_router.config import load_registry
from nim_model_router.cost import estimate_request_cost


def test_estimate_request_cost():
    registry = load_registry()
    cost = estimate_request_cost(
        registry,
        task="fast",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost == pytest.approx(0.000075)


def test_estimate_request_cost_missing_usage():
    registry = load_registry()
    cost = estimate_request_cost(
        registry,
        task="fast",
        prompt_tokens=None,
        completion_tokens=None,
    )
    assert cost is None