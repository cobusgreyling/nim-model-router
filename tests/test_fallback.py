from nim_model_router.config import load_registry
from nim_model_router.fallback import apply_latency_preference, resolve_ab_model
from nim_model_router.router import ModelRouter
from nim_model_router.types import RouteDecision, TaskType


def test_resolve_ab_model_disabled():
    registry = load_registry()
    model, note = resolve_ab_model(registry.tasks["fast"])
    assert model == registry.tasks["fast"].model
    assert note is None


def test_apply_latency_preference_switches_model():
    registry = load_registry()
    router = ModelRouter(registry)
    router.update_model_latency(registry.tasks["fast"].model, 50)
    router.update_model_latency(registry.tasks["agentic"].model, 9000)

    decision = RouteDecision(
        task=TaskType.AGENTIC,
        model=registry.tasks["agentic"].model,
        reason="test",
        fallback_models=[registry.tasks["fast"].model],
    )
    tuned = apply_latency_preference(
        decision,
        registry=registry,
        model_latencies=router._model_latencies,
    )
    assert tuned.model == registry.tasks["fast"].model
