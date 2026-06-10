from nim_model_router.config import load_registry
from nim_model_router.policies import apply_policies, validate_registry_policies
from nim_model_router.types import ClassificationResult, TaskType


def test_block_ultra_for_short_prompts():
    registry = load_registry()
    result = apply_policies(
        ClassificationResult(task=TaskType.REASONING, reason="kw", confidence=0.9),
        prompt_chars=50,
        token_estimate=20,
        policies=registry.policies,
    )
    assert result.task == TaskType.GENERAL


def test_validate_registry_policies_ok():
    registry = load_registry()
    warnings = validate_registry_policies(registry)
    assert warnings == []
