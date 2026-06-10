from __future__ import annotations

from typing import Any

from nim_model_router.types import ClassificationResult, Registry, RoutePolicies, TaskType

EXPENSIVE_TASKS = {
    TaskType.AGENTIC,
    TaskType.REASONING,
    TaskType.LONG_CONTEXT,
    TaskType.CODING,
}


def apply_policies(
    result: ClassificationResult,
    *,
    prompt_chars: int,
    token_estimate: int,
    policies: RoutePolicies,
) -> ClassificationResult:
    """Adjust classification based on configured request policies."""
    if (
        policies.block_ultra_for_short_prompts
        and result.task in {TaskType.REASONING, TaskType.LONG_CONTEXT}
        and prompt_chars <= policies.short_prompt_max_chars
    ):
        return ClassificationResult(
            task=TaskType.GENERAL,
            reason=f"policy: short prompt ({prompt_chars} chars) blocked ultra task",
            confidence=0.9,
        )

    max_tokens = policies.max_prompt_tokens.get(result.task.value)
    if max_tokens is not None and token_estimate > max_tokens:
        return ClassificationResult(
            task=TaskType.LONG_CONTEXT,
            reason=(
                f"policy: {token_estimate} tokens exceeds {result.task.value} limit {max_tokens}"
            ),
            confidence=0.85,
        )

    if (
        policies.prefer_fast_when_uncertain
        and result.confidence < policies.uncertain_confidence_threshold
        and result.task in EXPENSIVE_TASKS
    ):
        return ClassificationResult(
            task=TaskType.GENERAL,
            reason="policy: low confidence; prefer cheaper general tier",
            confidence=result.confidence,
        )

    return result


def estimate_prompt_metrics(
    payload: dict[str, Any], text: str, token_estimate: int
) -> tuple[int, int]:
    prompt_chars = len(text)
    if "messages" in payload:
        return prompt_chars, token_estimate
    raw_input = payload.get("input", "")
    if isinstance(raw_input, str):
        return len(raw_input), token_estimate
    return prompt_chars, token_estimate


def validate_registry_policies(registry: Registry) -> list[str]:
    warnings: list[str] = []
    for task_name, limit in registry.policies.max_prompt_tokens.items():
        if task_name not in registry.tasks:
            warnings.append(f"policy max_prompt_tokens references unknown task '{task_name}'")
        if limit <= 0:
            warnings.append(f"policy max_prompt_tokens for '{task_name}' must be positive")
    return warnings
