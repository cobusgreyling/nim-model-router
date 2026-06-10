from __future__ import annotations

from nim_model_router.types import Registry


def estimate_request_cost(
    registry: Registry,
    *,
    task: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Estimate USD cost from token usage and the task's configured rate."""
    task_cfg = registry.tasks.get(task)
    if task_cfg is None or task_cfg.cost_per_1m_tokens is None:
        return None

    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    if total_tokens <= 0:
        return None

    return (total_tokens / 1_000_000) * task_cfg.cost_per_1m_tokens
