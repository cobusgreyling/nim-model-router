from __future__ import annotations

import random

from nim_model_router.types import Registry, RouteDecision, TaskConfig, TaskType


def resolve_ab_model(task_cfg: TaskConfig) -> tuple[str, str | None]:
    ab = task_cfg.ab_test
    if not ab.enabled or not ab.variants:
        return task_cfg.model, None

    total = sum(max(0, variant.weight) for variant in ab.variants)
    if total <= 0:
        return task_cfg.model, None

    pick = random.randint(1, total)
    cumulative = 0
    for variant in ab.variants:
        cumulative += max(0, variant.weight)
        if pick <= cumulative:
            return variant.model, f"A/B variant ({variant.weight}/{total})"
    return task_cfg.model, None


def build_fallback_models(task: TaskType, registry: Registry) -> list[str]:
    task_cfg = registry.tasks.get(task.value)
    if not task_cfg:
        return []

    models: list[str] = []
    seen: set[str] = set()
    primary, _ = resolve_ab_model(task_cfg)
    seen.add(primary)

    for fallback_task in task_cfg.fallbacks:
        fallback_cfg = registry.tasks.get(fallback_task)
        if not fallback_cfg:
            continue
        fallback_model, _ = resolve_ab_model(fallback_cfg)
        if fallback_model not in seen:
            models.append(fallback_model)
            seen.add(fallback_model)
    return models


def apply_latency_preference(
    decision: RouteDecision,
    *,
    registry: Registry,
    model_latencies: dict[str, float],
) -> RouteDecision:
    if not registry.latency_routing or not model_latencies:
        return decision

    task_cfg = registry.tasks.get(decision.task.value)
    if not task_cfg:
        return decision

    candidates = [decision.model, *decision.fallback_models]
    available = [model for model in candidates if model in model_latencies]
    if len(available) < 2:
        return decision

    fastest = min(available, key=lambda model: model_latencies[model])
    if fastest == decision.model:
        return decision

    remaining = [model for model in candidates if model != fastest]
    return decision.model_copy(
        update={
            "model": fastest,
            "fallback_models": remaining,
            "reason": f"{decision.reason}; latency preference → {fastest}",
        }
    )
