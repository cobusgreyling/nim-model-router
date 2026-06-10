from __future__ import annotations

from typing import Any

from nim_model_router.classifier import classify_from_payload
from nim_model_router.fallback import (
    apply_latency_preference,
    build_fallback_models,
    resolve_ab_model,
)
from nim_model_router.types import ClassificationResult, Registry, RouteDecision, TaskType


def resolve_task_name(raw: str | None) -> TaskType | None:
    if not raw:
        return None
    normalized = raw.strip().lower().replace("-", "_")
    try:
        return TaskType(normalized)
    except ValueError:
        return None


def resolve_alias(model: str | None, registry: Registry) -> TaskType | None:
    if not model:
        return None
    alias_target = registry.aliases.get(model)
    if alias_target:
        return resolve_task_name(alias_target)
    if model.startswith("nim-router/"):
        suffix = model.removeprefix("nim-router/")
        return resolve_task_name(suffix)
    return None


class ModelRouter:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._model_latencies: dict[str, float] = {}

    def update_model_latency(self, model: str, latency_ms: float) -> None:
        previous = self._model_latencies.get(model)
        if previous is None:
            self._model_latencies[model] = latency_ms
        else:
            self._model_latencies[model] = (previous * 0.8) + (latency_ms * 0.2)

    def _decision_from_task(
        self,
        task: TaskType,
        *,
        reason: str,
        confidence: float,
        requested_model: str | None,
        ab_note: str | None = None,
        allow_latency_tuning: bool = True,
    ) -> RouteDecision:
        task_cfg = self.registry.tasks.get(task.value)
        if not task_cfg:
            fallback = self.registry.tasks[TaskType.FAST.value]
            return RouteDecision(
                task=TaskType.FAST,
                model=fallback.model,
                reason=f"unknown task '{task.value}'; fell back to fast",
                confidence=confidence,
                extra_body=fallback.extra_body,
                alias=requested_model,
                fallback_models=build_fallback_models(TaskType.FAST, self.registry),
                endpoint_path=fallback.endpoint,
            )

        model, ab_reason = resolve_ab_model(task_cfg)
        full_reason = reason if not ab_reason else f"{reason}; {ab_reason}"
        if ab_note:
            full_reason = f"{full_reason}; {ab_note}"

        decision = RouteDecision(
            task=task,
            model=model,
            reason=full_reason,
            confidence=confidence,
            extra_body=dict(task_cfg.extra_body),
            alias=requested_model,
            fallback_models=build_fallback_models(task, self.registry),
            endpoint_path=task_cfg.endpoint,
        )
        if allow_latency_tuning:
            return apply_latency_preference(
                decision,
                registry=self.registry,
                model_latencies=self._model_latencies,
            )
        return decision

    def route_chat(
        self,
        payload: dict[str, Any],
        *,
        task_header: str | None = None,
        force_task: TaskType | None = None,
        classification: ClassificationResult | None = None,
    ) -> RouteDecision:
        requested_model = payload.get("model")
        alias_task = resolve_alias(requested_model, self.registry)

        if force_task and force_task is not TaskType.AUTO:
            return self._decision_from_task(
                force_task,
                reason=f"explicit task override: {force_task.value}",
                confidence=1.0,
                requested_model=requested_model,
                allow_latency_tuning=False,
            )

        if task_header:
            header_task = resolve_task_name(task_header)
            if header_task and header_task is not TaskType.AUTO:
                return self._decision_from_task(
                    header_task,
                    reason=f"X-NIM-Task header: {header_task.value}",
                    confidence=1.0,
                    requested_model=requested_model,
                    allow_latency_tuning=False,
                )

        if alias_task and alias_task is not TaskType.AUTO:
            return self._decision_from_task(
                alias_task,
                reason=f"model alias '{requested_model}'",
                confidence=1.0,
                requested_model=requested_model,
                allow_latency_tuning=False,
            )

        if alias_task is TaskType.AUTO or requested_model in (None, "", "nim-router/auto"):
            result = classification or classify_from_payload(payload, self.registry)
            return self._decision_from_task(
                result.task,
                reason=result.reason,
                confidence=result.confidence,
                requested_model=requested_model,
            )

        if requested_model and "/" in requested_model:
            return RouteDecision(
                task=TaskType.AUTO,
                model=requested_model,
                reason="passthrough to requested NIM model",
                confidence=1.0,
                alias=requested_model,
            )

        result = classification or classify_from_payload(payload, self.registry)
        return self._decision_from_task(
            result.task,
            reason=result.reason,
            confidence=result.confidence,
            requested_model=requested_model,
        )

    def route_embedding(
        self,
        payload: dict[str, Any],
        *,
        task_header: str | None = None,
        classification: ClassificationResult | None = None,
    ) -> RouteDecision:
        if task_header:
            task = resolve_task_name(task_header)
            if task == TaskType.RERANK:
                return self._decision_from_task(
                    TaskType.RERANK,
                    reason="X-NIM-Task header: rerank",
                    confidence=1.0,
                    requested_model=payload.get("model"),
                )

        requested_model = payload.get("model")
        alias_task = resolve_alias(requested_model, self.registry)
        if alias_task == TaskType.RERANK:
            return self._decision_from_task(
                TaskType.RERANK,
                reason=f"model alias '{requested_model}'",
                confidence=1.0,
                requested_model=requested_model,
            )

        if alias_task == TaskType.EMBEDDING:
            return self._decision_from_task(
                TaskType.EMBEDDING,
                reason=f"model alias '{requested_model}'",
                confidence=1.0,
                requested_model=requested_model,
            )

        if requested_model and requested_model not in self.registry.aliases:
            return RouteDecision(
                task=TaskType.EMBEDDING,
                model=requested_model,
                reason="passthrough to requested embedding model",
                confidence=1.0,
                alias=requested_model,
            )

        if requested_model in (None, "", "nim-router/auto"):
            result = classification or classify_from_payload(payload, self.registry)
            if result.task == TaskType.RERANK:
                return self._decision_from_task(
                    TaskType.RERANK,
                    reason=result.reason,
                    confidence=result.confidence,
                    requested_model=requested_model,
                )

        return self._decision_from_task(
            TaskType.EMBEDDING,
            reason="default embedding route",
            confidence=1.0,
            requested_model=requested_model,
        )

    def route_rerank(
        self,
        payload: dict[str, Any],
        *,
        task_header: str | None = None,
        classification: ClassificationResult | None = None,
    ) -> RouteDecision:
        if task_header:
            task = resolve_task_name(task_header)
            if task and task is not TaskType.AUTO:
                return self._decision_from_task(
                    task,
                    reason=f"X-NIM-Task header: {task.value}",
                    confidence=1.0,
                    requested_model=payload.get("model"),
                )

        requested_model = payload.get("model")
        alias_task = resolve_alias(requested_model, self.registry)
        if alias_task and alias_task is not TaskType.AUTO:
            return self._decision_from_task(
                alias_task,
                reason=f"model alias '{requested_model}'",
                confidence=1.0,
                requested_model=requested_model,
            )

        if requested_model and requested_model not in self.registry.aliases:
            return RouteDecision(
                task=TaskType.RERANK,
                model=requested_model,
                reason="passthrough to requested rerank model",
                confidence=1.0,
                alias=requested_model,
                endpoint_path="/ranking",
            )

        if requested_model in (None, "", "nim-router/auto"):
            result = classification or classify_from_payload(payload, self.registry)
            task = result.task if result.task == TaskType.RERANK else TaskType.RERANK
            return self._decision_from_task(
                task,
                reason=result.reason if task == TaskType.RERANK else "default rerank route",
                confidence=result.confidence,
                requested_model=requested_model,
            )

        return self._decision_from_task(
            TaskType.RERANK,
            reason="default rerank route",
            confidence=1.0,
            requested_model=requested_model,
        )

    def list_router_models(self) -> list[dict[str, str]]:
        models: list[dict[str, str]] = []
        for alias, task_name in sorted(self.registry.aliases.items()):
            task_cfg = self.registry.tasks.get(task_name)
            models.append(
                {
                    "id": alias,
                    "object": "model",
                    "owned_by": "nim-model-router",
                    "task": task_name,
                    "resolved_model": task_cfg.model if task_cfg else "",
                    "description": task_cfg.description if task_cfg else "",
                }
            )
        for task_name, task_cfg in sorted(self.registry.tasks.items()):
            models.append(
                {
                    "id": task_cfg.model,
                    "object": "model",
                    "owned_by": "nvidia-nim",
                    "task": task_name,
                    "resolved_model": task_cfg.model,
                    "description": task_cfg.description,
                }
            )
        return models

    def cost_summary(self) -> dict[str, float | None]:
        return {task_name: cfg.cost_per_1m_tokens for task_name, cfg in self.registry.tasks.items()}
