from __future__ import annotations

from typing import Any

from nim_model_router.classifier import classify_from_payload
from nim_model_router.types import Registry, RouteDecision, TaskType


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

    def route_chat(
        self,
        payload: dict[str, Any],
        *,
        task_header: str | None = None,
        force_task: TaskType | None = None,
    ) -> RouteDecision:
        requested_model = payload.get("model")
        alias_task = resolve_alias(requested_model, self.registry)

        if force_task and force_task is not TaskType.AUTO:
            task = force_task
            reason = f"explicit task override: {task.value}"
        elif task_header:
            header_task = resolve_task_name(task_header)
            if header_task and header_task is not TaskType.AUTO:
                task = header_task
                reason = f"X-NIM-Task header: {task.value}"
            else:
                task, reason = classify_from_payload(payload, self.registry.classifier)
        elif alias_task and alias_task is not TaskType.AUTO:
            task = alias_task
            reason = f"model alias '{requested_model}'"
        elif alias_task is TaskType.AUTO or requested_model in (None, "", "nim-router/auto"):
            task, reason = classify_from_payload(payload, self.registry.classifier)
        elif requested_model and "/" in requested_model:
            # Passthrough: caller asked for a concrete NIM model ID.
            return RouteDecision(
                task=TaskType.AUTO,
                model=requested_model,
                reason="passthrough to requested NIM model",
                alias=requested_model,
            )
        else:
            task, reason = classify_from_payload(payload, self.registry.classifier)

        task_cfg = self.registry.tasks.get(task.value)
        if not task_cfg:
            fallback = self.registry.tasks[TaskType.FAST.value]
            return RouteDecision(
                task=TaskType.FAST,
                model=fallback.model,
                reason=f"unknown task '{task.value}'; fell back to fast",
                extra_body=fallback.extra_body,
                alias=requested_model,
            )

        return RouteDecision(
            task=task,
            model=task_cfg.model,
            reason=reason,
            extra_body=dict(task_cfg.extra_body),
            alias=requested_model,
        )

    def route_embedding(
        self, payload: dict[str, Any], *, task_header: str | None = None
    ) -> RouteDecision:
        if task_header:
            task = resolve_task_name(task_header)
            if task == TaskType.RERANK:
                cfg = self.registry.tasks[TaskType.RERANK.value]
                return RouteDecision(
                    task=TaskType.RERANK,
                    model=cfg.model,
                    reason="X-NIM-Task header: rerank",
                    extra_body=cfg.extra_body,
                )

        requested_model = payload.get("model")
        alias_task = resolve_alias(requested_model, self.registry)
        if alias_task == TaskType.RERANK:
            cfg = self.registry.tasks[TaskType.RERANK.value]
            return RouteDecision(
                task=TaskType.RERANK,
                model=cfg.model,
                reason=f"model alias '{requested_model}'",
                extra_body=cfg.extra_body,
                alias=requested_model,
            )
        if requested_model and requested_model not in self.registry.aliases:
            return RouteDecision(
                task=TaskType.EMBEDDING,
                model=requested_model,
                reason="passthrough to requested embedding model",
                alias=requested_model,
            )

        cfg = self.registry.tasks[TaskType.EMBEDDING.value]
        return RouteDecision(
            task=TaskType.EMBEDDING,
            model=cfg.model,
            reason="default embedding route",
            extra_body=cfg.extra_body,
            alias=requested_model,
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
