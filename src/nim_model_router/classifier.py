from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any

import tiktoken

from nim_model_router.policies import apply_policies
from nim_model_router.types import ClassificationResult, ClassifierConfig, Registry, TaskType

TokenEstimator = Callable[[str], int]
PluginClassifier = Callable[..., ClassificationResult]


@lru_cache(maxsize=1)
def _token_encoder():
    return tiktoken.get_encoding("cl100k_base")


def _default_token_estimator(text: str) -> int:
    try:
        return max(1, len(_token_encoder().encode(text)))
    except Exception:
        return max(1, len(text) // 4)


def _parse_llm_classifier_json(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    data = json.loads(stripped)
    return data if isinstance(data, dict) else {}


def _flatten_messages(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type == "image_url":
                    parts.append("[image]")
                elif block_type == "input_audio":
                    parts.append("[audio]")
    return "\n".join(parts)


def _system_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return "\n".join(parts)


def _last_user_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(texts)
    return ""


def _keyword_score(text: str, keywords: list[str]) -> tuple[float, str | None]:
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() in lowered:
            return 0.9, keyword
    return 0.0, None


def _load_plugin_classifier(name: str) -> PluginClassifier | None:
    for entry in entry_points(group="nim_model_router.classifiers"):
        if entry.name == name:
            loaded = entry.load()
            return loaded if callable(loaded) else None
    module_path, _, attr = name.partition(":")
    if not attr:
        return None
    module = importlib.import_module(module_path)
    loaded = getattr(module, attr, None)
    return loaded if callable(loaded) else None


def classify_request(
    *,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    input_text: str | None = None,
    config: ClassifierConfig,
    policies_registry: Registry | None = None,
    token_estimator: TokenEstimator | None = None,
) -> ClassificationResult:
    estimate_tokens = token_estimator or _default_token_estimator

    if tools:
        return ClassificationResult(
            task=TaskType.AGENTIC,
            reason="request includes tool definitions",
            confidence=1.0,
        )

    text = input_text if input_text is not None else _flatten_messages(messages)
    user_text = _last_user_text(messages) if messages else (input_text or "")
    system_text = _system_text(messages) if messages else ""
    probe = user_text or text
    lowered = probe.lower()

    if system_text:
        system_score, system_kw = _keyword_score(system_text, config.coding_keywords)
        if system_kw:
            result = ClassificationResult(
                task=TaskType.CODING,
                reason=f"system prompt matched coding keyword '{system_kw}'",
                confidence=max(0.85, system_score),
            )
            return _maybe_apply_policies(
                result, policies_registry, probe, text, estimate_tokens
            )

    if config.plugin_classifier:
        plugin = _load_plugin_classifier(config.plugin_classifier)
        if plugin:
            result = plugin(
                messages=messages,
                input_text=input_text,
                text=text,
                probe=probe,
                config=config,
            )
            if isinstance(result, ClassificationResult):
                return _maybe_apply_policies(
                    result, policies_registry, probe, text, estimate_tokens
                )

    token_estimate = estimate_tokens(text)

    for keyword in config.rerank_keywords:
        if keyword.lower() in lowered:
            result = ClassificationResult(
                task=TaskType.RERANK,
                reason=f"matched rerank keyword '{keyword}'",
                confidence=0.88,
            )
            return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)

    if token_estimate >= config.long_context_token_threshold:
        result = ClassificationResult(
            task=TaskType.LONG_CONTEXT,
            reason=(
                f"estimated prompt size {token_estimate} tokens exceeds "
                f"{config.long_context_token_threshold}"
            ),
            confidence=0.92,
        )
        return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)

    reasoning_score, reasoning_kw = _keyword_score(probe, config.reasoning_keywords)
    if reasoning_kw:
        result = ClassificationResult(
            task=TaskType.REASONING,
            reason=f"matched reasoning keyword '{reasoning_kw}'",
            confidence=reasoning_score,
        )
        return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)

    coding_score, coding_kw = _keyword_score(probe, config.coding_keywords)
    if coding_kw:
        result = ClassificationResult(
            task=TaskType.CODING,
            reason=f"matched coding keyword '{coding_kw}'",
            confidence=coding_score,
        )
        return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)

    if len(probe.strip()) <= config.fast_max_chars and "\n" not in probe.strip():
        result = ClassificationResult(
            task=TaskType.FAST,
            reason=f"short prompt ({len(probe.strip())} chars)",
            confidence=0.95,
        )
        return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)

    if re.search(r"\b(agent|workflow|tool|function call|multi-step)\b", lowered):
        result = ClassificationResult(
            task=TaskType.AGENTIC,
            reason="agentic phrasing detected",
            confidence=0.75,
        )
        return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)

    result = ClassificationResult(
        task=TaskType.GENERAL,
        reason="default to general for ambiguous tasks",
        confidence=0.6,
    )
    return _maybe_apply_policies(result, policies_registry, probe, text, estimate_tokens)


def _maybe_apply_policies(
    result: ClassificationResult,
    registry: Registry | None,
    probe: str,
    text: str,
    estimate_tokens: TokenEstimator,
) -> ClassificationResult:
    if registry is None:
        return result
    adjusted = apply_policies(
        result,
        prompt_chars=len(probe.strip()),
        token_estimate=estimate_tokens(text),
        policies=registry.policies,
    )
    return adjusted


def classify_from_payload(
    payload: dict[str, Any],
    registry: Registry,
    *,
    token_estimator: TokenEstimator | None = None,
) -> ClassificationResult:
    if "messages" in payload:
        return classify_request(
            messages=payload.get("messages"),
            tools=payload.get("tools"),
            config=registry.classifier,
            policies_registry=registry,
            token_estimator=token_estimator,
        )
    if "input" in payload:
        raw_input = payload["input"]
        if isinstance(raw_input, str):
            text = raw_input
        elif isinstance(raw_input, list):
            text = "\n".join(str(item) for item in raw_input)
        else:
            text = json.dumps(raw_input)
        return classify_request(
            input_text=text,
            config=registry.classifier,
            policies_registry=registry,
            token_estimator=token_estimator,
        )
    if "query" in payload and "documents" in payload:
        return ClassificationResult(
            task=TaskType.RERANK,
            reason="rerank payload detected (query + documents)",
            confidence=1.0,
        )
    return ClassificationResult(
        task=TaskType.FAST,
        reason="empty payload; using fast fallback",
        confidence=0.5,
    )


async def classify_with_llm(
    *,
    messages: list[dict[str, Any]] | None,
    input_text: str | None,
    config: ClassifierConfig,
    api_key: str,
    base_url: str,
) -> ClassificationResult | None:
    if not config.use_llm_classifier:
        return None

    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None

    prompt = input_text or _flatten_messages(messages)
    if not prompt.strip():
        return None

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=config.llm_classifier_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user request into one task: fast, general, agentic, "
                    "reasoning, long_context, coding, embedding, rerank. "
                    'Reply with JSON: {"task": "...", "reason": "...", "confidence": 0.0}'
                ),
            },
            {"role": "user", "content": prompt[:4000]},
        ],
        temperature=0,
        max_tokens=120,
    )
    content = response.choices[0].message.content or ""
    try:
        data = _parse_llm_classifier_json(content)
    except (json.JSONDecodeError, TypeError):
        return None
    task_name = str(data.get("task", "general")).lower().replace("-", "_")
    try:
        task = TaskType(task_name)
    except ValueError:
        task = TaskType.GENERAL
    return ClassificationResult(
        task=task,
        reason=f"llm classifier: {data.get('reason', 'classified')}",
        confidence=float(data.get("confidence", 0.8)),
    )
