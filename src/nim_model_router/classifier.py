from __future__ import annotations

import json
import re
from typing import Any

from nim_model_router.types import ClassifierConfig, TaskType


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token for English prose.
    return max(1, len(text) // 4)


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
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
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


def classify_request(
    *,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    input_text: str | None = None,
    config: ClassifierConfig,
) -> tuple[TaskType, str]:
    """Return (task, human-readable reason)."""
    if tools:
        return TaskType.AGENTIC, "request includes tool definitions"

    text = input_text if input_text is not None else _flatten_messages(messages)
    user_text = _last_user_text(messages) if messages else (input_text or "")
    probe = user_text or text
    lowered = probe.lower()

    token_estimate = _estimate_tokens(text)
    if token_estimate >= config.long_context_token_threshold:
        return (
            TaskType.LONG_CONTEXT,
            f"estimated prompt size {token_estimate} tokens exceeds "
            f"{config.long_context_token_threshold}",
        )

    for keyword in config.reasoning_keywords:
        if keyword.lower() in lowered:
            return TaskType.REASONING, f"matched reasoning keyword '{keyword}'"

    for keyword in config.coding_keywords:
        if keyword.lower() in lowered:
            return TaskType.CODING, f"matched coding keyword '{keyword}'"

    if len(probe.strip()) <= config.fast_max_chars and "\n" not in probe.strip():
        return TaskType.FAST, f"short prompt ({len(probe.strip())} chars)"

    if re.search(r"\b(agent|workflow|tool|function call|multi-step)\b", lowered):
        return TaskType.AGENTIC, "agentic phrasing detected"

    return TaskType.AGENTIC, "default to agentic for general tasks"


def classify_from_payload(
    payload: dict[str, Any], config: ClassifierConfig
) -> tuple[TaskType, str]:
    if "messages" in payload:
        return classify_request(
            messages=payload.get("messages"),
            tools=payload.get("tools"),
            config=config,
        )
    if "input" in payload:
        raw_input = payload["input"]
        if isinstance(raw_input, str):
            text = raw_input
        elif isinstance(raw_input, list):
            text = "\n".join(str(item) for item in raw_input)
        else:
            text = json.dumps(raw_input)
        return classify_request(input_text=text, config=config)
    return TaskType.FAST, "empty payload; using fast fallback"
