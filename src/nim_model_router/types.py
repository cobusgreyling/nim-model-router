from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    AUTO = "auto"
    FAST = "fast"
    AGENTIC = "agentic"
    REASONING = "reasoning"
    LONG_CONTEXT = "long_context"
    CODING = "coding"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class TaskConfig(BaseModel):
    model: str
    description: str = ""
    priority: int = 1
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    task: TaskType
    model: str
    reason: str
    extra_body: dict[str, Any] = Field(default_factory=dict)
    alias: str | None = None


class ClassifierConfig(BaseModel):
    long_context_token_threshold: int = 12000
    reasoning_keywords: list[str] = Field(default_factory=list)
    coding_keywords: list[str] = Field(default_factory=list)
    fast_max_chars: int = 120


class Registry(BaseModel):
    tasks: dict[str, TaskConfig]
    aliases: dict[str, str]
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)


class RouteLogEntry(BaseModel):
    task: str
    model: str
    reason: str
    latency_ms: float
    prompt_chars: int
    has_tools: bool
    streamed: bool
    status_code: int
    upstream_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None