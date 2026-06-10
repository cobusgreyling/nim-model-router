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
    GENERAL = "general"


class ABVariant(BaseModel):
    model: str
    weight: int = 50


class ABTestConfig(BaseModel):
    enabled: bool = False
    variants: list[ABVariant] = Field(default_factory=list)


class TaskConfig(BaseModel):
    model: str
    description: str = ""
    priority: int = 1
    cost_per_1m_tokens: float | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    fallbacks: list[str] = Field(default_factory=list)
    ab_test: ABTestConfig = Field(default_factory=ABTestConfig)
    endpoint: str | None = None


class RoutePolicies(BaseModel):
    max_prompt_tokens: dict[str, int] = Field(default_factory=dict)
    block_ultra_for_short_prompts: bool = True
    short_prompt_max_chars: int = 200
    prefer_fast_when_uncertain: bool = True
    uncertain_confidence_threshold: float = 0.55


class RouteDecision(BaseModel):
    task: TaskType
    model: str
    reason: str
    confidence: float = 1.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    alias: str | None = None
    fallback_models: list[str] = Field(default_factory=list)
    endpoint_path: str | None = None


class ClassificationResult(BaseModel):
    task: TaskType
    reason: str
    confidence: float = 1.0


class ClassifierConfig(BaseModel):
    long_context_token_threshold: int = 12000
    reasoning_keywords: list[str] = Field(default_factory=list)
    coding_keywords: list[str] = Field(default_factory=list)
    fast_max_chars: int = 120
    rerank_keywords: list[str] = Field(default_factory=list)
    use_llm_classifier: bool = False
    llm_classifier_model: str = "meta/llama-3.1-8b-instruct"
    plugin_classifier: str | None = None


class Registry(BaseModel):
    tasks: dict[str, TaskConfig]
    aliases: dict[str, str]
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    policies: RoutePolicies = Field(default_factory=RoutePolicies)
    latency_routing: bool = True


class RouteLogEntry(BaseModel):
    task: str
    model: str
    reason: str
    confidence: float = 1.0
    latency_ms: float
    prompt_chars: int
    has_tools: bool
    streamed: bool
    status_code: int
    upstream_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    fallback_used: bool = False
