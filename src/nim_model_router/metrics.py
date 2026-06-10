from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS_TOTAL = Counter(
    "nim_router_requests_total",
    "Total proxied requests",
    ["endpoint", "task", "model", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "nim_router_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint", "task", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
UPSTREAM_LATENCY = Histogram(
    "nim_router_upstream_latency_seconds",
    "Upstream NIM latency",
    ["endpoint", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
FALLBACKS_TOTAL = Counter(
    "nim_router_fallbacks_total",
    "Upstream fallbacks attempted",
    ["endpoint", "model"],
)


def record_request(
    *,
    endpoint: str,
    task: str,
    model: str,
    status_code: int,
    latency_seconds: float,
    upstream_latency_seconds: float | None = None,
    fallback_used: bool = False,
) -> None:
    REQUESTS_TOTAL.labels(
        endpoint=endpoint,
        task=task,
        model=model,
        status_code=str(status_code),
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint, task=task, model=model).observe(latency_seconds)
    if upstream_latency_seconds is not None:
        UPSTREAM_LATENCY.labels(endpoint=endpoint, model=model).observe(upstream_latency_seconds)
    if fallback_used:
        FALLBACKS_TOTAL.labels(endpoint=endpoint, model=model).inc()


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
