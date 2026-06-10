from __future__ import annotations

import threading
import time
from collections import Counter
from pathlib import Path

from nim_model_router.types import RouteLogEntry


class RouteLogStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: list[RouteLogEntry] = []

    def append(self, entry: RouteLogEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(entry.model_dump_json() + "\n")

    def summary(self) -> dict[str, object]:
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return {
                "total_requests": 0,
                "by_task": {},
                "by_model": {},
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0,
                "fallback_rate": 0.0,
            }

        by_task = Counter(entry.task for entry in entries)
        by_model = Counter(entry.model for entry in entries)
        avg_latency = sum(entry.latency_ms for entry in entries) / len(entries)
        avg_confidence = sum(entry.confidence for entry in entries) / len(entries)
        avg_upstream = [
            entry.upstream_latency_ms for entry in entries if entry.upstream_latency_ms is not None
        ]
        fallback_count = sum(1 for entry in entries if entry.fallback_used)
        model_latency: dict[str, float] = {}
        model_counts: Counter[str] = Counter()
        for entry in entries:
            if entry.upstream_latency_ms is None:
                continue
            model_counts[entry.model] += 1
            previous = model_latency.get(entry.model, 0.0)
            model_latency[entry.model] = previous + entry.upstream_latency_ms
        for model, total in model_counts.items():
            model_latency[model] = round(total / model_counts[model], 2)

        return {
            "total_requests": len(entries),
            "by_task": dict(by_task),
            "by_model": dict(by_model),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_confidence": round(avg_confidence, 3),
            "avg_upstream_latency_ms": round(sum(avg_upstream) / len(avg_upstream), 2)
            if avg_upstream
            else None,
            "fallback_rate": round(fallback_count / len(entries), 4),
            "model_latency_ms": model_latency,
            "recent": [entry.model_dump() for entry in entries[-10:]],
        }


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return self.elapsed_ms
