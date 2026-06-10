from pathlib import Path

from nim_model_router.logging_store import RouteLogStore, Timer
from nim_model_router.types import RouteLogEntry


def test_route_log_store_summary(tmp_path: Path):
    store = RouteLogStore(tmp_path / "router.log.jsonl")
    store.append(
        RouteLogEntry(
            task="fast",
            model="meta/llama",
            reason="short",
            confidence=0.9,
            latency_ms=12.5,
            prompt_chars=2,
            has_tools=False,
            streamed=False,
            status_code=200,
            upstream_latency_ms=10.0,
            prompt_tokens=1,
            completion_tokens=1,
        )
    )
    summary = store.summary()
    assert summary["total_requests"] == 1
    assert summary["by_task"]["fast"] == 1
    assert (tmp_path / "router.log.jsonl").exists()


def test_timer_tracks_elapsed():
    timer = Timer()
    timer.stop()
    assert timer.elapsed_ms >= 0