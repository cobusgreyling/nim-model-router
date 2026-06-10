from nim_model_router.metrics import record_request


def test_record_request_increments():
    record_request(
        endpoint="chat",
        task="fast",
        model="meta/llama",
        status_code=200,
        latency_seconds=0.1,
        upstream_latency_seconds=0.05,
        fallback_used=True,
    )
