from app.cache import ResponseCache
from app.monitoring import MetricsCollector


def test_cache_metrics_are_linked():
    metrics = MetricsCollector()
    cache = ResponseCache(ttl_seconds=60, metrics=metrics)

    assert cache.get("hello world") is None
    cache.set("hello world", "answer")
    assert cache.get("hello world") == "answer"

    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1
    assert metrics.summary["cache_hit_rate"] == "50.00%"
