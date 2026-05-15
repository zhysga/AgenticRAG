"""Prometheus metrics helpers for backend (optional)."""
from __future__ import annotations

from contextlib import contextmanager
import time

try:
    from prometheus_client import Counter, Histogram  # type: ignore
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    _PROM_AVAILABLE = False

    class _Noop:
        def __call__(self, *args, **kwargs):
            return self
        def observe(self, *args, **kwargs):
            return None

    def Counter(name: str, doc: str, *args, **kwargs):  # type: ignore
        return _Noop()

    def Histogram(name: str, doc: str, *args, **kwargs):  # type: ignore
        return _Noop()

# Counters
CACHE_HIT = Counter(
    "nodes_cache_hit_total",
    "Total node cache hits when initializing EnhancedRAGEngine",
)

# Histograms
RETRIEVE_LATENCY = Histogram(
    "hybrid_retrieve_latency_seconds",
    "Latency for hybrid retrieval path",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)


@contextmanager
def time_retrieve():
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        try:
            RETRIEVE_LATENCY.observe(dt)
        except Exception:
            pass
