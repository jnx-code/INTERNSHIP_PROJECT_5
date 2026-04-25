"""
src/monitoring/tracker.py
Simple in-memory request & performance tracker.
"""
import time
from collections import defaultdict, deque
from threading import Lock


class MetricsTracker:
    """Thread-safe in-process metrics store."""

    def __init__(self, history_size: int = 1000):
        self._lock           = Lock()
        self._total_requests = 0
        self._errors         = 0
        self._latencies: deque[float] = deque(maxlen=history_size)
        self._label_counts   = defaultdict(int)
        self._start_time     = time.time()

    def record(self, label: str, latency_ms: float, error: bool = False):
        with self._lock:
            self._total_requests += 1
            self._latencies.append(latency_ms)
            if error:
                self._errors += 1
            else:
                self._label_counts[label] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
            uptime_s = time.time() - self._start_time
            return {
                "uptime_seconds":    round(uptime_s, 1),
                "total_requests":    self._total_requests,
                "error_count":       self._errors,
                "error_rate_pct":    round(
                    100 * self._errors / max(self._total_requests, 1), 2
                ),
                "avg_latency_ms":    round(sum(lats) / len(lats), 1) if lats else 0,
                "p95_latency_ms":    round(sorted(lats)[int(len(lats) * 0.95)], 1) if lats else 0,
                "label_distribution": dict(self._label_counts),
            }


# Singleton used by the API
tracker = MetricsTracker()
