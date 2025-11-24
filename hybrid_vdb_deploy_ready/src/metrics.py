from __future__ import annotations
from dataclasses import dataclass, asdict
import time
from typing import Dict


@dataclass
class MetricsSnapshot:
    total_queries: int = 0
    local_hits: int = 0
    cloud_hits: int = 0
    cumulative_latency_ms: float = 0.0

    prediction_hits: int = 0
    prediction_misses: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["avg_latency_ms"] = (
            self.cumulative_latency_ms / self.total_queries if self.total_queries else 0.0
        )
        d["local_hit_rate"] = (
            self.local_hits / self.total_queries if self.total_queries else 0.0
        )
        d["prediction_accuracy"] = (
            self.prediction_hits / (self.prediction_hits + self.prediction_misses)
            if (self.prediction_hits + self.prediction_misses)
            else 0.0
        )
        return d


class Metrics:
    def __init__(self):
        self.current = MetricsSnapshot()

    def record_query(self, latency_ms: float, source: str) -> None:
        self.current.total_queries += 1
        self.current.cumulative_latency_ms += latency_ms
        if source == "local":
            self.current.local_hits += 1
        elif source == "cloud":
            self.current.cloud_hits += 1

    def record_prediction(self, hit: bool) -> None:
        if hit:
            self.current.prediction_hits += 1
        else:
            self.current.prediction_misses += 1

    def snapshot(self) -> Dict:
        return self.current.to_dict()
