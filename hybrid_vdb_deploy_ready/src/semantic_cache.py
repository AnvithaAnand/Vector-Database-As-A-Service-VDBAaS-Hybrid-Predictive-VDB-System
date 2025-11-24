from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import datetime as dt

from .config import EMBEDDING_DIM


@dataclass
class SemanticCluster:
    centroid: np.ndarray
    momentum: float = 0.0
    vector_ids: List[str] = field(default_factory=list)
    last_activity: dt.datetime = field(default_factory=dt.datetime.utcnow)


class SemanticCache:
    """Tracks which semantic regions are currently 'hot'.

    This is a very light‑weight online clustering mechanism with momentum.
    """

    def __init__(self, distance_threshold: float = 0.3):
        self.distance_threshold = distance_threshold
        self.clusters: List[SemanticCluster] = []

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype("float32")
        b = b.astype("float32")
        return 1.0 - float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    def update_with_vector(self, vec: np.ndarray, vec_id: str) -> None:
        if vec.ndim == 2:
            vec = vec[0]
        if not self.clusters:
            self.clusters.append(SemanticCluster(centroid=vec, momentum=1.0, vector_ids=[vec_id]))
            return

        best = None
        best_dist = 1e9
        for c in self.clusters:
            d = self._cosine_distance(c.centroid, vec)
            if d < best_dist:
                best = c
                best_dist = d

        if best is not None and best_dist < self.distance_threshold:
            # strengthen existing cluster
            best.momentum += 1.0
            best.centroid = 0.9 * best.centroid + 0.1 * vec
            best.vector_ids.append(vec_id)
            best.last_activity = dt.datetime.utcnow()
        else:
            # create new cluster
            self.clusters.append(SemanticCluster(centroid=vec, momentum=1.0, vector_ids=[vec_id]))

    def decay(self, factor: float = 0.95) -> None:
        now = dt.datetime.utcnow()
        alive = []
        for c in self.clusters:
            minutes = (now - c.last_activity).total_seconds() / 60.0
            c.momentum *= factor ** max(minutes, 0.0)
            if c.momentum > 0.1:
                alive.append(c)
        self.clusters = alive

    def find_hot_cluster(self, vec: np.ndarray) -> Optional[SemanticCluster]:
        if not self.clusters:
            return None
        if vec.ndim == 2:
            vec = vec[0]
        best = None
        best_dist = 1e9
        for c in self.clusters:
            d = self._cosine_distance(c.centroid, vec)
            if d < best_dist:
                best_dist = d
                best = c
        return best
