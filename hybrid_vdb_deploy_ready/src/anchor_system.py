from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import datetime as dt

from .config import (
    EMBEDDING_DIM,
    ANCHOR_DISTANCE_THRESHOLD,
    PREDICTION_HIT_THRESHOLD,
    WEAK_DECAY,
    MEDIUM_DECAY,
    STRONG_DECAY,
)


class AnchorType:
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    PERMANENT = "PERMANENT"


@dataclass
class Prediction:
    vector: np.ndarray
    created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)


@dataclass
class Anchor:
    id: int
    centroid: np.ndarray
    type: str = AnchorType.WEAK
    strength: float = 15.0
    hit_count: int = 0
    query_history: List[str] = field(default_factory=list)
    last_hit_time: dt.datetime = field(default_factory=dt.datetime.utcnow)
    predictions: List[Prediction] = field(default_factory=list)

    def promotion_check(self) -> None:
        if self.strength > 90:
            self.type = AnchorType.PERMANENT
        elif self.strength > 60:
            self.type = AnchorType.STRONG
        elif self.strength > 25:
            self.type = AnchorType.MEDIUM
        else:
            self.type = AnchorType.WEAK


class AnchorSystem:
    """Core 'brain' that tracks anchors and generates predictions."""

    def __init__(self):
        self.anchors: Dict[int, Anchor] = {}
        self._next_id = 0

    # --- utility -----------------------------------------------------
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype("float32")
        b = b.astype("float32")
        return 1.0 - float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    # --- main API ----------------------------------------------------
    def process_query(self, query_vec: np.ndarray, query_text: str) -> Anchor:
        """Either strengthen an existing anchor or create a new one."""
        if query_vec.ndim == 2:
            query_vec = query_vec[0]

        # find closest anchor
        best = None
        best_dist = 1e9
        for a in self.anchors.values():
            d = self._cosine_distance(a.centroid, query_vec)
            if d < best_dist:
                best = a
                best_dist = d

        if best is not None and best_dist < ANCHOR_DISTANCE_THRESHOLD:
            # strengthen existing anchor
            best.strength += 5.0
            best.hit_count += 1
            best.centroid = 0.9 * best.centroid + 0.1 * query_vec
            best.query_history.append(query_text)
            best.last_hit_time = dt.datetime.utcnow()
            best.promotion_check()
            return best

        # otherwise create new anchor
        anchor_id = self._next_id
        self._next_id += 1
        a = Anchor(id=anchor_id, centroid=query_vec, query_history=[query_text])
        self.anchors[a.id] = a
        return a

    def generate_predictions(self, anchor: Anchor, k: int = 3) -> List[Prediction]:
        """Generate K synthetic prediction vectors around the anchor centroid.

        In a production system these would come from actual semantic neighbours
        (e.g., via k‑NN on historical queries). Here we simply sample around the
        centroid with small Gaussian noise.
        """
        center = anchor.centroid
        preds = []
        for _ in range(k):
            noise = np.random.normal(0, 0.05, size=center.shape).astype("float32")
            preds.append(Prediction(vector=center + noise))
        anchor.predictions = preds
        return preds

    def check_prediction_hit(self, query_vec: np.ndarray) -> Optional[Anchor]:
        if query_vec.ndim == 2:
            query_vec = query_vec[0]

        for a in self.anchors.values():
            for p in a.predictions:
                d = self._cosine_distance(p.vector, query_vec)
                sim = 1.0 - d
                if sim >= PREDICTION_HIT_THRESHOLD:
                    # reward the anchor
                    a.strength += 10.0
                    a.hit_count += 1
                    a.last_hit_time = dt.datetime.utcnow()
                    a.promotion_check()
                    return a
        return None

    def decay(self) -> None:
        """Apply strength decay and prune very weak anchors."""
        now = dt.datetime.utcnow()
        to_delete = []
        for a in self.anchors.values():
            age_hours = (now - a.last_hit_time).total_seconds() / 3600.0
            if a.type == AnchorType.PERMANENT:
                continue
            if a.type == AnchorType.WEAK:
                a.strength *= WEAK_DECAY ** max(age_hours, 0.0)
            elif a.type == AnchorType.MEDIUM:
                a.strength *= MEDIUM_DECAY ** max(age_hours, 0.0)
            elif a.type == AnchorType.STRONG:
                a.strength *= STRONG_DECAY ** max(age_hours, 0.0)
            if a.strength < 5.0:
                to_delete.append(a.id)
        for aid in to_delete:
            del self.anchors[aid]
