from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .local_vdb import LocalVDB
from .config import HOT_PARTITION_CAPACITY


class StorageEngine:
    """Two‑tier storage over a LocalVDB.

    - Hot partition (small list kept in RAM with linear search)
    - Backing indices (permanent + dynamic) via LocalVDB
    """

    def __init__(self):
        self.local_vdb = LocalVDB()
        self.hot_vectors = np.empty((0, self.local_vdb.permanent.dim), dtype="float32")
        self.hot_ids: List[str] = []
        self.hot_capacity = HOT_PARTITION_CAPACITY

    def add_hot(self, vecs: np.ndarray, ids: List[str]) -> None:
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        self.hot_vectors = np.vstack([self.hot_vectors, vecs])
        self.hot_ids.extend(ids)
        # simple FIFO eviction
        if self.hot_vectors.shape[0] > self.hot_capacity:
            overflow = self.hot_vectors.shape[0] - self.hot_capacity
            self.hot_vectors = self.hot_vectors[overflow:, :]
            self.hot_ids = self.hot_ids[overflow:]

    def add_permanent(self, vecs: np.ndarray, ids: List[str]) -> None:
        self.local_vdb.add_permanent(vecs, ids)

    def add_dynamic(self, vecs: np.ndarray, ids: List[str]) -> None:
        self.local_vdb.add_dynamic(vecs, ids)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        # 1. hot partition
        if self.hot_vectors.shape[0] > 0:
            q = query.astype("float32")
            if q.ndim == 1:
                q = q[None, :]
            v = self.hot_vectors
            v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            scores = (v_norm @ q_norm.T)[:, 0]
            idx = np.argsort(-scores)[:k]
            hot_res = [(self.hot_ids[i], float(scores[i])) for i in idx]
        else:
            hot_res = []

        # 2. backing indices
        ids, scores = self.local_vdb.search(query, k)
        backing = list(zip(ids, scores))

        combined = hot_res + backing
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:k]
        if not combined:
            return [], []
        ids, scores = zip(*combined)
        return list(ids), list(scores)

    def compact(self) -> None:
        """Placeholder for disk compaction; no‑op in this in‑memory prototype."""
        return
