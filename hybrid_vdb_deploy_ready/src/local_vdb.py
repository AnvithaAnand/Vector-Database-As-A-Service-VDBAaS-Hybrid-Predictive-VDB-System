from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .config import EMBEDDING_DIM

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    _HAS_FAISS = False


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm


class SimpleIndex:
    """Small wrapper around FAISS or a NumPy brute‑force index.

    This is intentionally minimal – just enough to show the idea.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype="float32")
        self.ids: List[str] = []

        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None

    def add(self, vecs: np.ndarray, ids: List[str]) -> None:
        assert vecs.shape[1] == self.dim
        vecs = vecs.astype("float32")
        if _HAS_FAISS:
            faiss.normalize_L2(vecs)
            self.index.add(vecs)
        self.vectors = np.vstack([self.vectors, vecs])
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        if self.vectors.shape[0] == 0:
            return [], []
        query = query.astype("float32")
        if query.ndim == 1:
            query = query[None, :]
        if _HAS_FAISS:
            faiss.normalize_L2(query)
            scores, idx = self.index.search(query, k)
            idx = idx[0]
            scores = scores[0]
        else:
            # cosine similarity via NumPy
            vnorm = _normalize(self.vectors)
            qnorm = _normalize(query)[0]
            scores = (vnorm @ qnorm).astype("float32")
            idx = np.argsort(-scores)[:k]
            scores = scores[idx]
        ids = [self.ids[i] for i in idx if i < len(self.ids)]
        return ids, scores.tolist()


class LocalVDB:
    """Two‑tier local vector store: permanent + dynamic.

    This class does **not** do any persistence – it is purely in‑memory and
    intended for demonstration.
    """

    def __init__(self):
        self.permanent = SimpleIndex()
        self.dynamic = SimpleIndex()

    def add_permanent(self, vecs: np.ndarray, ids: List[str]) -> None:
        self.permanent.add(vecs, ids)

    def add_dynamic(self, vecs: np.ndarray, ids: List[str]) -> None:
        self.dynamic.add(vecs, ids)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[List[str], List[float]]:
        """Search permanent then dynamic and merge results by score."""
        p_ids, p_scores = self.permanent.search(query, k)
        d_ids, d_scores = self.dynamic.search(query, k)

        combined = list(zip(p_ids, p_scores)) + list(zip(d_ids, d_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:k]
        if not combined:
            return [], []
        ids, scores = zip(*combined)
        return list(ids), list(scores)
