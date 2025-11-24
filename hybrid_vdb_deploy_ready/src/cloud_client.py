from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import os

try:
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None

from .config import EMBEDDING_DIM


class CloudClient:
    """Thin wrapper around a remote vector DB.

    For the purposes of this portfolio project, the client gracefully falls
    back to a **mock in‑memory store** if Qdrant is not configured. This
    allows the rest of the system to run unchanged.
    """

    def __init__(self):
        api_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        collection = os.getenv("QDRANT_COLLECTION", "hybrid_vdb_demo")

        self.collection_name = collection
        if api_url and api_key and QdrantClient is not None:
            self._mock = False
            self.client = QdrantClient(url=api_url, api_key=api_key)
        else:
            # Mock mode – very small in‑memory dataset
            self._mock = True
            self.client = None
            self._mock_vectors = np.random.randn(256, EMBEDDING_DIM).astype("float32")
            self._mock_payloads = [
                {"id": f"doc_{i}", "text": f"Mock document {i}"} for i in range(256)
            ]

    # In a real system, query_vector would be used directly. Here we just
    # approximate behaviour.
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if self._mock:
            # cosine similarity
            q = query_vector.astype("float32")
            if q.ndim == 1:
                q = q[None, :]
            v = self._mock_vectors
            v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            scores = (v_norm @ q_norm.T)[:, 0]
            idx = np.argsort(-scores)[:k]
            out = []
            for i in idx:
                item = dict(self._mock_payloads[i])
                item["score"] = float(scores[i])
                item["vector"] = v[i]
                out.append(item)
            return out

        # Real Qdrant path (not exercised in tests)
        res = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=k,
        )
        out = []
        for point in res:
            payload = point.payload or {}
            payload.setdefault("id", str(point.id))
            payload["score"] = float(point.score)
            # vectors would typically be retrieved in a second call;
            # omitted here for brevity
            out.append(payload)
        return out
