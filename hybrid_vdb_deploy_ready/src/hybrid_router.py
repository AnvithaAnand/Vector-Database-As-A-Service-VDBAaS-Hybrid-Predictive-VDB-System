from __future__ import annotations
import time
from typing import Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None

from .anchor_system import AnchorSystem
from .storage_engine import StorageEngine
from .semantic_cache import SemanticCache
from .cloud_client import CloudClient
from .metrics import Metrics
from .config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM


class HybridRouter:
    """Orchestrates query flow between anchors, local storage and cloud."""

    def __init__(self):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run `pip install sentence-transformers`."
            )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.anchor_system = AnchorSystem()
        self.storage = StorageEngine()
        self.semantic_cache = SemanticCache()
        self.cloud = CloudClient()
        self.metrics = Metrics()

    # --- core API ----------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode(text)
        return np.asarray(vec, dtype="float32")

    def search(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        t0 = time.time()
        q_vec = self._embed(query_text)

        prediction_anchor = self.anchor_system.check_prediction_hit(q_vec)
        prediction_hit = prediction_anchor is not None
        self.metrics.record_prediction(hit=prediction_hit)

        # 1. try local
        ids, scores = self.storage.search(q_vec, k)
        source = "local" if ids else "cloud"

        if not ids:
            # 2. fall back to cloud
            res = self.cloud.search(q_vec, k)
            ids = [r["id"] for r in res]
            scores = [r["score"] for r in res]
            vectors = np.stack([r["vector"] for r in res], axis=0).astype("float32")

            # 3. feed into storage as dynamic / hot
            self.storage.add_dynamic(vectors, ids)
            self.storage.add_hot(vectors, ids)

        # 4. update anchors & semantic cache
        anchor = self.anchor_system.process_query(q_vec, query_text)
        # more predictions for stronger anchors
        k_pred = 3 if anchor.type == "WEAK" else 5 if anchor.type == "MEDIUM" else 7
        preds = self.anchor_system.generate_predictions(anchor, k=k_pred)

        # track semantic clusters using first vector
        if ids:
            self.semantic_cache.update_with_vector(q_vec, ids[0])

        latency_ms = (time.time() - t0) * 1000.0
        self.metrics.record_query(latency_ms=latency_ms, source=source)

        return {
            "query": query_text,
            "ids": ids,
            "scores": scores,
            "source": source,
            "latency_ms": latency_ms,
            "anchor_id": anchor.id,
            "anchor_type": anchor.type,
            "prediction_hit": prediction_hit,
            "metrics": self.metrics.snapshot(),
        }
