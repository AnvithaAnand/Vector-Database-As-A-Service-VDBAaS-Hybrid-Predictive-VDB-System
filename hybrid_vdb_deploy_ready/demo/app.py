from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from ..src.hybrid_router import HybridRouter

app = FastAPI(title="Hybrid VDB Demo")
router = HybridRouter()


class QueryRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/search")
def search(req: QueryRequest) -> Dict[str, Any]:
    return router.search(req.query, k=req.k)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
