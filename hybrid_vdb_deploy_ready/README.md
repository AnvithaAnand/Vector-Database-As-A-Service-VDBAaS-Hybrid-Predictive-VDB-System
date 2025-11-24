# Hybrid Vector Database as a Service (VDBAaS)

This repository contains a **prototype implementation** of a Hybrid Vector Database with
*anchor‑based predictive learning* and a local‑first retrieval strategy.

The goal of the project is to:
- Minimize end‑to‑end query latency for vector search
- Reduce dependency on a remote / cloud vector database
- Learn query patterns over time and proactively prefetch likely future vectors

## High‑Level Architecture

- `HybridRouter` orchestrates the query flow.
- `AnchorSystem` learns semantic regions ("anchors") and trajectories between queries.
- `StorageEngine` manages a two‑tier local store (permanent + dynamic).
- `SemanticCache` tracks "hot" semantic clusters with momentum.
- `CloudClient` is a thin wrapper over a remote VDB (e.g., Qdrant Cloud).
- `Metrics` records hit‑rates, latency, and learning curves.
- `Scheduler` runs lightweight background maintenance jobs.
- `demo/app.py` exposes a small FastAPI service for interactive querying.

> This is intentionally lightweight and dependency‑minimal so it can be run on a laptop.
> It is meant as a **systems design + ML infra** showcase rather than a production system.

## Quick Start

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the demo API
uvicorn demo.app:app --reload

# 4. Issue a query (in another terminal)
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "What is diabetes?"}'
```

The first few queries will likely hit the **cloud backend** (simulated by `CloudClient`),
but as the system observes more traffic, the **local hit‑rate** should improve.

## Folder Layout

```text
hybrid_vdb/
  README.md
  requirements.txt
  src/
    anchor_system.py
    cloud_client.py
    config.py
    hybrid_router.py
    local_vdb.py
    metrics.py
    scheduler.py
    semantic_cache.py
    storage_engine.py
  demo/
    app.py
  diagrams/
    architecture.png
    anchor_lifecycle.png
    data_flow.png
  examples/
    query_walkthrough.md
  tests/
    test_anchor_system.py
```

## Disclaimer

- The cloud calls are intentionally simplified and stubbed so the project can run without
  real credentials. You can plug in your own Qdrant / Milvus / Vespa backends if desired.
- FAISS usage is optional; if it is not installed, the system falls back to a NumPy
  brute‑force search implementation.

