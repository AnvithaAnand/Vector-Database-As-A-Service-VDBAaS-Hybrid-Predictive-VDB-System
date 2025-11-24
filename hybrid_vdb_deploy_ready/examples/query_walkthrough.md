# Query Walkthrough

This document walks through a single query as it flows through the system.

1. A user calls `/search` with the text: `"What is diabetes?"`.
2. `HybridRouter` embeds the text using `all-MiniLM-L6-v2` into a 384‑dimensional vector.
3. The vector is sent to `AnchorSystem.check_prediction_hit` to see whether it matches
   any previously generated prediction vectors.
4. On cold start there is no match, so the system proceeds to search the local
   `StorageEngine`, which initially has no vectors.
5. The router then falls back to `CloudClient.search`, which in this prototype
   returns results from a small mock corpus.
6. The vectors from the cloud response are inserted into the dynamic index and hot
   partition inside `StorageEngine`.
7. `AnchorSystem.process_query` either strengthens an existing anchor near this
   vector or creates a new WEAK anchor.
8. Several synthetic predictions are generated around the anchor centroid. Future
   queries that land near those predictions will count as prediction hits.
9. `Metrics` is updated with the latency and whether this was a local or cloud hit.
10. The API returns the IDs, scores, latency and a small snapshot of current metrics.
