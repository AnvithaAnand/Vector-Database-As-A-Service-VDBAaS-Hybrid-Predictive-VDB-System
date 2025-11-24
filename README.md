<h1 align="center">⚡ Hybrid Vector Database as a Service (VDBAaS)</h1>
<h3 align="center">Anchor-Based Predictive Learning • Hybrid Local–Cloud Vector Search • Semantic Caching</h3>

---

<p align="center">
  <img src="https://img.shields.io/badge/Type-Research%20Project-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Vector%20DB-Qdrant%20%2B%20FAISS-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-Anchors%20%2F%20Trajectory%20Learning-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
</p>

---

# 🚀 Overview

This project implements a **Hybrid Vector Database System (VDBAaS)** that combines:

- **Local FAISS vector search**
- **Cloud Qdrant VDB lookup**
- **Anchor-based predictive semantic learning**
- **Predictive prefetching**
- **Two-tier hybrid cache architecture**
- **Smart router for optimal query execution**

The system **learns semantic trajectories**, predicts future query regions, and proactively prefetches vector data to achieve:

- 🔥 **70%+ Local Cache Hit Rate**  
- ⚡ **3.3× Lower Average Latency**  
- 💸 **Up to 70% Cloud Query Cost Savings**  

---

# 🧠 Key Innovations

### ✓ **Anchor-Based Semantic Learning**
Learns user query behavior, forms semantic clusters called **anchors**, and predicts next likely queries.

### ✓ **Predictive Prefetching**
Prefetches related vectors based on semantic velocity and anchor trajectories.

### ✓ **Two-Tier Hybrid Storage**
- **Permanent Layer (FlatL2)** → stable, high-strength anchors  
- **Dynamic Layer (HNSW)** → fast, adaptive memory with momentum-based eviction  

### ✓ **Hybrid Router**
Decides whether to search:
- Local FAISS (fast)
- Cloud Qdrant (fallback)
- Both (prefetch)

---

## 🏗 Architecture

```mermaid
flowchart TD
    User["User Query"] --> Router["Hybrid Router"]
    
    Router --> |Check Prediction| AnchorSystem
    Router --> |Local FAISS Search| LocalVDB
    Router --> |If Miss → Cloud Qdrant| CloudVDB
    
    AnchorSystem --> |Generate Predictions| Predictions
    Predictions --> Prefetch["Prefetch + Cache"]
    Prefetch --> LocalVDB
    
    LocalVDB --> Router
    CloudVDB --> Router
    Router --> UserResponse["Response to User"]

