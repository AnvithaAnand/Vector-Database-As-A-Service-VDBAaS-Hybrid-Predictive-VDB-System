from pathlib import Path

# Embedding configuration
EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Storage configuration
HOT_PARTITION_CAPACITY = 1000
PERMANENT_CAPACITY = 30_000
DYNAMIC_CAPACITY = 70_000

# Anchor configuration
ANCHOR_DISTANCE_THRESHOLD = 0.35   # cosine distance threshold to join existing anchor
PREDICTION_HIT_THRESHOLD = 0.85
WEAK_DECAY = 0.5
MEDIUM_DECAY = 0.8
STRONG_DECAY = 0.9

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Misc
RANDOM_SEED = 42
