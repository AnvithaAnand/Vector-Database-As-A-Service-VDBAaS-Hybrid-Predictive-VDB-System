import numpy as np
from hybrid_vdb.src.anchor_system import AnchorSystem, AnchorType


def test_anchor_creation_and_promotion():
    sys = AnchorSystem()
    v1 = np.ones(384, dtype="float32")
    a1 = sys.process_query(v1, "first")
    assert a1.type == AnchorType.WEAK

    # hit the same region repeatedly to strengthen the anchor
    for _ in range(10):
        a1 = sys.process_query(v1, "repeat")

    assert a1.strength > 25
    assert a1.type in (AnchorType.MEDIUM, AnchorType.STRONG, AnchorType.PERMANENT)


def test_prediction_hit_and_decay():
    sys = AnchorSystem()
    v = np.ones(384, dtype="float32")
    anchor = sys.process_query(v, "base")

    preds = sys.generate_predictions(anchor, k=1)
    hit_vec = preds[0].vector
    hit_anchor = sys.check_prediction_hit(hit_vec)
    assert hit_anchor is not None
    assert hit_anchor.id == anchor.id

    # Decay should not immediately delete a reasonably strong anchor
    pre_strength = anchor.strength
    sys.decay()
    assert anchor.id in sys.anchors
    assert sys.anchors[anchor.id].strength <= pre_strength
