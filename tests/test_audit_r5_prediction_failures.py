"""The R5 review categories follow the greedy matcher and stay descriptive."""

from scripts.analysis.audit_r5_prediction_failures import audit_image


def test_exact_match_and_high_confidence_background_candidate():
    gt = [[0, 0.2, 0.2, 0.1, 0.1]]
    predictions = [
        [0, 0.2, 0.2, 0.1, 0.1, 0.9],
        [0, 0.8, 0.8, 0.1, 0.1, 0.8],
    ]
    misses, fps, tp = audit_image("a", gt, predictions, 0.5)
    assert tp == 1 and misses == []
    assert len(fps) == 1 and fps[0]["reason"] == "background_or_unlabeled"


def test_low_score_localization_and_other_class_are_distinct():
    gt = [
        [0, 0.2, 0.2, 0.1, 0.1],
        [1, 0.5, 0.5, 0.1, 0.1],
        [2, 0.8, 0.8, 0.1, 0.1],
    ]
    predictions = [
        [0, 0.2, 0.2, 0.1, 0.1, 0.1],
        [1, 0.54, 0.5, 0.1, 0.1, 0.9],
        [3, 0.8, 0.8, 0.1, 0.1, 0.9],
    ]
    misses, fps, tp = audit_image("a", gt, predictions, 0.25)
    assert tp == 0
    assert [item["reason"] for item in misses] == [
        "below_operating_confidence", "localization_candidate_0p1_to_0p5",
        "other_class_candidate",
    ]
    assert any(item["reason"] == "other_class_overlap" for item in fps)
