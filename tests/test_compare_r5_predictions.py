"""Focused checks for the fixed-val R5 scorecard's matching and macro AP."""

import pytest

from scripts.analysis.compare_r5_predictions import compare, operating_recall, size_bin


def test_original_image_size_bins_and_greedy_recall():
    shape = (1000, 1000)
    small = [0, 0.1, 0.1, 0.02, 0.02]
    medium = [0, 0.5, 0.5, 0.05, 0.05]
    large = [0, 0.8, 0.8, 0.2, 0.2]
    assert [size_bin(row, shape) for row in (small, medium, large)] == [
        "small", "medium", "large"
    ]
    report = operating_recall(
        {"a": [small, medium, large]},
        {"a": [[0, 0.1, 0.1, 0.02, 0.02, 0.9],
               [0, 0.1, 0.1, 0.02, 0.02, 0.8],
               [0, 0.8, 0.8, 0.2, 0.2, 0.9]]},
        {"a": shape},
    )
    assert [report["by_size"][name]["tp"] for name in ("small", "medium", "large")] == [1, 0, 1]
    assert report["by_class"]["person"]["recall"] == pytest.approx(2 / 3)


def test_macro_ap_delta_is_one_twelfth_of_single_class_gain():
    gt = {"a": [[11, 0.5, 0.5, 0.2, 0.2]]}
    baseline = {"a": []}
    candidate = {"a": [[11, 0.5, 0.5, 0.2, 0.2, 0.9]]}
    report = compare(gt, baseline, candidate, {"a": (100, 100)})
    assert report["baseline_map50_95"] == 0
    assert report["candidate_map50_95"] == pytest.approx(1 / 12)
    assert report["per_class"][11]["delta"] == pytest.approx(1)
    assert report["candidate_operating_recall"]["by_class"]["tricycle"]["tp"] == 1
