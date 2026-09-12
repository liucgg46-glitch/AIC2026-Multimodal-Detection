import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "analyze_e001_errors.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_e001_errors", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _box(class_id, xyxy, confidence=None):
    row = {
        "class_id": class_id,
        "box_xyxy": np.asarray(xyxy, dtype=float),
        "bbox_xywhn": [0.5, 0.5, 0.2, 0.2],
        "area_ref": 1000.0,
        "size_bin": "small",
    }
    if confidence is not None:
        row["confidence"] = confidence
    return row


def test_box_iou_exact_cases():
    assert MODULE.box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert MODULE.box_iou([0, 0, 1, 1], [1, 1, 2, 2]) == 0.0
    assert MODULE.box_iou([0, 0, 1, 1], [0.5, 0, 1.5, 1]) == 1 / 3


def test_size_category_boundaries_are_fixed():
    assert MODULE.size_category(255.999) == "tiny"
    assert MODULE.size_category(256) == "small"
    assert MODULE.size_category(1024) == "medium"
    assert MODULE.size_category(9216) == "large"


def test_operating_point_matching_covers_tp_fp_fn_and_class_confusion():
    ground_truth = [
        _box(0, [0.1, 0.1, 0.3, 0.3]),
        _box(1, [0.6, 0.6, 0.8, 0.8]),
    ]
    predictions = [
        _box(0, [0.1, 0.1, 0.3, 0.3], 0.9),
        _box(2, [0.6, 0.6, 0.8, 0.8], 0.8),
        _box(0, [0.0, 0.7, 0.1, 0.8], 0.7),
        _box(1, [0.6, 0.6, 0.8, 0.8], 0.1),
    ]

    pred_rows, gt_rows = MODULE.match_at_operating_point(predictions, ground_truth)

    assert [row["status"] for row in pred_rows] == ["TP", "FP", "FP"]
    assert pred_rows[1]["fp_reason"] == "class_confusion"
    assert pred_rows[2]["fp_reason"] == "background"
    assert [row["status"] for row in gt_rows] == ["TP", "FN"]


def test_decorate_box_uses_letterbox_reference_area():
    item = {"class_id": 0, "bbox_xywhn": [0.5, 0.5, 0.1, 0.1]}
    decorated = MODULE.decorate_box(item, width=1920, height=1080)
    assert decorated["area_ref"] == 640 * 0.1 * 360 * 0.1
    assert decorated["size_bin"] == "medium"


def test_clean_ground_truth_is_verified_against_local_labels(tmp_path):
    (tmp_path / "sample.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.1 0.2 0.3 0.4\n", encoding="utf-8"
    )
    exported = [{
        "image_stem": "sample",
        "ground_truth": [
            {"class_id": 1, "bbox_xywhn": [0.1, 0.2, 0.3, 0.4]},
            {"class_id": 0, "bbox_xywhn": [0.5, 0.5, 0.2, 0.2]},
        ],
    }]

    result = MODULE.validate_clean_ground_truth(exported, tmp_path)

    assert result["clean_label_match"] is True
    assert result["clean_label_objects_checked"] == 2


def test_clean_ground_truth_mismatch_fails(tmp_path):
    (tmp_path / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    exported = [{
        "image_stem": "sample",
        "ground_truth": [{"class_id": 1, "bbox_xywhn": [0.5, 0.5, 0.2, 0.2]}],
    }]

    with pytest.raises(ValueError, match="differs from labels_clean"):
        MODULE.validate_clean_ground_truth(exported, tmp_path)


def test_historical_metrics_use_best_map50_95_row(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.8,0.5,0.6,0.3\n"
        "2,0.7,0.6,0.65,0.4\n",
        encoding="utf-8",
    )

    metrics = MODULE.read_historical_best_metrics(results)

    assert metrics["best_epoch"] == 2
    assert metrics["precision"] == 0.7
    assert metrics["mAP50-95"] == 0.4
