import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "competition_evaluator", Path(__file__).resolve().parents[1] / "scripts/analysis/evaluate_submission.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_perfect_predictions_have_exact_unit_ap():
    gt = {"a": [[c, .5, .5, .2, .2] for c in range(12)]}
    pred = {"a": [r + [.9] for r in gt["a"]]}
    assert MODULE.evaluate(gt, pred)["map50_95"] == pytest.approx(1)


def test_low_confidence_tp_extends_recall_and_duplicate_is_not_tp():
    gt = {"a": [[0, .2, .2, .1, .1], [0, .8, .8, .1, .1]]}
    pred = {"a": [gt["a"][0] + [.9], gt["a"][0] + [.8], gt["a"][1] + [.1]]}
    low = MODULE.evaluate(gt, pred)["per_class"][0]["ap50"]
    high = MODULE.evaluate(gt, pred, conf=.25)["per_class"][0]["ap50"]
    assert low == pytest.approx((51 + 50 * 2 / 3) / 101)
    assert high == pytest.approx(51 / 101)


def test_greedy_matching_selects_unmatched_gt():
    gt = {"a": [[0, .45, .5, .4, .4], [0, .55, .5, .4, .4]]}
    pred = {"a": [[0, .45, .5, .4, .4, .9], [0, .47, .5, .4, .4, .8]]}
    assert MODULE.evaluate(gt, pred)["per_class"][0]["ap50"] == pytest.approx(1)


def test_limit_is_per_image_across_classes():
    gt = {"a": [[0, .5, .5, .2, .2]]}
    pred = {"a": [[1, .5, .5, .2, .2, .9]] * 100 + [gt["a"][0] + [.8]]}
    assert MODULE.evaluate(gt, pred)["map50_95"] == 0


def test_missing_empty_txt_is_rejected():
    with pytest.raises(ValueError, match="exactly match"):
        MODULE.evaluate({"a": []}, {})


def test_invalid_class_and_nonfinite_rows_rejected(tmp_path):
    path = tmp_path / "a.txt"
    for text in ["0.5 .5 .5 .2 .2 .9", "0 .5 .5 .2 .2 nan", "0 .5 .5 0 .2 .9"]:
        path.write_text(text)
        with pytest.raises(ValueError):
            MODULE.read_rows(path, True)
