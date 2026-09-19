from __future__ import annotations

import ast
import importlib.util
import json
import sys
import warnings
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "depth_preprocessing.py"
SPEC = importlib.util.spec_from_file_location("depth_preprocessing", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_depth_reader_is_independent_of_global_imread_patch(tmp_path, monkeypatch):
    expected = np.array([[0, 1, 300, 19999]], dtype=np.uint16)
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), expected)
    def patched_imread(*args, **kwargs):
        raise AssertionError("Depth reader must not use the globally patched cv2.imread")
    monkeypatch.setattr(cv2, "imread", patched_imread)
    actual = MODULE.read_image_unchanged(path)
    assert actual.dtype == np.uint16 and actual.ndim == 2
    np.testing.assert_array_equal(actual, expected)


def write_png(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), values)
    return path


def fitted_params(low: int = 10, high: int = 100) -> object:
    return MODULE.PercentileParams(
        candidate="percentile",
        p_low=2,
        p_high=98,
        low_value=low,
        high_value=high,
        valid_pixel_count=100,
        method=MODULE.QUANTILE_METHOD,
        zero_excluded=True,
        source_scope="train_png_only",
        png_count=1,
        train_split_normalized_sha256="a" * 64,
    )


def test_public_contract_and_python38_ast() -> None:
    assert MODULE.VALID_CANDIDATES == ("validmask", "percentile", "log", "inverse")
    assert ast.parse(SCRIPT.read_text(encoding="utf-8"), feature_version=(3, 8))
    for name in (
        "make_valid_mask",
        "fit_train_png_percentiles",
        "apply_percentile",
        "apply_log_depth",
        "apply_inverse_depth",
        "apply_candidate",
    ):
        assert callable(getattr(MODULE, name))


def test_canonical_split_identity_matches_repository() -> None:
    assert MODULE.normalized_split_sha256(REPO_ROOT / "data/splits/train.txt") == MODULE.CANONICAL_TRAIN_NORMALIZED_SHA256
    assert MODULE.normalized_split_sha256(REPO_ROOT / "data/splits/val.txt") == MODULE.CANONICAL_VAL_NORMALIZED_SHA256


def test_valid_mask_contract_determinism_and_input_immutability() -> None:
    depth = np.array([[0, 1], [65535, 0]], dtype=np.uint16)
    before = depth.copy()
    first = MODULE.make_valid_mask(depth)
    second = MODULE.make_valid_mask(depth)
    assert first.dtype == np.bool_ and first.shape == depth.shape
    assert np.array_equal(first, [[False, True], [True, False]])
    assert np.array_equal(first, second)
    assert np.array_equal(depth, before)


@pytest.mark.parametrize(
    ("depth", "source_format"),
    [
        (np.zeros((2, 2), dtype=np.uint8), "png"),
        (np.zeros((2, 2, 3), dtype=np.uint16), "png"),
        (np.zeros((2, 2), dtype=np.uint16), "jpg"),
    ],
)
def test_physical_candidates_reject_non_png_uint16_2d(
    depth: np.ndarray, source_format: str
) -> None:
    for transform in (
        MODULE.make_valid_mask,
        MODULE.apply_log_depth,
        MODULE.apply_inverse_depth,
    ):
        with pytest.raises(MODULE.DepthPreprocessingError):
            transform(depth, source_format=source_format)


def test_percentile_nearest_rank_zero_exclusion_and_determinism(tmp_path: Path) -> None:
    values = np.concatenate(
        (np.zeros(20, dtype=np.uint16), np.arange(1, 101, dtype=np.uint16))
    )
    first = write_png(tmp_path / "first.png", values[:60].reshape(6, 10))
    second = write_png(tmp_path / "second.PNG", values[60:].reshape(6, 10))
    params_a = MODULE.fit_train_png_percentiles([first, second])
    params_b = MODULE.fit_train_png_percentiles([second, first])
    assert params_a.valid_pixel_count == 100
    assert (params_a.low_value, params_a.high_value) == (2, 98)
    assert params_a.p_low == 2 and params_a.p_high == 98
    assert params_a.zero_excluded is True
    assert params_a.method == MODULE.QUANTILE_METHOD
    assert params_a.canonical_json() == params_b.canonical_json()
    assert params_a.identity_sha256() == params_b.identity_sha256()


def test_percentile_fit_reads_only_supplied_train_png(tmp_path: Path) -> None:
    train = write_png(
        tmp_path / "train.png", np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    )
    val = write_png(tmp_path / "val.png", np.full((10, 10), 65000, dtype=np.uint16))
    before = MODULE.fit_train_png_percentiles([train])
    write_png(val, np.ones((10, 10), dtype=np.uint16))
    after = MODULE.fit_train_png_percentiles([train])
    assert before == after
    assert (before.low_value, before.high_value) == (2, 98)


def test_percentile_fit_rejects_jpg_duplicates_count_and_decode(tmp_path: Path) -> None:
    png = write_png(
        tmp_path / "depth.png", np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    )
    jpg = tmp_path / "depth.jpg"
    assert cv2.imwrite(str(jpg), np.ones((4, 4, 3), dtype=np.uint8))
    with pytest.raises(MODULE.DepthPreprocessingError, match="只接受 train PNG"):
        MODULE.fit_train_png_percentiles([jpg])
    with pytest.raises(MODULE.DepthPreprocessingError, match="重复"):
        MODULE.fit_train_png_percentiles([png, png])
    with pytest.raises(MODULE.DepthPreprocessingError, match="数量错误"):
        MODULE.fit_train_png_percentiles([png], expected_count=2)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(MODULE.DepthPreprocessingError, match="解码失败"):
        MODULE.fit_train_png_percentiles([broken])


def test_percentile_fit_rejects_empty_and_degenerate_distribution(tmp_path: Path) -> None:
    empty = write_png(tmp_path / "empty.png", np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(MODULE.DepthPreprocessingError, match="没有 valid"):
        MODULE.fit_train_png_percentiles([empty])
    constant = write_png(tmp_path / "constant.png", np.full((4, 4), 7, dtype=np.uint16))
    with pytest.raises(MODULE.DepthPreprocessingError, match="分布退化"):
        MODULE.fit_train_png_percentiles([constant])


def test_percentile_transform_formula_mask_clipping_and_immutability() -> None:
    depth = np.array([[0, 1, 10], [55, 100, 200]], dtype=np.uint16)
    before = depth.copy()
    params = fitted_params()
    first, mask = MODULE.apply_percentile(depth, params)
    second, second_mask = MODULE.apply_percentile(depth, params)
    expected = np.array([[0, 0, 0], [0.5, 1, 1]], dtype=np.float32)
    assert first.dtype == np.float32 and mask.dtype == np.bool_
    assert first.shape == depth.shape and np.array_equal(first, expected)
    assert np.array_equal(mask, depth > 0)
    assert mask[0, 1] and first[0, 1] == 0
    assert np.isfinite(first).all() and 0 <= float(first.min()) <= float(first.max()) <= 1
    assert np.array_equal(first, second) and np.array_equal(mask, second_mask)
    assert np.array_equal(depth, before)


def test_percentile_requires_valid_frozen_params() -> None:
    depth = np.array([[1, 100]], dtype=np.uint16)
    params = fitted_params()
    with pytest.raises(FrozenInstanceError):
        params.low_value = 9
    with pytest.raises(MODULE.DepthPreprocessingError):
        MODULE.apply_candidate(depth, "percentile")
    with pytest.raises(MODULE.DepthPreprocessingError, match="high_value"):
        MODULE.apply_percentile(depth, replace(params, high_value=10))
    with pytest.raises(MODULE.DepthPreprocessingError, match="冻结"):
        MODULE.apply_percentile(depth, replace(params, p_low=1))


def test_log_formula_monotonic_finite_deterministic_and_input_unchanged() -> None:
    depth = np.array([[0, 1, 2, 100, 65535]], dtype=np.uint16)
    before = depth.copy()
    first, mask = MODULE.apply_log_depth(depth)
    second, _ = MODULE.apply_log_depth(depth)
    expected = np.zeros(depth.shape, dtype=np.float32)
    expected[mask] = np.log1p(depth[mask].astype(np.float32))
    assert first.dtype == np.float32 and first.shape == depth.shape
    assert first[0, 0] == 0 and np.array_equal(first, expected)
    assert np.isfinite(first).all() and np.all(np.diff(first[mask]) > 0)
    assert np.array_equal(first, second) and np.array_equal(depth, before)


def test_inverse_formula_warning_free_monotonic_and_no_hidden_scaling() -> None:
    depth = np.array([[0, 1, 2, 100, 65535]], dtype=np.uint16)
    before = depth.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first, mask = MODULE.apply_inverse_depth(depth)
    second, _ = MODULE.apply_inverse_depth(depth)
    expected = np.zeros(depth.shape, dtype=np.float32)
    expected[mask] = np.float32(1.0) / depth[mask].astype(np.float32)
    assert first.dtype == np.float32 and first.shape == depth.shape
    assert first[0, 0] == 0 and first[0, 1] == 1.0
    assert np.array_equal(first, expected)
    assert np.isfinite(first).all() and np.all(np.diff(first[mask]) < 0)
    assert np.array_equal(first, second) and np.array_equal(depth, before)


def test_candidates_are_mutually_exclusive_and_unknown_is_rejected() -> None:
    depth = np.array([[0, 10, 100]], dtype=np.uint16)
    params = fitted_params()
    with pytest.raises(MODULE.DepthPreprocessingError, match="combined"):
        MODULE.apply_candidate(depth, ["log", "inverse"])
    with pytest.raises(MODULE.DepthPreprocessingError, match="Unknown"):
        MODULE.apply_candidate(depth, "compat8")
    with pytest.raises(MODULE.DepthPreprocessingError, match="只能用于"):
        MODULE.apply_candidate(depth, "log", params)
    result = MODULE.apply_candidate(depth, "validmask")
    assert result.values.dtype == np.bool_
    assert np.array_equal(result.values, result.valid_mask)


def test_metadata_and_parameter_serialization_are_deterministic_and_portable() -> None:
    params = fitted_params()
    payloads: List[str] = []
    for candidate in MODULE.VALID_CANDIDATES:
        metadata = MODULE.candidate_metadata(
            candidate, params if candidate == "percentile" else None
        )
        first = MODULE.canonical_metadata_json(metadata)
        second = MODULE.canonical_metadata_json(metadata)
        assert first == second
        assert json.loads(first)["candidate"] == candidate
        payloads.append(first.lower())
    combined = "".join(payloads)
    assert "timestamp" not in combined and "uuid" not in combined
    assert "d:\\" not in combined and "/home/" not in combined
    percentile = json.loads(payloads[1])
    assert percentile["p_low"] == 2 and percentile["p_high"] == 98
    assert percentile["quantile_method"] == MODULE.QUANTILE_METHOD
    assert percentile["fit_scope"] == "train_png_only"
    assert percentile["parameter_identity_sha256"] == params.identity_sha256()
