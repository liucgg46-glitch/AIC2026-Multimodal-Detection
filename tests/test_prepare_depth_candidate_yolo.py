from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "prepare_depth_candidate_yolo.py"
SPEC = importlib.util.spec_from_file_location("prepare_depth_candidate_yolo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_png(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), values)
    return path


def write_jpg(path: Path) -> Path:
    image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return path


def digest_tree(root: Path) -> List[Tuple[str, int, str]]:
    return [
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


@pytest.fixture
def canonical_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Dict[str, Path]:
    root = tmp_path / "repo"
    depth_dir = root / MODULE.c2.CANONICAL_DEPTH_RELATIVE
    label_dir = root / MODULE.c2.CANONICAL_LABEL_RELATIVE
    train_split = root / MODULE.c2.CANONICAL_TRAIN_SPLIT_RELATIVE
    val_split = root / MODULE.c2.CANONICAL_VAL_SPLIT_RELATIVE
    depth_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    values = np.concatenate(
        (np.zeros(20, dtype=np.uint16), np.arange(1, 101, dtype=np.uint16))
    ).reshape(10, 12)
    train_png = write_png(depth_dir / "train_png.png", values)
    train_jpg = write_jpg(depth_dir / "train_jpg.jpg")
    val_values = np.array(
        [[0, 1, 299, 300, 301, 1000], [5000, 10000, 15000, 19999, 20, 50]],
        dtype=np.uint16,
    )
    write_png(depth_dir / "val_png.png", val_values)
    for stem in ("train_png", "train_jpg", "val_png"):
        write_text_lf(label_dir / f"{stem}.txt", "0 0.5 0.5 0.1 0.1\n")
    write_text_lf(train_split, "train_png\ntrain_jpg\n")
    write_text_lf(val_split, "val_png\n")

    for module in (MODULE, MODULE.c2, MODULE.c3):
        monkeypatch.setattr(module, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        MODULE.c2,
        "CANONICAL_TRAIN_NORMALIZED_SHA256",
        MODULE.c2.normalized_split_sha256(train_split),
    )
    monkeypatch.setattr(
        MODULE.c2,
        "CANONICAL_VAL_NORMALIZED_SHA256",
        MODULE.c2.normalized_split_sha256(val_split),
    )
    labels_identity, _ = MODULE.c2._aggregate_paths(list(label_dir.glob("*.txt")))
    monkeypatch.setattr(
        MODULE.c2, "CANONICAL_LABELS_CLEAN_SHA256", labels_identity["aggregate_sha256"]
    )
    monkeypatch.setattr(MODULE.c2, "EXPECTED_TRAIN_PNG_COUNT", 1)
    params = MODULE.c3.fit_train_png_percentiles(
        [train_png],
        expected_count=1,
        train_split_normalized_sha256=MODULE.c2.normalized_split_sha256(train_split),
    )
    monkeypatch.setattr(MODULE, "FROZEN_PERCENTILE_LOW_MM", params.low_value)
    monkeypatch.setattr(MODULE, "FROZEN_PERCENTILE_HIGH_MM", params.high_value)
    monkeypatch.setattr(
        MODULE, "FROZEN_PERCENTILE_IDENTITY_SHA256", params.identity_sha256()
    )
    return {
        "root": root,
        "depth_dir": depth_dir,
        "label_dir": label_dir,
        "train_split": train_split,
        "val_split": val_split,
        "train_png": train_png,
        "train_jpg": train_jpg,
    }


def build_candidate(
    paths: Dict[str, Path], candidate: str, force: bool = False
) -> Dict[str, object]:
    output = paths["root"] / MODULE.DEPTH_TRAINABLE_RELATIVE / candidate
    return MODULE.build_view(
        paths["depth_dir"],
        paths["label_dir"],
        paths["train_split"],
        paths["val_split"],
        output,
        candidate,
        force=force,
        expected_train_count=2,
        expected_val_count=1,
        expected_png_count=2,
        expected_jpg_count=1,
        expected_train_png_count=1,
        expected_train_jpg_count=1,
        expected_val_png_count=1,
        expected_val_jpg_count=0,
        expected_png_shapes=((10, 12), (2, 6)),
        expected_jpg_shapes=((8, 12, 3),),
    )


def test_public_contract_and_python38_ast() -> None:
    assert MODULE.SUPPORTED_CANDIDATES == (
        "validmask",
        "percentile",
        "log",
        "inverse",
    )
    ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT), feature_version=(3, 8))
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import depth_preprocessing as c3" in source
    assert "IMREAD_UNCHANGED" in source


def test_unit_floor_quantizer_exact_boundaries() -> None:
    values = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]], dtype=np.float32)
    result = MODULE.quantize_unit_float_to_uint8_floor(values)
    assert np.array_equal(result, [[0, 0, 127, 255, 255]])
    with pytest.raises(MODULE.DepthCandidateViewError):
        MODULE.quantize_unit_float_to_uint8_floor(np.array([np.nan], dtype=np.float32))


def test_half_up_rounding_locks_tie_behavior() -> None:
    values = np.array([0.0, 0.49, 0.5, 1.5, 2.5, 3.49], dtype=np.float64)
    assert np.array_equal(
        MODULE.round_half_up_nonnegative(values), [0, 0, 1, 2, 3, 3]
    )
    with pytest.raises(MODULE.DepthCandidateViewError):
        MODULE.round_half_up_nonnegative(np.array([-0.1]))


def test_validmask_mapping() -> None:
    depth = np.array([[0, 1], [299, 19999]], dtype=np.uint16)
    before = depth.copy()
    gray = MODULE.render_candidate_gray(depth, "validmask")
    assert gray.dtype == np.uint8 and gray.shape == depth.shape
    assert np.array_equal(gray, [[0, 255], [255, 255]])
    assert np.array_equal(depth, before)


def test_percentile_reuses_c3_and_floor_mapping() -> None:
    params = MODULE.c3.PercentileParams(
        candidate="percentile",
        p_low=2,
        p_high=98,
        low_value=100,
        high_value=200,
        valid_pixel_count=4,
        method=MODULE.c3.QUANTILE_METHOD,
        zero_excluded=True,
        source_scope="train_png_only",
        png_count=1,
        train_split_normalized_sha256="a" * 64,
    )
    depth = np.array([[0, 100, 150, 200]], dtype=np.uint16)
    expected_float, _ = MODULE.c3.apply_percentile(depth, params)
    result = MODULE.render_candidate_gray(depth, "percentile", params)
    assert np.array_equal(
        result, MODULE.quantize_unit_float_to_uint8_floor(expected_float)
    )
    assert np.array_equal(result, [[0, 0, 127, 255]])


def test_log_fixed_mapping_boundaries_monotonic_and_max_guard() -> None:
    depth = np.array([[0, 1, 2, 300, 1000, 19999]], dtype=np.uint16)
    result = MODULE.render_candidate_gray(depth, "log")
    assert result.dtype == np.uint8 and result[0, 0] == 0
    assert result[0, 1] == 0 and result[0, -1] == 255
    assert np.all(np.diff(result[0, 1:].astype(np.int16)) >= 0)
    assert np.isfinite(result).all()
    with pytest.raises(MODULE.DepthCandidateViewError, match="超过"):
        MODULE.render_candidate_gray(np.array([[20000]], dtype=np.uint16), "log")


def test_inverse_frozen_boundaries_midpoint_and_monotonicity() -> None:
    depth = np.array(
        [[0, 1, 299, 300, 301, 1000, 5000, 10000, 19999, 20000]],
        dtype=np.uint16,
    )
    before = depth.copy()
    output, valid = MODULE.quantize_inverse_depth(depth)
    assert np.array_equal(output[0, :4], [0, 255, 255, 255])
    assert output[0, -2] == 1 and output[0, -1] == 1
    d = 1000.0
    scaled = 254.0 * ((1.0 / d - 1.0 / 19999.0) / (1.0 / 300.0 - 1.0 / 19999.0))
    expected = 1 + int(np.floor(scaled + 0.5))
    assert int(output[0, 5]) == expected
    assert np.all(output[valid] >= 1)
    assert np.all(np.diff(output[0, 1:].astype(np.int16)) <= 0)
    assert np.array_equal(depth, before)


@pytest.mark.parametrize("candidate", MODULE.SUPPORTED_CANDIDATES)
def test_all_candidate_builds_have_exact_images_labels_and_manifest(
    canonical_project: Dict[str, Path], candidate: str
) -> None:
    source_raw = digest_tree(canonical_project["root"] / "data/raw")
    source_labels = digest_tree(canonical_project["label_dir"])
    split_bytes = (
        canonical_project["train_split"].read_bytes(),
        canonical_project["val_split"].read_bytes(),
    )
    result = build_candidate(canonical_project, candidate)
    output = canonical_project["root"] / MODULE.DEPTH_TRAINABLE_RELATIVE / candidate
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert manifest["candidate"] == candidate and manifest["total_count"] == 3
    assert manifest["train_count"] == 2 and manifest["val_count"] == 1
    assert manifest["output_dtype"] == "uint8" and manifest["output_channels"] == 3
    assert config["channels"] == 3 and config["train"] == "images/train"
    assert config["val"] == "images/val" and "path" not in config
    assert len(config["names"]) == 12
    for path in (output / "images").rglob("*.*"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        if path.suffix.lower() == ".png":
            assert np.array_equal(image[..., 0], image[..., 1])
            assert np.array_equal(image[..., 1], image[..., 2])
    staged_jpg = output / "images/train/train_jpg.jpg"
    assert staged_jpg.read_bytes() == canonical_project["train_jpg"].read_bytes()
    assert digest_tree(output / "labels") == [
        (f"train/{name}", size, digest)
        if name.startswith("train_")
        else (f"val/{name}", size, digest)
        for name, size, digest in digest_tree(canonical_project["label_dir"])
    ]
    assert digest_tree(canonical_project["root"] / "data/raw") == source_raw
    assert digest_tree(canonical_project["label_dir"]) == source_labels
    assert split_bytes == (
        canonical_project["train_split"].read_bytes(),
        canonical_project["val_split"].read_bytes(),
    )
    assert result["methods"]["label_byte_preserving_copy"] == 3


def test_percentile_manifest_locks_frozen_parameters(
    canonical_project: Dict[str, Path]
) -> None:
    result = build_candidate(canonical_project, "percentile")
    params = result["percentile_params"]
    conversion = result["manifest"]["conversion"]
    assert conversion["p_low"] == 2 and conversion["p_high"] == 98
    assert conversion["low_mm"] == params.low_value
    assert conversion["high_mm"] == params.high_value
    assert conversion["c3_parameter_identity_sha256"] == params.identity_sha256()
    assert conversion["quantization"] == "floor(clip(x,0,1)*255)"


def test_inverse_manifest_records_frozen_policy(
    canonical_project: Dict[str, Path]
) -> None:
    manifest = build_candidate(canonical_project, "inverse")["manifest"]
    conversion = manifest["conversion"]
    assert conversion["near_clip_mm"] == 300
    assert conversion["far_clip_mm"] == 19999
    assert conversion["valid_output_range"] == [1, 255]
    assert conversion["train_dependent_statistics"] is False
    assert conversion["percentile_used"] is False and conversion["log_used"] is False
    assert "half-away-from-zero" in conversion["rounding_rule"]


def test_manifest_is_deterministic_relative_and_has_no_runtime_identity(
    canonical_project: Dict[str, Path]
) -> None:
    build_candidate(canonical_project, "validmask")
    output = canonical_project["root"] / MODULE.DEPTH_TRAINABLE_RELATIVE / "validmask"
    first_manifest = (output / "manifest.json").read_bytes()
    first_yaml = (output / "data.yaml").read_bytes()
    build_candidate(canonical_project, "validmask", force=True)
    assert first_manifest == (output / "manifest.json").read_bytes()
    assert first_yaml == (output / "data.yaml").read_bytes()
    text = first_manifest.decode("utf-8").lower()
    assert "timestamp" not in text and "uuid" not in text
    assert str(canonical_project["root"]).lower().replace("\\", "/") not in text.replace("\\", "/")
    manifest = json.loads(first_manifest)
    for record in manifest["source_file_list"]:
        for key in (
            "source_image_project_relative",
            "source_label_project_relative",
            "staged_image_view_relative",
        ):
            assert not Path(record[key]).is_absolute() and "\\" not in record[key]


def test_force_rebuild_preserves_sibling_and_sources(
    canonical_project: Dict[str, Path]
) -> None:
    build_candidate(canonical_project, "log")
    trainable = canonical_project["root"] / MODULE.DEPTH_TRAINABLE_RELATIVE
    sibling = trainable / "compat8"
    sentinel = sibling / "keep.txt"
    write_text_lf(sentinel, "unchanged")
    obsolete = trainable / "log/obsolete.txt"
    write_text_lf(obsolete, "remove")
    raw_before, labels_before = digest_tree(canonical_project["root"] / "data/raw"), digest_tree(canonical_project["label_dir"])
    build_candidate(canonical_project, "log", force=True)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not obsolete.exists()
    assert digest_tree(canonical_project["root"] / "data/raw") == raw_before
    assert digest_tree(canonical_project["label_dir"]) == labels_before


@pytest.mark.parametrize("bad", ["compat8", "log,inverse", "unknown"])
def test_unknown_or_combined_candidate_rejected(bad: str) -> None:
    with pytest.raises(MODULE.DepthCandidateViewError):
        MODULE.canonical_output_root(bad)


def test_noncanonical_output_rejected(canonical_project: Dict[str, Path]) -> None:
    with pytest.raises(MODULE.DepthCandidateViewError, match="canonical view"):
        MODULE.validate_output_path(
            canonical_project["root"] / "data/processed/depth_trainable/compat8",
            "validmask",
            canonical_project["depth_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
        )


def test_unsafe_temp_cleanup_never_calls_rmtree(
    canonical_project: Dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = canonical_project["root"] / MODULE.DEPTH_TRAINABLE_RELATIVE / ".log-unsafe"
    staging.mkdir(parents=True)
    external = canonical_project["root"].parent / "external"
    external.mkdir()
    original, removed = MODULE._resolve_path, []

    def redirected(path: Path) -> Path:
        return external if MODULE._normalized_absolute(path) == MODULE._normalized_absolute(staging) else original(path)

    monkeypatch.setattr(MODULE, "_resolve_path", redirected)
    monkeypatch.setattr(MODULE.shutil, "rmtree", lambda path: removed.append(path))
    with pytest.raises(MODULE.DepthCandidateViewError, match="越出"):
        MODULE._remove_temporary_staging(staging, "log")
    assert removed == []


def test_cli_defaults_are_candidate_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--candidate", "inverse"])
    args = MODULE.parse_args()
    assert args.candidate == "inverse" and args.output_root is None and not args.force
