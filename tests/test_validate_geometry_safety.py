import ast
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "data" / "validate_geometry_safety.py"
SPEC = importlib.util.spec_from_file_location("validate_geometry_safety", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_image(path: Path, shape=(8, 12, 3)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    assert cv2.imwrite(str(path), image)
    return path


def write_label(path: Path, text="0 0.5 0.5 0.1 0.1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return path


@pytest.fixture
def synthetic_view(tmp_path: Path):
    source = tmp_path / "source"
    staging = tmp_path / "view"
    labels = tmp_path / "labels_clean"
    for subset, stems in (("train", ["train_a", "train_b"]), ("val", ["val_a"])):
        for stem in stems:
            write_image(source / (stem + ".png"))
            write_image(staging / "images" / subset / (stem + ".png"))
            write_label(labels / (stem + ".txt"), "" if stem == "val_a" else "0 0.5 0.5 0.1 0.1\n")
            write_label(staging / "labels" / subset / (stem + ".txt"), "" if stem == "val_a" else "0 0.5 0.5 0.1 0.1\n")
    spec = MODULE.ViewSpec("synthetic", "depth", source, staging)
    return spec, ["train_a", "train_b"], ["val_a"], labels


def validate(parts):
    return MODULE.validate_view(*parts)


def test_same_spatial_shape_passes(synthetic_view) -> None:
    result = validate(synthetic_view)
    assert result["geometry_pass"] is True
    assert result["dimension_mismatch_count"] == 0


@pytest.mark.parametrize("shape", [(8, 13, 3), (9, 12, 3)])
def test_changed_width_or_height_fails(synthetic_view, shape) -> None:
    spec = synthetic_view[0]
    write_image(spec.staging_dir / "images/train/train_a.png", shape)
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["dimension_mismatch_count"] == 1


def test_channel_change_with_same_height_width_passes(synthetic_view) -> None:
    spec = synthetic_view[0]
    write_image(spec.staging_dir / "images/train/train_a.png", (8, 12))
    assert validate(synthetic_view)["geometry_pass"] is True


def test_label_single_byte_change_fails(synthetic_view) -> None:
    spec = synthetic_view[0]
    write_label(spec.staging_dir / "labels/train/train_a.txt", "0 0.5 0.5 0.1 0.2\n")
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["label_mismatch_count"] == 1


@pytest.mark.parametrize("kind", ["image", "label"])
def test_missing_image_or_label_fails(synthetic_view, kind) -> None:
    spec = synthetic_view[0]
    path = spec.staging_dir / ("images" if kind == "image" else "labels") / "train" / ("train_a.png" if kind == "image" else "train_a.txt")
    path.unlink()
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["missing_count"] == 1


def test_extra_image_fails(synthetic_view) -> None:
    spec = synthetic_view[0]
    write_image(spec.staging_dir / "images/train/extra.png")
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["extra_count"] == 1


def test_duplicate_stem_fails(synthetic_view) -> None:
    spec = synthetic_view[0]
    write_image(spec.staging_dir / "images/train/train_a.jpg")
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["duplicate_count"] == 1


def test_casefold_collision_fails(synthetic_view) -> None:
    spec = synthetic_view[0]
    write_image(spec.staging_dir / "images/train/TRAIN_A.jpg")
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["casefold_collision_count"] > 0


def test_split_membership_change_fails(synthetic_view) -> None:
    spec = synthetic_view[0]
    (spec.staging_dir / "images/train/train_a.png").rename(spec.staging_dir / "images/train/wrong.png")
    result = validate(synthetic_view)
    assert result["geometry_pass"] is False
    assert result["split_mismatch_count"] > 0


def test_train_val_overlap_fails(synthetic_view) -> None:
    spec, train, val, labels = synthetic_view
    write_image(spec.staging_dir / "images/val/train_a.png")
    write_label(spec.staging_dir / "labels/val/train_a.txt")
    result = MODULE.validate_view(spec, train, val, labels)
    assert result["geometry_pass"] is False
    assert result["train_val_overlap_count"] == 1


def test_cross_subset_casefold_collision_fails(synthetic_view) -> None:
    spec, train, val, labels = synthetic_view
    write_image(spec.staging_dir / "images/val/TRAIN_A.png")
    write_label(spec.staging_dir / "labels/val/TRAIN_A.txt")
    result = MODULE.validate_view(spec, train, val, labels)
    assert result["geometry_pass"] is False
    assert result["casefold_collision_count"] > 0


@pytest.mark.parametrize(
    "text",
    [
        "0 0.5 0.5 0.1\n",
        "12 0.5 0.5 0.1 0.1\n",
        "0 nan 0.5 0.1 0.1\n",
        "0 0.5 inf 0.1 0.1\n",
        "0 0.5 0.5 0 0.1\n",
        "0 0.5 0.5 0.1 -0.1\n",
    ],
)
def test_invalid_yolo_rows_are_errors(tmp_path: Path, text: str) -> None:
    write_label(tmp_path / "labels/sample.txt", text)
    result = MODULE.validate_yolo_labels(tmp_path / "labels")
    assert result["bbox_validity_error_count"] == 1


def test_bbox_edge_overflow_is_warning_only(tmp_path: Path) -> None:
    write_label(tmp_path / "labels/sample.txt", "0 0.01 0.5 0.1 0.1\n")
    result = MODULE.validate_yolo_labels(tmp_path / "labels")
    assert result["bbox_validity_error_count"] == 0
    assert result["bbox_edge_warning_count"] == 1


def test_empty_label_is_valid(tmp_path: Path) -> None:
    write_label(tmp_path / "labels/empty.txt", "")
    result = MODULE.validate_yolo_labels(tmp_path / "labels")
    assert result["empty_label_count"] == 1
    assert result["bbox_validity_error_count"] == 0


def test_jpg_without_orientation_is_safe(tmp_path: Path) -> None:
    image = Image.new("RGB", (12, 8), color="gray")
    image.save(tmp_path / "plain.jpg")
    result = MODULE.inspect_exif_orientation(tmp_path)
    assert result == {
        "jpg_count": 1,
        "orientation_missing_count": 1,
        "orientation_default_count": 0,
        "orientation_nondefault_count": 0,
    }


def test_nondefault_jpg_orientation_is_reported(tmp_path: Path) -> None:
    image = Image.new("RGB", (12, 8), color="gray")
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "rotated.jpg", exif=exif)
    result = MODULE.inspect_exif_orientation(tmp_path)
    assert result["orientation_nondefault_count"] == 1


def test_deterministic_json(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    payload = {"z": 1, "text": "可复现", "nested": {"b": False, "a": 0}}
    MODULE.write_deterministic_json(output, payload)
    first = output.read_bytes()
    MODULE.write_deterministic_json(output, payload)
    assert output.read_bytes() == first
    assert json.loads(first.decode("utf-8")) == payload


def test_python38_static_syntax() -> None:
    ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH), feature_version=(3, 8))
    test_path = Path(__file__).resolve()
    ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path), feature_version=(3, 8))
