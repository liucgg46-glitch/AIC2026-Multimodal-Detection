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


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "prepare_depth_yolo.py"
REPO_ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_depth_yolo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_png_depth(path: Path, values: np.ndarray | None = None) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = np.array([[0, 255, 256, 19999], [1, 511, 1024, 65535]], dtype=np.uint16)
    assert cv2.imwrite(str(path), values)
    return values


def write_jpg_depth(path: Path, value: int = 42, channels: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 12, channels) if channels > 1 else (8, 12)
    image = np.full(shape, value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def digest_tree(root: Path) -> List[Tuple[str, int, str]]:
    return [
        (path.relative_to(root).as_posix(), path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    ]


def configure_contract_hashes(monkeypatch: pytest.MonkeyPatch, paths: Dict[str, Path]) -> None:
    monkeypatch.setattr(MODULE, "CANONICAL_TRAIN_NORMALIZED_SHA256", MODULE.normalized_split_sha256(paths["train_split"]))
    monkeypatch.setattr(MODULE, "CANONICAL_VAL_NORMALIZED_SHA256", MODULE.normalized_split_sha256(paths["val_split"]))
    labels = list(MODULE.index_labels(paths["label_dir"]).values())
    identity, _ = MODULE._aggregate_paths(labels)
    monkeypatch.setattr(MODULE, "CANONICAL_LABELS_CLEAN_SHA256", identity["aggregate_sha256"])


@pytest.fixture
def canonical_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    root = tmp_path / "repo"
    depth_dir = root / MODULE.CANONICAL_DEPTH_RELATIVE
    label_dir = root / MODULE.CANONICAL_LABEL_RELATIVE
    raw_labels = root / "data/raw/train/labels"
    train_split = root / MODULE.CANONICAL_TRAIN_SPLIT_RELATIVE
    val_split = root / MODULE.CANONICAL_VAL_SPLIT_RELATIVE
    output = root / MODULE.CANONICAL_OUTPUT_RELATIVE
    for directory in (depth_dir, label_dir, raw_labels, output.parent):
        directory.mkdir(parents=True, exist_ok=True)
    write_png_depth(depth_dir / "train_png.png")
    write_jpg_depth(depth_dir / "train_jpg.jpg")
    write_png_depth(depth_dir / "val_png.png", np.array([[0, 256], [512, 1024]], dtype=np.uint16))
    for stem in ("train_png", "train_jpg", "val_png"):
        write_text_lf(label_dir / f"{stem}.txt", "" if stem == "val_png" else "0 0.5 0.5 0.1 0.1\n")
    write_text_lf(train_split, "train_png\ntrain_jpg\n")
    write_text_lf(val_split, "val_png\n")
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", root)
    paths = {"root": root, "depth_dir": depth_dir, "label_dir": label_dir, "raw_labels": raw_labels, "train_split": train_split, "val_split": val_split, "output": output}
    configure_contract_hashes(monkeypatch, paths)
    return paths


def build_canonical(paths: Dict[str, Path], force: bool = False) -> Dict[str, object]:
    return MODULE.build_view(
        paths["depth_dir"], paths["label_dir"], paths["train_split"], paths["val_split"], paths["output"],
        force=force, expected_train_count=2, expected_val_count=1, expected_png_count=2,
        expected_jpg_count=1, expected_train_png_count=1, expected_train_jpg_count=1,
        expected_val_png_count=1, expected_val_jpg_count=0,
        expected_png_shapes=((2, 4), (2, 2)), expected_jpg_shapes=((8, 12, 3),),
    )


def test_official_contract_matches_repository() -> None:
    assert MODULE.CANONICAL_DEPTH_RELATIVE.as_posix() == "data/raw/train/depth"
    assert MODULE.normalized_split_sha256(REPO_ROOT / "data/splits/train.txt") == MODULE.CANONICAL_TRAIN_NORMALIZED_SHA256
    assert MODULE.normalized_split_sha256(REPO_ROOT / "data/splits/val.txt") == MODULE.CANONICAL_VAL_NORMALIZED_SHA256
    depth = list((REPO_ROOT / MODULE.CANONICAL_DEPTH_RELATIVE).iterdir())
    assert sum(path.suffix.lower() == ".png" for path in depth) == 1851
    assert sum(path.suffix.lower() in {".jpg", ".jpeg"} for path in depth) == 149


def test_split_identity_is_newline_independent_and_content_sensitive() -> None:
    variants = [b"a\nb\n", b"a\r\nb\r\n", b"a\r\nb\n"]
    hashes = [hashlib.sha256(MODULE.normalize_split_content(item)).hexdigest() for item in variants]
    assert len(set(hashes)) == 1
    assert hashlib.sha256(MODULE.normalize_split_content(b"b\na\n")).hexdigest() != hashes[0]


@pytest.mark.parametrize("content", ["same\nsame\n", "ABC\nabc\n"])
def test_split_duplicate_and_casefold_collision_rejected(tmp_path: Path, content: str) -> None:
    split = tmp_path / "split.txt"
    write_text_lf(split, content)
    with pytest.raises(MODULE.DepthDatasetViewError, match="重复或仅大小写不同"):
        MODULE.read_stems(split, expected_count=2)


def test_canonical_build_counts_and_empty_label(canonical_project: Dict[str, Path]) -> None:
    result = build_canonical(canonical_project)
    validation = result["validation"]
    assert (validation["train_count"], validation["val_count"]) == (2, 1)
    assert (validation["png_count"], validation["jpg_count"]) == (2, 1)
    assert (canonical_project["output"] / "labels/val/val_png.txt").read_bytes() == b""


def test_png_conversion_is_exact_lossless_and_preserves_zero(canonical_project: Dict[str, Path]) -> None:
    source = cv2.imread(str(canonical_project["depth_dir"] / "train_png.png"), cv2.IMREAD_UNCHANGED)
    build_canonical(canonical_project)
    staged_path = canonical_project["output"] / "images/train/train_png.png"
    staged = cv2.imread(str(staged_path), cv2.IMREAD_UNCHANGED)
    expected = (source >> 8).astype(np.uint8)
    assert staged.dtype == np.uint8 and staged.shape == source.shape + (3,)
    assert np.array_equal(staged[..., 0], expected)
    assert np.array_equal(staged[..., 0], staged[..., 1]) and np.array_equal(staged[..., 1], staged[..., 2])
    assert np.all(staged[..., 0][source == 0] == 0)
    assert staged_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_jpg_and_labels_are_byte_preserving_isolated_copies(canonical_project: Dict[str, Path]) -> None:
    build_canonical(canonical_project)
    pairs = [
        (canonical_project["depth_dir"] / "train_jpg.jpg", canonical_project["output"] / "images/train/train_jpg.jpg"),
        (canonical_project["label_dir"] / "train_jpg.txt", canonical_project["output"] / "labels/train/train_jpg.txt"),
    ]
    for source, staged in pairs:
        assert source.read_bytes() == staged.read_bytes()
        assert not os.path.samefile(source, staged)


def test_every_output_is_uint8_three_channel_and_finite(canonical_project: Dict[str, Path]) -> None:
    build_canonical(canonical_project)
    for path in (canonical_project["output"] / "images").rglob("*.*"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        assert np.isfinite(image).all() and 0 <= int(image.min()) <= int(image.max()) <= 255


def test_manifest_and_yaml_are_deterministic(canonical_project: Dict[str, Path]) -> None:
    first = build_canonical(canonical_project)
    first_manifest = (canonical_project["output"] / "manifest.json").read_bytes()
    first_yaml = (canonical_project["output"] / "data.yaml").read_bytes()
    second = build_canonical(canonical_project, force=True)
    assert first_manifest == (canonical_project["output"] / "manifest.json").read_bytes()
    assert first_yaml == (canonical_project["output"] / "data.yaml").read_bytes()
    assert first["manifest"] == second["manifest"]
    manifest = json.loads(first_manifest)
    assert manifest["representation"] == "compat8" and manifest["isolation"] is True
    assert manifest["preprocessing"] == "fixed_format_compatibility"
    assert manifest["png_strategy"]["conversion"] == "uint8 = uint16 >> 8; channels = repeat(gray, 3)"
    assert manifest["jpg_strategy"]["physical_unit"] == "unknown"
    assert manifest["physical_scale_policy"] == "PNG and JPG DO NOT share a confirmed physical scale."
    assert manifest["train_split_normalized_sha256"] == MODULE.normalized_split_sha256(canonical_project["train_split"])
    assert manifest["val_split_normalized_sha256"] == MODULE.normalized_split_sha256(canonical_project["val_split"])
    assert b"created_at" not in first_manifest and b"uuid" not in first_manifest.lower()
    for record in manifest["source_file_list"]:
        for key in ("source_image_project_relative", "source_label_project_relative", "staged_image_view_relative"):
            assert not Path(record[key]).is_absolute() and "\\" not in record[key]
    config = yaml.safe_load(first_yaml)
    assert config["channels"] == 3 and config["train"] == "images/train" and config["val"] == "images/val"
    assert tuple(config["names"].values()) == MODULE.CLASS_NAMES


def test_jpg_manifest_records_byte_copy_and_unknown_unit(canonical_project: Dict[str, Path]) -> None:
    manifest = build_canonical(canonical_project)["manifest"]
    record = next(item for item in manifest["source_file_list"] if item["stem"] == "train_jpg")
    assert record["source_format"] == "jpg_uint8_3ch"
    assert record["conversion_branch"] == "byte_preserving_copy" and record["physical_unit"] == "unknown"
    assert record["source_sha256"] == record["output_sha256"]


@pytest.mark.parametrize("label_kind", ["raw_labels", "custom"])
def test_noncanonical_labels_rejected(canonical_project: Dict[str, Path], label_kind: str) -> None:
    label_dir = canonical_project["raw_labels"] if label_kind == "raw_labels" else canonical_project["root"] / "labels-other"
    label_dir.mkdir(exist_ok=True)
    with pytest.raises(MODULE.DepthDatasetViewError, match="label-dir 必须使用项目 canonical 路径"):
        MODULE.build_view(canonical_project["depth_dir"], label_dir, canonical_project["train_split"], canonical_project["val_split"], canonical_project["output"])


@pytest.mark.parametrize("split_key", ["train_split", "val_split"])
def test_replacement_split_rejected(canonical_project: Dict[str, Path], split_key: str) -> None:
    replacement = canonical_project["root"] / f"other-{split_key}.txt"
    replacement.write_bytes(canonical_project[split_key].read_bytes())
    train = replacement if split_key == "train_split" else canonical_project["train_split"]
    val = replacement if split_key == "val_split" else canonical_project["val_split"]
    with pytest.raises(MODULE.DepthDatasetViewError, match="必须使用项目 canonical 路径"):
        MODULE.validate_canonical_inputs(canonical_project["label_dir"], train, val)


def test_split_content_tampering_rejected(canonical_project: Dict[str, Path]) -> None:
    write_text_lf(canonical_project["train_split"], "train_jpg\ntrain_png\n")
    with pytest.raises(MODULE.DepthDatasetViewError, match="normalized logical-content"):
        MODULE.validate_canonical_inputs(canonical_project["label_dir"], canonical_project["train_split"], canonical_project["val_split"])


def test_labels_identity_tampering_rejected(canonical_project: Dict[str, Path]) -> None:
    with (canonical_project["label_dir"] / "train_png.txt").open("a", encoding="utf-8") as stream:
        stream.write("1 0.5 0.5 0.2 0.2\n")
    with pytest.raises(MODULE.DepthDatasetViewError, match="labels_clean identity"):
        build_canonical(canonical_project)


def test_overlap_validation(canonical_project: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    write_text_lf(canonical_project["val_split"], "train_png\n")
    configure_contract_hashes(monkeypatch, canonical_project)
    with pytest.raises(MODULE.DepthDatasetViewError, match="train/val split 存在重复"):
        build_canonical(canonical_project)


@pytest.mark.parametrize("kind", ["source", "label", "extra_source", "extra_label"])
def test_missing_and_extra_sources_are_rejected(canonical_project: Dict[str, Path], kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if kind == "source":
        (canonical_project["depth_dir"] / "val_png.png").unlink()
    elif kind == "label":
        (canonical_project["label_dir"] / "val_png.txt").unlink()
    elif kind == "extra_source":
        write_png_depth(canonical_project["depth_dir"] / "extra.png")
    else:
        write_text_lf(canonical_project["label_dir"] / "extra.txt", "")
        configure_contract_hashes(monkeypatch, canonical_project)
    with pytest.raises(MODULE.DepthDatasetViewError, match="不一致"):
        build_canonical(canonical_project)
    assert not canonical_project["output"].exists()


@pytest.mark.parametrize("names", [("same.png", "same.jpg"), ("ABC.png", "abc.jpg")])
def test_duplicate_and_casefold_depth_stem_rejected(canonical_project: Dict[str, Path], names: Tuple[str, str]) -> None:
    write_png_depth(canonical_project["depth_dir"] / names[0])
    write_jpg_depth(canonical_project["depth_dir"] / names[1])
    with pytest.raises(MODULE.DepthDatasetViewError, match="casefold stem"):
        MODULE.index_depth_images(canonical_project["depth_dir"])


@pytest.mark.parametrize("mutation", ["png_uint8", "png_three", "jpg_gray", "jpg_uint16", "broken"])
def test_invalid_source_encoding_rejected(canonical_project: Dict[str, Path], mutation: str) -> None:
    if mutation.startswith("png"):
        path = canonical_project["depth_dir"] / "train_png.png"
        image = np.ones((8, 12), dtype=np.uint8) if mutation == "png_uint8" else np.ones((8, 12, 3), dtype=np.uint16)
        assert cv2.imwrite(str(path), image)
        message = "PNG Depth"
    elif mutation in {"jpg_gray", "jpg_uint16"}:
        path = canonical_project["depth_dir"] / "train_jpg.jpg"
        if mutation == "jpg_gray":
            write_jpg_depth(path, channels=1)
        else:
            success, encoded = cv2.imencode(".png", np.ones((8, 12), dtype=np.uint16))
            assert success
            path.write_bytes(encoded.tobytes())
        message = "JPG Depth"
    else:
        path = canonical_project["depth_dir"] / "train_png.png"
        path.write_bytes(b"not an image")
        message = "解码失败"
    with pytest.raises(MODULE.DepthDatasetViewError, match=message):
        build_canonical(canonical_project)


def test_unexpected_source_shape_is_reported(canonical_project: Dict[str, Path]) -> None:
    write_png_depth(canonical_project["depth_dir"] / "train_png.png", np.ones((3, 3), dtype=np.uint16))
    with pytest.raises(MODULE.DepthDatasetViewError, match="source shape 异常"):
        build_canonical(canonical_project)


def test_format_counts_are_strict(canonical_project: Dict[str, Path]) -> None:
    with pytest.raises(MODULE.DepthDatasetViewError, match="PNG/JPG 数量错误"):
        MODULE.build_view(
            canonical_project["depth_dir"], canonical_project["label_dir"], canonical_project["train_split"], canonical_project["val_split"], canonical_project["output"],
            expected_train_count=2, expected_val_count=1, expected_png_count=1, expected_jpg_count=2,
            expected_train_png_count=1, expected_train_jpg_count=1, expected_val_png_count=0, expected_val_jpg_count=1,
            expected_png_shapes=((2, 4), (2, 2)), expected_jpg_shapes=((8, 12, 3),),
        )


def test_force_rebuild_preserves_sources_and_sibling(canonical_project: Dict[str, Path]) -> None:
    build_canonical(canonical_project)
    sibling = canonical_project["output"].parent / "future-representation"
    sentinel = sibling / "keep.txt"
    write_text_lf(sentinel, "unchanged")
    write_text_lf(canonical_project["output"] / "obsolete.txt", "remove")
    raw_before, labels_before = digest_tree(canonical_project["root"] / "data/raw"), digest_tree(canonical_project["label_dir"])
    split_before = (canonical_project["train_split"].read_bytes(), canonical_project["val_split"].read_bytes())
    build_canonical(canonical_project, force=True)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not (canonical_project["output"] / "obsolete.txt").exists()
    assert digest_tree(canonical_project["root"] / "data/raw") == raw_before
    assert digest_tree(canonical_project["label_dir"]) == labels_before
    assert split_before == (canonical_project["train_split"].read_bytes(), canonical_project["val_split"].read_bytes())


@pytest.mark.parametrize("candidate", ["project", "raw", "depth", "labels", "processed", "root"])
def test_output_boundary_rejects_noncanonical_paths(canonical_project: Dict[str, Path], candidate: str) -> None:
    candidates = {"project": canonical_project["root"], "raw": canonical_project["root"] / "data/raw", "depth": canonical_project["depth_dir"], "labels": canonical_project["label_dir"], "processed": canonical_project["root"] / "data/processed", "root": canonical_project["output"].parent}
    with pytest.raises(MODULE.DepthDatasetViewError, match="输出目录"):
        MODULE.validate_output_path(candidates[candidate], canonical_project["depth_dir"], canonical_project["label_dir"], canonical_project["train_split"], canonical_project["val_split"])


@pytest.mark.parametrize("redirect_key", ["processed", "trainable", "output"])
def test_redirected_staging_rejected_before_mutation(canonical_project: Dict[str, Path], monkeypatch: pytest.MonkeyPatch, redirect_key: str) -> None:
    targets = {"processed": canonical_project["root"] / "data/processed", "trainable": canonical_project["output"].parent, "output": canonical_project["output"]}
    target = targets[redirect_key]
    external = canonical_project["root"].parent / f"external-{redirect_key}"
    sentinel = external / "sentinel.txt"
    write_text_lf(sentinel, "unchanged")
    original = MODULE._resolve_path
    def redirected(path: Path) -> Path:
        return external if MODULE._normalized_absolute(path) == MODULE._normalized_absolute(target) else original(path)
    monkeypatch.setattr(MODULE, "_resolve_path", redirected)
    with pytest.raises(MODULE.DepthDatasetViewError, match="逃出|重定向"):
        build_canonical(canonical_project, force=True)
    assert sentinel.read_text(encoding="utf-8") == "unchanged" and not canonical_project["output"].exists()


def test_unsafe_temp_cleanup_never_calls_rmtree(canonical_project: Dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    staging = canonical_project["output"].parent / ".compat8-unsafe"
    staging.mkdir()
    external = canonical_project["root"].parent / "external-temp"
    sentinel = external / "sentinel.txt"
    write_text_lf(sentinel, "unchanged")
    original, removed = MODULE._resolve_path, []
    def redirected(path: Path) -> Path:
        return external if MODULE._normalized_absolute(path) == MODULE._normalized_absolute(staging) else original(path)
    monkeypatch.setattr(MODULE, "_resolve_path", redirected)
    monkeypatch.setattr(MODULE.shutil, "rmtree", lambda path: removed.append(path))
    with pytest.raises(MODULE.DepthDatasetViewError, match="临时 staging 越出"):
        MODULE._remove_temporary_staging(staging)
    assert removed == [] and sentinel.read_text(encoding="utf-8") == "unchanged"


def test_only_compat8_is_accepted(canonical_project: Dict[str, Path]) -> None:
    with pytest.raises(MODULE.DepthDatasetViewError, match="仅实现 representation=compat8"):
        MODULE.build_view(canonical_project["depth_dir"], canonical_project["label_dir"], canonical_project["train_split"], canonical_project["val_split"], canonical_project["output"], representation="percentile")


def test_cli_defaults_and_no_hardlink_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = MODULE.parse_args()
    assert args.depth_dir == "data/raw/train/depth" and args.label_dir == "data/processed/train/labels_clean"
    assert args.train_split == "data/splits/train.txt" and args.val_split == "data/splits/val.txt"
    assert args.output_root == "data/processed/depth_trainable/compat8" and args.representation == "compat8"
    assert not hasattr(args, "link_mode")


def test_python38_parse_and_lf_writer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT), feature_version=(3, 8))
    assert ".write_text(" not in source and "IMREAD_UNCHANGED" in source
