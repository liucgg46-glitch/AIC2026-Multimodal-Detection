"""Build the canonical, isolated Ultralytics Depth-only compat8 dataset view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DEPTH_RELATIVE = Path("data/raw/train/depth")
CANONICAL_LABEL_RELATIVE = Path("data/processed/train/labels_clean")
CANONICAL_TRAIN_SPLIT_RELATIVE = Path("data/splits/train.txt")
CANONICAL_VAL_SPLIT_RELATIVE = Path("data/splits/val.txt")
CANONICAL_OUTPUT_RELATIVE = Path("data/processed/depth_trainable/compat8")
CANONICAL_DEPTH_TRAIN_PARENT_RELATIVE = Path("data/raw/train")
DEPTH_TRAINABLE_ROOT_RELATIVE = Path("data/processed/depth_trainable")

# SHA-256 of normalized logical content, intentionally independent of CRLF/LF.
CANONICAL_TRAIN_NORMALIZED_SHA256 = "20b0c1fb09a6848a4700a5adc8ce1f9a6d9040a48626a1759020d7f986450969"
CANONICAL_VAL_NORMALIZED_SHA256 = "336165b3509b6a0516b052c02fd98d152441300e8ae757eb0d27ca68a053ee48"
CANONICAL_LABELS_CLEAN_SHA256 = "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"

DEPTH_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_TRAIN_COUNT = 1600
EXPECTED_VAL_COUNT = 400
EXPECTED_PNG_COUNT = 1851
EXPECTED_JPG_COUNT = 149
EXPECTED_TRAIN_PNG_COUNT = 1478
EXPECTED_TRAIN_JPG_COUNT = 122
EXPECTED_VAL_PNG_COUNT = 373
EXPECTED_VAL_JPG_COUNT = 27
EXPECTED_PNG_SHAPE = (1080, 1920)
EXPECTED_JPG_SHAPE = (360, 640, 3)
MANIFEST_SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0"
PNG_COMPRESSION = 3

# Canonical class-id order already used by the E001 RGB preparation pipeline.
CLASS_NAMES = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
)


class DepthDatasetViewError(ValueError):
    """Raised when the canonical Depth dataset contract is invalid."""


def canonical_paths() -> Dict[str, Path]:
    """Construct trusted lexical paths from the script-derived project root."""
    root = _resolve_path(PROJECT_ROOT)
    return {
        "project_root": root,
        "data": root / "data",
        "data_processed": root / "data/processed",
        "raw_root": root / "data/raw",
        "depth_train_parent": root / CANONICAL_DEPTH_TRAIN_PARENT_RELATIVE,
        "depth_dir": root / CANONICAL_DEPTH_RELATIVE,
        "label_dir": root / CANONICAL_LABEL_RELATIVE,
        "train_split": root / CANONICAL_TRAIN_SPLIT_RELATIVE,
        "val_split": root / CANONICAL_VAL_SPLIT_RELATIVE,
        "depth_trainable_root": root / DEPTH_TRAINABLE_ROOT_RELATIVE,
        "output_root": root / CANONICAL_OUTPUT_RELATIVE,
    }


def project_path(value: Union[str, Path]) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _normalized_absolute(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _resolve_path(path: Path) -> Path:
    """Resolve symlinks and junction-like redirections for safety checks."""
    return path.resolve()


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(_resolve_path(first))) == os.path.normcase(str(_resolve_path(second)))


def _is_within(path: Path, parent: Path) -> bool:
    resolved_path = _resolve_path(path)
    resolved_parent = _resolve_path(parent)
    if _same_path(resolved_path, resolved_parent):
        return True
    try:
        resolved_path.relative_to(resolved_parent)
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    left = _resolve_path(first)
    right = _resolve_path(second)
    if _same_path(left, right):
        return True
    try:
        right.relative_to(left)
        return True
    except ValueError:
        pass
    try:
        left.relative_to(right)
        return True
    except ValueError:
        return False


def _require_canonical_path(candidate: Path, expected: Path, label: str) -> None:
    """Require both checkout-relative spelling and resolved identity to be canonical."""
    if _normalized_absolute(candidate) != _normalized_absolute(expected):
        raise DepthDatasetViewError(f"{label} 必须使用项目 canonical 路径: {expected}")
    if not _same_path(candidate, expected):
        raise DepthDatasetViewError(f"{label} resolve 后不是项目 canonical 路径: {expected}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_split_content(content: Union[str, bytes]) -> bytes:
    """Normalize only newline encoding and retain exactly one final line terminator."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DepthDatasetViewError("Split 必须是 UTF-8 文本") from exc
    else:
        text = content.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    logical_lines = text.split("\n")
    if text.endswith("\n"):
        logical_lines = logical_lines[:-1]
    return ("\n".join(logical_lines) + "\n").encode("utf-8")


def normalized_split_sha256(path: Path) -> str:
    """Hash the canonical logical split content independent of newline style."""
    return hashlib.sha256(normalize_split_content(path.read_bytes())).hexdigest()


def project_relative_path(path: Path) -> str:
    """Return a machine-independent project-relative path for manifest identity."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise DepthDatasetViewError(f"Manifest 身份路径必须位于项目根目录: {path}") from exc


def validate_canonical_inputs(label_dir: Path, train_split: Path, val_split: Path) -> None:
    paths = canonical_paths()
    _require_canonical_path(label_dir, paths["label_dir"], "label-dir")
    _require_canonical_path(train_split, paths["train_split"], "train split")
    _require_canonical_path(val_split, paths["val_split"], "val split")
    if (
        not train_split.is_file()
        or normalized_split_sha256(train_split) != CANONICAL_TRAIN_NORMALIZED_SHA256
    ):
        raise DepthDatasetViewError(
            "canonical train split normalized logical-content SHA-256 不一致；"
            "拒绝使用被替换或篡改的 split"
        )
    if (
        not val_split.is_file()
        or normalized_split_sha256(val_split) != CANONICAL_VAL_NORMALIZED_SHA256
    ):
        raise DepthDatasetViewError(
            "canonical val split normalized logical-content SHA-256 不一致；"
            "拒绝使用被替换或篡改的 split"
        )


def validate_staging_boundary(output_root: Path) -> None:
    """Anchor the writable staging boundary at PROJECT_ROOT plus fixed relative paths."""
    paths = canonical_paths()
    project_root = _resolve_path(paths["project_root"])
    expected_output = paths["output_root"]
    if _normalized_absolute(output_root) != _normalized_absolute(expected_output):
        raise DepthDatasetViewError(f"输出目录必须是 canonical compat8 view: {expected_output}")

    critical_paths = (
        paths["project_root"],
        paths["data"],
        paths["data_processed"],
        paths["depth_trainable_root"],
        paths["output_root"],
    )
    for path in critical_paths:
        if not _is_within(path, project_root):
            raise DepthDatasetViewError(
                f"canonical staging 关键路径 resolve 后逃出 PROJECT_ROOT: {path}"
            )

    resolved_staging_root = _resolve_path(paths["depth_trainable_root"])
    resolved_output = _resolve_path(output_root)
    if resolved_output.parent != resolved_staging_root:
        raise DepthDatasetViewError(
            "canonical compat8 view 的父目录被 symlink 或 junction 重定向"
        )
    expected_resolved_location = resolved_staging_root / "compat8"
    if _normalized_absolute(resolved_output) != _normalized_absolute(expected_resolved_location):
        raise DepthDatasetViewError("canonical compat8 view 不得通过 symlink 或 junction 重定向")
    if output_root.is_symlink():
        raise DepthDatasetViewError("canonical compat8 view 不得是 symlink 或重定向路径")


def validate_output_path(
    output_root: Path,
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
) -> None:
    """Reject non-canonical output and every source/output overlap after resolution."""
    validate_staging_boundary(output_root)
    paths = canonical_paths()
    exact_forbidden = {
        "project root": paths["project_root"],
        "data": paths["data"],
        "data/processed": paths["data_processed"],
        "depth_trainable root": paths["depth_trainable_root"],
    }
    for label, forbidden in exact_forbidden.items():
        if _same_path(output_root, forbidden):
            raise DepthDatasetViewError(f"输出目录禁止指向 {label}: {output_root}")

    overlap_sources = {
        "raw Depth source": depth_dir,
        "labels_clean": label_dir,
        "train split": train_split,
        "val split": val_split,
        "data/raw": paths["raw_root"],
        "canonical source parent": paths["depth_train_parent"],
    }
    for label, source in overlap_sources.items():
        if _paths_overlap(output_root, source):
            raise DepthDatasetViewError(f"输出目录与受保护的 {label} 路径重叠: {output_root}")

def validate_force_target(
    output_root: Path,
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
) -> None:
    """Revalidate the single deletable view immediately before --force removal."""
    validate_output_path(output_root, depth_dir, label_dir, train_split, val_split)
    expected = PROJECT_ROOT / CANONICAL_OUTPUT_RELATIVE
    if _normalized_absolute(output_root) != _normalized_absolute(expected):
        raise DepthDatasetViewError("--force 只能重建 canonical compat8 view")
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise DepthDatasetViewError("--force 目标必须是非 symlink 的 canonical compat8 目录")


def _validate_temporary_staging(staging: Path) -> None:
    validate_staging_boundary(canonical_paths()["output_root"])
    paths = canonical_paths()
    project_root = _resolve_path(paths["project_root"])
    allowed = _resolve_path(paths["depth_trainable_root"])
    resolved_staging = _resolve_path(staging)
    if (
        staging.is_symlink()
        or resolved_staging.parent != allowed
        or not _is_within(resolved_staging, project_root)
    ):
        raise DepthDatasetViewError(f"临时 staging 越出 canonical depth_trainable 根目录: {staging}")
    if not staging.name.startswith(".compat8-"):
        raise DepthDatasetViewError(f"拒绝清理非 compat8 临时目录: {staging}")


def _remove_temporary_staging(staging: Path) -> None:
    if staging.exists():
        _validate_temporary_staging(staging)
        shutil.rmtree(staging)


def read_stems(split_path: Path, expected_count: int) -> List[str]:
    if not split_path.is_file():
        raise DepthDatasetViewError(f"Split 文件不存在: {split_path}")
    stems = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not stems:
        raise DepthDatasetViewError(f"Split 文件为空，拒绝回退到全部训练数据: {split_path}")
    invalid = [stem for stem in stems if Path(stem).name != stem or Path(stem).suffix]
    if invalid:
        raise DepthDatasetViewError(f"Split 每行必须是无目录、无扩展名的 stem；非法值: {invalid[:5]}")
    folded = Counter(stem.casefold() for stem in stems)
    duplicates = sorted(stem for stem in stems if folded[stem.casefold()] > 1)
    if duplicates:
        raise DepthDatasetViewError(f"Split 包含重复或仅大小写不同的 stem: {duplicates[:5]}")
    if len(stems) != expected_count:
        raise DepthDatasetViewError(
            f"Split 数量错误: {split_path}，期望 {expected_count}，实际 {len(stems)}"
        )
    return stems


def _index_unique_files(directory: Path, extensions: Set[str], kind: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise DepthDatasetViewError(f"{kind} 目录不存在: {directory}")
    by_identity: Dict[str, List[Path]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
        if path.is_file() and path.suffix.lower() in extensions:
            by_identity.setdefault(path.stem.casefold(), []).append(path)
    ambiguous = {identity: paths for identity, paths in by_identity.items() if len(paths) > 1}
    if ambiguous:
        _, paths = next(iter(sorted(ambiguous.items())))
        raise DepthDatasetViewError(
            f"同一个 casefold stem 匹配到多个 {kind} 文件: {[path.name for path in paths]}"
        )
    return {paths[0].stem: paths[0] for paths in by_identity.values()}


def index_depth_images(depth_dir: Path) -> Dict[str, Path]:
    return _index_unique_files(depth_dir, DEPTH_IMAGE_EXTENSIONS, "Depth 图像")


def index_labels(label_dir: Path) -> Dict[str, Path]:
    return _index_unique_files(label_dir, {".txt"}, "labels_clean")


def _set_mismatch_message(kind: str, expected: Set[str], actual: Set[str]) -> Optional[str]:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return None
    return f"{kind} 与 fixed split 不一致；缺失: {missing[:5]}；split 外额外 stem: {extra[:5]}"


def validate_depth_encoding(path: Path) -> Dict[str, Any]:
    """Decode one source without losing its original bit depth and validate its branch."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise DepthDatasetViewError(f"Depth 图像解码失败: {path}")
    suffix = path.suffix.lower()
    if suffix == ".png":
        if image.dtype != np.uint16 or image.ndim != 2:
            raise DepthDatasetViewError(
                f"PNG Depth 必须是 uint16 single-channel；{path.name} 实际 dtype={image.dtype}, "
                f"shape={image.shape}"
            )
        source_format = "png_uint16_1ch"
        conversion = "uint8 = uint16 >> 8; channels = repeat(gray, 3)"
        physical_unit = "source depth units (mm per dataset specification)"
    elif suffix in {".jpg", ".jpeg"}:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise DepthDatasetViewError(
                f"JPG Depth 必须是 uint8 3-channel；{path.name} 实际 dtype={image.dtype}, "
                f"shape={image.shape}"
            )
        source_format = "jpg_uint8_3ch"
        conversion = "byte_preserving_copy"
        physical_unit = "unknown"
    else:
        raise DepthDatasetViewError(f"不支持的 Depth 扩展名: {path.name}")
    return {
        "source_bytes": path.stat().st_size,
        "source_channels": 1 if image.ndim == 2 else int(image.shape[2]),
        "source_dtype": str(image.dtype),
        "source_format": source_format,
        "source_sha256": sha256_file(path),
        "source_shape": [int(value) for value in image.shape],
        "conversion_branch": conversion,
        "physical_unit": physical_unit,
    }


def _aggregate_paths(paths: List[Path]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes = {}
    total_bytes = 0
    for path in sorted(paths, key=lambda item: (item.name.casefold(), item.name)):
        size = path.stat().st_size
        file_hash = sha256_file(path)
        digest.update(f"{path.name}\0{size}\0{file_hash}\n".encode("utf-8"))
        total_bytes += size
        file_hashes[path.name] = file_hash
    return (
        {
            "aggregate_sha256": digest.hexdigest(),
            "file_count": len(paths),
            "total_bytes": total_bytes,
        },
        file_hashes,
    )


def validate_sources(
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    expected_train_count: int = EXPECTED_TRAIN_COUNT,
    expected_val_count: int = EXPECTED_VAL_COUNT,
    expected_png_count: int = EXPECTED_PNG_COUNT,
    expected_jpg_count: int = EXPECTED_JPG_COUNT,
    expected_train_png_count: int = EXPECTED_TRAIN_PNG_COUNT,
    expected_train_jpg_count: int = EXPECTED_TRAIN_JPG_COUNT,
    expected_val_png_count: int = EXPECTED_VAL_PNG_COUNT,
    expected_val_jpg_count: int = EXPECTED_VAL_JPG_COUNT,
    expected_png_shapes: Tuple[Tuple[int, ...], ...] = (EXPECTED_PNG_SHAPE,),
    expected_jpg_shapes: Tuple[Tuple[int, ...], ...] = (EXPECTED_JPG_SHAPE,),
) -> Dict[str, Any]:
    validate_canonical_inputs(label_dir, train_split, val_split)
    train_stems = read_stems(train_split, expected_train_count)
    val_stems = read_stems(val_split, expected_val_count)
    overlap = sorted(set(train_stems) & set(val_stems))
    if overlap:
        raise DepthDatasetViewError(f"train/val split 存在重复 stem: {overlap[:5]}")
    expected_stems = set(train_stems) | set(val_stems)
    expected_total = expected_train_count + expected_val_count
    if len(expected_stems) != expected_total:
        raise DepthDatasetViewError(
            f"合并 split 唯一 stem 数量错误；期望 {expected_total}，实际 {len(expected_stems)}"
        )

    images = index_depth_images(depth_dir)
    labels = index_labels(label_dir)
    image_mismatch = _set_mismatch_message("Depth 图像", expected_stems, set(images))
    if image_mismatch:
        raise DepthDatasetViewError(image_mismatch)
    label_mismatch = _set_mismatch_message("labels_clean", expected_stems, set(labels))
    if label_mismatch:
        raise DepthDatasetViewError(label_mismatch)

    labels_identity, label_hashes = _aggregate_paths(list(labels.values()))
    if labels_identity["aggregate_sha256"] != CANONICAL_LABELS_CLEAN_SHA256:
        raise DepthDatasetViewError("canonical labels_clean identity 不一致；拒绝使用被替换或篡改的标签")
    source_info = {
        stem: validate_depth_encoding(images[stem]) for stem in sorted(expected_stems)
    }
    png_shapes = {
        tuple(info["source_shape"])
        for info in source_info.values()
        if info["source_format"] == "png_uint16_1ch"
    }
    jpg_shapes = {
        tuple(info["source_shape"])
        for info in source_info.values()
        if info["source_format"] == "jpg_uint8_3ch"
    }
    if png_shapes != set(expected_png_shapes) or jpg_shapes != set(expected_jpg_shapes):
        raise DepthDatasetViewError(
            "Depth source shape 异常；"
            f"期望 PNG={sorted(expected_png_shapes)}, JPG={sorted(expected_jpg_shapes)}；"
            f"实际 PNG={sorted(png_shapes)}, JPG={sorted(jpg_shapes)}"
        )

    def format_counts(stems: List[str]) -> Tuple[int, int]:
        png = sum(images[stem].suffix.lower() == ".png" for stem in stems)
        return png, len(stems) - png

    png_count, jpg_count = format_counts(train_stems + val_stems)
    train_png_count, train_jpg_count = format_counts(train_stems)
    val_png_count, val_jpg_count = format_counts(val_stems)
    actual_counts = (
        png_count,
        jpg_count,
        train_png_count,
        train_jpg_count,
        val_png_count,
        val_jpg_count,
    )
    expected_counts = (
        expected_png_count,
        expected_jpg_count,
        expected_train_png_count,
        expected_train_jpg_count,
        expected_val_png_count,
        expected_val_jpg_count,
    )
    if actual_counts != expected_counts:
        raise DepthDatasetViewError(
            "Depth PNG/JPG 数量错误；"
            f"期望 total/train/val={expected_counts}，实际={actual_counts}"
        )

    entries = {
        subset: [(stem, images[stem], labels[stem]) for stem in stems]
        for subset, stems in (("train", train_stems), ("val", val_stems))
    }
    return {
        "entries": entries,
        "images": images,
        "labels": labels,
        "label_hashes": label_hashes,
        "labels_identity": labels_identity,
        "source_info": source_info,
        "train_count": len(train_stems),
        "val_count": len(val_stems),
        "total_count": len(expected_stems),
        "png_count": png_count,
        "jpg_count": jpg_count,
        "train_png_count": train_png_count,
        "train_jpg_count": train_jpg_count,
        "val_png_count": val_png_count,
        "val_jpg_count": val_jpg_count,
    }


def copy_isolated(source: Path, destination: Path) -> None:
    """Create an isolated byte-preserving copy."""
    shutil.copy2(source, destination)


def convert_png_compat8(source: Path, destination: Path) -> None:
    """Apply the fixed C2 bit-depth compatibility conversion and lossless encoding."""
    depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise DepthDatasetViewError(f"PNG Depth 转换前格式变化或解码失败: {source}")
    gray = np.right_shift(depth, 8).astype(np.uint8)
    output = np.repeat(gray[..., None], 3, axis=2)
    success = cv2.imwrite(
        str(destination),
        output,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    )
    if not success:
        raise DepthDatasetViewError(f"compat8 PNG 写入失败: {destination}")
    decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint8
        or decoded.shape != output.shape
        or not np.array_equal(decoded, output)
    ):
        raise DepthDatasetViewError(f"compat8 PNG lossless 验证失败: {destination}")


def build_source_records(validation: Dict[str, Any], staging: Path) -> List[Dict[str, Any]]:
    subset_by_stem = {
        stem: subset
        for subset in ("train", "val")
        for stem, _, _ in validation["entries"][subset]
    }
    records = []
    for stem in sorted(validation["images"]):
        subset = subset_by_stem[stem]
        image = validation["images"][stem]
        label = validation["labels"][stem]
        staged_image_relative = Path("images") / subset / image.name
        staged_label_relative = Path("labels") / subset / f"{stem}.txt"
        staged_image = staging / staged_image_relative
        staged_label = staging / staged_label_relative
        output = cv2.imread(str(staged_image), cv2.IMREAD_UNCHANGED)
        if output is None or output.dtype != np.uint8 or output.ndim != 3 or output.shape[2] != 3:
            raise DepthDatasetViewError(
                f"compat8 输出必须是 uint8 3-channel: {staged_image}"
            )
        info = validation["source_info"][stem]
        output_sha256 = sha256_file(staged_image)
        if sha256_file(staged_label) != validation["label_hashes"][label.name]:
            raise DepthDatasetViewError(f"staged label 不是 byte-preserving copy: {staged_label}")
        if info["source_format"] == "jpg_uint8_3ch" and output_sha256 != info["source_sha256"]:
            raise DepthDatasetViewError(f"staged JPG 不是 byte-preserving copy: {staged_image}")
        records.append(
            {
                "conversion_branch": info["conversion_branch"],
                "label_bytes": label.stat().st_size,
                "label_sha256": validation["label_hashes"][label.name],
                "output_bytes": staged_image.stat().st_size,
                "output_channels": 3,
                "output_dtype": str(output.dtype),
                "output_sha256": output_sha256,
                "output_shape": [int(value) for value in output.shape],
                "physical_unit": info["physical_unit"],
                "source_bytes": info["source_bytes"],
                "source_channels": info["source_channels"],
                "source_dtype": info["source_dtype"],
                "source_extension": image.suffix.lower(),
                "source_format": info["source_format"],
                "source_image_project_relative": project_relative_path(image),
                "source_sha256": info["source_sha256"],
                "source_shape": info["source_shape"],
                "source_label_project_relative": project_relative_path(label),
                "staged_image_view_relative": staged_image_relative.as_posix(),
                "staged_label_view_relative": staged_label_relative.as_posix(),
                "stem": stem,
                "subset": subset,
            }
        )
    return records


def _aggregate_records(records: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for record in records:
        if prefix == "source":
            path_key, size_key, hash_key = (
                "source_image_project_relative",
                "source_bytes",
                "source_sha256",
            )
        elif prefix == "output":
            path_key, size_key, hash_key = (
                "staged_image_view_relative",
                "output_bytes",
                "output_sha256",
            )
        else:
            path_key, size_key, hash_key = (
                "source_label_project_relative",
                "label_bytes",
                "label_sha256",
            )
        size = record[size_key]
        file_hash = record[hash_key]
        filename = Path(record[path_key]).name
        digest.update(f"{filename}\0{size}\0{file_hash}\n".encode("utf-8"))
        total_bytes += size
    return {"aggregate_sha256": digest.hexdigest(), "file_count": len(records), "total_bytes": total_bytes}


def build_manifest(
    validation: Dict[str, Any],
    staging: Path,
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    methods: Counter,
) -> Dict[str, Any]:
    records = build_source_records(validation, staging)
    return {
        "default_isolation_policy": "copy",
        "isolation": True,
        "jpg_count": validation["jpg_count"],
        "jpg_strategy": {
            "conversion": "byte_preserving_copy",
            "output_channels": 3,
            "output_dtype": "uint8",
            "physical_unit": "unknown",
            "source_channels": 3,
            "source_dtype": "uint8",
            "source_format": "jpg_uint8_3ch",
        },
        "label_dir": project_relative_path(label_dir),
        "labels_clean_identity": _aggregate_records(records, "label"),
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "materialization_counts": dict(sorted(methods.items())),
        "output_channels": 3,
        "output_dtype": "uint8",
        "output_identity": _aggregate_records(records, "output"),
        "png_count": validation["png_count"],
        "png_strategy": {
            "conversion": "uint8 = uint16 >> 8; channels = repeat(gray, 3)",
            "invalid_zero_preserved": True,
            "output_channels": 3,
            "output_dtype": "uint8",
            "physical_semantics": "source physical depth; derived loader-compatible view",
            "source_channels": 1,
            "source_dtype": "uint16",
            "source_format": "png_uint16_1ch",
        },
        "preprocessing": "fixed_format_compatibility",
        "physical_scale_policy": "PNG and JPG DO NOT share a confirmed physical scale.",
        "representation": "compat8",
        "script_version": SCRIPT_VERSION,
        "source_file_list": records,
        "source_depth_dir": project_relative_path(depth_dir),
        "source_depth_identity": _aggregate_records(records, "source"),
        "total_count": validation["total_count"],
        "train_count": validation["train_count"],
        "train_jpg_count": validation["train_jpg_count"],
        "train_png_count": validation["train_png_count"],
        "train_split_path": project_relative_path(train_split),
        "train_split_normalized_sha256": normalized_split_sha256(train_split),
        "val_count": validation["val_count"],
        "val_jpg_count": validation["val_jpg_count"],
        "val_png_count": validation["val_png_count"],
        "val_split_path": project_relative_path(val_split),
        "val_split_normalized_sha256": normalized_split_sha256(val_split),
    }


def _write_text_lf(path: Path, text: str) -> None:
    """Write deterministic UTF-8/LF text using the Python 3.8-compatible API."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def build_view(
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    output_root: Path,
    representation: str = "compat8",
    force: bool = False,
    expected_train_count: int = EXPECTED_TRAIN_COUNT,
    expected_val_count: int = EXPECTED_VAL_COUNT,
    expected_png_count: int = EXPECTED_PNG_COUNT,
    expected_jpg_count: int = EXPECTED_JPG_COUNT,
    expected_train_png_count: int = EXPECTED_TRAIN_PNG_COUNT,
    expected_train_jpg_count: int = EXPECTED_TRAIN_JPG_COUNT,
    expected_val_png_count: int = EXPECTED_VAL_PNG_COUNT,
    expected_val_jpg_count: int = EXPECTED_VAL_JPG_COUNT,
    expected_png_shapes: Tuple[Tuple[int, ...], ...] = (EXPECTED_PNG_SHAPE,),
    expected_jpg_shapes: Tuple[Tuple[int, ...], ...] = (EXPECTED_JPG_SHAPE,),
) -> Dict[str, Any]:
    """Validate canonical inputs before atomically publishing an isolated compat8 view."""
    validate_output_path(output_root, depth_dir, label_dir, train_split, val_split)
    if representation != "compat8":
        raise DepthDatasetViewError(f"当前 C2 仅实现 representation=compat8: {representation}")
    validate_canonical_inputs(label_dir, train_split, val_split)
    if output_root.exists() and not force:
        raise DepthDatasetViewError(f"输出目录已存在；如需重建请显式使用 --force: {output_root}")

    validation = validate_sources(
        depth_dir,
        label_dir,
        train_split,
        val_split,
        expected_train_count=expected_train_count,
        expected_val_count=expected_val_count,
        expected_png_count=expected_png_count,
        expected_jpg_count=expected_jpg_count,
        expected_train_png_count=expected_train_png_count,
        expected_train_jpg_count=expected_train_jpg_count,
        expected_val_png_count=expected_val_png_count,
        expected_val_jpg_count=expected_val_jpg_count,
        expected_png_shapes=expected_png_shapes,
        expected_jpg_shapes=expected_jpg_shapes,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".compat8-", dir=str(output_root.parent)))
    methods = Counter()
    staging_is_safe = False
    try:
        _validate_temporary_staging(staging)
        staging_is_safe = True
        for subset in ("train", "val"):
            image_out = staging / "images" / subset
            label_out = staging / "labels" / subset
            image_out.mkdir(parents=True)
            label_out.mkdir(parents=True)
            for stem, image, label in validation["entries"][subset]:
                destination = image_out / f"{stem}{image.suffix.lower()}"
                if image.suffix.lower() == ".png":
                    convert_png_compat8(image, destination)
                    methods["png_compat8_conversion"] += 1
                else:
                    copy_isolated(image, destination)
                    methods["jpg_byte_preserving_copy"] += 1
                copy_isolated(label, label_out / f"{stem}.txt")
                methods["label_copy"] += 1

        dataset_yaml = {
            "path": output_root.resolve().as_posix(),
            "train": "images/train",
            "val": "images/val",
            "channels": 3,
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }
        _write_text_lf(
            staging / "data.yaml",
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        )
        manifest = build_manifest(
            validation,
            staging,
            depth_dir,
            label_dir,
            train_split,
            val_split,
            methods,
        )
        _write_text_lf(
            staging / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

        if output_root.exists():
            validate_force_target(output_root, depth_dir, label_dir, train_split, val_split)
            shutil.rmtree(output_root)
        validate_output_path(output_root, depth_dir, label_dir, train_split, val_split)
        _validate_temporary_staging(staging)
        staging.replace(output_root)
    except Exception:
        if staging_is_safe:
            _remove_temporary_staging(staging)
        raise
    return {"manifest": manifest, "methods": methods, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dir", default=CANONICAL_DEPTH_RELATIVE.as_posix())
    parser.add_argument("--label-dir", default=CANONICAL_LABEL_RELATIVE.as_posix())
    parser.add_argument("--train-split", default=CANONICAL_TRAIN_SPLIT_RELATIVE.as_posix())
    parser.add_argument("--val-split", default=CANONICAL_VAL_SPLIT_RELATIVE.as_posix())
    parser.add_argument("--output-root", default=CANONICAL_OUTPUT_RELATIVE.as_posix())
    parser.add_argument("--representation", default="compat8")
    parser.add_argument("--force", action="store_true", help="Rebuild only the canonical compat8 view.")
    return parser.parse_args()


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        output_root = project_path(args.output_root)
        result = build_view(
            project_path(args.depth_dir),
            project_path(args.label_dir),
            project_path(args.train_split),
            project_path(args.val_split),
            output_root,
            representation=args.representation,
            force=args.force,
        )
    except (DepthDatasetViewError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    manifest = result["manifest"]
    print(f"Depth YOLO compat8 兼容视图已生成: {output_root}")
    print(f"train/val: {manifest['train_count']}/{manifest['val_count']}")
    print(f"PNG/JPG: {manifest['png_count']}/{manifest['jpg_count']}")
    print(f"物化方式: {dict(result['methods'])}")
    print(f"隔离 staging: {manifest['isolation']}")
    print(f"Ultralytics data 配置: {output_root / 'data.yaml'}")
    print(f"Deterministic manifest: {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
