"""Build isolated Ultralytics views for the four frozen C3 Depth candidates."""

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
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import depth_preprocessing as c3  # noqa: E402
import prepare_depth_yolo as c2  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPTH_TRAINABLE_RELATIVE = Path("data/processed/depth_trainable")
SUPPORTED_CANDIDATES = c3.VALID_CANDIDATES
FROZEN_PERCENTILE_LOW_MM = 1638
FROZEN_PERCENTILE_HIGH_MM = 18819
FROZEN_PERCENTILE_IDENTITY_SHA256 = (
    "aab4008fb51154016d7703de6af186e39267c322b71e93e791aa28e1b646f6ab"
)
D_MIN_VALID_MM = 1
D_NEAR_MM = 300
D_FAR_MM = 19999
L_MIN = np.float32(np.log1p(np.float32(D_MIN_VALID_MM)))
L_MAX = np.float32(np.log1p(np.float32(D_FAR_MM)))
INVERSE_NEAR = np.float32(1.0 / D_NEAR_MM)
INVERSE_FAR = np.float32(1.0 / D_FAR_MM)
PNG_COMPRESSION = 3
MANIFEST_SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0"


class DepthCandidateViewError(ValueError):
    """Raised when a C4 build would violate the frozen data or safety contract."""


def project_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _normalized_absolute(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _resolve_path(path: Path) -> Path:
    return path.resolve()


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(_resolve_path(first))) == os.path.normcase(
        str(_resolve_path(second))
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        _resolve_path(path).relative_to(_resolve_path(parent))
        return True
    except ValueError:
        return False


def _paths_overlap(first: Path, second: Path) -> bool:
    left, right = _resolve_path(first), _resolve_path(second)
    if _same_path(left, right):
        return True
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _require_candidate(candidate: str) -> None:
    if candidate not in SUPPORTED_CANDIDATES:
        raise DepthCandidateViewError(f"不支持或组合的 C4 candidate: {candidate}")


def canonical_output_root(candidate: str) -> Path:
    _require_candidate(candidate)
    return PROJECT_ROOT / DEPTH_TRAINABLE_RELATIVE / candidate


def project_relative_path(path: Path) -> str:
    try:
        return _resolve_path(path).relative_to(_resolve_path(PROJECT_ROOT)).as_posix()
    except ValueError as exc:
        raise DepthCandidateViewError(f"Manifest 路径必须位于项目内: {path}") from exc


def _reject_redirection(path: Path, label: str) -> None:
    if path.exists() and (
        path.is_symlink()
        or _normalized_absolute(_resolve_path(path)) != _normalized_absolute(path)
    ):
        raise DepthCandidateViewError(f"{label} 不得是 symlink 或 junction 重定向: {path}")


def validate_output_path(
    output_root: Path,
    candidate: str,
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
) -> None:
    """Allow exactly one canonical sibling and reject redirected or overlapping paths."""
    _require_candidate(candidate)
    expected = canonical_output_root(candidate)
    if _normalized_absolute(output_root) != _normalized_absolute(expected):
        raise DepthCandidateViewError(f"输出目录必须是 candidate canonical view: {expected}")
    root = PROJECT_ROOT
    processed = root / "data/processed"
    trainable = root / DEPTH_TRAINABLE_RELATIVE
    for path, label in (
        (root, "project root"),
        (processed, "data/processed"),
        (trainable, "depth_trainable root"),
        (output_root, "candidate output"),
    ):
        _reject_redirection(path, label)
    if not _is_within(trainable, root) or not _is_within(output_root, root):
        raise DepthCandidateViewError("C4 staging resolve 后逃出 PROJECT_ROOT")
    if _resolve_path(output_root).parent != _resolve_path(trainable):
        raise DepthCandidateViewError("candidate output 父目录被重定向")
    if output_root.name != candidate or output_root.name == "compat8":
        raise DepthCandidateViewError("C4 只能管理当前四个 candidate sibling")
    for source_label, source in (
        ("raw Depth", depth_dir),
        ("labels_clean", label_dir),
        ("train split", train_split),
        ("val split", val_split),
        ("data/raw", root / "data/raw"),
    ):
        if _paths_overlap(output_root, source):
            raise DepthCandidateViewError(f"输出目录与 {source_label} 重叠: {output_root}")


def validate_force_target(
    output_root: Path,
    candidate: str,
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
) -> None:
    validate_output_path(
        output_root, candidate, depth_dir, label_dir, train_split, val_split
    )
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise DepthCandidateViewError("--force 目标必须是当前 candidate 的普通目录")


def _validate_temporary_staging(staging: Path, candidate: str) -> None:
    trainable = PROJECT_ROOT / DEPTH_TRAINABLE_RELATIVE
    _reject_redirection(trainable, "depth_trainable root")
    if (
        staging.is_symlink()
        or _resolve_path(staging).parent != _resolve_path(trainable)
        or not _is_within(staging, PROJECT_ROOT)
        or not staging.name.startswith(f".{candidate}-")
    ):
        raise DepthCandidateViewError(f"临时 staging 越出当前 candidate 安全边界: {staging}")


def _remove_temporary_staging(staging: Path, candidate: str) -> None:
    if staging.exists():
        _validate_temporary_staging(staging, candidate)
        shutil.rmtree(staging)


def validate_module_roots() -> None:
    if not _same_path(PROJECT_ROOT, c2.PROJECT_ROOT) or not _same_path(
        PROJECT_ROOT, c3.PROJECT_ROOT
    ):
        raise DepthCandidateViewError("C2/C3 与 C4 PROJECT_ROOT 不一致")


def round_half_up_nonnegative(values: np.ndarray) -> np.ndarray:
    """Round finite nonnegative values with ties toward positive infinity.

    For this nonnegative domain, ``floor(x + 0.5)`` is deterministic
    round-half-away-from-zero and deliberately avoids ties-to-even behavior.
    """
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise DepthCandidateViewError("half-up rounding 只接受 finite nonnegative values")
    return np.floor(array + np.float64(0.5))


def quantize_unit_float_to_uint8_floor(values: np.ndarray) -> np.ndarray:
    """Return floor(clip(float32(values), 0, 1) * float32(255)) as uint8."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise DepthCandidateViewError("unit quantization 只接受 finite floating array")
    unit = np.clip(array.astype(np.float32), np.float32(0.0), np.float32(1.0))
    return np.floor(unit * np.float32(255.0)).astype(np.uint8)


def quantize_inverse_depth(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply C3 reciprocal then the frozen 300..19999 mm half-up uint8 mapping."""
    inverse, valid = c3.apply_inverse_depth(depth)
    output = np.zeros(depth.shape, dtype=np.uint8)
    clipped_inverse = np.clip(inverse[valid], INVERSE_FAR, INVERSE_NEAR).astype(
        np.float64
    )
    scaled = np.float64(254.0) * (
        (clipped_inverse - np.float64(INVERSE_FAR))
        / np.float64(INVERSE_NEAR - INVERSE_FAR)
    )
    quantized = np.float64(1.0) + round_half_up_nonnegative(scaled)
    output[valid] = np.clip(quantized, 1, 255).astype(np.uint8)
    if np.any(output[valid] < 1):
        raise DepthCandidateViewError("inverse valid pixel 被错误量化为 0")
    return output, valid


def render_candidate_gray(
    depth: np.ndarray,
    candidate: str,
    percentile_params: Optional[c3.PercentileParams] = None,
) -> np.ndarray:
    """Call one C3 pure transform and produce its frozen C4 uint8 gray view."""
    _require_candidate(candidate)
    if candidate == "validmask":
        if percentile_params is not None:
            raise DepthCandidateViewError("PercentileParams 只能用于 percentile")
        return c3.make_valid_mask(depth).astype(np.uint8) * np.uint8(255)
    if candidate == "percentile":
        if percentile_params is None:
            raise DepthCandidateViewError("percentile candidate 缺少 C3 frozen params")
        values, _ = c3.apply_percentile(depth, percentile_params)
        return quantize_unit_float_to_uint8_floor(values)
    if percentile_params is not None:
        raise DepthCandidateViewError("PercentileParams 只能用于 percentile")
    if candidate == "log":
        if int(depth.max()) > D_FAR_MM:
            raise DepthCandidateViewError(
                f"canonical PNG max 超过 {D_FAR_MM} mm，拒绝静默 clip"
            )
        values, valid = c3.apply_log_depth(depth)
        unit = np.zeros(depth.shape, dtype=np.float32)
        unit[valid] = (values[valid] - L_MIN) / np.float32(L_MAX - L_MIN)
        return quantize_unit_float_to_uint8_floor(unit)
    return quantize_inverse_depth(depth)[0]


def fit_frozen_percentile_params(
    validation: Dict[str, Any], train_split: Path
) -> c3.PercentileParams:
    paths = [
        image
        for _, image, _ in validation["entries"]["train"]
        if image.suffix.lower() == ".png"
    ]
    params = c3.fit_train_png_percentiles(
        paths,
        expected_count=c2.EXPECTED_TRAIN_PNG_COUNT,
        train_split_normalized_sha256=c2.normalized_split_sha256(train_split),
    )
    if (
        params.low_value != FROZEN_PERCENTILE_LOW_MM
        or params.high_value != FROZEN_PERCENTILE_HIGH_MM
        or params.identity_sha256() != FROZEN_PERCENTILE_IDENTITY_SHA256
    ):
        raise DepthCandidateViewError(
            "C3 percentile 参数与冻结的 P2/P98/identity 不一致"
        )
    return params


def write_candidate_png(
    source: Path,
    destination: Path,
    candidate: str,
    percentile_params: Optional[c3.PercentileParams],
) -> None:
    depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
        raise DepthCandidateViewError(f"PNG 必须是 IMREAD_UNCHANGED uint16 2D: {source}")
    if int(depth.max()) > D_FAR_MM:
        raise DepthCandidateViewError(
            f"canonical PNG max 超过 {D_FAR_MM} mm: {source.name}"
        )
    gray = render_candidate_gray(depth, candidate, percentile_params)
    output = np.repeat(gray[..., None], 3, axis=2)
    if output.shape[:2] != depth.shape:
        raise DepthCandidateViewError(f"C4 不得改变 PNG geometry: {source}")
    if not cv2.imwrite(
        str(destination), output, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
    ):
        raise DepthCandidateViewError(f"candidate PNG 写入失败: {destination}")
    decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
    if (
        decoded is None
        or decoded.dtype != np.uint8
        or decoded.shape != output.shape
        or not np.array_equal(decoded, output)
        or not np.array_equal(decoded[..., 0], decoded[..., 1])
        or not np.array_equal(decoded[..., 1], decoded[..., 2])
    ):
        raise DepthCandidateViewError(f"candidate PNG 写盘回读验证失败: {destination}")


def copy_isolated(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_records(records: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    keys = {
        "source": ("source_image_project_relative", "source_bytes", "source_sha256"),
        "output": ("staged_image_view_relative", "output_bytes", "output_sha256"),
        "label": ("source_label_project_relative", "label_bytes", "label_sha256"),
    }
    path_key, size_key, hash_key = keys[prefix]
    digest, total = hashlib.sha256(), 0
    for record in records:
        size, file_hash = record[size_key], record[hash_key]
        digest.update(
            f"{Path(record[path_key]).name}\0{size}\0{file_hash}\n".encode("utf-8")
        )
        total += size
    return {
        "aggregate_sha256": digest.hexdigest(),
        "file_count": len(records),
        "total_bytes": total,
    }


def candidate_conversion_metadata(
    candidate: str, percentile_params: Optional[c3.PercentileParams]
) -> Dict[str, Any]:
    common = {
        "channel_mapping": "repeat_gray_x3",
        "input_dtype": "uint16",
        "input_unit": "mm",
        "invalid_value": 0,
        "output_channels": 3,
        "output_dtype": "uint8",
        "valid_rule": "depth_mm > 0",
    }
    if candidate == "validmask":
        return dict(
            common,
            formula="depth_mm > 0",
            valid_value=255,
            representation="validmask",
        )
    if candidate == "percentile":
        if percentile_params is None:
            raise DepthCandidateViewError("percentile manifest 缺少 params")
        return dict(
            common,
            c3_parameter_identity_sha256=percentile_params.identity_sha256(),
            float_range=[0.0, 1.0],
            formula="C3 P2/P98 clip and normalize",
            high_mm=percentile_params.high_value,
            low_mm=percentile_params.low_value,
            p_high=percentile_params.p_high,
            p_low=percentile_params.p_low,
            quantization="floor(clip(x,0,1)*255)",
            representation="percentile",
        )
    if candidate == "log":
        return dict(
            common,
            formula="log1p(depth_mm)",
            mapping_max=float(L_MAX),
            mapping_min=float(L_MIN),
            model_range_max_mm=D_FAR_MM,
            model_range_min_mm=D_MIN_VALID_MM,
            quantization="floor(clip((log-L_MIN)/(L_MAX-L_MIN),0,1)*255)",
            representation="log",
        )
    return dict(
        common,
        custom_loader=False,
        far_clip_mm=D_FAR_MM,
        formula="reciprocal_depth_fixed_near_far_mapping",
        log_used=False,
        near_clip_mm=D_NEAR_MM,
        percentile_used=False,
        representation="inverse",
        rounding_rule="round-half-away-from-zero for nonnegative values via floor(x+0.5)",
        train_dependent_statistics=False,
        valid_output_range=[1, 255],
    )


def build_source_records(
    validation: Dict[str, Any], staging: Path, candidate: str
) -> List[Dict[str, Any]]:
    subset_by_stem = {
        stem: subset
        for subset in ("train", "val")
        for stem, _, _ in validation["entries"][subset]
    }
    records = []
    for stem in sorted(validation["images"]):
        subset = subset_by_stem[stem]
        image, label = validation["images"][stem], validation["labels"][stem]
        image_relative = Path("images") / subset / f"{stem}{image.suffix.lower()}"
        label_relative = Path("labels") / subset / f"{stem}.txt"
        output_image, output_label = staging / image_relative, staging / label_relative
        decoded = cv2.imread(str(output_image), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.dtype != np.uint8 or decoded.ndim != 3 or decoded.shape[2] != 3:
            raise DepthCandidateViewError(f"C4 输出不是 uint8 HxWx3: {output_image}")
        info = validation["source_info"][stem]
        output_hash = sha256_file(output_image)
        label_hash = sha256_file(output_label)
        if label_hash != validation["label_hashes"][label.name]:
            raise DepthCandidateViewError(f"staged label 不是 byte-preserving copy: {output_label}")
        if info["source_format"] == "jpg_uint8_3ch" and output_hash != info["source_sha256"]:
            raise DepthCandidateViewError(f"staged JPG 不是 byte-preserving copy: {output_image}")
        records.append(
            {
                "candidate": candidate,
                "conversion": (
                    "byte_preserving_passthrough"
                    if image.suffix.lower() in {".jpg", ".jpeg"}
                    else f"c3_{candidate}_to_c4_uint8_x3"
                ),
                "label_bytes": label.stat().st_size,
                "label_sha256": label_hash,
                "output_bytes": output_image.stat().st_size,
                "output_channels": 3,
                "output_dtype": str(decoded.dtype),
                "output_sha256": output_hash,
                "output_shape": [int(value) for value in decoded.shape],
                "physical_unit": (
                    "unknown" if image.suffix.lower() in {".jpg", ".jpeg"} else "mm"
                ),
                "source_bytes": info["source_bytes"],
                "source_dtype": info["source_dtype"],
                "source_format": info["source_format"],
                "source_image_project_relative": project_relative_path(image),
                "source_label_project_relative": project_relative_path(label),
                "source_sha256": info["source_sha256"],
                "source_shape": info["source_shape"],
                "staged_image_view_relative": image_relative.as_posix(),
                "staged_label_view_relative": label_relative.as_posix(),
                "stem": stem,
                "subset": subset,
            }
        )
    return records


def build_manifest(
    validation: Dict[str, Any],
    staging: Path,
    candidate: str,
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    methods: Counter,
    percentile_params: Optional[c3.PercentileParams],
) -> Dict[str, Any]:
    records = build_source_records(validation, staging, candidate)
    return {
        "candidate": candidate,
        "c3_metadata": c3.candidate_metadata(candidate, percentile_params),
        "conversion": candidate_conversion_metadata(candidate, percentile_params),
        "geometry_policy": "pixel_values_only; preserve source width and height; labels byte-preserving",
        "jpg_policy": {
            "conversion": "byte_preserving_passthrough",
            "output_channels": 3,
            "output_dtype": "uint8",
            "physical_unit": "unknown",
        },
        "label_dir": project_relative_path(label_dir),
        "labels_clean_identity": _aggregate_records(records, "label"),
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "materialization_counts": dict(sorted(methods.items())),
        "output_channels": 3,
        "output_dtype": "uint8",
        "output_identity": _aggregate_records(records, "output"),
        "representation": candidate,
        "script": "scripts/data/prepare_depth_candidate_yolo.py",
        "script_version": SCRIPT_VERSION,
        "source_depth_dir": project_relative_path(depth_dir),
        "source_depth_identity": _aggregate_records(records, "source"),
        "source_file_list": records,
        "total_count": validation["total_count"],
        "train_count": validation["train_count"],
        "train_jpg_count": validation["train_jpg_count"],
        "train_png_count": validation["train_png_count"],
        "train_split_normalized_sha256": c2.normalized_split_sha256(train_split),
        "train_split_path": project_relative_path(train_split),
        "val_count": validation["val_count"],
        "val_jpg_count": validation["val_jpg_count"],
        "val_png_count": validation["val_png_count"],
        "val_split_normalized_sha256": c2.normalized_split_sha256(val_split),
        "val_split_path": project_relative_path(val_split),
    }


def _write_text_lf(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def build_view(
    depth_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    output_root: Path,
    candidate: str,
    force: bool = False,
    expected_train_count: int = c2.EXPECTED_TRAIN_COUNT,
    expected_val_count: int = c2.EXPECTED_VAL_COUNT,
    expected_png_count: int = c2.EXPECTED_PNG_COUNT,
    expected_jpg_count: int = c2.EXPECTED_JPG_COUNT,
    expected_train_png_count: int = c2.EXPECTED_TRAIN_PNG_COUNT,
    expected_train_jpg_count: int = c2.EXPECTED_TRAIN_JPG_COUNT,
    expected_val_png_count: int = c2.EXPECTED_VAL_PNG_COUNT,
    expected_val_jpg_count: int = c2.EXPECTED_VAL_JPG_COUNT,
    expected_png_shapes: Tuple[Tuple[int, ...], ...] = (c2.EXPECTED_PNG_SHAPE,),
    expected_jpg_shapes: Tuple[Tuple[int, ...], ...] = (c2.EXPECTED_JPG_SHAPE,),
) -> Dict[str, Any]:
    """Validate canonical inputs, build one temp sibling, then publish it."""
    validate_module_roots()
    validate_output_path(
        output_root, candidate, depth_dir, label_dir, train_split, val_split
    )
    if output_root.exists() and not force:
        raise DepthCandidateViewError(f"输出已存在；重建当前 view 需显式 --force: {output_root}")
    try:
        validation = c2.validate_sources(
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
    except c2.DepthDatasetViewError as exc:
        raise DepthCandidateViewError(str(exc)) from exc
    percentile_params = (
        fit_frozen_percentile_params(validation, train_split)
        if candidate == "percentile"
        else None
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{candidate}-", dir=str(output_root.parent))
    )
    methods, staging_is_safe = Counter(), False
    try:
        _validate_temporary_staging(staging, candidate)
        staging_is_safe = True
        for subset in ("train", "val"):
            image_out, label_out = staging / "images" / subset, staging / "labels" / subset
            image_out.mkdir(parents=True)
            label_out.mkdir(parents=True)
            for stem, image, label in validation["entries"][subset]:
                destination = image_out / f"{stem}{image.suffix.lower()}"
                if image.suffix.lower() == ".png":
                    write_candidate_png(
                        image, destination, candidate, percentile_params
                    )
                    methods[f"png_{candidate}_conversion"] += 1
                else:
                    copy_isolated(image, destination)
                    methods["jpg_byte_preserving_passthrough"] += 1
                copy_isolated(label, label_out / f"{stem}.txt")
                methods["label_byte_preserving_copy"] += 1
        dataset_yaml = {
            "train": "images/train",
            "val": "images/val",
            "channels": 3,
            "names": {index: name for index, name in enumerate(c2.CLASS_NAMES)},
        }
        _write_text_lf(
            staging / "data.yaml",
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        )
        manifest = build_manifest(
            validation,
            staging,
            candidate,
            depth_dir,
            label_dir,
            train_split,
            val_split,
            methods,
            percentile_params,
        )
        _write_text_lf(
            staging / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if output_root.exists():
            validate_force_target(
                output_root, candidate, depth_dir, label_dir, train_split, val_split
            )
            shutil.rmtree(output_root)
        validate_output_path(
            output_root, candidate, depth_dir, label_dir, train_split, val_split
        )
        _validate_temporary_staging(staging, candidate)
        staging.replace(output_root)
    except Exception:
        if staging_is_safe:
            _remove_temporary_staging(staging, candidate)
        raise
    return {
        "manifest": manifest,
        "methods": methods,
        "percentile_params": percentile_params,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dir", default=c2.CANONICAL_DEPTH_RELATIVE.as_posix())
    parser.add_argument("--label-dir", default=c2.CANONICAL_LABEL_RELATIVE.as_posix())
    parser.add_argument(
        "--train-split", default=c2.CANONICAL_TRAIN_SPLIT_RELATIVE.as_posix()
    )
    parser.add_argument("--val-split", default=c2.CANONICAL_VAL_SPLIT_RELATIVE.as_posix())
    parser.add_argument("--output-root")
    parser.add_argument("--candidate", required=True, choices=SUPPORTED_CANDIDATES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_console_encoding() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    candidate = args.candidate
    output_root = (
        project_path(args.output_root)
        if args.output_root
        else canonical_output_root(candidate)
    )
    result = build_view(
        project_path(args.depth_dir),
        project_path(args.label_dir),
        project_path(args.train_split),
        project_path(args.val_split),
        output_root,
        candidate,
        force=args.force,
    )
    manifest = result["manifest"]
    print(f"C4 {candidate} view 已生成: {output_root}")
    print(f"train/val: {manifest['train_count']}/{manifest['val_count']}")
    jpg_count = manifest["train_jpg_count"] + manifest["val_jpg_count"]
    print(f"PNG/JPG: {manifest['total_count'] - jpg_count}/{jpg_count}")
    print(f"data.yaml: {output_root / 'data.yaml'}")
    print(f"manifest.json: {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
