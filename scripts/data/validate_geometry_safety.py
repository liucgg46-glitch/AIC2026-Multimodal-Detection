"""Validate that C1-C4 staging preserves the Visible supervision geometry."""

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

import cv2
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
RAW_TRAIN = PROJECT_ROOT / "data" / "raw" / "train"
LABELS_CLEAN = PROJECT_ROOT / "data" / "processed" / "train" / "labels_clean"
TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.txt"
VAL_SPLIT = PROJECT_ROOT / "data" / "splits" / "val.txt"
JSON_OUTPUT = PROJECT_ROOT / "outputs" / "analysis" / "modality_geometry_safety.json"
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_TRAIN_SHA256 = "20b0c1fb09a6848a4700a5adc8ce1f9a6d9040a48626a1759020d7f986450969"
EXPECTED_VAL_SHA256 = "336165b3509b6a0516b052c02fd98d152441300e8ae757eb0d27ca68a053ee48"


class GeometrySafetyError(RuntimeError):
    """Raised when the canonical geometry safety contract fails."""


class ViewSpec(NamedTuple):
    view: str
    modality: str
    source_dir: Path
    staging_dir: Path


def canonical_views(project_root: Path = PROJECT_ROOT) -> List[ViewSpec]:
    """Return the six reviewed staging views without inspecting model code."""
    processed = project_root / "data" / "processed"
    depth_root = processed / "depth_trainable"
    return [
        ViewSpec("ir_raw3", "infrared", project_root / "data/raw/train/infrared", processed / "ir_trainable/raw3"),
        ViewSpec("depth_compat8", "depth", project_root / "data/raw/train/depth", depth_root / "compat8"),
        ViewSpec("depth_validmask", "depth", project_root / "data/raw/train/depth", depth_root / "validmask"),
        ViewSpec("depth_percentile", "depth", project_root / "data/raw/train/depth", depth_root / "percentile"),
        ViewSpec("depth_log", "depth", project_root / "data/raw/train/depth", depth_root / "log"),
        ViewSpec("depth_inverse", "depth", project_root / "data/raw/train/depth", depth_root / "inverse"),
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GeometrySafetyError(message)


def normalize_split_content(content: bytes) -> bytes:
    """Match the project's frozen logical-content split hash rule."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GeometrySafetyError("Split 必须是 UTF-8 文本") from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if text.endswith("\n"):
        lines = lines[:-1]
    return ("\n".join(lines) + "\n").encode("utf-8")


def normalized_split_sha256(path: Path) -> str:
    return hashlib.sha256(normalize_split_content(path.read_bytes())).hexdigest()


def read_split(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8-sig")
    stems = [line.strip() for line in text.splitlines() if line.strip()]
    require(all(Path(stem).name == stem and Path(stem).suffix == "" for stem in stems), "Split 必须只包含 stem")
    return stems


def index_files(directory: Path, extensions: Set[str]) -> Dict[str, Any]:
    """Index exact stems while retaining duplicate and case-fold collision evidence."""
    require(directory.is_dir(), "目录不存在: {}".format(directory))
    exact: Dict[str, List[Path]] = {}
    folded: Dict[str, Set[str]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
        if path.is_file() and path.suffix.lower() in extensions:
            exact.setdefault(path.stem, []).append(path)
            folded.setdefault(path.stem.casefold(), set()).add(path.stem)
    return {
        "exact": exact,
        "file_count": sum(len(paths) for paths in exact.values()),
        "duplicate_count": sum(max(0, len(paths) - 1) for paths in exact.values()),
        "casefold_collision_count": sum(1 for stems in folded.values() if len(stems) > 1),
    }


def decode_spatial_shape(path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise GeometrySafetyError("图像解码失败: {}".format(path))
    return int(image.shape[0]), int(image.shape[1])


def shape_summary(shapes: Iterable[Tuple[int, int]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for height, width in shapes:
        key = "{}x{}".format(height, width)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def casefold_collision_count(stems: Iterable[str]) -> int:
    groups: Dict[str, Set[str]] = {}
    for stem in stems:
        groups.setdefault(stem.casefold(), set()).add(stem)
    return sum(1 for spellings in groups.values() if len(spellings) > 1)


def _single(index: Dict[str, Any], stem: str) -> Optional[Path]:
    paths = index["exact"].get(stem, [])
    return paths[0] if len(paths) == 1 else None


def validate_view(
    spec: ViewSpec,
    train_stems: Sequence[str],
    val_stems: Sequence[str],
    labels_clean: Path,
) -> Dict[str, Any]:
    """Directly decode source/staged files and compare canonical label bytes."""
    source = index_files(spec.source_dir, IMAGE_EXTENSIONS)
    canonical_labels = index_files(labels_clean, {".txt"})
    images = {
        "train": index_files(spec.staging_dir / "images/train", IMAGE_EXTENSIONS),
        "val": index_files(spec.staging_dir / "images/val", IMAGE_EXTENSIONS),
    }
    labels = {
        "train": index_files(spec.staging_dir / "labels/train", {".txt"}),
        "val": index_files(spec.staging_dir / "labels/val", {".txt"}),
    }
    expected_by_subset = {"train": set(train_stems), "val": set(val_stems)}
    expected_all = expected_by_subset["train"] | expected_by_subset["val"]

    missing_samples: Set[Tuple[str, str]] = set()
    extra_paths = 0
    split_mismatch = 0
    dimension_mismatch = 0
    label_mismatch = 0
    source_shapes: List[Tuple[int, int]] = []
    staged_shapes: List[Tuple[int, int]] = []

    source_keys = set(source["exact"])
    canonical_label_keys = set(canonical_labels["exact"])
    extra_paths += sum(len(source["exact"][stem]) for stem in source_keys - expected_all)
    extra_paths += sum(len(canonical_labels["exact"][stem]) for stem in canonical_label_keys - expected_all)

    for subset in ("train", "val"):
        expected = expected_by_subset[subset]
        image_keys = set(images[subset]["exact"])
        label_keys = set(labels[subset]["exact"])
        extra_paths += sum(len(images[subset]["exact"][stem]) for stem in image_keys - expected)
        extra_paths += sum(len(labels[subset]["exact"][stem]) for stem in label_keys - expected)
        split_mismatch += len(expected ^ image_keys) + len(expected ^ label_keys)

        for stem in sorted(expected):
            source_path = _single(source, stem)
            staged_path = _single(images[subset], stem)
            canonical_label = _single(canonical_labels, stem)
            staged_label = _single(labels[subset], stem)
            if None in (source_path, staged_path, canonical_label, staged_label):
                missing_samples.add((subset, stem))
                continue
            source_shape = decode_spatial_shape(source_path)  # type: ignore[arg-type]
            staged_shape = decode_spatial_shape(staged_path)  # type: ignore[arg-type]
            source_shapes.append(source_shape)
            staged_shapes.append(staged_shape)
            if source_shape != staged_shape:
                dimension_mismatch += 1
            if canonical_label.read_bytes() != staged_label.read_bytes():  # type: ignore[union-attr]
                label_mismatch += 1

    actual_train = set(images["train"]["exact"]) | set(labels["train"]["exact"])
    actual_val = set(images["val"]["exact"]) | set(labels["val"]["exact"])
    overlap_count = len(actual_train & actual_val)
    duplicate_count = source["duplicate_count"] + canonical_labels["duplicate_count"]
    casefold_count = source["casefold_collision_count"] + canonical_labels["casefold_collision_count"]
    for subset in ("train", "val"):
        duplicate_count += images[subset]["duplicate_count"] + labels[subset]["duplicate_count"]
        casefold_count += images[subset]["casefold_collision_count"] + labels[subset]["casefold_collision_count"]
    casefold_count += casefold_collision_count(actual_train | actual_val)

    result = {
        "view": spec.view,
        "modality": spec.modality,
        "sample_count": images["train"]["file_count"] + images["val"]["file_count"],
        "train_count": images["train"]["file_count"],
        "val_count": images["val"]["file_count"],
        "dimension_mismatch_count": dimension_mismatch,
        "label_mismatch_count": label_mismatch,
        "missing_count": len(missing_samples),
        "extra_count": extra_paths,
        "duplicate_count": duplicate_count,
        "casefold_collision_count": casefold_count,
        "split_mismatch_count": split_mismatch,
        "train_val_overlap_count": overlap_count,
        "source_shape_summary": shape_summary(source_shapes),
        "staged_shape_summary": shape_summary(staged_shapes),
    }
    result["geometry_pass"] = all(
        result[key] == 0
        for key in (
            "dimension_mismatch_count",
            "label_mismatch_count",
            "missing_count",
            "extra_count",
            "duplicate_count",
            "casefold_collision_count",
            "split_mismatch_count",
            "train_val_overlap_count",
        )
    ) and result["sample_count"] == len(expected_all)
    return result


def validate_yolo_labels(label_dir: Path, class_count: int = 12) -> Dict[str, int]:
    """Apply the existing DATA_SPEC error/warning boundary without rewriting labels."""
    index = index_files(label_dir, {".txt"})
    row_count = 0
    empty_count = 0
    validity_errors = 0
    edge_warnings = 0
    for stem in sorted(index["exact"]):
        paths = index["exact"][stem]
        if len(paths) != 1:
            validity_errors += 1
            continue
        lines = [line.strip() for line in paths[0].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            empty_count += 1
        for line in lines:
            row_count += 1
            fields = line.split()
            if len(fields) != 5:
                validity_errors += 1
                continue
            try:
                class_value = float(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError:
                validity_errors += 1
                continue
            values = (class_value, x, y, width, height)
            if not all(math.isfinite(value) for value in values):
                validity_errors += 1
                continue
            if not class_value.is_integer() or not 0 <= int(class_value) < class_count:
                validity_errors += 1
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                validity_errors += 1
                continue
            if (
                x - width / 2.0 < 0.0
                or x + width / 2.0 > 1.0
                or y - height / 2.0 < 0.0
                or y + height / 2.0 > 1.0
            ):
                edge_warnings += 1
    return {
        "label_file_count": index["file_count"],
        "bbox_row_count": row_count,
        "empty_label_count": empty_count,
        "bbox_validity_error_count": validity_errors,
        "bbox_edge_warning_count": edge_warnings,
    }


def inspect_exif_orientation(directory: Path) -> Dict[str, int]:
    files = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}),
        key=lambda path: (path.name.casefold(), path.name),
    )
    missing = 0
    default = 0
    nondefault = 0
    for path in files:
        with Image.open(path) as image:
            orientation = image.getexif().get(274)
        if orientation is None:
            missing += 1
        elif orientation == 1:
            default += 1
        else:
            nondefault += 1
    return {
        "jpg_count": len(files),
        "orientation_missing_count": missing,
        "orientation_default_count": default,
        "orientation_nondefault_count": nondefault,
    }


def git_head_sha(project_root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project_root), check=True, capture_output=True, text=True
    )
    sha = result.stdout.strip()
    require(len(sha) == 40 and all(char in "0123456789abcdef" for char in sha), "Git HEAD 不是完整 SHA")
    return sha


def source_snapshot(paths: Sequence[Path]) -> Dict[str, Tuple[int, int]]:
    snapshot: Dict[str, Tuple[int, int]] = {}
    for base in paths:
        require(base.exists(), "受保护路径不存在: {}".format(base))
        entries = [base] if base.is_file() else sorted(path for path in base.rglob("*") if path.is_file())
        for path in entries:
            stat = path.stat()
            snapshot[str(path.resolve())] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def python38_static_compatibility() -> str:
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH), feature_version=(3, 8))
    return "passed"


def write_deterministic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def run_validation(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    labels_clean = project_root / "data/processed/train/labels_clean"
    train_split = project_root / "data/splits/train.txt"
    val_split = project_root / "data/splits/val.txt"
    train_stems = read_split(train_split)
    val_stems = read_split(val_split)
    train_sha = normalized_split_sha256(train_split)
    val_sha = normalized_split_sha256(val_split)
    split_overlap = len(set(train_stems) & set(val_stems))
    split_duplicates = len(train_stems) - len(set(train_stems)) + len(val_stems) - len(set(val_stems))
    split_casefold_collisions = casefold_collision_count(train_stems + val_stems)
    require(train_sha == EXPECTED_TRAIN_SHA256, "train split normalized SHA256 不一致")
    require(val_sha == EXPECTED_VAL_SHA256, "val split normalized SHA256 不一致")
    require(len(train_stems) == 1600 and len(val_stems) == 400, "fixed split 数量不一致")
    require(
        split_overlap == 0 and split_duplicates == 0 and split_casefold_collisions == 0,
        "fixed split overlap/duplicate/casefold collision 不为 0",
    )

    protected = {
        "data_raw_unchanged": project_root / "data/raw",
        "labels_clean_unchanged": labels_clean,
        "train_split_unchanged": train_split,
        "val_split_unchanged": val_split,
    }
    before = {name: source_snapshot([path]) for name, path in protected.items()}
    views = [validate_view(spec, train_stems, val_stems, labels_clean) for spec in canonical_views(project_root)]
    bbox = validate_yolo_labels(labels_clean)
    exif = {
        "infrared_jpg": inspect_exif_orientation(project_root / "data/raw/train/infrared"),
        "depth_jpg": inspect_exif_orientation(project_root / "data/raw/train/depth"),
    }
    after = {name: source_snapshot([path]) for name, path in protected.items()}
    source_safety = {name: before[name] == after[name] for name in protected}
    source_unchanged = all(source_safety.values())

    payload: Dict[str, Any] = {
        "schema": "aic2026.modality_geometry_safety.v1",
        "git_head_sha": git_head_sha(project_root),
        "supervision_frame": "visible",
        "staging_spatial_transform_applied": False,
        "labels_transformed": False,
        "ir_registration_guaranteed": False,
        "depth_registration_guaranteed": False,
        "ir_risk": "input-supervision residual misalignment",
        "depth_risk": "input-supervision residual misalignment",
        "split_identity": {
            "train_count": len(train_stems),
            "val_count": len(val_stems),
            "train_normalized_sha256": train_sha,
            "val_normalized_sha256": val_sha,
            "duplicate_count": split_duplicates,
            "casefold_collision_count": split_casefold_collisions,
            "train_val_overlap_count": split_overlap,
        },
        "bbox_summary": bbox,
        "exif_orientation": exif,
        "views": views,
        "source_safety": source_safety,
        "python_3_8_static_syntax": python38_static_compatibility(),
    }
    require(all(view["geometry_pass"] for view in views), "至少一个正式 view 未通过 geometry contract")
    require(bbox["bbox_validity_error_count"] == 0, "labels_clean 存在 validity error")
    require(source_unchanged, "验证期间受保护 source 发生变化")
    return payload


def main() -> None:
    payload = run_validation()
    write_deterministic_json(JSON_OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
