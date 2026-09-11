"""Build the canonical, isolated Ultralytics IR-only raw3 dataset view."""

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
CANONICAL_IR_RELATIVE = Path("data/raw/train/infrared")
CANONICAL_LABEL_RELATIVE = Path("data/processed/train/labels_clean")
CANONICAL_TRAIN_SPLIT_RELATIVE = Path("data/splits/train.txt")
CANONICAL_VAL_SPLIT_RELATIVE = Path("data/splits/val.txt")
CANONICAL_OUTPUT_RELATIVE = Path("data/processed/ir_trainable/raw3")
CANONICAL_IR_TRAIN_PARENT_RELATIVE = Path("data/raw/train")
IR_TRAINABLE_ROOT_RELATIVE = Path("data/processed/ir_trainable")

# SHA-256 of normalized logical content, intentionally independent of CRLF/LF.
CANONICAL_TRAIN_NORMALIZED_SHA256 = "20b0c1fb09a6848a4700a5adc8ce1f9a6d9040a48626a1759020d7f986450969"
CANONICAL_VAL_NORMALIZED_SHA256 = "336165b3509b6a0516b052c02fd98d152441300e8ae757eb0d27ca68a053ee48"
CANONICAL_LABELS_CLEAN_SHA256 = "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"

IR_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_TRAIN_COUNT = 1600
EXPECTED_VAL_COUNT = 400
MANIFEST_SCHEMA_VERSION = 3
SCRIPT_VERSION = "1.2"

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


class IRDatasetViewError(ValueError):
    """Raised when the canonical IR dataset contract is invalid."""


def canonical_paths() -> Dict[str, Path]:
    """Construct trusted lexical paths from the script-derived project root."""
    root = _resolve_path(PROJECT_ROOT)
    return {
        "project_root": root,
        "data": root / "data",
        "data_processed": root / "data/processed",
        "raw_root": root / "data/raw",
        "ir_train_parent": root / CANONICAL_IR_TRAIN_PARENT_RELATIVE,
        "ir_dir": root / CANONICAL_IR_RELATIVE,
        "label_dir": root / CANONICAL_LABEL_RELATIVE,
        "train_split": root / CANONICAL_TRAIN_SPLIT_RELATIVE,
        "val_split": root / CANONICAL_VAL_SPLIT_RELATIVE,
        "ir_trainable_root": root / IR_TRAINABLE_ROOT_RELATIVE,
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
        raise IRDatasetViewError(f"{label} 必须使用项目 canonical 路径: {expected}")
    if not _same_path(candidate, expected):
        raise IRDatasetViewError(f"{label} resolve 后不是项目 canonical 路径: {expected}")


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
            raise IRDatasetViewError("Split 必须是 UTF-8 文本") from exc
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
        raise IRDatasetViewError(f"Manifest 身份路径必须位于项目根目录: {path}") from exc


def validate_canonical_inputs(label_dir: Path, train_split: Path, val_split: Path) -> None:
    paths = canonical_paths()
    _require_canonical_path(label_dir, paths["label_dir"], "label-dir")
    _require_canonical_path(train_split, paths["train_split"], "train split")
    _require_canonical_path(val_split, paths["val_split"], "val split")
    if (
        not train_split.is_file()
        or normalized_split_sha256(train_split) != CANONICAL_TRAIN_NORMALIZED_SHA256
    ):
        raise IRDatasetViewError(
            "canonical train split normalized logical-content SHA-256 不一致；"
            "拒绝使用被替换或篡改的 split"
        )
    if (
        not val_split.is_file()
        or normalized_split_sha256(val_split) != CANONICAL_VAL_NORMALIZED_SHA256
    ):
        raise IRDatasetViewError(
            "canonical val split normalized logical-content SHA-256 不一致；"
            "拒绝使用被替换或篡改的 split"
        )


def validate_staging_boundary(output_root: Path) -> None:
    """Anchor the writable staging boundary at PROJECT_ROOT plus fixed relative paths."""
    paths = canonical_paths()
    project_root = _resolve_path(paths["project_root"])
    expected_output = paths["output_root"]
    if _normalized_absolute(output_root) != _normalized_absolute(expected_output):
        raise IRDatasetViewError(f"输出目录必须是 canonical raw3 view: {expected_output}")

    critical_paths = (
        paths["project_root"],
        paths["data"],
        paths["data_processed"],
        paths["ir_trainable_root"],
        paths["output_root"],
    )
    for path in critical_paths:
        if not _is_within(path, project_root):
            raise IRDatasetViewError(
                f"canonical staging 关键路径 resolve 后逃出 PROJECT_ROOT: {path}"
            )

    resolved_staging_root = _resolve_path(paths["ir_trainable_root"])
    resolved_output = _resolve_path(output_root)
    if resolved_output.parent != resolved_staging_root:
        raise IRDatasetViewError(
            "canonical raw3 view 的父目录被 symlink 或 junction 重定向"
        )
    expected_resolved_location = resolved_staging_root / "raw3"
    if _normalized_absolute(resolved_output) != _normalized_absolute(expected_resolved_location):
        raise IRDatasetViewError("canonical raw3 view 不得通过 symlink 或 junction 重定向")
    if output_root.is_symlink():
        raise IRDatasetViewError("canonical raw3 view 不得是 symlink 或重定向路径")


def validate_output_path(
    output_root: Path,
    ir_dir: Path,
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
        "ir_trainable root": paths["ir_trainable_root"],
    }
    for label, forbidden in exact_forbidden.items():
        if _same_path(output_root, forbidden):
            raise IRDatasetViewError(f"输出目录禁止指向 {label}: {output_root}")

    overlap_sources = {
        "raw IR source": ir_dir,
        "labels_clean": label_dir,
        "train split": train_split,
        "val split": val_split,
        "data/raw": paths["raw_root"],
        "canonical source parent": paths["ir_train_parent"],
    }
    for label, source in overlap_sources.items():
        if _paths_overlap(output_root, source):
            raise IRDatasetViewError(f"输出目录与受保护的 {label} 路径重叠: {output_root}")

def validate_force_target(
    output_root: Path,
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
) -> None:
    """Revalidate the single deletable view immediately before --force removal."""
    validate_output_path(output_root, ir_dir, label_dir, train_split, val_split)
    expected = PROJECT_ROOT / CANONICAL_OUTPUT_RELATIVE
    if _normalized_absolute(output_root) != _normalized_absolute(expected):
        raise IRDatasetViewError("--force 只能重建 canonical raw3 view")
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise IRDatasetViewError("--force 目标必须是非 symlink 的 canonical raw3 目录")


def _validate_temporary_staging(staging: Path) -> None:
    validate_staging_boundary(canonical_paths()["output_root"])
    paths = canonical_paths()
    project_root = _resolve_path(paths["project_root"])
    allowed = _resolve_path(paths["ir_trainable_root"])
    resolved_staging = _resolve_path(staging)
    if (
        staging.is_symlink()
        or resolved_staging.parent != allowed
        or not _is_within(resolved_staging, project_root)
    ):
        raise IRDatasetViewError(f"临时 staging 越出 canonical ir_trainable 根目录: {staging}")
    if not staging.name.startswith(".raw3-"):
        raise IRDatasetViewError(f"拒绝清理非 raw3 临时目录: {staging}")


def _remove_temporary_staging(staging: Path) -> None:
    if staging.exists():
        _validate_temporary_staging(staging)
        shutil.rmtree(staging)


def read_stems(split_path: Path, expected_count: int) -> List[str]:
    if not split_path.is_file():
        raise IRDatasetViewError(f"Split 文件不存在: {split_path}")
    stems = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not stems:
        raise IRDatasetViewError(f"Split 文件为空，拒绝回退到全部训练数据: {split_path}")
    invalid = [stem for stem in stems if Path(stem).name != stem or Path(stem).suffix]
    if invalid:
        raise IRDatasetViewError(f"Split 每行必须是无目录、无扩展名的 stem；非法值: {invalid[:5]}")
    folded = Counter(stem.casefold() for stem in stems)
    duplicates = sorted(stem for stem in stems if folded[stem.casefold()] > 1)
    if duplicates:
        raise IRDatasetViewError(f"Split 包含重复或仅大小写不同的 stem: {duplicates[:5]}")
    if len(stems) != expected_count:
        raise IRDatasetViewError(
            f"Split 数量错误: {split_path}，期望 {expected_count}，实际 {len(stems)}"
        )
    return stems


def _index_unique_files(directory: Path, extensions: Set[str], kind: str) -> Dict[str, Path]:
    if not directory.is_dir():
        raise IRDatasetViewError(f"{kind} 目录不存在: {directory}")
    by_identity: Dict[str, List[Path]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
        if path.is_file() and path.suffix.lower() in extensions:
            by_identity.setdefault(path.stem.casefold(), []).append(path)
    ambiguous = {identity: paths for identity, paths in by_identity.items() if len(paths) > 1}
    if ambiguous:
        _, paths = next(iter(sorted(ambiguous.items())))
        raise IRDatasetViewError(
            f"同一个 casefold stem 匹配到多个 {kind} 文件: {[path.name for path in paths]}"
        )
    return {paths[0].stem: paths[0] for paths in by_identity.values()}


def index_ir_images(ir_dir: Path) -> Dict[str, Path]:
    return _index_unique_files(ir_dir, IR_IMAGE_EXTENSIONS, "IR 图像")


def index_labels(label_dir: Path) -> Dict[str, Path]:
    return _index_unique_files(label_dir, {".txt"}, "labels_clean")


def _set_mismatch_message(kind: str, expected: Set[str], actual: Set[str]) -> Optional[str]:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return None
    return f"{kind} 与 fixed split 不一致；缺失: {missing[:5]}；split 外额外 stem: {extra[:5]}"


def validate_ir_encoding(image_paths: List[Path]) -> None:
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IRDatasetViewError(f"IR 图像解码失败: {path}")
        if image.dtype != np.uint8:
            raise IRDatasetViewError(f"raw3 要求 uint8 IR；{path.name} 实际为 {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise IRDatasetViewError(f"raw3 要求 3-channel IR；{path.name} 实际 shape={image.shape}")


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
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    expected_train_count: int = EXPECTED_TRAIN_COUNT,
    expected_val_count: int = EXPECTED_VAL_COUNT,
    check_image_encoding: bool = True,
) -> Dict[str, Any]:
    validate_canonical_inputs(label_dir, train_split, val_split)
    train_stems = read_stems(train_split, expected_train_count)
    val_stems = read_stems(val_split, expected_val_count)
    overlap = sorted(set(train_stems) & set(val_stems))
    if overlap:
        raise IRDatasetViewError(f"train/val split 存在重复 stem: {overlap[:5]}")
    expected_stems = set(train_stems) | set(val_stems)
    expected_total = expected_train_count + expected_val_count
    if len(expected_stems) != expected_total:
        raise IRDatasetViewError(
            f"合并 split 唯一 stem 数量错误；期望 {expected_total}，实际 {len(expected_stems)}"
        )

    images = index_ir_images(ir_dir)
    labels = index_labels(label_dir)
    image_mismatch = _set_mismatch_message("IR 图像", expected_stems, set(images))
    if image_mismatch:
        raise IRDatasetViewError(image_mismatch)
    label_mismatch = _set_mismatch_message("labels_clean", expected_stems, set(labels))
    if label_mismatch:
        raise IRDatasetViewError(label_mismatch)

    labels_identity, label_hashes = _aggregate_paths(list(labels.values()))
    if labels_identity["aggregate_sha256"] != CANONICAL_LABELS_CLEAN_SHA256:
        raise IRDatasetViewError("canonical labels_clean identity 不一致；拒绝使用被替换或篡改的标签")
    if check_image_encoding:
        validate_ir_encoding([images[stem] for stem in sorted(expected_stems)])

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
        "train_count": len(train_stems),
        "val_count": len(val_stems),
        "total_count": len(expected_stems),
        "png_count": sum(path.suffix.lower() == ".png" for path in images.values()),
        "jpg_count": sum(path.suffix.lower() in {".jpg", ".jpeg"} for path in images.values()),
    }


def materialize(source: Path, destination: Path, mode: str) -> str:
    if mode not in {"auto", "hardlink", "copy"}:
        raise IRDatasetViewError(f"不支持的 link mode: {mode}")
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def effective_link_mode(methods: Counter) -> str:
    used = sorted(mode for mode, count in methods.items() if count)
    return used[0] if len(used) == 1 else "mixed"


def build_source_records(validation: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        records.append(
            {
                "image_bytes": image.stat().st_size,
                "image_sha256": sha256_file(image),
                "label_bytes": label.stat().st_size,
                "label_sha256": validation["label_hashes"][label.name],
                "source_image_project_relative": project_relative_path(image),
                "source_label_project_relative": project_relative_path(label),
                "staged_image_view_relative": (Path("images") / subset / image.name).as_posix(),
                "staged_label_view_relative": (Path("labels") / subset / label.name).as_posix(),
                "stem": stem,
                "subset": subset,
            }
        )
    return records


def _aggregate_records(records: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for record in records:
        path_key = "source_image_project_relative" if prefix == "image" else "source_label_project_relative"
        size = record[f"{prefix}_bytes"]
        file_hash = record[f"{prefix}_sha256"]
        filename = Path(record[path_key]).name
        digest.update(f"{filename}\0{size}\0{file_hash}\n".encode("utf-8"))
        total_bytes += size
    return {"aggregate_sha256": digest.hexdigest(), "file_count": len(records), "total_bytes": total_bytes}


def build_manifest(
    validation: Dict[str, Any],
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    link_mode_requested: str,
    methods: Counter,
) -> Dict[str, Any]:
    records = build_source_records(validation)
    effective = effective_link_mode(methods)
    return {
        "channels": 3,
        "dtype": "uint8",
        "image_byte_preserving": True,
        "isolation": effective == "copy",
        "jpg_count": validation["jpg_count"],
        "label_dir": project_relative_path(label_dir),
        "labels_clean_identity": _aggregate_records(records, "label"),
        "link_mode_counts": dict(sorted(methods.items())),
        "link_mode_effective": effective,
        "link_mode_requested": link_mode_requested,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "png_count": validation["png_count"],
        "preprocessing": "none",
        "representation": "raw3",
        "script_version": SCRIPT_VERSION,
        "source_file_list": records,
        "source_ir_dir": project_relative_path(ir_dir),
        "source_ir_identity": _aggregate_records(records, "image"),
        "total_count": validation["total_count"],
        "train_count": validation["train_count"],
        "train_split_path": project_relative_path(train_split),
        "train_split_normalized_sha256": normalized_split_sha256(train_split),
        "val_count": validation["val_count"],
        "val_split_path": project_relative_path(val_split),
        "val_split_normalized_sha256": normalized_split_sha256(val_split),
    }


def _write_text_lf(path: Path, text: str) -> None:
    """Write deterministic UTF-8/LF text using the Python 3.8-compatible API."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def build_view(
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    output_root: Path,
    representation: str = "raw3",
    link_mode: str = "copy",
    force: bool = False,
    expected_train_count: int = EXPECTED_TRAIN_COUNT,
    expected_val_count: int = EXPECTED_VAL_COUNT,
) -> Dict[str, Any]:
    """Validate canonical inputs before atomically publishing an isolated raw3 view."""
    validate_output_path(output_root, ir_dir, label_dir, train_split, val_split)
    if representation != "raw3":
        raise IRDatasetViewError(f"当前 C1 仅实现 representation=raw3: {representation}")
    if link_mode not in {"auto", "hardlink", "copy"}:
        raise IRDatasetViewError(f"不支持的 link mode: {link_mode}")
    validate_canonical_inputs(label_dir, train_split, val_split)
    if output_root.exists() and not force:
        raise IRDatasetViewError(f"输出目录已存在；如需重建请显式使用 --force: {output_root}")

    validation = validate_sources(
        ir_dir,
        label_dir,
        train_split,
        val_split,
        expected_train_count=expected_train_count,
        expected_val_count=expected_val_count,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".raw3-", dir=str(output_root.parent)))
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
                methods[materialize(image, image_out / f"{stem}{image.suffix}", link_mode)] += 1
                methods[materialize(label, label_out / f"{stem}.txt", link_mode)] += 1

        dataset_yaml = {
            "path": output_root.resolve().as_posix(),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }
        _write_text_lf(
            staging / "data.yaml",
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        )
        manifest = build_manifest(
            validation,
            ir_dir,
            label_dir,
            train_split,
            val_split,
            link_mode,
            methods,
        )
        _write_text_lf(
            staging / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

        if output_root.exists():
            validate_force_target(output_root, ir_dir, label_dir, train_split, val_split)
            shutil.rmtree(output_root)
        validate_output_path(output_root, ir_dir, label_dir, train_split, val_split)
        _validate_temporary_staging(staging)
        staging.replace(output_root)
    except Exception:
        if staging_is_safe:
            _remove_temporary_staging(staging)
        raise
    return {"manifest": manifest, "methods": methods, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", default=CANONICAL_IR_RELATIVE.as_posix())
    parser.add_argument("--label-dir", default=CANONICAL_LABEL_RELATIVE.as_posix())
    parser.add_argument("--train-split", default=CANONICAL_TRAIN_SPLIT_RELATIVE.as_posix())
    parser.add_argument("--val-split", default=CANONICAL_VAL_SPLIT_RELATIVE.as_posix())
    parser.add_argument("--output-root", default=CANONICAL_OUTPUT_RELATIVE.as_posix())
    parser.add_argument("--representation", default="raw3")
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "auto"),
        default="copy",
        help=(
            "Materialization mode. Default 'copy' is the isolated E002 staging mode. "
            "Explicit 'hardlink' shares file identity/inode with source and is non-isolated; "
            "'auto' may also select hardlink."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rebuild only the canonical raw3 view.")
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
            project_path(args.ir_dir),
            project_path(args.label_dir),
            project_path(args.train_split),
            project_path(args.val_split),
            output_root,
            representation=args.representation,
            link_mode=args.link_mode,
            force=args.force,
        )
    except (IRDatasetViewError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    manifest = result["manifest"]
    print(f"IR YOLO raw3 兼容视图已生成: {output_root}")
    print(f"train/val: {manifest['train_count']}/{manifest['val_count']}")
    print(f"PNG/JPG: {manifest['png_count']}/{manifest['jpg_count']}")
    print(f"映射方式: {dict(result['methods'])}")
    print(f"隔离 staging: {manifest['isolation']}")
    print(f"Ultralytics data 配置: {output_root / 'data.yaml'}")
    print(f"Deterministic manifest: {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
