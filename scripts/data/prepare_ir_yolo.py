"""Build a deterministic Ultralytics-compatible IR-only dataset view."""

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
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IR_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "infrared"
DEFAULT_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "train" / "labels_clean"
DEFAULT_TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.txt"
DEFAULT_VAL_SPLIT = PROJECT_ROOT / "data" / "splits" / "val.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "ir_trainable" / "raw3"
ALLOWED_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "ir_trainable"

IR_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_TRAIN_COUNT = 1600
EXPECTED_VAL_COUNT = 400
MANIFEST_SCHEMA_VERSION = 1
SCRIPT_VERSION = "1.0"

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
    """Raised when the IR source, split, or output contract is invalid."""


def project_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def manifest_path(path: Path) -> str:
    """Return a stable project-relative path when possible, using POSIX separators."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_stems(split_path: Path, expected_count: int) -> list[str]:
    if not split_path.is_file():
        raise IRDatasetViewError(f"Split 文件不存在: {split_path}")

    stems = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not stems:
        raise IRDatasetViewError(f"Split 文件为空，拒绝回退到全部训练数据: {split_path}")

    invalid = [stem for stem in stems if Path(stem).name != stem or Path(stem).suffix]
    if invalid:
        raise IRDatasetViewError(f"Split 每行必须是无目录、无扩展名的 stem；非法值: {invalid[:5]}")

    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise IRDatasetViewError(f"Split 包含重复 stem: {duplicates[:5]}")
    if len(stems) != expected_count:
        raise IRDatasetViewError(
            f"Split 数量错误: {split_path}，期望 {expected_count}，实际 {len(stems)}"
        )
    return stems


def index_ir_images(ir_dir: Path) -> dict[str, Path]:
    if not ir_dir.is_dir():
        raise IRDatasetViewError(f"IR 图像目录不存在: {ir_dir}")

    by_stem: dict[str, list[Path]] = {}
    for path in sorted(ir_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IR_IMAGE_EXTENSIONS:
            by_stem.setdefault(path.stem, []).append(path)

    ambiguous = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if ambiguous:
        stem, paths = next(iter(sorted(ambiguous.items())))
        raise IRDatasetViewError(
            f"同一个 stem 匹配到多个 IR 图像: {stem} -> {[path.name for path in paths]}"
        )
    return {stem: paths[0] for stem, paths in by_stem.items()}


def index_labels(label_dir: Path) -> dict[str, Path]:
    if not label_dir.is_dir():
        raise IRDatasetViewError(f"labels_clean 目录不存在: {label_dir}")
    return {
        path.stem: path
        for path in sorted(label_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.suffix.lower() == ".txt"
    }


def _set_mismatch_message(kind: str, expected: set[str], actual: set[str]) -> str | None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return None
    return f"{kind} 与 fixed split 不一致；缺失: {missing[:5]}；split 外额外 stem: {extra[:5]}"


def validate_ir_encoding(image_paths: list[Path]) -> None:
    """Verify raw3 inputs without converting or rewriting their pixels."""
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IRDatasetViewError(f"IR 图像解码失败: {path}")
        if image.dtype != np.uint8:
            raise IRDatasetViewError(f"raw3 要求 uint8 IR；{path.name} 实际为 {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise IRDatasetViewError(f"raw3 要求 3-channel IR；{path.name} 实际 shape={image.shape}")


def validate_sources(
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    *,
    expected_train_count: int = EXPECTED_TRAIN_COUNT,
    expected_val_count: int = EXPECTED_VAL_COUNT,
    check_image_encoding: bool = True,
) -> dict[str, Any]:
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

    if check_image_encoding:
        validate_ir_encoding([images[stem] for stem in sorted(expected_stems)])

    entries = {
        subset: [(stem, images[stem], labels[stem]) for stem in stems]
        for subset, stems in (("train", train_stems), ("val", val_stems))
    }
    png_count = sum(path.suffix.lower() == ".png" for path in images.values())
    jpg_count = sum(path.suffix.lower() in {".jpg", ".jpeg"} for path in images.values())
    return {
        "entries": entries,
        "images": images,
        "labels": labels,
        "train_stems": train_stems,
        "val_stems": val_stems,
        "train_count": len(train_stems),
        "val_count": len(val_stems),
        "total_count": len(expected_stems),
        "png_count": png_count,
        "jpg_count": jpg_count,
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


def _aggregate_identity(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for record in records:
        name = record[f"{prefix}_file"]
        size = record[f"{prefix}_bytes"]
        file_hash = record[f"{prefix}_sha256"]
        digest.update(f"{name}\0{size}\0{file_hash}\n".encode("utf-8"))
        total_bytes += size
    return {
        "aggregate_sha256": digest.hexdigest(),
        "file_count": len(records),
        "total_bytes": total_bytes,
    }


def build_source_records(validation: dict[str, Any]) -> list[dict[str, Any]]:
    subset_by_stem = {
        stem: subset
        for subset in ("train", "val")
        for stem, _, _ in validation["entries"][subset]
    }
    records: list[dict[str, Any]] = []
    for stem in sorted(validation["images"]):
        image = validation["images"][stem]
        label = validation["labels"][stem]
        records.append(
            {
                "image_bytes": image.stat().st_size,
                "image_file": image.name,
                "image_sha256": sha256_file(image),
                "label_bytes": label.stat().st_size,
                "label_file": label.name,
                "label_sha256": sha256_file(label),
                "stem": stem,
                "subset": subset_by_stem[stem],
            }
        )
    return records


def effective_link_mode(methods: Counter[str]) -> str:
    used = sorted(mode for mode, count in methods.items() if count)
    return used[0] if len(used) == 1 else "mixed"


def build_manifest(
    validation: dict[str, Any],
    *,
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    link_mode_requested: str,
    methods: Counter[str],
) -> dict[str, Any]:
    records = build_source_records(validation)
    return {
        "channels": 3,
        "dtype": "uint8",
        "image_byte_preserving": True,
        "jpg_count": validation["jpg_count"],
        "label_dir": manifest_path(label_dir),
        "labels_clean_identity": _aggregate_identity(records, "label"),
        "link_mode_counts": dict(sorted(methods.items())),
        "link_mode_effective": effective_link_mode(methods),
        "link_mode_requested": link_mode_requested,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "png_count": validation["png_count"],
        "preprocessing": "none",
        "representation": "raw3",
        "script_version": SCRIPT_VERSION,
        "source_file_list": records,
        "source_ir_dir": manifest_path(ir_dir),
        "source_ir_identity": _aggregate_identity(records, "image"),
        "total_count": validation["total_count"],
        "train_count": validation["train_count"],
        "train_split_path": manifest_path(train_split),
        "train_split_sha256": sha256_file(train_split),
        "val_count": validation["val_count"],
        "val_split_path": manifest_path(val_split),
        "val_split_sha256": sha256_file(val_split),
    }


def ensure_output_scope(output_root: Path, allowed_root: Path) -> None:
    try:
        output_root.resolve().relative_to(allowed_root.resolve())
    except ValueError as exc:
        raise IRDatasetViewError(f"输出目录必须位于 {allowed_root}: {output_root}") from exc


def build_view(
    ir_dir: Path,
    label_dir: Path,
    train_split: Path,
    val_split: Path,
    output_root: Path,
    *,
    representation: str = "raw3",
    link_mode: str = "auto",
    force: bool = False,
    expected_train_count: int = EXPECTED_TRAIN_COUNT,
    expected_val_count: int = EXPECTED_VAL_COUNT,
    allowed_output_root: Path | None = None,
) -> dict[str, Any]:
    """Validate all inputs before atomically publishing a byte-preserving raw3 view."""
    if representation != "raw3":
        raise IRDatasetViewError(f"当前 C1 仅实现 representation=raw3: {representation}")
    if link_mode not in {"auto", "hardlink", "copy"}:
        raise IRDatasetViewError(f"不支持的 link mode: {link_mode}")
    if allowed_output_root is not None:
        ensure_output_scope(output_root, allowed_output_root)
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
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    methods: Counter[str] = Counter()
    try:
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
        (staging / "data.yaml").write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        manifest = build_manifest(
            validation,
            ir_dir=ir_dir,
            label_dir=label_dir,
            train_split=train_split,
            val_split=val_split,
            link_mode_requested=link_mode,
            methods=methods,
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {"manifest": manifest, "methods": methods, "validation": validation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", default="data/raw/train/infrared")
    parser.add_argument("--label-dir", default="data/processed/train/labels_clean")
    parser.add_argument("--train-split", default="data/splits/train.txt")
    parser.add_argument("--val-split", default="data/splits/val.txt")
    parser.add_argument("--output-root", default="data/processed/ir_trainable/raw3")
    parser.add_argument("--representation", default="raw3")
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--force", action="store_true")
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
            allowed_output_root=ALLOWED_OUTPUT_ROOT,
        )
    except (IRDatasetViewError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    manifest = result["manifest"]
    print(f"IR YOLO raw3 兼容视图已生成: {output_root}")
    print(f"train/val: {manifest['train_count']}/{manifest['val_count']}")
    print(f"PNG/JPG: {manifest['png_count']}/{manifest['jpg_count']}")
    print(f"映射方式: {dict(result['methods'])}")
    print(f"Ultralytics data 配置: {output_root / 'data.yaml'}")
    print(f"Deterministic manifest: {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
