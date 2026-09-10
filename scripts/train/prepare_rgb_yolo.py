"""Build a disposable Ultralytics-compatible RGB dataset view."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = [
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
]


class DatasetViewError(ValueError):
    """Raised when the source data or split contract is invalid."""


def project_path(value: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_stems(split_path: Path) -> list[str]:
    if not split_path.is_file():
        raise DatasetViewError(f"Split 文件不存在: {split_path}")

    stems = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not stems:
        raise DatasetViewError(f"Split 文件为空，拒绝回退到全部训练数据: {split_path}")

    invalid = [stem for stem in stems if Path(stem).name != stem or Path(stem).suffix]
    if invalid:
        raise DatasetViewError(
            f"Split 每行必须是无目录、无扩展名的 stem；非法值: {invalid[:5]}"
        )

    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        raise DatasetViewError(f"Split 包含重复 stem: {duplicates[:5]}")
    return stems


def index_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        raise DatasetViewError(f"RGB 图像目录不存在: {image_dir}")

    by_stem: dict[str, list[Path]] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem.setdefault(path.stem, []).append(path)

    ambiguous = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if ambiguous:
        stem, paths = next(iter(sorted(ambiguous.items())))
        raise DatasetViewError(
            f"同一个 stem 匹配到多个 RGB 图像: {stem} -> {[p.name for p in paths]}"
        )
    return {stem: paths[0] for stem, paths in by_stem.items()}


def validate_sources(
    data_root: Path,
    train_split: Path,
    val_split: Path,
    label_dir: Path | None = None,
) -> dict[str, list[tuple[str, Path, Path]]]:
    train_stems = read_stems(train_split)
    val_stems = read_stems(val_split)
    overlap = sorted(set(train_stems) & set(val_stems))
    if overlap:
        raise DatasetViewError(f"train/val split 存在重复 stem: {overlap[:5]}")

    images = index_images(data_root / "visible")
    label_dir = data_root / "labels" if label_dir is None else label_dir
    if not label_dir.is_dir():
        raise DatasetViewError(f"标签目录不存在: {label_dir}")

    result: dict[str, list[tuple[str, Path, Path]]] = {}
    for subset, stems in (("train", train_stems), ("val", val_stems)):
        entries: list[tuple[str, Path, Path]] = []
        for stem in stems:
            image = images.get(stem)
            if image is None:
                raise DatasetViewError(f"{subset} split 找不到 RGB 图像: {stem}")
            label = label_dir / f"{stem}.txt"
            if not label.is_file():
                raise DatasetViewError(f"{subset} split 找不到标签: {stem}.txt")
            entries.append((stem, image, label))
        result[subset] = entries
    return result


def materialize(source: Path, destination: Path, mode: str) -> str:
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def build_view(
    data_root: Path,
    train_split: Path,
    val_split: Path,
    output_root: Path,
    *,
    label_dir: Path | None = None,
    link_mode: str = "auto",
    force: bool = False,
) -> Counter[str]:
    """Validate all inputs, then atomically publish the generated view."""
    entries = validate_sources(data_root, train_split, val_split, label_dir)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not force:
        raise DatasetViewError(f"输出目录已存在；如需重建请显式使用 --force: {output_root}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    methods: Counter[str] = Counter()
    try:
        for subset, subset_entries in entries.items():
            image_out = staging / "images" / subset
            label_out = staging / "labels" / subset
            image_out.mkdir(parents=True)
            label_out.mkdir(parents=True)
            for stem, image, label in subset_entries:
                methods[materialize(image, image_out / f"{stem}{image.suffix}", link_mode)] += 1
                methods[materialize(label, label_out / f"{stem}.txt", link_mode)] += 1

        dataset_yaml = {
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        }
        (staging / "data.yaml").write_text(
            yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/train")
    parser.add_argument(
        "--label-dir",
        default=None,
        help="标签目录；默认使用 <data-root>/labels",
    )
    parser.add_argument("--train-split", default="data/splits/train.txt")
    parser.add_argument("--val-split", default="data/splits/val.txt")
    parser.add_argument("--output-root", default="data/processed/rgb_yolo")
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_console_encoding() -> None:
    """Keep Chinese diagnostics readable in redirected Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        data_root = project_path(args.data_root)
        label_dir = project_path(args.label_dir) if args.label_dir is not None else None
        train_split = project_path(args.train_split)
        val_split = project_path(args.val_split)
        output_root = project_path(args.output_root)
        methods = build_view(
            data_root,
            train_split,
            val_split,
            output_root,
            label_dir=label_dir,
            link_mode=args.link_mode,
            force=args.force,
        )
    except (DatasetViewError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"RGB YOLO 兼容视图已生成: {output_root}")
    print(f"映射方式: {dict(methods)}")
    print(f"Ultralytics data 配置: {output_root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
