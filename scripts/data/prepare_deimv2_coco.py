"""Export the fixed clean-label RGB split as a deterministic COCO dataset view."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage can", "uav", "tricycle",
]
EXPECTED_LABEL_SHA256 = "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"
TRAIN_COUNT = 1600
VAL_COUNT = 400
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class CocoExportError(ValueError):
    """Raised when the fixed data contract cannot be exported safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_labels(paths: Iterable[Path]) -> Dict[str, object]:
    paths = sorted(paths, key=lambda item: (item.name.casefold(), item.name))
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        size = path.stat().st_size
        digest.update(("%s\0%d\0%s\n" % (path.name, size, sha256(path))).encode("utf-8"))
        total_bytes += size
    return {"aggregate_sha256": digest.hexdigest(), "file_count": len(paths), "total_bytes": total_bytes}


def read_split(path: Path, expected: int) -> List[str]:
    stems = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(stems) != expected or len(set(stems)) != expected:
        raise CocoExportError("固定 split 数量或唯一性错误: %s" % path)
    if any(Path(stem).name != stem or Path(stem).suffix for stem in stems):
        raise CocoExportError("split 必须只包含无扩展名 stem: %s" % path)
    return stems


def image_index(path: Path) -> Dict[str, Path]:
    images: Dict[str, List[Path]] = {}
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(item.stem, []).append(item)
    ambiguous = [stem for stem, values in images.items() if len(values) != 1]
    if ambiguous:
        raise CocoExportError("RGB stem 匹配多个文件: %s" % ambiguous[:5])
    return {stem: values[0] for stem, values in images.items()}


def read_yolo_labels(path: Path, width: int, height: int) -> List[Tuple[int, List[float]]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise CocoExportError("标签字段数错误: %s:%d" % (path, line_number))
        class_id = int(fields[0])
        cx, cy, box_width, box_height = [float(value) for value in fields[1:]]
        if class_id not in range(len(CLASS_NAMES)) or not all(0 <= value <= 1 for value in (cx, cy, box_width, box_height)):
            raise CocoExportError("标签值越界: %s:%d" % (path, line_number))
        if box_width <= 0 or box_height <= 0:
            raise CocoExportError("标签宽高必须为正: %s:%d" % (path, line_number))
        x = max(0.0, (cx - box_width / 2) * width)
        y = max(0.0, (cy - box_height / 2) * height)
        x2 = min(float(width), (cx + box_width / 2) * width)
        y2 = min(float(height), (cy + box_height / 2) * height)
        if x2 <= x or y2 <= y:
            raise CocoExportError("裁剪后空框: %s:%d" % (path, line_number))
        result.append((class_id, [x, y, x2 - x, y2 - y]))
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


def build_coco_view(
    visible_dir: Path, label_dir: Path, train_split: Path, val_split: Path,
    output_root: Path, link_mode: str = "auto", force: bool = False,
) -> Dict[str, object]:
    train_stems, val_stems = read_split(train_split, TRAIN_COUNT), read_split(val_split, VAL_COUNT)
    if set(train_stems) & set(val_stems):
        raise CocoExportError("train/val split 重叠")
    images = image_index(visible_dir)
    label_paths = [label_dir / (stem + ".txt") for stem in train_stems + val_stems]
    if any(not path.is_file() for path in label_paths):
        raise CocoExportError("labels_clean 文件缺失")
    label_identity = aggregate_labels(label_paths)
    if label_identity["aggregate_sha256"] != EXPECTED_LABEL_SHA256:
        raise CocoExportError("labels_clean aggregate SHA256 不一致")
    missing = [stem for stem in train_stems + val_stems if stem not in images]
    if missing:
        raise CocoExportError("RGB 图像缺失: %s" % missing[:5])
    if output_root.exists() and not force:
        raise CocoExportError("输出已存在；重建必须显式 --force: %s" % output_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s-" % output_root.name, dir=str(output_root.parent)))
    methods: Counter[str] = Counter()
    subset_stats = {}
    try:
        (staging / "annotations").mkdir(parents=True)
        next_annotation_id = 1
        global_image_id = 1
        for subset, stems in (("train", train_stems), ("val", val_stems)):
            image_out = staging / "images" / subset
            image_out.mkdir(parents=True)
            coco_images, annotations = [], []
            for stem in stems:
                source = images[stem]
                image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                if image is None or image.ndim < 2:
                    raise CocoExportError("无法解码 RGB: %s" % source)
                height, width = image.shape[:2]
                destination_name = stem + source.suffix.lower()
                methods[materialize(source, image_out / destination_name, link_mode)] += 1
                coco_images.append({"id": global_image_id, "file_name": destination_name,
                                    "width": width, "height": height})
                for class_id, bbox in read_yolo_labels(label_dir / (stem + ".txt"), width, height):
                    annotations.append({
                        "id": next_annotation_id, "image_id": global_image_id,
                        "category_id": class_id, "bbox": bbox,
                        "area": bbox[2] * bbox[3], "iscrowd": 0,
                    })
                    next_annotation_id += 1
                global_image_id += 1
            payload = {
                "info": {"description": "AIC2026 fixed split with labels_clean", "version": "1"},
                "licenses": [], "images": coco_images, "annotations": annotations,
                "categories": [{"id": index, "name": name, "supercategory": "object"}
                               for index, name in enumerate(CLASS_NAMES)],
            }
            annotation_path = staging / "annotations" / ("instances_%s.json" % subset)
            annotation_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                                       encoding="utf-8")
            subset_stats[subset] = {
                "images": len(coco_images), "annotations": len(annotations),
                "annotation_sha256": sha256(annotation_path),
            }
        manifest = {
            "schema_version": 1, "representation": "rgb_coco_labels_clean_v1",
            "category_id_range": [0, 11], "train_count": TRAIN_COUNT, "val_count": VAL_COUNT,
            "labels_identity": label_identity, "subsets": subset_stats,
            "materialization": dict(sorted(methods.items())),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8")
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-dir", default="data/raw/train/visible")
    parser.add_argument("--label-dir", default="data/processed/train/labels_clean")
    parser.add_argument("--train-split", default="data/splits/train.txt")
    parser.add_argument("--val-split", default="data/splits/val.txt")
    parser.add_argument("--output-root", default="data/processed/deimv2_rgb_coco")
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_coco_view(
            project_path(args.visible_dir), project_path(args.label_dir),
            project_path(args.train_split), project_path(args.val_split),
            project_path(args.output_root), args.link_mode, args.force,
        )
    except (CocoExportError, OSError, ValueError) as exc:
        print("错误: %s" % exc)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
