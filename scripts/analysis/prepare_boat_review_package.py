"""Build a portable, read-only visual review package for every class-1 GT image.

Source visible images and labels_clean are copied, never edited. The browser
viewer draws exact normalized YOLO boxes on a canvas and exports review CSV.
R8 model-disagreement hints are advisory, not proposed corrections.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = (
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage can", "uav", "tricycle",
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRAIN_COUNT = 1600
VAL_COUNT = 400
BOAT_IMAGE_COUNT = 62
BOAT_BOX_COUNT = 135
REVIEW_FIELDS = (
    "stem", "row_index", "original_class", "original_bbox", "proposed_class",
    "proposed_bbox", "decision", "visual_rationale", "split", "image_file",
    "r8_reason", "r8_other_class_id", "r8_other_class_iou",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> List[str]:
    stems = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate stems in split: %s" % path)
    return stems


def index_images(directory: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError("Duplicate visible image stem: %s" % path.stem)
            images[path.stem] = path
    return images


def parse_label(path: Path) -> List[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError("Expected five YOLO fields: %s:%d" % (path, line_number))
        cls = int(values[0])
        box = [float(value) for value in values[1:]]
        if not 0 <= cls < len(CLASS_NAMES) or not all(math.isfinite(value) for value in box):
            raise ValueError("Invalid label class or box: %s:%d" % (path, line_number))
        rows.append({"row_index": line_number, "class_id": cls, "class_name": CLASS_NAMES[cls],
                     "bbox": box, "bbox_text": " ".join(values[1:])})
    return rows


def r8_hint(stem: str, box: Sequence[float], misses: Dict[str, List[dict]]) -> dict:
    matches = [item for item in misses.get(stem, [])
               if int(item["class_id"]) == 1 and
               max(abs(float(item[field]) - value)
                   for field, value in zip(("x", "y", "w", "h"), box)) < 1e-7]
    if len(matches) > 1:
        raise ValueError("Ambiguous R8 hint for %s %s" % (stem, box))
    if not matches:
        return {"r8_reason": "", "r8_other_class_id": "", "r8_other_class_iou": ""}
    item = matches[0]
    return {"r8_reason": item["reason"], "r8_other_class_id": item["other_class_id"],
            "r8_other_class_iou": item["other_class_iou"]}


def collect_records(labels_dir: Path, visible_dir: Path, train_split: Path, val_split: Path,
                    r8_dir: Path) -> Tuple[List[dict], List[dict], Dict[str, Path]]:
    train = set(read_split(train_split))
    val = set(read_split(val_split))
    if len(train) != TRAIN_COUNT or len(val) != VAL_COUNT or train & val:
        raise ValueError("Expected fixed disjoint 1600/400 split")
    summary = json.loads((r8_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("images") != VAL_COUNT:
        raise ValueError("R8 audit does not cover fixed 400-image val")
    misses: Dict[str, List[dict]] = {}
    with (r8_dir / "gt_misses_at_conf_0p001.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            if int(item["class_id"]) == 1:
                misses.setdefault(item["stem"], []).append(item)
    images = index_images(visible_dir)
    review_rows: List[dict] = []
    image_records: List[dict] = []
    selected_images: Dict[str, Path] = {}
    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda path: path.name):
        all_boxes = parse_label(label_path)
        boat_boxes = [item for item in all_boxes if item["class_id"] == 1]
        if not boat_boxes:
            continue
        stem = label_path.stem
        if stem not in train and stem not in val:
            raise ValueError("Boat image absent from fixed split: %s" % stem)
        if stem not in images:
            raise ValueError("Missing visible image: %s" % stem)
        source_image = images[stem]
        selected_images[stem] = source_image
        split = "val" if stem in val else "train"
        for item in boat_boxes:
            review_rows.append({
                "stem": stem, "row_index": item["row_index"], "original_class": 1,
                "original_bbox": item["bbox_text"], "proposed_class": "",
                "proposed_bbox": "", "decision": "unreviewed", "visual_rationale": "",
                "split": split, "image_file": source_image.name,
                **r8_hint(stem, item["bbox"], misses),
            })
        image_records.append({
            "stem": stem, "split": split, "image_file": source_image.name,
            "boxes": all_boxes, "boat_count": len(boat_boxes),
            "r8_flag_count": sum(bool(r8_hint(stem, item["bbox"], misses)["r8_reason"])
                                 for item in boat_boxes),
        })
    if len(image_records) != BOAT_IMAGE_COUNT or len(review_rows) != BOAT_BOX_COUNT:
        raise ValueError("Unexpected boat coverage: %d images, %d boxes" %
                         (len(image_records), len(review_rows)))
    return image_records, review_rows, selected_images


def write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_package(output: Path, labels_dir: Path, visible_dir: Path, train_split: Path,
                  val_split: Path, r8_dir: Path, template_path: Path) -> dict:
    if output.exists():
        raise ValueError("Review package already exists: %s" % output)
    image_records, review_rows, selected_images = collect_records(
        labels_dir, visible_dir, train_split, val_split, r8_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s-" % output.name, dir=output.parent))
    try:
        (staging / "images").mkdir()
        (staging / "labels_clean_snapshot").mkdir()
        image_sha = {}
        label_sha = {}
        for stem, path in selected_images.items():
            target = staging / "images" / path.name
            shutil.copyfile(path, target)
            image_sha[path.name] = sha256(target)
            source_label = labels_dir / (stem + ".txt")
            target_label = staging / "labels_clean_snapshot" / source_label.name
            shutil.copyfile(source_label, target_label)
            label_sha[source_label.name] = sha256(target_label)
        write_csv(staging / "review.csv", review_rows)
        payload = {"images": image_records, "review_rows": review_rows,
                   "class_names": list(CLASS_NAMES), "review_fields": list(REVIEW_FIELDS)}
        embedded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
        template = template_path.read_text(encoding="utf-8")
        if template.count("__REVIEW_DATA_JSON__") != 1:
            raise ValueError("Review HTML template missing unique payload placeholder")
        (staging / "index.html").write_bytes(
            template.replace("__REVIEW_DATA_JSON__", embedded).encode("utf-8"))
        manifest = {
            "schema_version": 1, "source": "labels_clean class-1 objects, fixed 1600/400 split",
            "image_count": len(image_records), "boat_box_count": len(review_rows),
            "train_images": sum(item["split"] == "train" for item in image_records),
            "val_images": sum(item["split"] == "val" for item in image_records),
            "r8_summary_sha256": sha256(r8_dir / "summary.json"),
            "r8_misses_sha256": sha256(r8_dir / "gt_misses_at_conf_0p001.csv"),
            "image_sha256": image_sha, "label_snapshot_sha256": label_sha,
            "row_index": "1-based physical line number in labels_clean_snapshot/<stem>.txt",
            "bbox_format": "normalized YOLO x_center y_center width height; original text retained",
        }
        (staging / "manifest.json").write_bytes((
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8"))
        staging.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/train/labels_clean")
    parser.add_argument("--visible", type=Path, default=ROOT / "data/raw/train/visible")
    parser.add_argument("--train-split", type=Path, default=ROOT / "data/splits/train.txt")
    parser.add_argument("--val-split", type=Path, default=ROOT / "data/splits/val.txt")
    parser.add_argument("--r8-audit", type=Path, default=ROOT / "outputs/analysis/R8_R5_FAILURE_AUDIT")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis/R8_BOAT_REVIEW_PACKAGE")
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).with_name("boat_review_template.html"))
    args = parser.parse_args()
    manifest = build_package(args.output, args.labels, args.visible, args.train_split,
                             args.val_split, args.r8_audit, args.template)
    print("Boat review package: %s" % args.output)
    print("Images: %d, boxes: %d, train/val images: %d/%d" % (
        manifest["image_count"], manifest["boat_box_count"],
        manifest["train_images"], manifest["val_images"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
