"""Compare exported COCO JSON semantics and raw RGB identity across machines."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.prepare_deimv2_coco import image_index, read_split, sha256


def digest_lines(lines: List[str]) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def load_subset(dataset_root: Path, subset: str) -> Dict[str, Any]:
    path = dataset_root / "annotations" / ("instances_%s.json" % subset)
    if not path.is_file():
        path = dataset_root / ("instances_%s.json" % subset)
    return json.loads(path.read_text(encoding="utf-8"))


def first_semantic_diff(left: Any, right: Any, path: str = "$") -> Optional[Dict[str, Any]]:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right, "reason": "type"}
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {"path": "%s.%s" % (path, key), "left": left.get(key),
                        "right": right.get(key), "reason": "missing_key"}
            difference = first_semantic_diff(left[key], right[key], "%s.%s" % (path, key))
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left": len(left), "right": len(right), "reason": "length"}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_semantic_diff(a, b, "%s[%d]" % (path, index))
            if difference is not None:
                return difference
        return None
    if left != right:
        return {"path": path, "left": left, "right": right, "reason": "value"}
    return None


def subset_identity(
    dataset_root: Path, visible_dir: Path, split_path: Path, subset: str,
) -> Dict[str, Any]:
    stems = read_split(split_path, 1600 if subset == "train" else 400)
    images = image_index(visible_dir)
    coco = load_subset(dataset_root, subset)
    coco_images = coco["images"]
    if len(coco_images) != len(stems):
        raise ValueError("COCO image count differs from fixed split")
    raw_lines: List[str] = []
    dimension_lines: List[str] = []
    image_records = []
    for stem, record in zip(stems, coco_images):
        source = images[stem]
        if Path(record["file_name"]).stem != stem:
            raise ValueError("COCO image order differs from fixed split at %s" % stem)
        raw_hash = sha256(source)
        raw_lines.append("%s\0%s\0%d\0%s\n" % (stem, source.name, source.stat().st_size, raw_hash))
        dimension_lines.append("%s\0%d\0%d\n" % (record["file_name"], record["width"], record["height"]))
        image_records.append({"stem": stem, "name": source.name, "raw_sha256": raw_hash,
                              "width": record["width"], "height": record["height"]})
    return {
        "subset": subset,
        "raw_visible_aggregate_sha256": digest_lines(raw_lines),
        "dimension_fingerprint_sha256": digest_lines(dimension_lines),
        "dimension_fingerprint_format": "file_name\\0width\\0height\\n in fixed split order",
        "raw_visible_aggregate_format": "stem\\0file_name\\0byte_size\\0file_sha256\\n in fixed split order",
        "first_three_images": image_records[:3],
        "annotation_sha256": sha256(dataset_root / "annotations" / ("instances_%s.json" % subset)),
        "images": len(coco_images), "annotations": len(coco["annotations"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/processed/deimv2_rgb_coco")
    parser.add_argument("--visible-dir", type=Path, default=ROOT / "data/raw/train/visible")
    parser.add_argument("--train-split", type=Path, default=ROOT / "data/splits/train.txt")
    parser.add_argument("--val-split", type=Path, default=ROOT / "data/splits/val.txt")
    parser.add_argument("--compare-root", type=Path,
                        help="Another COCO export with instances_train.json and instances_val.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: Dict[str, Any] = {
        "python": platform.python_version(), "opencv": cv2.__version__, "subsets": {},
    }
    for subset, split in (("train", args.train_split), ("val", args.val_split)):
        identity = subset_identity(args.dataset_root, args.visible_dir, split, subset)
        if args.compare_root is not None:
            identity["first_semantic_diff"] = first_semantic_diff(
                load_subset(args.dataset_root, subset), load_subset(args.compare_root, subset)
            )
            identity["compare_annotation_sha256"] = sha256(
                (args.compare_root / "annotations" / ("instances_%s.json" % subset))
                if (args.compare_root / "annotations" / ("instances_%s.json" % subset)).is_file()
                else (args.compare_root / ("instances_%s.json" % subset))
            )
        report["subsets"][subset] = identity
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
