"""Compare fixed-val RGB prediction exports using the local AIC scorer.

AP follows scripts/analysis/evaluate_submission.py (a PDF reconstruction, not
the official scorer). Size recall is a diagnostic at conf=0.25 and IoU=0.50;
COCO's original-image area boundaries of 32^2 and 96^2 pixels are used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.evaluate_submission import evaluate, ious, read_rows


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = (
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage can", "uav", "tricycle",
)
EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_fixed_val(split: Path, labels: Path, images: Path, prediction_dir: Path):
    stems = [line.strip() for line in split.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(stems) != 400 or len(set(stems)) != 400:
        raise ValueError("Expected the fixed 400-image validation split")
    index: Dict[str, Path] = {}
    for image in images.iterdir():
        if image.is_file() and image.suffix.lower() in EXTENSIONS:
            if image.stem in index:
                raise ValueError("Duplicate visible image stem: %s" % image.stem)
            index[image.stem] = image
    if set(index) != set(stems):
        raise ValueError("Image directory must contain exactly the fixed val stems")
    if {p.stem for p in prediction_dir.glob("*.txt")} != set(stems):
        raise ValueError("Prediction directory must contain exactly the fixed val TXT files")
    gt = {stem: read_rows(labels / (stem + ".txt")) for stem in stems}
    pred = {stem: read_rows(prediction_dir / (stem + ".txt"), prediction=True) for stem in stems}
    dimensions = {}
    for stem, path in index.items():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Cannot read image: %s" % path)
        dimensions[stem] = image.shape[:2]
    return gt, pred, dimensions


def size_bin(row: Sequence[float], shape: Sequence[int]) -> str:
    area = row[3] * row[4] * shape[0] * shape[1]
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def operating_recall(gt: dict, pred: dict, dimensions: dict, conf: float = 0.25) -> dict:
    support = defaultdict(int)
    matched = defaultdict(int)
    class_support = defaultdict(int)
    class_matched = defaultdict(int)
    for stem, ground_truth in gt.items():
        shape = dimensions[stem]
        for row in ground_truth:
            support[size_bin(row, shape)] += 1
            class_support[int(row[0])] += 1
        for cls in range(12):
            gt_rows = [(i, row) for i, row in enumerate(ground_truth) if int(row[0]) == cls]
            if not gt_rows:
                continue
            used = np.zeros(len(gt_rows), dtype=bool)
            predictions = sorted(
                (row for row in pred[stem] if int(row[0]) == cls and row[5] >= conf),
                key=lambda row: -row[5],
            )[:100]
            for row in predictions:
                overlaps = ious(row[1:5], [g[1][1:5] for g in gt_rows])
                overlaps[used] = -1
                best = int(overlaps.argmax())
                if overlaps[best] >= 0.5:
                    used[best] = True
                    matched[size_bin(gt_rows[best][1], shape)] += 1
                    class_matched[cls] += 1
    return {
        "confidence": conf, "iou": 0.5,
        "by_size": {name: {"gt": support[name], "tp": matched[name],
                           "recall": matched[name] / support[name] if support[name] else None}
                    for name in ("small", "medium", "large")},
        "by_class": {CLASS_NAMES[cls]: {"gt": class_support[cls], "tp": class_matched[cls],
                                        "recall": class_matched[cls] / class_support[cls]
                                        if class_support[cls] else None}
                     for cls in range(12)},
    }


def compare(gt: dict, baseline: dict, candidate: dict, dimensions: dict) -> dict:
    base_ap = evaluate(gt, baseline, conf=0.001, max_det=100)
    candidate_ap = evaluate(gt, candidate, conf=0.001, max_det=100)
    base_recall = operating_recall(gt, baseline, dimensions)
    candidate_recall = operating_recall(gt, candidate, dimensions)
    classes: List[dict] = []
    for cls in range(12):
        before = base_ap["per_class"][cls]
        after = candidate_ap["per_class"][cls]
        classes.append({
            "class_id": cls, "name": CLASS_NAMES[cls], "gt": before["gt"],
            "baseline_ap50_95": before["ap50_95"],
            "candidate_ap50_95": after["ap50_95"],
            "delta": after["ap50_95"] - before["ap50_95"],
        })
    return {
        "scope": "fixed 400-image val; local PDF-reconstructed AIC AP, not official leaderboard",
        "baseline_map50_95": base_ap["map50_95"],
        "candidate_map50_95": candidate_ap["map50_95"],
        "delta": candidate_ap["map50_95"] - base_ap["map50_95"],
        "per_class": classes,
        "baseline_operating_recall": base_recall,
        "candidate_operating_recall": candidate_recall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=ROOT / "data/processed/rgb_yolo_clean/images/val")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/train/labels_clean")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/val.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt, baseline, dimensions = load_fixed_val(args.split, args.labels, args.images, args.baseline)
    _, candidate, _ = load_fixed_val(args.split, args.labels, args.images, args.candidate)
    report = compare(gt, baseline, candidate, dimensions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("baseline=%.6f candidate=%.6f delta=%+.6f" % (
        report["baseline_map50_95"], report["candidate_map50_95"], report["delta"]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
