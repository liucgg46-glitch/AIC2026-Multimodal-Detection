"""Reproducible fixed-val R5 error attribution from competition TXT predictions.

Categories are geometric heuristics for image review, not assertions that a
prediction or annotation is wrong. Matching uses confidence-ordered greedy
same-class IoU >= 0.50, like the local AIC PDF reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.evaluate_submission import ious, read_rows  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
CLASS_NAMES = (
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage can", "uav", "tricycle",
)


def best_overlap(box: Sequence[float], rows: Sequence[Sequence[float]]) -> Tuple[float, int]:
    if not rows:
        return 0.0, -1
    overlaps = ious(box, [row[1:5] for row in rows])
    index = int(overlaps.argmax())
    return float(overlaps[index]), index


def audit_image(stem: str, ground_truth: List[List[float]], predictions: List[List[float]],
                conf: float) -> Tuple[List[dict], List[dict], int]:
    misses: List[dict] = []
    false_positives: List[dict] = []
    tp = 0
    for cls in range(12):
        gt_indices = [i for i, row in enumerate(ground_truth) if int(row[0]) == cls]
        gt_rows = [ground_truth[i] for i in gt_indices]
        same_predictions = [row for row in predictions if int(row[0]) == cls]
        other_gt = [row for row in ground_truth if int(row[0]) != cls]
        other_predictions = [row for row in predictions if int(row[0]) != cls]
        used = set()
        for row in sorted((p for p in same_predictions if p[5] >= conf), key=lambda p: -p[5]):
            available = ious(row[1:5], [g[1:5] for g in gt_rows])
            for index in used:
                available[index] = -1
            if len(available) and available.max() >= 0.5:
                used.add(int(available.argmax()))
                tp += 1
                continue
            same_iou, _ = best_overlap(row[1:5], gt_rows)
            other_iou, other_index = best_overlap(row[1:5], other_gt)
            if same_iou >= 0.5:
                reason = "duplicate_or_assignment_conflict"
            elif other_iou >= 0.5:
                reason = "other_class_overlap"
            elif same_iou >= 0.1:
                reason = "localization_overlap_0p1_to_0p5"
            else:
                reason = "background_or_unlabeled"
            false_positives.append({
                "stem": stem, "class_id": cls, "class_name": CLASS_NAMES[cls],
                "confidence": row[5], "reason": reason, "same_class_iou": same_iou,
                "other_class_iou": other_iou,
                "other_class_id": int(other_gt[other_index][0]) if other_index >= 0 else "",
                "x": row[1], "y": row[2], "w": row[3], "h": row[4],
            })
        for index, row in enumerate(gt_rows):
            if index in used:
                continue
            same_iou, _ = best_overlap(row[1:5], same_predictions)
            other_iou, other_index = best_overlap(row[1:5], other_predictions)
            best_same_score = max(
                (p[5] for p in same_predictions if best_overlap(row[1:5], [p])[0] >= 0.5),
                default=0.0,
            )
            if 0 < best_same_score < conf:
                reason = "below_operating_confidence"
            elif same_iou >= 0.5:
                reason = "assignment_conflict"
            elif other_iou >= 0.5:
                reason = "other_class_candidate"
            elif same_iou >= 0.1:
                reason = "localization_candidate_0p1_to_0p5"
            else:
                reason = "no_nearby_candidate"
            misses.append({
                "stem": stem, "class_id": cls, "class_name": CLASS_NAMES[cls],
                "reason": reason, "same_class_iou": same_iou,
                "best_same_confidence_at_iou_0p5": best_same_score,
                "other_class_iou": other_iou,
                "other_class_id": int(other_predictions[other_index][0]) if other_index >= 0 else "",
                "x": row[1], "y": row[2], "w": row[3], "h": row[4],
            })
    return misses, false_positives, tp


def load_fixed_val(prediction_dir: Path, labels_dir: Path, split: Path):
    stems = [line.strip() for line in split.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(stems) != 400 or len(set(stems)) != 400:
        raise ValueError("Expected the fixed 400-image val split")
    if {path.stem for path in prediction_dir.glob("*.txt")} != set(stems):
        raise ValueError("Prediction directory does not contain exactly the fixed val stems")
    gt = {s: read_rows(labels_dir / (s + ".txt")) for s in stems}
    pred = {s: read_rows(prediction_dir / (s + ".txt"), prediction=True) for s in stems}
    if any(len(rows) > 100 for rows in pred.values()):
        raise ValueError("At least one image exceeds the competition max_det=100")
    digest = hashlib.sha256()
    for stem in sorted(stems):
        for folder in (labels_dir, prediction_dir):
            digest.update(stem.encode("utf-8"))
            digest.update(hashlib.sha256((folder / (stem + ".txt")).read_bytes()).digest())
    return stems, gt, pred, digest.hexdigest()


def summarize(stems: List[str], gt: dict, pred: dict, input_sha256: str) -> Tuple[dict, List[dict], List[dict]]:
    analyses = {}
    review_misses: List[dict] = []
    review_fps: List[dict] = []
    for conf in (0.001, 0.25, 0.5):
        misses: List[dict] = []
        fps: List[dict] = []
        tp = 0
        for stem in stems:
            miss, fp, matched = audit_image(stem, gt[stem], pred[stem], conf)
            misses.extend(miss)
            fps.extend(fp)
            tp += matched
        analyses[str(conf)] = {
            "gt": sum(map(len, gt.values())), "predictions_above_conf":
            sum(sum(row[5] >= conf for row in pred[s]) for s in stems),
            "tp": tp, "fn": len(misses), "fp": len(fps),
            "fn_reasons": dict(Counter(item["reason"] for item in misses)),
            "fp_reasons": dict(Counter(item["reason"] for item in fps)),
            "fn_by_class": {CLASS_NAMES[cls]: dict(Counter(
                item["reason"] for item in misses if item["class_id"] == cls))
                for cls in range(12)},
            "fp_by_class": {CLASS_NAMES[cls]: dict(Counter(
                item["reason"] for item in fps if item["class_id"] == cls))
                for cls in range(12)},
        }
        if conf == 0.001:
            review_misses = misses
        if conf == 0.5:
            review_fps = fps
    report = {
        "scope": "fixed val; geometric review heuristics, not adjudicated label errors",
        "input_sha256": input_sha256, "images": len(stems), "analyses": analyses,
    }
    return report, review_misses, review_fps


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/train/labels_clean")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/val.txt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists: %s" % args.output)
    stems, gt, pred, digest = load_fixed_val(args.predictions, args.labels, args.split)
    report, misses, fps = summarize(stems, gt, pred, digest)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    write_csv(args.output / "gt_misses_at_conf_0p001.csv", misses, [
        "stem", "class_id", "class_name", "reason", "same_class_iou",
        "best_same_confidence_at_iou_0p5", "other_class_iou", "other_class_id",
        "x", "y", "w", "h",
    ])
    write_csv(args.output / "false_positives_at_conf_0p5.csv", fps, [
        "stem", "class_id", "class_name", "confidence", "reason", "same_class_iou",
        "other_class_iou", "other_class_id", "x", "y", "w", "h",
    ])
    print(json.dumps({"output": str(args.output), "analyses": report["analyses"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
