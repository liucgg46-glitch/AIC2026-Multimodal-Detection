"""Reproducible E001 CLEAN error analysis on the fixed 400-image validation set.

This script consumes exported validation predictions and ground truth. It does
not train a model, inspect prelim_test, or modify labels/splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = REPO_ROOT / "outputs" / "analysis" / "e001_exports"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "raw" / "train" / "visible"
DEFAULT_LABEL_DIR = REPO_ROOT / "data" / "processed" / "train" / "labels_clean"
DEFAULT_VAL_SPLIT = REPO_ROOT / "data" / "splits" / "val.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "e001_error_analysis"
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
SIZE_ORDER = ("tiny", "small", "medium", "large")
REFERENCE_IMGSZ = 640
OPERATING_CONF = 0.25
MATCH_IOU = 0.50
HIGH_CONF_FP = 0.50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
UNAVAILABLE = "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def read_historical_best_metrics(results_path: Path) -> dict[str, Any]:
    """Read the canonical historical result without mixing it with re-validation."""
    frame = pd.read_csv(results_path)
    frame.columns = [column.strip() for column in frame.columns]
    metric_columns = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
    }
    missing = [column for column in metric_columns.values() if column not in frame]
    if missing:
        raise ValueError(f"Historical results.csv lacks columns: {missing}")
    best = frame.loc[frame[metric_columns["mAP50-95"]].idxmax()]
    return {
        "source": "E001 historical training results.csv; row with maximum mAP50-95",
        "best_epoch": int(best["epoch"]),
        **{name: float(best[column]) for name, column in metric_columns.items()},
    }


def xywhn_to_xyxy(box: Iterable[float]) -> np.ndarray:
    x, y, w, h = (float(value) for value in box)
    return np.asarray([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=float)


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    a = np.asarray(list(box_a), dtype=float)
    b = np.asarray(list(box_b), dtype=float)
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def size_category(area_ref: float) -> str:
    if area_ref < 256:
        return "tiny"
    if area_ref < 1024:
        return "small"
    if area_ref < 9216:
        return "medium"
    return "large"


def index_images(image_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem: {path.stem}")
            result[path.stem] = path
    return result


def validate_clean_ground_truth(
    ground_truth: list[dict[str, Any]], label_dir: Path
) -> dict[str, Any]:
    """Confirm that exported GT exactly represents the local clean labels."""
    checked_objects = 0
    for image_row in ground_truth:
        stem = image_row["image_stem"]
        label_path = label_dir / f"{stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing clean label for fixed val: {label_path}")
        local_rows = []
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid clean-label row in {label_path.name}: {line}")
            local_rows.append((int(values[0]), *(round(float(value), 8) for value in values[1:])))
        export_rows = [
            (
                int(box["class_id"]),
                *(round(float(value), 8) for value in box["bbox_xywhn"]),
            )
            for box in image_row["ground_truth"]
        ]
        if Counter(local_rows) != Counter(export_rows):
            raise ValueError(f"Exported GT differs from labels_clean: {stem}")
        checked_objects += len(local_rows)
    return {
        "clean_label_match": True,
        "clean_label_images_checked": len(ground_truth),
        "clean_label_objects_checked": checked_objects,
    }


def decorate_box(item: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    result = dict(item)
    result["box_xyxy"] = xywhn_to_xyxy(item["bbox_xywhn"])
    _, _, norm_w, norm_h = (float(value) for value in item["bbox_xywhn"])
    scale = min(REFERENCE_IMGSZ / width, REFERENCE_IMGSZ / height)
    area_ref = norm_w * width * scale * norm_h * height * scale
    result["area_ref"] = area_ref
    result["size_bin"] = size_category(area_ref)
    return result


def validate_inputs(
    export_dir: Path,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    val_stems: list[str],
    per_class: pd.DataFrame,
) -> dict[str, Any]:
    pred_stems = [row["image_stem"] for row in predictions]
    gt_stems = [row["image_stem"] for row in ground_truth]
    expected = set(val_stems)
    if len(val_stems) != 400 or len(expected) != 400:
        raise ValueError("Fixed val split must contain 400 unique stems")
    if set(pred_stems) != expected or len(pred_stems) != 400:
        raise ValueError("Prediction stems do not exactly match fixed val")
    if set(gt_stems) != expected or len(gt_stems) != 400:
        raise ValueError("Ground-truth stems do not exactly match fixed val")
    if len(per_class) != len(CLASS_NAMES):
        raise ValueError("Per-class metrics must contain exactly 12 rows")

    gt_support = Counter(
        int(box["class_id"]) for row in ground_truth for box in row["ground_truth"]
    )
    metric_support = {
        int(row.class_id): int(row.GT_support) for row in per_class.itertuples()
    }
    if dict(sorted(gt_support.items())) != metric_support:
        raise ValueError("Per-class GT support does not match val_ground_truth.json")

    all_predictions = [box for row in predictions for box in row["predictions"]]
    if not all_predictions:
        raise ValueError("No validation predictions found")
    min_conf = min(float(box["confidence"]) for box in all_predictions)
    if min_conf > 0.01:
        raise ValueError("Predictions were exported at too high a confidence threshold")
    for box in all_predictions:
        if int(box["class_id"]) not in range(len(CLASS_NAMES)):
            raise ValueError("Prediction contains invalid class id")
        if len(box["bbox_xywhn"]) != 4:
            raise ValueError("Prediction bbox must be normalized xywh")

    source_files = (
        "args.yaml",
        "results.csv",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "validation_metrics.json",
        "val_predictions.json",
        "val_ground_truth.json",
    )
    for name in source_files:
        if not (export_dir / name).is_file():
            raise FileNotFoundError(export_dir / name)
    return {
        "val_images": 400,
        "ground_truth_objects": sum(gt_support.values()),
        "prediction_objects": len(all_predictions),
        "minimum_export_confidence": min_conf,
        "fixed_val_match": True,
        "source_files": {name: sha256(export_dir / name) for name in source_files},
    }


def match_at_operating_point(
    predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used_gt: set[int] = set()
    pred_results: list[dict[str, Any]] = []
    candidates = sorted(
        [p for p in predictions if float(p["confidence"]) >= OPERATING_CONF],
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    for pred_index, pred in enumerate(candidates):
        same_class = [
            (gt_index, box_iou(pred["box_xyxy"], gt["box_xyxy"]))
            for gt_index, gt in enumerate(ground_truth)
            if gt_index not in used_gt and int(gt["class_id"]) == int(pred["class_id"])
        ]
        best_index, best_iou = max(same_class, key=lambda pair: pair[1], default=(-1, 0.0))
        status = "FP"
        matched_gt = -1
        if best_index >= 0 and best_iou >= MATCH_IOU:
            status = "TP"
            matched_gt = best_index
            used_gt.add(best_index)

        any_matches = [
            (gt_index, box_iou(pred["box_xyxy"], gt["box_xyxy"]))
            for gt_index, gt in enumerate(ground_truth)
        ]
        any_index, any_iou = max(any_matches, key=lambda pair: pair[1], default=(-1, 0.0))
        reason = ""
        if status == "FP":
            if any_index >= 0 and any_iou >= MATCH_IOU:
                if int(ground_truth[any_index]["class_id"]) != int(pred["class_id"]):
                    reason = "class_confusion"
                else:
                    reason = "duplicate_detection"
            elif any_iou >= 0.10:
                reason = "localization"
            else:
                reason = "background"
        pred_results.append(
            {
                **pred,
                "prediction_index": pred_index,
                "status": status,
                "matched_gt_index": matched_gt,
                "matched_iou": best_iou if status == "TP" else 0.0,
                "best_any_gt_index": any_index,
                "best_any_iou": any_iou,
                "fp_reason": reason,
            }
        )

    gt_results: list[dict[str, Any]] = []
    for gt_index, gt in enumerate(ground_truth):
        matched = next(
            (row for row in pred_results if row["matched_gt_index"] == gt_index), None
        )
        overlaps = [
            (pred, box_iou(pred["box_xyxy"], gt["box_xyxy"]))
            for pred in predictions
        ]
        best_pred, best_any_iou = max(overlaps, key=lambda pair: pair[1], default=(None, 0.0))
        gt_results.append(
            {
                **gt,
                "ground_truth_index": gt_index,
                "status": "TP" if matched else "FN",
                "matched_confidence": float(matched["confidence"]) if matched else 0.0,
                "matched_iou": float(matched["matched_iou"]) if matched else 0.0,
                "best_prediction_class": (
                    int(best_pred["class_id"]) if best_pred is not None else -1
                ),
                "best_prediction_confidence": (
                    float(best_pred["confidence"]) if best_pred is not None else 0.0
                ),
                "best_prediction_iou": best_any_iou,
            }
        )
    return pred_results, gt_results


def touching_boundary(box_xyxy: Iterable[float], tolerance: float = 0.01) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return x1 <= tolerance or y1 <= tolerance or x2 >= 1 - tolerance or y2 >= 1 - tolerance


def image_statistics(image: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def max_pairwise_iou(boxes: list[dict[str, Any]]) -> float:
    result = 0.0
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            result = max(result, box_iou(boxes[first]["box_xyxy"], boxes[second]["box_xyxy"]))
    return result


def draw_overlay(
    image_path: Path,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {image_path}")
    height, width = image.shape[:2]
    canvas = image.copy()
    for gt in gt_rows:
        x1, y1, x2, y2 = gt["box_xyxy"]
        points = tuple(int(round(v)) for v in (x1 * width, y1 * height, x2 * width, y2 * height))
        cv2.rectangle(canvas, points[:2], points[2:], (0, 210, 0), 2)
        label = f"GT {CLASS_NAMES[int(gt['class_id'])]} {gt['status']}"
        cv2.putText(canvas, label, (points[0], max(18, points[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 0), 1, cv2.LINE_AA)
    for pred in pred_rows:
        x1, y1, x2, y2 = pred["box_xyxy"]
        points = tuple(int(round(v)) for v in (x1 * width, y1 * height, x2 * width, y2 * height))
        color = (230, 120, 20) if pred["status"] == "TP" else (20, 20, 230)
        cv2.rectangle(canvas, points[:2], points[2:], color, 2)
        label = f"P {CLASS_NAMES[int(pred['class_id'])]} {pred['confidence']:.2f} {pred['status']}"
        cv2.putText(canvas, label, (points[0], min(height - 5, points[1] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    banner = np.full((42, width, 3), 245, dtype=np.uint8)
    cv2.putText(banner, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), np.vstack([banner, canvas]))


def plot_per_class(metrics: pd.DataFrame, output_dir: Path) -> None:
    ordered = metrics.sort_values("AP50-95")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.barh(y - 0.18, ordered["AP50"], height=0.36, label="AP50", color="#4C78A8")
    ax.barh(y + 0.18, ordered["AP50-95"], height=0.36, label="AP50-95", color="#F58518")
    ax.set_yticks(y, ordered["class_name"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average precision")
    ax.set_title("E001 CLEAN per-class validation AP")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"per_class_ap.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_relation(metrics: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    support = metrics["GT_support"].astype(float)
    ap = metrics["AP50-95"].astype(float)
    log_support = np.log10(support)
    pearson = float(np.corrcoef(log_support, ap)[0, 1])
    spearman = float(pd.Series(support).rank().corr(pd.Series(ap).rank()))
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.scatter(support, ap, s=55, color="#4C78A8", edgecolor="white", linewidth=0.8)
    for _, row in metrics.iterrows():
        ax.annotate(row["class_name"], (row["GT_support"], row["AP50-95"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("GT support (log scale)")
    ax.set_ylabel("AP50-95")
    ax.set_ylim(0, 0.65)
    ax.set_title(f"Class frequency vs AP (Spearman r={spearman:.3f})")
    ax.grid(alpha=0.25)
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"class_frequency_vs_ap.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pearson_log_support_vs_ap50_95": pearson, "spearman_support_vs_ap50_95": spearman}


def plot_size_performance(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [row["size_bin"] for row in rows]
    precision = [row["precision_at_conf_0_25_iou_0_50"] for row in rows]
    recall = [row["recall_at_conf_0_25_iou_0_50"] for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.bar(x - 0.18, precision, width=0.36, label="Precision", color="#59A14F")
    ax.bar(x + 0.18, recall, width=0.36, label="Recall", color="#E15759")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title("Internal size-binned operating-point performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"size_performance.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    export_dir = args.export_dir.resolve()
    image_dir = args.image_dir.resolve()
    label_dir = args.label_dir.resolve()
    output_dir = args.output_dir.resolve()
    hard_dir = output_dir / "hard_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    hard_dir.mkdir(parents=True, exist_ok=True)

    predictions_raw = read_json(export_dir / "val_predictions.json")
    ground_truth_raw = read_json(export_dir / "val_ground_truth.json")
    per_class = pd.read_csv(export_dir / "per_class_metrics.csv")
    validation_metrics = read_json(export_dir / "validation_metrics.json")
    historical_metrics = read_historical_best_metrics(export_dir / "results.csv")
    val_stems = [
        Path(line.strip()).stem
        for line in args.val_split.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    validation = validate_inputs(export_dir, predictions_raw, ground_truth_raw, val_stems, per_class)
    validation.update(validate_clean_ground_truth(ground_truth_raw, label_dir))
    image_paths = index_images(image_dir)
    missing_images = sorted(set(val_stems) - set(image_paths))
    if missing_images:
        raise FileNotFoundError(f"Missing fixed-val images: {missing_images[:5]}")

    prediction_by_stem = {row["image_stem"]: row["predictions"] for row in predictions_raw}
    gt_by_stem = {row["image_stem"]: row["ground_truth"] for row in ground_truth_raw}
    all_pred_results: list[dict[str, Any]] = []
    all_gt_results: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    matched_by_stem: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    for stem in val_stems:
        image = cv2.imread(str(image_paths[stem]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable fixed-val image: {image_paths[stem]}")
        height, width = image.shape[:2]
        predictions = [decorate_box(item, width, height) for item in prediction_by_stem[stem]]
        ground_truth = [decorate_box(item, width, height) for item in gt_by_stem[stem]]
        pred_results, gt_results = match_at_operating_point(predictions, ground_truth)
        for row in pred_results:
            row["image_stem"] = stem
        for row in gt_results:
            row["image_stem"] = stem
            row["touches_boundary"] = touching_boundary(row["box_xyxy"])
        mean_luma, contrast = image_statistics(image)
        image_rows.append(
            {
                "image_stem": stem,
                "width": width,
                "height": height,
                "mean_luminance": mean_luma,
                "grayscale_std": contrast,
                "gt_count": len(gt_results),
                "tp_count": sum(row["status"] == "TP" for row in gt_results),
                "fn_count": sum(row["status"] == "FN" for row in gt_results),
                "fp_count": sum(row["status"] == "FP" for row in pred_results),
                "max_gt_pairwise_iou": max_pairwise_iou(gt_results),
            }
        )
        matched_by_stem[stem] = (pred_results, gt_results)
        all_pred_results.extend(pred_results)
        all_gt_results.extend(gt_results)

    per_class = per_class.sort_values("class_id")
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    confusion = pd.read_csv(export_dir / "confusion_matrix.csv", index_col=0)
    confusion.to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    imported_confusion_tp = int(round(sum(float(confusion.loc[name, name]) for name in CLASS_NAMES)))
    imported_confusion_misses = int(round(sum(float(confusion.loc["background", name]) for name in CLASS_NAMES)))
    confusion_pairs: list[dict[str, Any]] = []
    for predicted in CLASS_NAMES:
        for true in CLASS_NAMES:
            if predicted == true:
                continue
            count = int(round(float(confusion.loc[predicted, true])))
            if count:
                confusion_pairs.append({"true_class": true, "predicted_class": predicted, "count": count})
    confusion_pairs.sort(key=lambda row: (-row["count"], row["true_class"], row["predicted_class"]))
    write_csv(output_dir / "confusion_pairs.csv", confusion_pairs, ("true_class", "predicted_class", "count"))
    shutil.copy2(export_dir / "confusion_matrix.png", output_dir / "confusion_matrix.png")
    shutil.copy2(export_dir / "confusion_matrix_normalized.png", output_dir / "confusion_matrix_normalized.png")

    high_fp = [row for row in all_pred_results if row["status"] == "FP" and row["confidence"] >= HIGH_CONF_FP]
    high_fp.sort(key=lambda row: float(row["confidence"]), reverse=True)
    high_fp_rows = [
        {
            "image_stem": row["image_stem"],
            "class_id": row["class_id"],
            "class_name": CLASS_NAMES[int(row["class_id"])],
            "confidence": row["confidence"],
            "bbox_xywhn": json.dumps(row["bbox_xywhn"]),
            "fp_reason": row["fp_reason"],
            "best_any_iou": row["best_any_iou"],
            "best_gt_class": (
                CLASS_NAMES[int(matched_by_stem[row["image_stem"]][1][row["best_any_gt_index"]]["class_id"])]
                if row["best_any_gt_index"] >= 0 else ""
            ),
        }
        for row in high_fp
    ]
    write_csv(output_dir / "high_confidence_fp.csv", high_fp_rows, high_fp_rows[0].keys() if high_fp_rows else ())

    fn_rows = [row for row in all_gt_results if row["status"] == "FN"]
    fn_rows.sort(key=lambda row: (SIZE_ORDER.index(row["size_bin"]), row["best_prediction_confidence"]))
    representative_fn = [
        {
            "image_stem": row["image_stem"],
            "class_id": row["class_id"],
            "class_name": CLASS_NAMES[int(row["class_id"])],
            "size_bin": row["size_bin"],
            "area_ref_px2": row["area_ref"],
            "touches_boundary": row["touches_boundary"],
            "bbox_xywhn": json.dumps(row["bbox_xywhn"]),
            "best_prediction_class": (
                CLASS_NAMES[int(row["best_prediction_class"])] if row["best_prediction_class"] >= 0 else ""
            ),
            "best_prediction_confidence": row["best_prediction_confidence"],
            "best_prediction_iou": row["best_prediction_iou"],
        }
        for row in fn_rows
    ]
    write_csv(output_dir / "representative_fn.csv", representative_fn, representative_fn[0].keys() if representative_fn else ())

    size_rows: list[dict[str, Any]] = []
    class_size_rows: list[dict[str, Any]] = []
    for size_bin in SIZE_ORDER:
        gt_subset = [row for row in all_gt_results if row["size_bin"] == size_bin]
        pred_subset = [row for row in all_pred_results if row["size_bin"] == size_bin]
        tp = sum(row["status"] == "TP" for row in gt_subset)
        fn = len(gt_subset) - tp
        fp = sum(row["status"] == "FP" for row in pred_subset)
        size_rows.append(
            {
                "size_bin": size_bin,
                "GT_support": len(gt_subset),
                "TP": tp,
                "FN": fn,
                "FP_by_prediction_size": fp,
                "precision_at_conf_0_25_iou_0_50": tp / (tp + fp) if tp + fp else 0.0,
                "recall_at_conf_0_25_iou_0_50": tp / len(gt_subset) if gt_subset else 0.0,
                "mean_matched_iou": float(np.mean([row["matched_iou"] for row in gt_subset if row["status"] == "TP"])) if tp else 0.0,
            }
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            subset = [row for row in gt_subset if int(row["class_id"]) == class_id]
            class_tp = sum(row["status"] == "TP" for row in subset)
            class_size_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "size_bin": size_bin,
                    "GT_support": len(subset),
                    "TP": class_tp,
                    "FN": len(subset) - class_tp,
                    "recall_at_conf_0_25_iou_0_50": class_tp / len(subset) if subset else "",
                }
            )
    write_csv(output_dir / "size_performance.csv", size_rows, size_rows[0].keys())
    write_csv(output_dir / "class_size_performance.csv", class_size_rows, class_size_rows[0].keys())

    image_frame = pd.DataFrame(image_rows)
    image_frame.to_csv(output_dir / "image_difficulty_metrics.csv", index=False, encoding="utf-8-sig")
    low_light_cutoff = float(image_frame["mean_luminance"].quantile(0.10))
    low_contrast_cutoff = float(image_frame["grayscale_std"].quantile(0.10))
    low_light_subset_count = int((image_frame["mean_luminance"] <= low_light_cutoff).sum())
    image_lookup = {row["image_stem"]: row for row in image_rows}

    hard_candidates: dict[str, list[str]] = {
        "tiny_target": [row["image_stem"] for row in fn_rows if row["size_bin"] == "tiny"],
        "occlusion": [row["image_stem"] for row in sorted(image_rows, key=lambda item: item["max_gt_pairwise_iou"], reverse=True) if row["fn_count"] > 0],
        "low_light": [row["image_stem"] for row in sorted(image_rows, key=lambda item: item["mean_luminance"]) if row["mean_luminance"] <= low_light_cutoff and row["fn_count"] > 0],
        "low_contrast": [row["image_stem"] for row in sorted(image_rows, key=lambda item: item["grayscale_std"]) if row["grayscale_std"] <= low_contrast_cutoff and row["fn_count"] > 0],
        "crowded_scene": [row["image_stem"] for row in sorted(image_rows, key=lambda item: item["gt_count"], reverse=True) if row["fn_count"] > 0],
        "similar_classes": [row["image_stem"] for row in high_fp if row["fp_reason"] == "class_confusion"],
        "boundary_object": [row["image_stem"] for row in fn_rows if row["touches_boundary"]],
    }
    hard_rows: list[dict[str, Any]] = []
    for category, candidates in hard_candidates.items():
        selected: list[str] = []
        for stem in candidates:
            if stem not in selected:
                selected.append(stem)
            if len(selected) == 3:
                break
        for rank, stem in enumerate(selected, start=1):
            pred_rows, gt_rows = matched_by_stem[stem]
            filename = f"{category}_{rank}_{stem}.jpg"
            details = image_lookup[stem]
            draw_overlay(
                image_paths[stem],
                gt_rows,
                pred_rows,
                hard_dir / filename,
                f"{category} | {stem} | GT={details['gt_count']} FN={details['fn_count']} FP={details['fp_count']}",
            )
            hard_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "image_stem": stem,
                    "preview": f"hard_cases/{filename}",
                    "GT_count": details["gt_count"],
                    "FN_count": details["fn_count"],
                    "FP_count": details["fp_count"],
                    "mean_luminance": details["mean_luminance"],
                    "grayscale_std": details["grayscale_std"],
                    "max_gt_pairwise_iou": details["max_gt_pairwise_iou"],
                }
            )
    write_csv(output_dir / "hard_cases.csv", hard_rows, hard_rows[0].keys())

    plot_per_class(per_class, output_dir)
    correlations = plot_frequency_relation(per_class, output_dir)
    plot_size_performance(size_rows, output_dir)
    correlations["size_order_vs_recall_spearman"] = float(
        pd.Series(range(len(SIZE_ORDER))).corr(
            pd.Series([row["recall_at_conf_0_25_iou_0_50"] for row in size_rows]).rank()
        )
    )

    lowest_ap = per_class.nsmallest(5, "AP50-95")
    fn_class_counts = Counter(int(row["class_id"]) for row in fn_rows)
    high_fp_reasons = Counter(row["fp_reason"] for row in high_fp)
    internal_tp = sum(row["status"] == "TP" for row in all_gt_results)
    provenance = {
        "historical_training_result": {
            **historical_metrics,
            "experiment": "E001_RGB_YOLO11N_CLEAN",
            "checkpoint_path": "runs/E001_RGB_YOLO11N_CLEAN/weights/best.pt",
            "checkpoint_sha256": UNAVAILABLE,
            "dataset": "official Train_2000; data/processed/train/labels_clean",
            "split": "data/splits/train.txt (1600) and data/splits/val.txt (400), seed=2026",
            "imgsz": 640,
            "device": "0 on NVIDIA GeForce RTX 3090 24GB server",
            "training_command": UNAVAILABLE,
            "python_version": UNAVAILABLE,
            "torch_version": UNAVAILABLE,
            "ultralytics_version": UNAVAILABLE,
            "config_evidence": "args.yaml in E001_CLEAN_ERROR_ANALYSIS_EXPORT and docs/EXPERIMENT_LOG.md",
        },
        "analysis_export_revalidation": {
            "metrics_source": "validation_metrics.json and per_class_metrics.csv in E001_CLEAN_ERROR_ANALYSIS_EXPORT",
            "checkpoint_path_or_identity": "canonical runs/E001_RGB_YOLO11N_CLEAN/weights/best.pt according to experiment record; exact binding/hash is unavailable in the export package",
            "validation_or_export_command": UNAVAILABLE,
            "dataset": "fixed val=400; exported GT independently matched data/processed/train/labels_clean",
            "imgsz": UNAVAILABLE,
            "prediction_export_conf": UNAVAILABLE,
            "validation_iou": UNAVAILABLE,
            "max_det": UNAVAILABLE,
            "device": UNAVAILABLE,
            "python_version": UNAVAILABLE,
            "torch_version": UNAVAILABLE,
            "ultralytics_version": UNAVAILABLE,
            "export_source": export_dir.name,
            "observed_minimum_prediction_confidence": validation["minimum_export_confidence"],
            "note": "args.yaml records the historical training run; it is not evidence of the separate re-validation/export command.",
            "metrics": validation_metrics,
        },
        "member_b_internal_matcher": {
            "implementation": "scripts/analysis/analyze_e001_errors.py",
            "prediction_confidence_gte": OPERATING_CONF,
            "matching_iou_gte": MATCH_IOU,
            "high_confidence_fp_gte": HIGH_CONF_FP,
            "TP": internal_tp,
            "FN": len(fn_rows),
            "high_confidence_FP": len(high_fp),
        },
        "imported_confusion": {
            "source": "confusion_matrix.csv in E001_CLEAN_ERROR_ANALYSIS_EXPORT",
            "threshold": UNAVAILABLE,
            "matching_settings": UNAVAILABLE,
            "diagonal_TP": imported_confusion_tp,
            "background_row_misses": imported_confusion_misses,
            "comparability_note": "Not the same evaluator/operating point as the member B internal matcher; do not equate the counts.",
        },
        "member_b_postprocessing_environment": {
            "python": "3.10.21",
            "torch": "2.14.0+cpu",
            "ultralytics": "8.4.144",
            "opencv": "5.0.0",
            "numpy": "2.2.6",
            "pandas": "2.3.3",
            "matplotlib": "3.10.9",
            "scope": "Local post-processing and tests only; not the training/re-validation environment.",
        },
    }
    summary = {
        "analysis_scope": "E001_RGB_YOLO11N_CLEAN fixed val only",
        "source_export_name": export_dir.name,
        "validation": validation,
        "operating_point": {"confidence": OPERATING_CONF, "iou": MATCH_IOU},
        "historical_training_metrics": historical_metrics,
        "analysis_export_revalidation_metrics": validation_metrics,
        "imported_confusion": provenance["imported_confusion"],
        "internal_matcher_TP": internal_tp,
        "false_negatives_at_operating_point": len(fn_rows),
        "high_confidence_false_positives": len(high_fp),
        "high_confidence_fp_reasons": dict(high_fp_reasons),
        "top_fn_classes": [
            {"class_id": class_id, "class_name": CLASS_NAMES[class_id], "count": count}
            for class_id, count in fn_class_counts.most_common(5)
        ],
        "lowest_ap50_95_classes": lowest_ap[["class_id", "class_name", "AP50-95", "GT_support"]].to_dict("records"),
        "size_performance": size_rows,
        "correlations": correlations,
        "hard_case_thresholds": {
            "low_light_luminance_p10": low_light_cutoff,
            "visible_defined_low_light_candidate_images": low_light_subset_count,
            "low_contrast_std_p10": low_contrast_cutoff,
        },
        "limitations": [
            "Per-class AP is copied from the canonical Ultralytics validation export.",
            "Size-binned precision/recall is an internal operating-point analysis, not official competition AP.",
            "Occlusion is selected by GT-overlap proxy and requires human confirmation.",
            "Correlations are exploratory and do not establish causality.",
        ],
    }
    write_json(output_dir / "analysis_summary.json", summary)
    write_json(output_dir / "source_validation.json", validation)
    write_json(output_dir / "provenance.json", provenance)

    major_confusions = confusion_pairs[:5]
    weakest = lowest_ap
    lines = [
        "# E001 CLEAN 固定验证集错误分析",
        "",
        "## 1. 范围与口径",
        "",
        "- 对象：`E001_RGB_YOLO11N_CLEAN`；",
        "- 数据：固定 `val.txt` 的 400 张图，未使用 prelim_test；",
        f"- GT：{validation['ground_truth_objects']} 个；低阈值预测：{validation['prediction_objects']} 个；",
        f"- CLEAN 校验：导出 GT 与本地 `labels_clean` 的 {validation['clean_label_images_checked']} 张、{validation['clean_label_objects_checked']} 个框逐项一致；",
        "- provenance 详见 `e001_error_analysis/provenance.json`；无法由现有证据恢复的字段统一记为 `unavailable`；",
        "- size bin 使用 `imgsz=640` letterbox 参考面积：tiny < 256，small [256,1024)，medium [1024,9216)，large >= 9216；",
        "- 未修改 labels_clean、fixed split 或任何数据文件。",
        "",
        "## 2. B1：两套指标与来源",
        "",
        "两套指标必须分开理解，本次 analysis export / re-validation 不覆盖 canonical historical training `results.csv`。",
        "",
        "| 指标来源 | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| E001 historical training `results.csv`（best epoch={historical_metrics['best_epoch']}） | {historical_metrics['precision']:.5f} | {historical_metrics['recall']:.5f} | {historical_metrics['mAP50']:.5f} | {historical_metrics['mAP50-95']:.5f} |",
        f"| B analysis export / re-validation | {validation_metrics['precision']:.5f} | {validation_metrics['recall']:.5f} | {validation_metrics['mAP50']:.5f} | {validation_metrics['mAP50-95']:.5f} |",
        "",
        "Historical result 直接取 export 中 `results.csv` 的最高 mAP50-95 行；checkpoint 记录为 `runs/E001_RGB_YOLO11N_CLEAN/weights/best.pt`，但 export 未包含 checkpoint/hash。re-validation 的指标来自 `validation_metrics.json`，逐类指标来自 `per_class_metrics.csv`。其 validation/export 命令、imgsz、conf、iou、max_det、device 及 Python/torch/Ultralytics 版本均无法由现有 export 确认，记为 `unavailable`。`args.yaml` 是历史训练配置，不能当作独立 re-validation 命令证据。",
        "",
        "### Re-validation per-class metrics",
        "",
        "| class | GT | Precision | Recall | AP50 | AP50-95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in per_class.iterrows():
        lines.append(f"| {row['class_name']} | {int(row['GT_support'])} | {row['precision']:.4f} | {row['recall']:.4f} | {row['AP50']:.4f} | {row['AP50-95']:.4f} |")
    lines.extend([
        "",
        "AP50-95 最低的类别为：" + "、".join(f"{row['class_name']} ({row['AP50-95']:.3f}, GT={int(row['GT_support'])})" for _, row in weakest.iterrows()) + "。",
        "",
        "## 3. B2：Confusion 与内部 FP/FN 的不同口径",
        "",
        "### Imported confusion",
        "",
        "- 来源：analysis export 中的 `confusion_matrix.csv`；",
        "- threshold：`unavailable`；matching 设置：`unavailable`；",
        f"- diagonal TP = {imported_confusion_tp}；background-row misses = {imported_confusion_misses}。",
        "",
        "### Member B internal FP/FN matcher",
        "",
        f"- 实现：当前 `scripts/analysis/analyze_e001_errors.py`；prediction confidence >= {OPERATING_CONF}；matching IoU >= {MATCH_IOU}；",
        f"- TP = {internal_tp}；FN = {len(fn_rows)}；",
        f"- high-confidence FP 定义为 confidence >= {HIGH_CONF_FP}，count = {len(high_fp)}。",
        "",
        "**Imported confusion 与内部 matcher 不是同一个 evaluator / operating point 的可直接比较结果，不应把两组 TP/FN 数字写成等式或要求一致。**",
        "",
        "主要非背景类别混淆：" + ("；".join(f"{row['true_class']} -> {row['predicted_class']} ({row['count']})" for row in major_confusions) if major_confusions else "未观察到明显类别间混淆") + "。",
        "完整证据见 `confusion_pairs.csv`、`high_confidence_fp.csv` 和 `representative_fn.csv`。",
        "",
        "## 4. B3：BBox-size Performance",
        "",
        "尺寸定义固定为 `imgsz=640` 的 letterbox 参考像素面积：tiny < 256 px²、small [256,1024) px²、medium [1024,9216) px²、large >= 9216 px²。以下是自定义内部工作点统计，不是标准 COCO size AP/Precision：TP/FN 按 GT size 归档，FP 按 prediction size 归档，因此表中 Precision 只用于内部诊断。",
        "",
        "| size | GT | TP | FN | FP(pred-size) | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in size_rows:
        lines.append(f"| {row['size_bin']} | {row['GT_support']} | {row['TP']} | {row['FN']} | {row['FP_by_prediction_size']} | {row['precision_at_conf_0_25_iou_0_50']:.4f} | {row['recall_at_conf_0_25_iou_0_50']:.4f} |")
    lines.extend([
        "",
        "## 5. B4：Hard Cases",
        "",
        f"`hard_cases.csv` 是定向错误案例清单，不代表固定 val 的总体分布。每类最多选择 3 个候选；派生 JPG 仅在本地按需生成且不纳入 Git。occlusion 只是 GT overlap proxy，不报告或声称正式 occlusion Recall。low-light 仅定义为 Visible 灰度均值 P10 候选子集（{low_light_subset_count} 张），目前没有正式 low-light Recall。",
        "",
        "## 6. B5：Data-performance Relation",
        "",
        f"类别在固定 val 中的 GT support 与 AP50-95 的 Spearman 相关系数为 {correlations['spearman_support_vs_ap50_95']:.3f}；log10(val GT support) 与 AP50-95 的 Pearson 相关系数为 {correlations['pearson_log_support_vs_ap50_95']:.3f}。这里的 frequency 仅指 val GT support，不是 training frequency；仅报告探索性相关，不解释为因果。",
        "",
        "## 7. B6：Multimodal Hypotheses",
        "",
        "1. 主要瓶颈是整体 Recall 明显低于 Precision、极低频类别不稳定、tiny/small 目标漏检、背景方向漏检多，以及 AP50 到 AP50-95 的定位性能下降。",
        "2. 假设：Visible-defined 低照度和低对比度候选中的漏检可能与 RGB 信息不足有关，尚未证明。",
        "3. E002 IR 假设：优先验证低照度、低对比度下的 person、animal 和其他高 FN 类别；若白天正常场景也同样漏检，则不能归因于 RGB 光照不足。",
        "4. E003 Depth 假设：优先验证拥挤、GT-overlap proxy 和前后景重叠样本；边界截断和 tiny 目标不应预设可由 Depth 自动解决。",
        "5. 重点观察低 AP/低 Recall 类别，以及 `hard_cases.csv` 中同时出现 FN 和高置信度 FP 的样本。",
        "6. Fusion 假设：首先做针对 Visible-defined low-light 与 GT-overlap proxy 子集的可证伪比较；若单模态 IR/Depth 未改善对应子集，不应直接增加 Fusion 复杂度。",
        "",
        "## 8. 限制",
        "",
        "- size Precision/Recall 是自定义固定工作点内部统计，不是标准 COCO size 指标或官方榜单指标；",
        "- hard cases 是定向错误候选，不代表总体分布；场景标签由图像统计和框几何筛选，仍需人工复核；",
        "- 12 类相关分析样本量小，只能形成假设；",
        "- 本报告没有使用 prelim_test，也没有启动新训练。",
        "",
    ])
    (output_dir.parent / "e001_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "report": str(output_dir.parent / 'e001_error_analysis.md'), "val_images": 400, "gt": len(all_gt_results), "predictions": validation['prediction_objects'], "fn": len(fn_rows), "high_conf_fp": len(high_fp)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
