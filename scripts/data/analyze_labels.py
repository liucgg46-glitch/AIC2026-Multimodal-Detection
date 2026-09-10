"""Validate and summarize AIC2026 YOLO training labels.

The script is read-only with respect to ``data/raw``. It validates the official
five-field label format, records every issue with file and line number, and
writes reproducible JSON/CSV reports under ``outputs/analysis``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = REPO_ROOT / "data" / "raw" / "train" / "labels"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "raw" / "train" / "visible"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
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
ISSUE_FIELDS = (
    "stem",
    "file",
    "line",
    "severity",
    "issue_type",
    "details",
    "content",
)
BOUNDARY_EPSILON = 1e-6
REFERENCE_IMAGE_SIZE = 640
SIZE_CATEGORY_ORDER = ("tiny", "small", "medium", "large")
SIZE_THRESHOLDS_PX2 = {
    "tiny_upper_exclusive": 256.0,
    "small_upper_exclusive": 1024.0,
    "medium_upper_exclusive": 9216.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AIC2026 labels without modifying official data."
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help=f"Label directory (default: {DEFAULT_LABEL_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Report directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Visible image directory used for pixel bbox sizes (default: {DEFAULT_IMAGE_DIR})",
    )
    return parser.parse_args()


def add_issue(
    issues: list[dict[str, Any]],
    *,
    path: Path,
    line_number: int | None,
    severity: str = "error",
    issue_type: str,
    details: str,
    content: str = "",
) -> None:
    issues.append(
        {
            "stem": path.stem,
            "file": str(path),
            "line": "" if line_number is None else line_number,
            "severity": severity,
            "issue_type": issue_type,
            "details": details,
            "content": content,
        }
    )


def index_image_sizes(image_dir: Path) -> dict[str, tuple[int, int]]:
    """Return case-insensitive stem -> (width, height) without changing images."""
    sizes: dict[str, tuple[int, int]] = {}
    for path in tqdm(
        sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()),
        desc="Reading image sizes",
        unit="file",
    ):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in sizes:
            raise ValueError(f"Duplicate visible-image stem: {path.stem}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read visible image: {path}")
        height, width = image.shape[:2]
        sizes[key] = (int(width), int(height))
    return sizes


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def classify_reference_area(area_px: float) -> str:
    """Classify a bbox at the fixed 640x640 letterbox reference scale."""
    if area_px < SIZE_THRESHOLDS_PX2["tiny_upper_exclusive"]:
        return "tiny"
    if area_px < SIZE_THRESHOLDS_PX2["small_upper_exclusive"]:
        return "small"
    if area_px < SIZE_THRESHOLDS_PX2["medium_upper_exclusive"]:
        return "medium"
    return "large"


def validate_labels(
    label_dir: Path, image_sizes: dict[str, tuple[int, int]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    label_paths = sorted(
        (
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".txt"
        ),
        key=lambda path: path.name.casefold(),
    )
    issues: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    class_object_counts: Counter[int] = Counter()
    class_image_stems: dict[int, set[str]] = defaultdict(set)
    objects_per_image: list[int] = []
    empty_label_files: list[str] = []
    total_nonempty_rows = 0
    valid_object_count = 0
    bbox_records: list[dict[str, Any]] = []

    for path in tqdm(label_paths, desc="Validating labels", unit="file"):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            add_issue(
                issues,
                path=path,
                line_number=None,
                issue_type="unreadable_label_file",
                details=f"{type(exc).__name__}: {exc}",
            )
            issue_counts["unreadable_label_file"] += 1
            objects_per_image.append(0)
            continue

        rows = [(number, line.strip()) for number, line in enumerate(text.splitlines(), 1)]
        nonempty_rows = [(number, line) for number, line in rows if line]
        if not nonempty_rows:
            empty_label_files.append(path.stem)
            objects_per_image.append(0)
            continue

        valid_in_image = 0
        seen_annotations: dict[tuple[int, float, float, float, float], int] = {}
        total_nonempty_rows += len(nonempty_rows)

        for line_number, line in nonempty_rows:
            fields = line.split()
            if len(fields) != 5:
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="invalid_field_count",
                    details=f"expected 5 fields, got {len(fields)}",
                    content=line,
                )
                issue_counts["invalid_field_count"] += 1
                continue

            class_token, *bbox_tokens = fields
            try:
                class_id = int(class_token)
            except ValueError:
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="invalid_class_id",
                    details="class_id must be an integer in [0, 11]",
                    content=line,
                )
                issue_counts["invalid_class_id"] += 1
                continue
            if not 0 <= class_id < len(CLASS_NAMES):
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="invalid_class_id",
                    details=f"class_id={class_id}, expected [0, 11]",
                    content=line,
                )
                issue_counts["invalid_class_id"] += 1
                continue

            try:
                x, y, width, height = (float(token) for token in bbox_tokens)
            except ValueError:
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="non_numeric_bbox",
                    details="x, y, width and height must be numeric",
                    content=line,
                )
                issue_counts["non_numeric_bbox"] += 1
                continue

            values = (x, y, width, height)
            if not all(math.isfinite(value) for value in values):
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="non_finite_bbox",
                    details=f"bbox contains NaN or infinity: {values}",
                    content=line,
                )
                issue_counts["non_finite_bbox"] += 1
                continue

            row_has_error = False
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="invalid_center_coordinate",
                    details=f"center=({x}, {y}), expected both values in [0, 1]",
                    content=line,
                )
                issue_counts["invalid_center_coordinate"] += 1
                row_has_error = True
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    issue_type="invalid_bbox_size",
                    details=f"size=({width}, {height}), expected both values in (0, 1]",
                    content=line,
                )
                issue_counts["invalid_bbox_size"] += 1
                row_has_error = True

            bbox_out_of_bounds = False
            x_min = x - width / 2.0
            x_max = x + width / 2.0
            y_min = y - height / 2.0
            y_max = y + height / 2.0
            if not row_has_error:
                if (
                    x_min < -BOUNDARY_EPSILON
                    or y_min < -BOUNDARY_EPSILON
                    or x_max > 1.0 + BOUNDARY_EPSILON
                    or y_max > 1.0 + BOUNDARY_EPSILON
                ):
                    bbox_out_of_bounds = True
                    add_issue(
                        issues,
                        path=path,
                        line_number=line_number,
                        severity="warning",
                        issue_type="bbox_out_of_bounds",
                        details=(
                            f"edges=({x_min:.8g}, {y_min:.8g}, "
                            f"{x_max:.8g}, {y_max:.8g})"
                        ),
                        content=line,
                    )
                    issue_counts["bbox_out_of_bounds"] += 1

            annotation = (class_id, x, y, width, height)
            if annotation in seen_annotations:
                add_issue(
                    issues,
                    path=path,
                    line_number=line_number,
                    severity="warning",
                    issue_type="duplicate_annotation",
                    details=f"duplicates line {seen_annotations[annotation]}",
                    content=line,
                )
                issue_counts["duplicate_annotation"] += 1
            else:
                seen_annotations[annotation] = line_number

            if not row_has_error:
                valid_object_count += 1
                valid_in_image += 1
                class_object_counts[class_id] += 1
                class_image_stems[class_id].add(path.stem)
                image_size = image_sizes.get(path.stem.casefold())
                if image_size is None:
                    raise ValueError(f"Missing visible image for label stem: {path.stem}")
                image_width, image_height = image_size
                bbox_width_px = width * image_width
                bbox_height_px = height * image_height
                letterbox_scale = min(
                    REFERENCE_IMAGE_SIZE / image_width,
                    REFERENCE_IMAGE_SIZE / image_height,
                )
                bbox_width_ref = bbox_width_px * letterbox_scale
                bbox_height_ref = bbox_height_px * letterbox_scale
                bbox_area_ref = bbox_width_ref * bbox_height_ref
                size_category = classify_reference_area(bbox_area_ref)
                bbox_records.append(
                    {
                        "stem": path.stem,
                        "line": line_number,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "image_width": image_width,
                        "image_height": image_height,
                        "norm_center_x": x,
                        "norm_center_y": y,
                        "norm_w": width,
                        "norm_h": height,
                        "relative_area": width * height,
                        "bbox_width_px": bbox_width_px,
                        "bbox_height_px": bbox_height_px,
                        "bbox_area_px": bbox_width_px * bbox_height_px,
                        "aspect_ratio_w_over_h": bbox_width_px / bbox_height_px,
                        "reference_imgsz": REFERENCE_IMAGE_SIZE,
                        "letterbox_scale": letterbox_scale,
                        "bbox_width_ref_px": bbox_width_ref,
                        "bbox_height_ref_px": bbox_height_ref,
                        "bbox_area_ref_px2": bbox_area_ref,
                        "size_category": size_category,
                        "bbox_out_of_bounds": bbox_out_of_bounds,
                        "overflow_left_px": max(0.0, -x_min * image_width),
                        "overflow_top_px": max(0.0, -y_min * image_height),
                        "overflow_right_px": max(0.0, (x_max - 1.0) * image_width),
                        "overflow_bottom_px": max(0.0, (y_max - 1.0) * image_height),
                    }
                )

        objects_per_image.append(valid_in_image)

    sorted_counts = sorted(objects_per_image)
    image_count = len(objects_per_image)
    median = None
    if image_count:
        middle = image_count // 2
        median = (
            float(sorted_counts[middle])
            if image_count % 2
            else (sorted_counts[middle - 1] + sorted_counts[middle]) / 2.0
        )

    class_counts = [
        {
            "class_id": class_id,
            "class_name": class_name,
            "object_count": class_object_counts[class_id],
            "image_count": len(class_image_stems[class_id]),
            "object_fraction": (
                class_object_counts[class_id] / valid_object_count
                if valid_object_count
                else 0.0
            ),
            "image_fraction": (
                len(class_image_stems[class_id]) / len(label_paths)
                if label_paths
                else 0.0
            ),
        }
        for class_id, class_name in enumerate(CLASS_NAMES)
    ]
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    nonzero_class_counts = [
        item["object_count"] for item in class_counts if item["object_count"] > 0
    ]
    size_counts = Counter(record["size_category"] for record in bbox_records)
    size_distribution = [
        {
            "size_category": category,
            "bbox_count": size_counts[category],
            "bbox_fraction": (
                size_counts[category] / valid_object_count
                if valid_object_count
                else 0.0
            ),
        }
        for category in SIZE_CATEGORY_ORDER
    ]
    class_size_distribution = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_records = [
            record for record in bbox_records if record["class_id"] == class_id
        ]
        category_counts = Counter(
            record["size_category"] for record in class_records
        )
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "object_count": len(class_records),
        }
        for category in SIZE_CATEGORY_ORDER:
            row[f"{category}_count"] = category_counts[category]
            row[f"{category}_fraction"] = (
                category_counts[category] / len(class_records)
                if class_records
                else 0.0
            )
        class_size_distribution.append(row)
    summary = {
        "label_dir": str(label_dir.resolve()),
        "label_file_count": len(label_paths),
        "empty_label_file_count": len(empty_label_files),
        "empty_label_stems": empty_label_files,
        "total_nonempty_rows": total_nonempty_rows,
        "valid_object_count": valid_object_count,
        "invalid_row_count": total_nonempty_rows - valid_object_count,
        "issue_count": len(issues),
        "error_count": error_count,
        "warning_count": warning_count,
        "issue_counts": dict(sorted(issue_counts.items())),
        "objects_per_image": {
            "min": min(objects_per_image) if objects_per_image else None,
            "max": max(objects_per_image) if objects_per_image else None,
            "mean": sum(objects_per_image) / image_count if image_count else None,
            "median": median,
        },
        "class_counts": class_counts,
        "class_imbalance": {
            "largest_class": max(
                class_counts, key=lambda item: item["object_count"], default=None
            ),
            "smallest_nonzero_class": min(
                (item for item in class_counts if item["object_count"] > 0),
                key=lambda item: item["object_count"],
                default=None,
            ),
            "max_to_min_nonzero_ratio": (
                max(nonzero_class_counts) / min(nonzero_class_counts)
                if nonzero_class_counts
                else None
            ),
        },
        "bbox_statistics": {
            "width_px": summarize_metric(
                [record["bbox_width_px"] for record in bbox_records]
            ),
            "height_px": summarize_metric(
                [record["bbox_height_px"] for record in bbox_records]
            ),
            "area_px": summarize_metric(
                [record["bbox_area_px"] for record in bbox_records]
            ),
            "relative_area": summarize_metric(
                [record["relative_area"] for record in bbox_records]
            ),
            "aspect_ratio_w_over_h": summarize_metric(
                [record["aspect_ratio_w_over_h"] for record in bbox_records]
            ),
            "reference_area_px2": summarize_metric(
                [record["bbox_area_ref_px2"] for record in bbox_records]
            ),
        },
        "size_categories": {
            "status": "team_standard_defined",
            "official_competition_standard": False,
            "reference_imgsz": REFERENCE_IMAGE_SIZE,
            "scaling": "scale=min(reference_imgsz/W, reference_imgsz/H)",
            "thresholds_px2": {
                "tiny": "area < 256",
                "small": "256 <= area < 1024",
                "medium": "1024 <= area < 9216",
                "large": "area >= 9216",
            },
            "distribution": size_distribution,
            "per_class_distribution": class_size_distribution,
        },
        "passed": error_count == 0,
        "notes": [
            "Empty label files are counted but are not treated as invalid.",
            "BBox edge checks use a 1e-6 tolerance for decimal rounding.",
            "Out-of-bounds edges and exact duplicates are quality warnings; they do not invalidate rows that satisfy the official five-field value rules.",
        ],
    }
    return summary, issues, bbox_records


def write_reports(
    output_dir: Path,
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
    bbox_records: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "label_check_summary.json"
    dataset_summary_path = output_dir / "dataset_summary.json"
    issues_path = output_dir / "label_issues.csv"
    class_counts_path = output_dir / "class_counts.csv"
    bbox_stats_path = output_dir / "bbox_stats.csv"
    size_distribution_path = output_dir / "size_distribution.csv"
    class_size_distribution_path = output_dir / "class_size_distribution.csv"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dataset_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with issues_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUE_FIELDS)
        writer.writeheader()
        writer.writerows(issues)
    with class_counts_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "class_id",
                "class_name",
                "object_count",
                "image_count",
                "object_fraction",
                "image_fraction",
            ),
        )
        writer.writeheader()
        writer.writerows(summary["class_counts"])
    bbox_fields = (
        "stem",
        "line",
        "class_id",
        "class_name",
        "image_width",
        "image_height",
        "norm_center_x",
        "norm_center_y",
        "norm_w",
        "norm_h",
        "relative_area",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_px",
        "aspect_ratio_w_over_h",
        "reference_imgsz",
        "letterbox_scale",
        "bbox_width_ref_px",
        "bbox_height_ref_px",
        "bbox_area_ref_px2",
        "size_category",
        "bbox_out_of_bounds",
        "overflow_left_px",
        "overflow_top_px",
        "overflow_right_px",
        "overflow_bottom_px",
    )
    with bbox_stats_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=bbox_fields)
        writer.writeheader()
        writer.writerows(bbox_records)
    with size_distribution_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("size_category", "bbox_count", "bbox_fraction"),
        )
        writer.writeheader()
        writer.writerows(summary["size_categories"]["distribution"])
    class_size_fields = ["class_id", "class_name", "object_count"]
    for category in SIZE_CATEGORY_ORDER:
        class_size_fields.extend(
            (f"{category}_count", f"{category}_fraction")
        )
    with class_size_distribution_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=class_size_fields)
        writer.writeheader()
        writer.writerows(summary["size_categories"]["per_class_distribution"])
    return (
        summary_path,
        dataset_summary_path,
        issues_path,
        class_counts_path,
        bbox_stats_path,
        size_distribution_path,
        class_size_distribution_path,
    )


def main() -> int:
    args = parse_args()
    label_dir = args.label_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not label_dir.is_dir():
        raise SystemExit(f"Label directory does not exist: {label_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {image_dir}")

    image_sizes = index_image_sizes(image_dir)
    summary, issues, bbox_records = validate_labels(label_dir, image_sizes)
    (
        summary_path,
        dataset_summary_path,
        issues_path,
        class_counts_path,
        bbox_stats_path,
        size_distribution_path,
        class_size_distribution_path,
    ) = write_reports(output_dir, summary, issues, bbox_records)

    print("\nLabel validation complete")
    print(f"  label files: {summary['label_file_count']}")
    print(f"  empty files: {summary['empty_label_file_count']}")
    print(f"  non-empty rows: {summary['total_nonempty_rows']}")
    print(f"  valid objects: {summary['valid_object_count']}")
    print(
        f"  issues: {summary['issue_count']} "
        f"({summary['error_count']} errors, {summary['warning_count']} warnings)"
    )
    print(f"  passed: {summary['passed']}")
    print(f"  summary: {summary_path}")
    print(f"  dataset summary: {dataset_summary_path}")
    print(f"  issues: {issues_path}")
    print(f"  class counts: {class_counts_path}")
    print(f"  bbox stats: {bbox_stats_path}")
    print(f"  size distribution: {size_distribution_path}")
    print(f"  class-size distribution: {class_size_distribution_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
