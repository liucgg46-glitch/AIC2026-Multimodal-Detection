"""Create the fixed AIC2026 train/val split and audit its distributions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from group_stratification import (
    build_feature_matrix,
    build_groups,
    select_validation_groups,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_DIR = REPO_ROOT / "data" / "raw" / "train" / "labels"
DEFAULT_BBOX_STATS = REPO_ROOT / "outputs" / "analysis" / "bbox_stats.csv"
DEFAULT_GROUP_PAIRS = REPO_ROOT / "outputs" / "analysis" / "group_candidate_pairs.csv"
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
DEFAULT_SEED = 2026
DEFAULT_VAL_FRACTION = 0.2
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
SIZE_CATEGORIES = ("tiny", "small", "medium", "large")
MAX_BASIC_CLASS_GAP_PP = 2.0
MAX_BASIC_SIZE_GAP_PP = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic 80/20 split and compare distributions."
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--bbox-stats", type=Path, default=DEFAULT_BBOX_STATS)
    parser.add_argument("--group-pairs", type=Path, default=DEFAULT_GROUP_PAIRS)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    return parser.parse_args()


def collect_stems(label_dir: Path) -> list[str]:
    stems = sorted(
        path.stem
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".txt"
    )
    if len(stems) != len({stem.casefold() for stem in stems}):
        raise ValueError("Label stems are not unique under case-insensitive matching")
    return stems


def create_split(
    stems: list[str], seed: int, val_fraction: float
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    shuffled = stems.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_fraction)
    val_stems = sorted(shuffled[:val_count])
    train_stems = sorted(shuffled[val_count:])
    return train_stems, val_stems


def load_bbox_records(path: Path, valid_stems: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = row["stem"]
            if stem not in valid_stems:
                raise ValueError(f"bbox_stats contains unknown stem: {stem}")
            class_id = int(row["class_id"])
            size_category = row["size_category"]
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"Invalid class_id in bbox_stats: {class_id}")
            if size_category not in SIZE_CATEGORIES:
                raise ValueError(f"Invalid size_category in bbox_stats: {size_category}")
            records.append(
                {
                    "stem": stem,
                    "class_id": class_id,
                    "size_category": size_category,
                }
            )
    return records


def safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def audit_split(
    all_stems: set[str],
    train_stems: set[str],
    val_stems: set[str],
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    split_stems = {
        "all": all_stems,
        "train": train_stems,
        "val": val_stems,
    }
    class_counts: dict[str, Counter[int]] = {}
    class_images: dict[str, dict[int, set[str]]] = {}
    size_counts: dict[str, Counter[str]] = {}
    total_objects: dict[str, int] = {}

    for split_name, stems in split_stems.items():
        selected = [record for record in records if record["stem"] in stems]
        total_objects[split_name] = len(selected)
        class_counts[split_name] = Counter(record["class_id"] for record in selected)
        size_counts[split_name] = Counter(
            record["size_category"] for record in selected
        )
        images: dict[int, set[str]] = defaultdict(set)
        for record in selected:
            images[record["class_id"]].add(record["stem"])
        class_images[split_name] = images

    class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        all_count = class_counts["all"][class_id]
        train_count = class_counts["train"][class_id]
        val_count = class_counts["val"][class_id]
        train_fraction = safe_fraction(train_count, total_objects["train"])
        val_fraction = safe_fraction(val_count, total_objects["val"])
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "all_object_count": all_count,
                "train_object_count": train_count,
                "val_object_count": val_count,
                "val_share_of_class": safe_fraction(val_count, all_count),
                "train_object_fraction": train_fraction,
                "val_object_fraction": val_fraction,
                "train_val_gap_percentage_points": abs(
                    train_fraction - val_fraction
                )
                * 100.0,
                "all_image_count": len(class_images["all"][class_id]),
                "train_image_count": len(class_images["train"][class_id]),
                "val_image_count": len(class_images["val"][class_id]),
            }
        )

    size_rows: list[dict[str, Any]] = []
    for category in SIZE_CATEGORIES:
        all_count = size_counts["all"][category]
        train_count = size_counts["train"][category]
        val_count = size_counts["val"][category]
        train_fraction = safe_fraction(train_count, total_objects["train"])
        val_fraction = safe_fraction(val_count, total_objects["val"])
        size_rows.append(
            {
                "size_category": category,
                "all_bbox_count": all_count,
                "train_bbox_count": train_count,
                "val_bbox_count": val_count,
                "val_share_of_category": safe_fraction(val_count, all_count),
                "train_bbox_fraction": train_fraction,
                "val_bbox_fraction": val_fraction,
                "train_val_gap_percentage_points": abs(
                    train_fraction - val_fraction
                )
                * 100.0,
            }
        )

    all_classes_covered = all(
        row["train_image_count"] > 0 and row["val_image_count"] > 0
        for row in class_rows
        if row["all_image_count"] > 0
    )
    all_sizes_covered = all(
        row["train_bbox_count"] > 0 and row["val_bbox_count"] > 0
        for row in size_rows
    )
    max_class_gap = max(
        (row["train_val_gap_percentage_points"] for row in class_rows),
        default=0.0,
    )
    max_size_gap = max(
        (row["train_val_gap_percentage_points"] for row in size_rows),
        default=0.0,
    )
    summary = {
        "sample_counts": {
            "all": len(all_stems),
            "train": len(train_stems),
            "val": len(val_stems),
        },
        "object_counts": total_objects,
        "consistency_check": {
            "definition": (
                "Internal diagnostic: every nonzero class and size category appears in both "
                "splits; maximum train-vs-val object-share gaps are <=2 percentage points."
            ),
            "official_competition_standard": False,
            "all_classes_present_in_train_and_val": all_classes_covered,
            "all_size_categories_present_in_train_and_val": all_sizes_covered,
            "max_class_gap_percentage_points": max_class_gap,
            "max_size_gap_percentage_points": max_size_gap,
            "class_gap_limit_percentage_points": MAX_BASIC_CLASS_GAP_PP,
            "size_gap_limit_percentage_points": MAX_BASIC_SIZE_GAP_PP,
            "basically_consistent": (
                all_classes_covered
                and all_sizes_covered
                and max_class_gap <= MAX_BASIC_CLASS_GAP_PP
                and max_size_gap <= MAX_BASIC_SIZE_GAP_PP
            ),
        },
        "class_distribution": class_rows,
        "size_distribution": size_rows,
    }
    return summary, class_rows, size_rows


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    label_dir = args.label_dir.resolve()
    bbox_stats_path = args.bbox_stats.resolve()
    group_pairs_path = args.group_pairs.resolve()
    split_dir = args.split_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    if not label_dir.is_dir():
        raise SystemExit(f"Label directory does not exist: {label_dir}")
    if not bbox_stats_path.is_file():
        raise SystemExit(f"Run analyze_labels.py first; missing: {bbox_stats_path}")
    if not group_pairs_path.is_file():
        raise SystemExit(f"Run analyze_groups.py first; missing: {group_pairs_path}")

    stems = collect_stems(label_dir)
    records = load_bbox_records(bbox_stats_path, set(stems))
    groups, grouping_metadata = build_groups(stems, group_pairs_path)
    group_features, feature_names = build_feature_matrix(groups, records)
    target_val_count = round(len(stems) * args.val_fraction)
    selected_group_indices, optimization_metadata = select_validation_groups(
        groups,
        group_features,
        val_sample_count=target_val_count,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    val_stems = sorted(
        stem
        for index, group in enumerate(groups)
        if index in selected_group_indices
        for stem in group.stems
    )
    train_stems = sorted(set(stems) - set(val_stems))
    if set(train_stems) & set(val_stems):
        raise RuntimeError("Train and val overlap")
    if set(train_stems) | set(val_stems) != set(stems):
        raise RuntimeError("Train and val do not cover all samples")

    summary, class_rows, size_rows = audit_split(
        set(stems), set(train_stems), set(val_stems), records
    )
    summary.update(
        {
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "method": "deterministic group-aware multilabel stratification",
            "source": "official data/raw/train labels only; test data excluded",
            "reference_imgsz": 640,
            "grouping": grouping_metadata,
            "optimization": {
                **optimization_metadata,
                "feature_names": feature_names,
            },
        }
    )

    group_crossing_count = 0
    group_rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        split_name = "val" if index in selected_group_indices else "train"
        if any(stem in set(val_stems) for stem in group.stems) and any(
            stem in set(train_stems) for stem in group.stems
        ):
            group_crossing_count += 1
        for stem in group.stems:
            group_rows.append(
                {
                    "stem": stem,
                    "group_id": group.group_id,
                    "group_type": group.group_type,
                    "group_size": group.size,
                    "split": split_name,
                }
            )
    summary["grouping"]["cross_split_group_count"] = group_crossing_count
    if group_crossing_count:
        raise RuntimeError(f"Group leakage detected: {group_crossing_count} groups")

    split_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    summary_path = analysis_dir / "split_summary.json"
    class_path = analysis_dir / "split_class_distribution.csv"
    size_path = analysis_dir / "split_size_distribution.csv"
    group_path = analysis_dir / "split_group_assignments.csv"
    write_lines(train_path, train_stems)
    write_lines(val_path, val_stems)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(class_path, class_rows)
    write_csv(size_path, size_rows)
    write_csv(group_path, sorted(group_rows, key=lambda row: row["stem"]))

    print("Fixed split created")
    print(f"  seed: {args.seed}")
    print(f"  train: {len(train_stems)} -> {train_path}")
    print(f"  val: {len(val_stems)} -> {val_path}")
    print(f"  groups: {len(groups)}; cross-split groups: {group_crossing_count}")
    print(
        "  basically consistent: "
        f"{summary['consistency_check']['basically_consistent']}"
    )
    print(f"  summary: {summary_path}")
    print(f"  class distribution: {class_path}")
    print(f"  size distribution: {size_path}")
    print(f"  group assignments: {group_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
