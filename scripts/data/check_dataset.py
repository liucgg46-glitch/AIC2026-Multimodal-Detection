"""Check integrity of the official AIC2026 multimodal training dataset.

This script is deliberately read-only with respect to ``data/raw``.  It scans
the four training folders, matches files by filename stem, verifies that images
can be decoded, compares spatial sizes, and summarizes modality metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_ROOT = REPO_ROOT / "data" / "raw" / "train"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis"
MODALITIES = ("visible", "infrared", "depth", "labels")
IMAGE_MODALITIES = ("visible", "infrared", "depth")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ISSUE_FIELDS = ("stem", "issue_type", "modality", "path", "details")


@dataclass(frozen=True)
class FileIndex:
    files: dict[str, Path]
    display_stems: dict[str, str]
    duplicates: dict[str, list[Path]]
    unexpected_files: list[Path]
    subdirectories: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check AIC2026 train-set integrity without modifying raw data."
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help=f"Training data root (default: {DEFAULT_TRAIN_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON/CSV reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def normalized_stem(path: Path) -> str:
    """Normalize stems for matching on Windows while retaining original names."""
    return path.stem.casefold()


def index_directory(directory: Path, allowed_extensions: set[str]) -> FileIndex:
    grouped: dict[str, list[Path]] = defaultdict(list)
    unexpected_files: list[Path] = []
    subdirectories: list[Path] = []

    for entry in sorted(directory.iterdir(), key=lambda p: p.name.casefold()):
        if entry.is_dir():
            subdirectories.append(entry)
            continue
        if entry.name == ".gitkeep":
            continue
        if entry.suffix.casefold() not in allowed_extensions:
            unexpected_files.append(entry)
            continue
        grouped[normalized_stem(entry)].append(entry)

    files: dict[str, Path] = {}
    display_stems: dict[str, str] = {}
    duplicates: dict[str, list[Path]] = {}
    for key, paths in grouped.items():
        display_stems[key] = paths[0].stem
        if len(paths) == 1:
            files[key] = paths[0]
        else:
            duplicates[key] = paths

    return FileIndex(files, display_stems, duplicates, unexpected_files, subdirectories)


def add_issue(
    issues: list[dict[str, str]],
    *,
    stem: str = "",
    issue_type: str,
    modality: str = "",
    path: Path | None = None,
    details: str = "",
) -> None:
    issues.append(
        {
            "stem": stem,
            "issue_type": issue_type,
            "modality": modality,
            "path": str(path) if path else "",
            "details": details,
        }
    )


def channel_count(image: np.ndarray) -> int:
    return 1 if image.ndim == 2 else int(image.shape[2])


def shape_string(image: np.ndarray) -> str:
    return "x".join(str(value) for value in image.shape)


def update_image_summary(
    summary: dict[str, Any], image: np.ndarray, extension: str
) -> None:
    summary["decoded_files"] += 1
    summary["dtype_counts"][str(image.dtype)] += 1
    summary["channel_counts"][str(channel_count(image))] += 1
    summary["shape_counts"][shape_string(image)] += 1
    summary["extension_counts"][extension.casefold()] += 1


def new_image_summary() -> dict[str, Any]:
    return {
        "decoded_files": 0,
        "dtype_counts": Counter(),
        "channel_counts": Counter(),
        "shape_counts": Counter(),
        "extension_counts": Counter(),
    }


def counters_to_dict(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {key: counters_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [counters_to_dict(item) for item in value]
    return value


def validate_expected_format(
    modality: str,
    stem: str,
    path: Path,
    image: np.ndarray,
    issues: list[dict[str, str]],
) -> None:
    channels = channel_count(image)
    if modality in {"visible", "infrared"}:
        if image.dtype != np.uint8 or channels != 3:
            add_issue(
                issues,
                stem=stem,
                issue_type="unexpected_image_format",
                modality=modality,
                path=path,
                details=f"expected uint8/3-channel, got {image.dtype}/{channels}-channel",
            )
    elif modality == "depth":
        extension = path.suffix.casefold()
        expected = (
            (image.dtype == np.uint16 and channels == 1)
            if extension == ".png"
            else (image.dtype == np.uint8 and channels in {1, 3})
        )
        if not expected:
            expected_description = (
                "uint16/1-channel for PNG"
                if extension == ".png"
                else "uint8/1-or-3-channel for JPEG"
            )
            add_issue(
                issues,
                stem=stem,
                issue_type="unexpected_depth_format",
                modality=modality,
                path=path,
                details=(
                    f"expected {expected_description}, "
                    f"got {image.dtype}/{channels}-channel"
                ),
            )


def inspect_dataset(train_root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    indexes: dict[str, FileIndex] = {}

    for modality in MODALITIES:
        directory = train_root / modality
        if not directory.is_dir():
            add_issue(
                issues,
                issue_type="missing_directory",
                modality=modality,
                path=directory,
            )
            continue

        extensions = {".txt"} if modality == "labels" else IMAGE_EXTENSIONS
        index = index_directory(directory, extensions)
        indexes[modality] = index

        for path in index.unexpected_files:
            add_issue(
                issues,
                issue_type="unexpected_file_extension",
                modality=modality,
                path=path,
                details=f"extension={path.suffix or '<none>'}",
            )
        for path in index.subdirectories:
            add_issue(
                issues,
                issue_type="unexpected_subdirectory",
                modality=modality,
                path=path,
            )
        for key, paths in index.duplicates.items():
            add_issue(
                issues,
                stem=index.display_stems[key],
                issue_type="duplicate_stem",
                modality=modality,
                details="; ".join(str(path) for path in paths),
            )

    stem_sets = {
        modality: set(index.files) | set(index.duplicates)
        for modality, index in indexes.items()
    }
    union_stems = set().union(*stem_sets.values()) if stem_sets else set()
    common_stems = set.intersection(*stem_sets.values()) if len(stem_sets) == 4 else set()

    for key in sorted(union_stems):
        display_stem = next(
            (
                index.display_stems[key]
                for index in indexes.values()
                if key in index.display_stems
            ),
            key,
        )
        for modality in MODALITIES:
            if key not in stem_sets.get(modality, set()):
                add_issue(
                    issues,
                    stem=display_stem,
                    issue_type="missing_modality_file",
                    modality=modality,
                )

    image_summaries = {modality: new_image_summary() for modality in IMAGE_MODALITIES}
    depth_accumulator = {
        "pixel_count": 0,
        "zero_count": 0,
        "sum": 0.0,
        "global_min": None,
        "global_max": None,
        "per_image_medians": [],
    }
    decoded_sizes_by_stem: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)

    image_jobs = [
        (modality, key, path)
        for modality in IMAGE_MODALITIES
        for key, path in indexes.get(
            modality, FileIndex({}, {}, {}, [], [])
        ).files.items()
    ]
    for modality, key, path in tqdm(image_jobs, desc="Decoding images", unit="file"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        display_stem = indexes[modality].display_stems[key]
        if image is None:
            add_issue(
                issues,
                stem=display_stem,
                issue_type="unreadable_image",
                modality=modality,
                path=path,
            )
            continue

        decoded_sizes_by_stem[key][modality] = tuple(image.shape[:2])
        update_image_summary(image_summaries[modality], image, path.suffix)
        validate_expected_format(modality, display_stem, path, image, issues)

        if modality == "depth":
            values = image.astype(np.float64, copy=False)
            count = int(values.size)
            depth_accumulator["pixel_count"] += count
            depth_accumulator["zero_count"] += int(np.count_nonzero(values == 0))
            depth_accumulator["sum"] += float(values.sum(dtype=np.float64))
            image_min = float(values.min())
            image_max = float(values.max())
            depth_accumulator["global_min"] = (
                image_min
                if depth_accumulator["global_min"] is None
                else min(depth_accumulator["global_min"], image_min)
            )
            depth_accumulator["global_max"] = (
                image_max
                if depth_accumulator["global_max"] is None
                else max(depth_accumulator["global_max"], image_max)
            )
            depth_accumulator["per_image_medians"].append(float(np.median(values)))

    size_mismatch_count = 0
    extension_mismatch_count = 0
    for key in sorted(common_stems):
        sizes = decoded_sizes_by_stem.get(key, {})
        if set(sizes) != set(IMAGE_MODALITIES):
            continue
        image_paths = {
            modality: indexes[modality].files.get(key) for modality in IMAGE_MODALITIES
        }
        extensions = {
            modality: path.suffix.casefold()
            for modality, path in image_paths.items()
            if path is not None
        }
        if len(extensions) == len(IMAGE_MODALITIES) and len(set(extensions.values())) != 1:
            extension_mismatch_count += 1
            display_stem = indexes["visible"].display_stems[key]
            add_issue(
                issues,
                stem=display_stem,
                issue_type="modality_extension_mismatch",
                details=", ".join(
                    f"{name}={extension}" for name, extension in extensions.items()
                ),
            )
        if len(set(sizes.values())) != 1:
            size_mismatch_count += 1
            display_stem = indexes["visible"].display_stems[key]
            add_issue(
                issues,
                stem=display_stem,
                issue_type="modality_size_mismatch",
                details=", ".join(f"{name}={size}" for name, size in sizes.items()),
            )

    pixel_count = depth_accumulator["pixel_count"]
    depth_summary = {
        "pixel_count": pixel_count,
        "min": depth_accumulator["global_min"],
        "max": depth_accumulator["global_max"],
        "mean": depth_accumulator["sum"] / pixel_count if pixel_count else None,
        "median_of_image_medians": (
            float(np.median(depth_accumulator["per_image_medians"]))
            if depth_accumulator["per_image_medians"]
            else None
        ),
        "zero_ratio": depth_accumulator["zero_count"] / pixel_count if pixel_count else None,
    }

    issue_counts = Counter(issue["issue_type"] for issue in issues)
    summary: dict[str, Any] = {
        "train_root": str(train_root.resolve()),
        "file_counts": {
            modality: len(index.files) + sum(len(v) for v in index.duplicates.values())
            for modality, index in indexes.items()
        },
        "unique_stem_counts": {
            modality: len(stems) for modality, stems in stem_sets.items()
        },
        "union_stem_count": len(union_stems),
        "common_stem_count": len(common_stems),
        "size_mismatch_count": size_mismatch_count,
        "extension_mismatch_count": extension_mismatch_count,
        "image_summaries": image_summaries,
        "depth_statistics": depth_summary,
        "issue_count": len(issues),
        "issue_counts": issue_counts,
        "passed": len(issues) == 0,
    }
    return counters_to_dict(summary), issues


def write_reports(
    output_dir: Path, summary: dict[str, Any], issues: Iterable[dict[str, str]]
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_check_summary.json"
    issues_path = output_dir / "dataset_issues.csv"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with issues_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUE_FIELDS)
        writer.writeheader()
        writer.writerows(issues)
    return summary_path, issues_path


def main() -> int:
    args = parse_args()
    train_root = args.train_root.resolve()
    output_dir = args.output_dir.resolve()

    if not train_root.is_dir():
        raise SystemExit(f"Training directory does not exist: {train_root}")

    summary, issues = inspect_dataset(train_root)
    summary_path, issues_path = write_reports(output_dir, summary, issues)

    print("\nDataset integrity check complete")
    print(f"  train root: {train_root}")
    print(f"  file counts: {summary['file_counts']}")
    print(f"  common stems: {summary['common_stem_count']}")
    print(f"  extension mismatches: {summary['extension_mismatch_count']}")
    print(f"  size mismatches: {summary['size_mismatch_count']}")
    print(f"  issues: {summary['issue_count']}")
    print(f"  summary: {summary_path}")
    print(f"  issues: {issues_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
