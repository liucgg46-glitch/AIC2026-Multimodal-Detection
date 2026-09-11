"""Analyze AIC2026 Visible, Infrared and Depth data without modifying sources.

The script performs full descriptive statistics for the selected fixed split.
Registration uses a deterministic, format-balanced sample because cross-modal
matching is substantially more expensive and can be unreliable. Results always
carry a reliability flag; unreliable shifts are excluded from aggregate errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "train"
DEFAULT_LABEL_DIR = REPO_ROOT / "data" / "processed" / "train" / "labels_clean"
DEFAULT_JSON = REPO_ROOT / "outputs" / "analysis" / "modalities_stats.json"
DEFAULT_CSV = REPO_ROOT / "outputs" / "analysis" / "modalities_stats.csv"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "analysis" / "modalities_summary.md"
DEFAULT_DEPTH_VISUALIZATION_DIR = REPO_ROOT / "outputs" / "visualization" / "depth_analysis"
SPLIT_DIR = REPO_ROOT / "data" / "splits"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PREFERRED_DEPTH_VISUALIZATION_STEMS = (
    "000002",
    "003127",
    "shuming_343_00000288",
    "shuming_342_00000275",
    "003125",
)


def portable_project_path(path: Path) -> str:
    """Return repository paths in portable POSIX form for generated artifacts."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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


def channel_count(image: np.ndarray) -> int:
    return 1 if image.ndim == 2 else int(image.shape[2])


def shape_text(image: np.ndarray) -> str:
    return "x".join(str(value) for value in image.shape)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    percentiles = np.percentile(array, (25, 50, 75, 90, 95))
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: np.ndarray) -> None:
        numeric = values.astype(np.float64, copy=False)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.total += float(finite.sum(dtype=np.float64))
        self.total_squared += float(np.square(finite).sum(dtype=np.float64))
        current_min = float(finite.min())
        current_max = float(finite.max())
        self.minimum = current_min if self.minimum is None else min(self.minimum, current_min)
        self.maximum = current_max if self.maximum is None else max(self.maximum, current_max)

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
        }


def histogram_quantile(histogram: np.ndarray, quantile: float) -> float | None:
    count = int(histogram.sum())
    if count == 0:
        return None
    target = quantile * (count - 1)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    return float(np.searchsorted(cumulative, target + 1, side="left"))


@dataclass
class BasicStats:
    files: int = 0
    decoded: int = 0
    suffixes: Counter[str] = field(default_factory=Counter)
    dtypes: Counter[str] = field(default_factory=Counter)
    shapes: Counter[str] = field(default_factory=Counter)
    channels: Counter[str] = field(default_factory=Counter)
    border_present: list[float] = field(default_factory=list)
    border_left: list[float] = field(default_factory=list)
    border_right: list[float] = field(default_factory=list)
    border_top: list[float] = field(default_factory=list)
    border_bottom: list[float] = field(default_factory=list)
    fov_ratio: list[float] = field(default_factory=list)
    low_pixel_ratio: list[float] = field(default_factory=list)
    lowest_fov: list[tuple[float, str]] = field(default_factory=list)

    def update(self, path: Path, image: np.ndarray, border: dict[str, float | bool]) -> None:
        self.files += 1
        self.decoded += 1
        self.suffixes[path.suffix.casefold()] += 1
        self.dtypes[str(image.dtype)] += 1
        self.shapes[shape_text(image)] += 1
        self.channels[str(channel_count(image))] += 1
        self.border_present.append(float(bool(border["present"])))
        self.border_left.append(float(border["left_px"]))
        self.border_right.append(float(border["right_px"]))
        self.border_top.append(float(border["top_px"]))
        self.border_bottom.append(float(border["bottom_px"]))
        self.fov_ratio.append(float(border["fov_ratio"]))
        self.low_pixel_ratio.append(float(border["low_pixel_ratio"]))
        self.lowest_fov.append((float(border["fov_ratio"]), path.stem))

    def result(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "decoded": self.decoded,
            "suffix_distribution": dict(sorted(self.suffixes.items())),
            "dtype_distribution": dict(sorted(self.dtypes.items())),
            "shape_distribution": dict(sorted(self.shapes.items())),
            "channel_distribution": dict(sorted(self.channels.items())),
            "border_occurrence_ratio": float(np.mean(self.border_present)) if self.border_present else None,
            "border_width_px": {
                "left": distribution(self.border_left),
                "right": distribution(self.border_right),
                "top": distribution(self.border_top),
                "bottom": distribution(self.border_bottom),
            },
            "effective_fov_ratio": distribution(self.fov_ratio),
            "low_pixel_ratio": distribution(self.low_pixel_ratio),
            "lowest_fov_stems": [
                {"stem": stem, "fov_ratio": ratio}
                for ratio, stem in sorted(self.lowest_fov)[:10]
            ],
        }


@dataclass
class InfraredStats:
    pixels: RunningMoments = field(default_factory=RunningMoments)
    image_min: list[float] = field(default_factory=list)
    image_max: list[float] = field(default_factory=list)
    image_mean: list[float] = field(default_factory=list)
    image_std: list[float] = field(default_factory=list)
    diff_bg: list[float] = field(default_factory=list)
    diff_br: list[float] = field(default_factory=list)
    diff_gr: list[float] = field(default_factory=list)
    diff_by_stem: list[tuple[float, str]] = field(default_factory=list)

    def update(self, stem: str, image: np.ndarray) -> dict[str, float | None]:
        self.pixels.update(image)
        values = image.astype(np.float64, copy=False)
        minimum, maximum = float(values.min()), float(values.max())
        mean, std = float(values.mean()), float(values.std())
        self.image_min.append(minimum)
        self.image_max.append(maximum)
        self.image_mean.append(mean)
        self.image_std.append(std)
        result: dict[str, float | None] = {
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "diff_bg": None,
            "diff_br": None,
            "diff_gr": None,
        }
        if image.ndim == 3 and image.shape[2] >= 3:
            signed = image[..., :3].astype(np.int32)
            diffs = (
                float(np.abs(signed[..., 0] - signed[..., 1]).mean()),
                float(np.abs(signed[..., 0] - signed[..., 2]).mean()),
                float(np.abs(signed[..., 1] - signed[..., 2]).mean()),
            )
            self.diff_bg.append(diffs[0])
            self.diff_br.append(diffs[1])
            self.diff_gr.append(diffs[2])
            max_diff = max(diffs)
            self.diff_by_stem.append((max_diff, stem))
            result.update(diff_bg=diffs[0], diff_br=diffs[1], diff_gr=diffs[2])
        return result

    def result(self, three_channel_count: int, single_channel_count: int, decoded: int) -> dict[str, Any]:
        max_diffs = [value for value, _ in self.diff_by_stem]
        diff_summary = distribution(max_diffs)
        q75 = diff_summary["p75"]
        q25 = diff_summary["p25"]
        threshold = None if q75 is None or q25 is None else float(q75 + 1.5 * (q75 - q25))
        all_outliers = [] if threshold is None else [
            {"stem": stem, "max_channel_mean_abs_diff": value}
            for value, stem in sorted(self.diff_by_stem, reverse=True)
            if value > threshold
        ]
        return {
            "global_pixels": self.pixels.result(),
            "per_image_min": distribution(self.image_min),
            "per_image_max": distribution(self.image_max),
            "per_image_mean": distribution(self.image_mean),
            "per_image_std": distribution(self.image_std),
            "three_channel_ratio": three_channel_count / decoded if decoded else None,
            "single_channel_ratio": single_channel_count / decoded if decoded else None,
            "channel_mean_absolute_difference": {
                "B_G": distribution(self.diff_bg),
                "B_R": distribution(self.diff_br),
                "G_R": distribution(self.diff_gr),
                "maximum_pair_per_image": diff_summary,
            },
            "channel_difference_outlier_threshold": threshold,
            "channel_difference_outlier_count": len(all_outliers),
            "channel_difference_outliers": all_outliers[:20],
            "largest_channel_difference_stems": [
                {"stem": stem, "max_channel_mean_abs_diff": value}
                for value, stem in sorted(self.diff_by_stem, reverse=True)[:10]
            ],
        }


@dataclass
class PngDepthStats:
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(65536, dtype=np.int64))
    zero_ratios: list[float] = field(default_factory=list)
    zero_by_stem: list[tuple[float, str]] = field(default_factory=list)
    total_pixels: int = 0
    below_300: int = 0
    range_300_20000: int = 0
    above_20000: int = 0
    nonzero_below_300: int = 0

    def update(self, stem: str, image: np.ndarray) -> dict[str, float]:
        flat = image.reshape(-1)
        self.histogram += np.bincount(flat, minlength=65536)
        count = int(flat.size)
        zero_ratio = float(np.count_nonzero(flat == 0) / count)
        self.total_pixels += count
        self.below_300 += int(np.count_nonzero(flat < 300))
        self.nonzero_below_300 += int(np.count_nonzero((flat > 0) & (flat < 300)))
        self.range_300_20000 += int(np.count_nonzero((flat >= 300) & (flat <= 20000)))
        self.above_20000 += int(np.count_nonzero(flat > 20000))
        self.zero_ratios.append(zero_ratio)
        self.zero_by_stem.append((zero_ratio, stem))
        return {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "median": float(np.median(flat)),
            "zero_ratio": zero_ratio,
        }

    def result(self) -> dict[str, Any]:
        total = self.total_pixels
        nonzero_hist = self.histogram.copy()
        nonzero_hist[0] = 0
        nonzero_count = int(nonzero_hist.sum())
        indices = np.arange(self.histogram.size, dtype=np.float64)
        all_mean = float(np.dot(self.histogram, indices) / total) if total else None
        nonzero_mean = float(np.dot(nonzero_hist, indices) / nonzero_count) if nonzero_count else None
        return {
            "total_pixels": total,
            "all_pixels": {
                "min": histogram_quantile(self.histogram, 0.0),
                "max": histogram_quantile(self.histogram, 1.0),
                "mean": all_mean,
                "median": histogram_quantile(self.histogram, 0.5),
                "p25": histogram_quantile(self.histogram, 0.25),
                "p75": histogram_quantile(self.histogram, 0.75),
                "p90": histogram_quantile(self.histogram, 0.90),
                "p95": histogram_quantile(self.histogram, 0.95),
            },
            "nonzero_pixels": {
                "count": nonzero_count,
                "mean": nonzero_mean,
                "median": histogram_quantile(nonzero_hist, 0.5),
                "p25": histogram_quantile(nonzero_hist, 0.25),
                "p75": histogram_quantile(nonzero_hist, 0.75),
                "p90": histogram_quantile(nonzero_hist, 0.90),
                "p95": histogram_quantile(nonzero_hist, 0.95),
            },
            "global_zero_ratio": float(self.histogram[0] / total) if total else None,
            "per_image_zero_ratio": distribution(self.zero_ratios),
            "all_pixel_range_ratios": {
                "below_300_including_zero": self.below_300 / total if total else None,
                "range_300_to_20000_inclusive": self.range_300_20000 / total if total else None,
                "above_20000": self.above_20000 / total if total else None,
            },
            "nonzero_below_300_ratio": self.nonzero_below_300 / nonzero_count if nonzero_count else None,
            "highest_zero_ratio_stems": [
                {"stem": stem, "zero_ratio": ratio}
                for ratio, stem in sorted(self.zero_by_stem, reverse=True)[:10]
            ],
        }


@dataclass
class JpgDepthStats:
    pixels: RunningMoments = field(default_factory=RunningMoments)
    image_mean: list[float] = field(default_factory=list)
    image_std: list[float] = field(default_factory=list)
    zero_ratios: list[float] = field(default_factory=list)
    dynamic_ranges: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    diff_bg: list[float] = field(default_factory=list)
    diff_br: list[float] = field(default_factory=list)
    diff_gr: list[float] = field(default_factory=list)

    def update(self, image: np.ndarray) -> dict[str, float | None]:
        self.pixels.update(image)
        values = image.astype(np.float64, copy=False)
        minimum, maximum = float(values.min()), float(values.max())
        mean, std = float(values.mean()), float(values.std())
        zero_ratio = float(np.count_nonzero(image == 0) / image.size)
        dynamic_range = maximum - minimum
        gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        counts = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        self.image_mean.append(mean)
        self.image_std.append(std)
        self.zero_ratios.append(zero_ratio)
        self.dynamic_ranges.append(dynamic_range)
        self.entropies.append(entropy)
        result: dict[str, float | None] = {
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "zero_ratio": zero_ratio,
            "dynamic_range": dynamic_range,
            "entropy_bits": entropy,
            "diff_bg": None,
            "diff_br": None,
            "diff_gr": None,
        }
        if image.ndim == 3 and image.shape[2] >= 3:
            signed = image[..., :3].astype(np.int32)
            diffs = (
                float(np.abs(signed[..., 0] - signed[..., 1]).mean()),
                float(np.abs(signed[..., 0] - signed[..., 2]).mean()),
                float(np.abs(signed[..., 1] - signed[..., 2]).mean()),
            )
            self.diff_bg.append(diffs[0])
            self.diff_br.append(diffs[1])
            self.diff_gr.append(diffs[2])
            result.update(diff_bg=diffs[0], diff_br=diffs[1], diff_gr=diffs[2])
        return result

    def result(self) -> dict[str, Any]:
        return {
            "global_pixels": self.pixels.result(),
            "per_image_mean": distribution(self.image_mean),
            "per_image_std": distribution(self.image_std),
            "per_image_zero_ratio": distribution(self.zero_ratios),
            "per_image_dynamic_range": distribution(self.dynamic_ranges),
            "per_image_grayscale_entropy_bits": distribution(self.entropies),
            "channel_mean_absolute_difference": {
                "B_G": distribution(self.diff_bg),
                "B_R": distribution(self.diff_br),
                "G_R": distribution(self.diff_gr),
            },
            "physical_depth_mapping": "unknown",
        }


def report_warning(messages: list[str], message: str) -> None:
    messages.append(message)
    warnings.warn(message, stacklevel=2)


def build_stem_map(
    directory: Path, suffixes: set[str], messages: list[str]
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    candidates: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.casefold() in suffixes:
            candidates.setdefault(path.stem, []).append(path)
    duplicates = {
        stem: [path.name for path in paths]
        for stem, paths in candidates.items()
        if len(paths) > 1
    }
    for stem, names in duplicates.items():
        report_warning(messages, f"Ambiguous stem {stem!r} in {directory}: {', '.join(names)}")
    mapping = {stem: paths[0] for stem, paths in candidates.items() if len(paths) == 1}
    return mapping, duplicates


def load_fixed_split(split: str) -> list[str]:
    path = SPLIT_DIR / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Fixed split does not exist: {path}")
    stems: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        stem = Path(value).stem
        if stem in seen:
            raise ValueError(f"Duplicate stem {stem!r} in {path} line {line_number}")
        seen.add(stem)
        stems.append(stem)
    return stems


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    if image.ndim == 3:
        return image[..., 0]
    raise ValueError(f"Cannot convert shape {image.shape} to grayscale")


def median_leading_runs(mask: np.ndarray, axis: int, reverse: bool = False) -> float:
    working = np.flip(mask, axis=axis) if reverse else mask
    if axis == 1:
        no_low = ~working
        runs = np.argmax(no_low, axis=1)
        usable = ~np.all(working, axis=1)
        if not np.any(usable):
            return float(working.shape[1])
        runs = runs[usable]
    else:
        no_low = ~working
        runs = np.argmax(no_low, axis=0)
        usable = ~np.all(working, axis=0)
        if not np.any(usable):
            return float(working.shape[0])
        runs = runs[usable]
    return float(np.median(runs))


def border_analysis(image: np.ndarray, modality: str, suffix: str) -> dict[str, float | bool | str]:
    if modality == "depth" and suffix == ".png" and image.ndim == 2 and image.dtype == np.uint16:
        low_mask = image == 0
        definition = "edge-continuous zero-valued pixels"
    else:
        gray = grayscale(image)
        if gray.dtype != np.uint8:
            finite = gray[np.isfinite(gray)]
            scale = float(np.percentile(finite, 99)) if finite.size else 1.0
            gray8 = np.clip(gray.astype(np.float32) * (255.0 / max(scale, 1.0)), 0, 255).astype(np.uint8)
        else:
            gray8 = gray
        low_mask = gray8 <= 8
        definition = "edge-continuous grayscale <= 8 pixels"

    height, width = low_mask.shape
    left = median_leading_runs(low_mask, axis=1)
    right = median_leading_runs(low_mask, axis=1, reverse=True)
    top = median_leading_runs(low_mask, axis=0)
    bottom = median_leading_runs(low_mask, axis=0, reverse=True)
    content_width = max(0.0, width - left - right)
    content_height = max(0.0, height - top - bottom)
    return {
        "definition": definition,
        "present": any(value >= 1.0 for value in (left, right, top, bottom)),
        "left_px": left,
        "right_px": right,
        "top_px": top,
        "bottom_px": bottom,
        "fov_ratio": content_width * content_height / (width * height),
        "low_pixel_ratio": float(low_mask.mean()),
    }


def normalize_depth_png(depth: np.ndarray) -> np.ndarray:
    result = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    if np.any(valid):
        low, high = (float(value) for value in np.percentile(depth[valid], (2, 98)))
        result[valid] = np.clip(
            (depth[valid].astype(np.float32) - low) * (255.0 / max(high - low, 1.0)),
            0,
            255,
        ).astype(np.uint8)
    return result


def normalize_valid_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalize valid analysis values to [0, 1] while leaving invalid pixels at zero."""
    result = np.zeros(values.shape, dtype=np.float32)
    if not np.any(valid):
        return result
    valid_values = values[valid].astype(np.float64, copy=False)
    low = float(valid_values.min())
    high = float(valid_values.max())
    if high > low:
        result[valid] = ((valid_values - low) / (high - low)).astype(np.float32)
    else:
        result[valid] = 1.0
    return result


def select_depth_visualization_stems(
    rows: list[dict[str, Any]], val_sample_count: int
) -> tuple[list[str], list[str]]:
    """Select requested examples plus deterministic val zero-ratio quantiles."""
    png_rows = {
        str(row["stem"]): row
        for row in rows
        if row.get("depth_kind") == "png_uint16_single"
    }
    selected = [stem for stem in PREFERRED_DEPTH_VISUALIZATION_STEMS if stem in png_rows]
    missing = [stem for stem in PREFERRED_DEPTH_VISUALIZATION_STEMS if stem not in png_rows]
    val_rows = sorted(
        (row for row in png_rows.values() if row.get("split") == "val"),
        key=lambda row: (float(row["depth_zero_ratio"]), str(row["stem"])),
    )
    count = min(max(val_sample_count, 0), len(val_rows))
    if count == 1:
        quantile_indices = [len(val_rows) // 2]
    elif count > 1:
        quantile_indices = [round(index * (len(val_rows) - 1) / (count - 1)) for index in range(count)]
    else:
        quantile_indices = []
    for target_index in quantile_indices:
        candidate_indices = sorted(
            range(len(val_rows)), key=lambda index: (abs(index - target_index), index)
        )
        for index in candidate_indices:
            stem = str(val_rows[index]["stem"])
            if stem not in selected:
                selected.append(stem)
                break
    return selected, missing


def write_png_depth_visualization(
    stem: str,
    visible_path: Path,
    depth_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Write a derived PNG Depth review figure without changing source arrays or files."""
    visible = read_image(visible_path)
    depth = read_image(depth_path)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(
            f"Depth visualization requires uint16 single-channel PNG, got {depth.dtype} {depth.shape}"
        )

    valid = depth > 0
    valid_count = int(np.count_nonzero(valid))
    valid_ratio = valid_count / depth.size
    zero_ratio = 1.0 - valid_ratio
    valid_values = depth[valid].astype(np.float64, copy=False)
    if valid_count:
        p2, p98 = (float(value) for value in np.percentile(valid_values, (2, 98)))
    else:
        p2, p98 = 0.0, 0.0

    raw_display = normalize_valid_values(depth, valid)
    clipped = np.zeros(depth.shape, dtype=np.float32)
    if valid_count:
        clipped_values = np.clip(valid_values, p2, p98)
        if p98 > p2:
            clipped[valid] = ((clipped_values - p2) / (p98 - p2)).astype(np.float32)
        else:
            clipped[valid] = 1.0

    # Log and inverse depth are derived visualization/preprocessing candidates, not physical depth.
    log_values = np.zeros(depth.shape, dtype=np.float64)
    inverse_values = np.zeros(depth.shape, dtype=np.float64)
    if valid_count:
        log_values[valid] = np.log1p(valid_values)
        inverse_values[valid] = 1.0 / valid_values
    log_display = normalize_valid_values(log_values, valid)
    inverse_display = normalize_valid_values(inverse_values, valid)

    def colorize(values: np.ndarray, colormap: int) -> np.ndarray:
        scaled = np.clip(np.rint(values * 255.0), 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, colormap)
        colored[~valid] = 0
        return colored

    def panel(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((440, 640, 3), dtype=np.uint8)
        canvas[80:, :] = resized
        cv2.putText(canvas, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        if subtitle:
            cv2.putText(
                canvas,
                subtitle,
                (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (205, 205, 205),
                1,
            )
        return canvas

    visible_bgr = (
        visible[..., :3]
        if visible.ndim == 3 and visible.shape[2] >= 3
        else cv2.cvtColor(visible.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    )
    valid_mask = np.where(valid, 255, 0).astype(np.uint8)
    panels = [
        panel(visible_bgr, "Visible reference"),
        panel(
            colorize(raw_display, cv2.COLORMAP_VIRIDIS),
            f"Raw Depth (normalized display, {depth.dtype})",
            f"min={int(depth.min())}, max={int(depth.max())}",
        ),
        panel(
            valid_mask,
            "Valid Mask (depth > 0)",
            f"valid={valid_ratio:.2%}, zero={zero_ratio:.2%}",
        ),
        panel(
            colorize(clipped, cv2.COLORMAP_VIRIDIS),
            f"Percentile Depth (P2-P98, {depth.dtype})",
            f"P2={p2:.1f}, P98={p98:.1f}",
        ),
        panel(
            colorize(log_display, cv2.COLORMAP_VIRIDIS),
            "Log Depth",
            "valid-only log1p display",
        ),
        panel(
            colorize(inverse_display, cv2.COLORMAP_MAGMA),
            "Inverse Depth",
            "valid-only 1/depth display",
        ),
    ]
    figure = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_depth_analysis.png"
    if not cv2.imwrite(str(output_path), figure):
        raise OSError(f"Failed to write image: {output_path}")
    return {
        "stem": stem,
        "output_path": portable_project_path(output_path),
        "valid_ratio": valid_ratio,
        "zero_ratio": zero_ratio,
        "p2": p2,
        "p98": p98,
    }


def c4_depth_anomaly_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    png_rows = [row for row in rows if row.get("depth_kind") == "png_uint16_single"]
    jpg_rows = [row for row in rows if row.get("depth_kind") == "jpg_uint8"]
    near_all_zero = [
        {"stem": row["stem"], "zero_ratio": row["depth_zero_ratio"]}
        for row in png_rows
        if float(row["depth_zero_ratio"]) >= 0.99
    ]
    all_black = [row["stem"] for row in png_rows if float(row["depth_max"]) == 0.0]
    low_dynamic = sorted(
        (
            {
                "stem": row["stem"],
                "dynamic_range": float(row["depth_max"]) - float(row["depth_min"]),
                "max": float(row["depth_max"]),
            }
            for row in png_rows
        ),
        key=lambda item: (item["dynamic_range"], item["stem"]),
    )[:5]
    outside_expected_range = [
        row["stem"]
        for row in png_rows
        if float(row["depth_max"]) > 20000.0 or float(row["depth_max"]) < 300.0
    ]
    return {
        "near_all_zero_threshold": 0.99,
        "near_all_zero": near_all_zero,
        "all_black": all_black,
        "lowest_dynamic_range": low_dynamic,
        "outside_expected_nonzero_max_range": outside_expected_range,
        "png_count": len(png_rows),
        "png_dtypes": dict(Counter(str(row["depth_dtype"]) for row in png_rows)),
        "png_shapes": dict(Counter(str(row["depth_shape"]) for row in png_rows)),
        "png_channels": dict(Counter(str(row["depth_channels"]) for row in png_rows)),
        "jpg_count": len(jpg_rows),
        "jpg_dtypes": dict(Counter(str(row["depth_dtype"]) for row in jpg_rows)),
        "jpg_shapes": dict(Counter(str(row["depth_shape"]) for row in jpg_rows)),
        "jpg_channels": dict(Counter(str(row["depth_channels"]) for row in jpg_rows)),
        "unexpected_representations": [
            {"stem": row["stem"], "kind": row["depth_kind"]}
            for row in rows
            if row.get("depth_kind") not in {"png_uint16_single", "jpg_uint8"}
        ],
    }


def registration_gray(image: np.ndarray, is_png_depth: bool) -> np.ndarray:
    if is_png_depth:
        return normalize_depth_png(image)
    gray = grayscale(image)
    if gray.dtype == np.uint8:
        return gray
    finite = gray[np.isfinite(gray)]
    result = np.zeros(gray.shape, dtype=np.uint8)
    if finite.size:
        low, high = (float(value) for value in np.percentile(finite, (2, 98)))
        result = np.clip(
            (gray.astype(np.float32) - low) * (255.0 / max(high - low, 1.0)), 0, 255
        ).astype(np.uint8)
    return result


def edge_representation(image: np.ndarray, max_dimension: int = 480) -> tuple[np.ndarray, float, int]:
    scale = min(1.0, max_dimension / max(image.shape))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    median = float(np.median(image))
    low = max(10, round(0.66 * median))
    high = max(low + 20, round(1.33 * median))
    binary = cv2.Canny(image, low, high)
    edge_count = int(np.count_nonzero(binary))
    edges = cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), 1.0)
    edges[:3, :] = 0
    edges[-3:, :] = 0
    edges[:, :3] = 0
    edges[:, -3:] = 0
    return edges, scale, edge_count


def local_edge_registration(
    visible: np.ndarray,
    modality: np.ndarray,
    pair: str,
    is_png_depth: bool = False,
    search_radius: int = 8,
) -> dict[str, Any]:
    if visible.shape[:2] != modality.shape[:2]:
        return {"reliable": False, "reason": "shape_mismatch", "method": "local_edge_correlation"}
    visible_gray = registration_gray(visible, False)
    modality_gray = registration_gray(modality, is_png_depth)
    visible_edges, scale, visible_edge_count = edge_representation(visible_gray)
    modality_edges, other_scale, modality_edge_count = edge_representation(modality_gray)
    if visible_edges.shape != modality_edges.shape or not math.isclose(scale, other_scale):
        return {"reliable": False, "reason": "resized_shape_mismatch", "method": "local_edge_correlation"}
    if min(visible_edge_count, modality_edge_count) < 100:
        return {
            "reliable": False,
            "reason": "insufficient_edges",
            "method": "local_edge_correlation",
            "visible_edge_count": visible_edge_count,
            "modality_edge_count": modality_edge_count,
        }

    padded = cv2.copyMakeBorder(
        visible_edges,
        search_radius,
        search_radius,
        search_radius,
        search_radius,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    response = cv2.matchTemplate(padded, modality_edges, cv2.TM_CCORR_NORMED)
    _, best_score, _, best_location = cv2.minMaxLoc(response)
    best_x, best_y = best_location
    masked = response.copy()
    masked[
        max(0, best_y - 1) : min(response.shape[0], best_y + 2),
        max(0, best_x - 1) : min(response.shape[1], best_x + 2),
    ] = -1
    second_score = float(masked.max())
    median_score = float(np.median(response))
    peak_margin = float(best_score - second_score)
    peak_ratio = float(best_score / max(median_score, 1e-8))
    dx = float((search_radius - best_x) / scale)
    dy = float((search_radius - best_y) / scale)
    at_boundary = best_x in {0, response.shape[1] - 1} or best_y in {0, response.shape[0] - 1}

    if pair == "visible_ir":
        minimum_score, minimum_margin = 0.20, 0.005
    else:
        minimum_score, minimum_margin = 0.10, 0.003
    reliable = bool(
        best_score >= minimum_score
        and peak_margin >= minimum_margin
        and peak_ratio >= 1.10
        and not at_boundary
    )
    reasons = []
    if best_score < minimum_score:
        reasons.append("low_edge_correlation")
    if peak_margin < minimum_margin:
        reasons.append("ambiguous_peak")
    if peak_ratio < 1.10:
        reasons.append("weak_peak_ratio")
    if at_boundary:
        reasons.append("search_boundary")
    score_component = np.clip((best_score - minimum_score) / max(0.40 - minimum_score, 1e-6), 0, 1)
    margin_component = np.clip(peak_margin / (minimum_margin * 3), 0, 1)
    ratio_component = np.clip((peak_ratio - 1.0) / 0.30, 0, 1)
    confidence = float(score_component * margin_component * ratio_component)
    return {
        "method": "local_edge_correlation",
        "dx": dx,
        "dy": dy,
        "magnitude": math.hypot(dx, dy),
        "confidence": confidence,
        "edge_correlation": float(best_score),
        "peak_margin": peak_margin,
        "peak_ratio": peak_ratio,
        "visible_edge_count": visible_edge_count,
        "modality_edge_count": modality_edge_count,
        "search_limit_original_px": float(search_radius / scale),
        "reliable": reliable,
        "reason": "reliable" if reliable else ",".join(reasons),
    }


def sanitize_registration_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove untrusted floating estimates from an unreliable serialized result."""
    serialized = dict(result)
    if not bool(serialized.get("reliable", False)):
        for key in (
            "dx",
            "dy",
            "magnitude",
            "confidence",
            "edge_correlation",
            "peak_margin",
            "peak_ratio",
        ):
            serialized[key] = None
    return serialized


def parse_labels(path: Path, messages: list[str]) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            report_warning(messages, f"Invalid label field count: {path}:{line_number}")
            continue
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError:
            report_warning(messages, f"Invalid label values: {path}:{line_number}")
            continue
        if not 0 <= class_id < len(CLASS_NAMES) or width <= 0 or height <= 0:
            report_warning(messages, f"Invalid clean label row: {path}:{line_number}")
            continue
        boxes.append((class_id, cx, cy, width, height))
    return boxes


def target_records(
    stem: str,
    labels: list[tuple[int, float, float, float, float]],
    depth: np.ndarray,
    depth_kind: str,
    registrations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    height, width = depth.shape[:2]
    for class_id, cx, cy, box_width, box_height in labels:
        x1 = max(0, min(width, math.floor((cx - box_width / 2) * width)))
        y1 = max(0, min(height, math.floor((cy - box_height / 2) * height)))
        x2 = max(0, min(width, math.ceil((cx + box_width / 2) * width)))
        y2 = max(0, min(height, math.ceil((cy + box_height / 2) * height)))
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / (width * height))
        record: dict[str, Any] = {
            "stem": stem,
            "class_id": class_id,
            "bbox_area_ratio": area_ratio,
            "physical_depth_median": None,
            "depth_valid_ratio": None,
        }
        if depth_kind == "png_uint16_single" and x2 > x1 and y2 > y1:
            crop = depth[y1:y2, x1:x2]
            valid = crop > 0
            valid_count = int(np.count_nonzero(valid))
            record["depth_valid_ratio"] = float(valid_count / crop.size)
            if valid_count >= 16 and record["depth_valid_ratio"] >= 0.25:
                record["physical_depth_median"] = float(np.median(crop[valid]))
        for pair, result in registrations.items():
            record[f"{pair}_reliable"] = bool(result.get("reliable", False))
            record[f"{pair}_magnitude"] = result.get("magnitude")
        records.append(record)
    return records


def balanced_registration_sample(stems: list[str], depth_map: dict[str, Path], count: int, seed: int) -> set[str]:
    if count <= 0 or count >= len(stems):
        return set(stems)
    groups = {
        "png": [stem for stem in stems if depth_map[stem].suffix.casefold() == ".png"],
        "jpg": [stem for stem in stems if depth_map[stem].suffix.casefold() in {".jpg", ".jpeg"}],
    }
    rng = random.Random(seed)
    selected: list[str] = []
    target_each = count // 2
    for name in ("png", "jpg"):
        group = sorted(groups[name])
        selected.extend(rng.sample(group, min(target_each, len(group))))
    remaining = [stem for stem in stems if stem not in set(selected)]
    needed = min(count, len(stems)) - len(selected)
    if needed > 0:
        selected.extend(rng.sample(sorted(remaining), needed))
    return set(selected)


def summarize_registration(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    attempted = [row for row in rows if row.get(f"{prefix}_reliable") is not None]
    reliable = [row for row in attempted if row.get(f"{prefix}_reliable")]
    reasons = Counter(row.get(f"{prefix}_reason", "unknown") for row in attempted if not row.get(f"{prefix}_reliable"))
    result = {
        "attempted": len(attempted),
        "reliable": len(reliable),
        "reliable_ratio": len(reliable) / len(attempted) if attempted else None,
        "dx_px": distribution(row[f"{prefix}_dx"] for row in reliable),
        "dy_px": distribution(row[f"{prefix}_dy"] for row in reliable),
        "magnitude_px": distribution(row[f"{prefix}_magnitude"] for row in reliable),
        "confidence": distribution(row[f"{prefix}_confidence"] for row in reliable),
        "unreliable_reasons": dict(reasons),
    }
    result["by_sample_encoding"] = {}
    for kind in sorted({row["depth_kind"] for row in attempted}):
        subset = [row for row in attempted if row["depth_kind"] == kind]
        good = [row for row in subset if row.get(f"{prefix}_reliable")]
        result["by_sample_encoding"][kind] = {
            "attempted": len(subset),
            "reliable": len(good),
            "reliable_ratio": len(good) / len(subset) if subset else None,
            "magnitude_px": distribution(row[f"{prefix}_magnitude"] for row in good),
        }
    return result


def grouped_target_registration(
    targets: list[dict[str, Any]], value_key: str, lower: float, upper: float, labels: tuple[str, str]
) -> dict[str, Any]:
    groups = {
        labels[0]: [record for record in targets if record[value_key] is not None and record[value_key] <= lower],
        labels[1]: [record for record in targets if record[value_key] is not None and record[value_key] >= upper],
    }
    result: dict[str, Any] = {}
    for name, records in groups.items():
        item: dict[str, Any] = {"target_count": len(records)}
        for pair in ("reg_ir", "reg_depth"):
            magnitudes = [
                record[f"{pair}_magnitude"]
                for record in records
                if record.get(f"{pair}_reliable") and record.get(f"{pair}_magnitude") is not None
            ]
            item[pair] = distribution(magnitudes)
        result[name] = item
    return result


def summarize_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    physical_values = [record["physical_depth_median"] for record in targets if record["physical_depth_median"] is not None]
    area_values = [record["bbox_area_ratio"] for record in targets]
    physical_summary = distribution(physical_values)
    area_summary = distribution(area_values)
    result: dict[str, Any] = {
        "method_note": (
            "Registration values are reliable image-level translations inherited by targets; "
            "they are not local per-bbox registrations."
        ),
        "physical_depth_coverage": len(physical_values) / len(targets) if targets else None,
        "physical_target_depth": physical_summary,
        "bbox_area_ratio": area_summary,
        "physical_depth_groups": None,
        "bbox_scale_groups": None,
    }
    if physical_values:
        low, high = (float(value) for value in np.percentile(physical_values, (33.333, 66.667)))
        result["physical_depth_thresholds"] = {"near_max": low, "far_min": high, "unit": "dataset depth units (mm per specification for PNG uint16)"}
        result["physical_depth_groups"] = grouped_target_registration(
            targets, "physical_depth_median", low, high, ("near", "far")
        )
    if area_values:
        low, high = (float(value) for value in np.percentile(area_values, (33.333, 66.667)))
        result["bbox_scale_thresholds"] = {"small_max": low, "large_min": high, "unit": "image area ratio"}
        result["bbox_scale_groups"] = grouped_target_registration(
            targets, "bbox_area_ratio", low, high, ("small_bbox", "large_bbox")
        )
    return result


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100 * float(value):.{digits}f}%"


def compact_distribution(stats: dict[str, Any], percent: bool = False) -> str:
    formatter = fmt_percent if percent else fmt_number
    return (
        f"mean={formatter(stats.get('mean'))}, median={formatter(stats.get('median'))}, "
        f"p90={formatter(stats.get('p90'))}, p95={formatter(stats.get('p95'))}, "
        f"min={formatter(stats.get('min'))}, max={formatter(stats.get('max'))}"
    )


def compact_distribution_with_count(stats: dict[str, Any]) -> str:
    return f"n={stats.get('count', 0)}, {compact_distribution(stats)}"


def markdown_counter(counter: dict[str, int]) -> str:
    return ", ".join(f"`{key}`: {value}" for key, value in counter.items()) or "none"


def registration_markdown(name: str, stats: dict[str, Any]) -> list[str]:
    lines = [
        f"### {name}",
        "",
        f"- Attempted: {stats['attempted']}",
        f"- Reliable: {stats['reliable']} ({fmt_percent(stats['reliable_ratio'])})",
        f"- Reliable displacement: {compact_distribution(stats['magnitude_px'])} px",
        f"- dx: {compact_distribution(stats['dx_px'])} px",
        f"- dy: {compact_distribution(stats['dy_px'])} px",
        f"- Unreliable reasons: {markdown_counter(stats['unreliable_reasons'])}",
    ]
    for encoding, item in stats["by_sample_encoding"].items():
        lines.append(
            f"- `{encoding}`: {item['reliable']}/{item['attempted']} reliable "
            f"({fmt_percent(item['reliable_ratio'])}); displacement "
            f"{compact_distribution(item['magnitude_px'])} px."
        )
    lines.append("")
    return lines


def generate_report(summary: dict[str, Any], c4_review: dict[str, Any] | None = None) -> str:
    """Render the reproducible C5 Chinese report from formal statistics and fixed guidance."""
    c4_review = c4_review or {}
    scope = summary["scope"]
    matching = summary["matching"]
    basic = summary["basic"]
    infrared = summary["infrared"]
    ir_basic = basic["infrared"]
    png = summary["depth"]["png_uint16_single"]
    jpg = summary["depth"]["jpg_uint8"]
    registration_ir = summary["registration"]["visible_ir"]
    registration_depth = summary["registration"]["visible_depth"]
    targets = summary["near_far"]
    near_group = targets["physical_depth_groups"]["near"]
    far_group = targets["physical_depth_groups"]["far"]
    thresholds = targets["physical_depth_thresholds"]
    anomalies = c4_review.get("anomalies", {})

    counts = matching["file_counts"]
    visible = basic["visible"]
    depth = basic["depth"]
    top_zero_stems = [item["stem"] for item in png["highest_zero_ratio_stems"][:4]]
    near_all_zero = anomalies.get("near_all_zero", [])
    lowest_dynamic = anomalies.get("lowest_dynamic_range", [])
    all_black = anomalies.get("all_black", [])
    unexpected = anomalies.get("unexpected_representations", [])
    unexpected_max = anomalies.get("outside_expected_nonzero_max_range", [])
    primary_near_zero = (
        f"`{near_all_zero[0]['stem']}`，zero ratio 约 "
        f"{fmt_percent(near_all_zero[0]['zero_ratio'])}。"
        if near_all_zero
        else "未发现。"
    )
    other_high_zero = "、".join(f"`{stem}`" for stem in top_zero_stems[1:]) or "未发现"
    minimum_dynamic = (
        f"`{lowest_dynamic[0]['stem']}`，动态范围为 "
        f"{fmt_number(lowest_dynamic[0]['dynamic_range'], 0)}。"
        if lowest_dynamic
        else "未评估。"
    )

    lines = [
        "# AIC2026 多模态数据分析报告",
        "",
        "本报告汇总成员 C 对 Visible、Infrared 与 Depth 数据的 C1～C4 检查结果。数值以 `outputs/analysis/modalities_stats.json` 的正式全量统计为准。实测结果、工程解释与后续实验假设在文中分别说明；任何预处理或融合方案均需通过固定划分上的受控实验验证，本文不对 mAP 提升作预判。",
        "",
        "## 1. 数据范围与约束",
        "",
        "**实测结果**",
        "",
        f"- Train 数据共 {scope['analyzed_stems']} 组；Visible、Infrared、Depth 与 `labels_clean` 的 filename stem 一一对应，共同 stem 和并集均为 {matching['common_stems']}，未发现缺失或重复 stem。",
        f"- 官方原始标签位于 `data/raw/train/labels/`，共 {counts['raw_labels']} 个 TXT；旧路径 `data/raw/train/labels/labels/` 当前不使用。",
        "- Bbox、near/far 和目标级分析统一使用 `data/processed/train/labels_clean/`。官方原始标签未被修改。",
        f"- 固定划分为 `data/splits/train.txt` {scope['fixed_train_count']} 组、`data/splits/val.txt` {scope['fixed_val_count']} 组，二者无重叠，不重新随机划分。",
        "- `PHASE_1_1000` 不参与训练、验证、调参或人工标注。",
        "",
        "**工程约束**",
        "",
        "- `data/raw/` 始终只读；分析派生文件写入 `outputs/analysis/` 或 `outputs/visualization/`。",
        "- 所有后续 E001～E006 实验必须沿用同一固定 train/val split，保证比较口径一致。",
        "- Visible GT bbox 的坐标基准始终是官方 Visible 图像，不得依据 IR 或 Depth 的残余偏移修改标注。",
        "",
        "## 2. 三模态基础对应关系",
        "",
        "| 模态 | 文件数 | 扩展名 | dtype | shape | 通道数 |",
        "|---|---:|---|---|---|---:|",
        f"| Visible | {counts['visible']} | PNG {visible['suffix_distribution']['.png']}，JPG {visible['suffix_distribution']['.jpg']} | uint8 | 1080×1920×3：{visible['shape_distribution']['1080x1920x3']}；360×640×3：{visible['shape_distribution']['360x640x3']} | 3 |",
        f"| Infrared | {counts['infrared']} | PNG {ir_basic['suffix_distribution']['.png']}，JPG {ir_basic['suffix_distribution']['.jpg']} | uint8 | 1080×1920×3：{ir_basic['shape_distribution']['1080x1920x3']}；360×640×3：{ir_basic['shape_distribution']['360x640x3']} | 3 |",
        f"| Depth | {counts['depth']} | PNG {summary['depth']['png_file_count']}，JPG {summary['depth']['jpg_file_count']} | PNG uint16；JPG uint8 | PNG 1080×1920；JPG 360×640×3 | PNG 1；JPG 3 |",
        "",
        f"三模态 width 和 height 一致率均为 {fmt_percent(matching['width_agreement_ratio'], 0)}。该结果仅说明同一 stem 的数组宽高一致，不能证明 RGB/Visible、Infrared 与 Depth 已达到逐像素严格配准。黑边、有效视场差异、残余位移以及有限的自动配准可靠率均表明：**尺寸一致不等于像素级严格对齐**。",
        "",
        "## 3. Infrared 数据特性",
        "",
        f"全部 {counts['infrared']} 张 Infrared 均为 uint8 三通道图像，全局数值范围为 {fmt_number(infrared['global_pixels']['min'], 0)}～{fmt_number(infrared['global_pixels']['max'], 0)}，全局均值为 {fmt_number(infrared['global_pixels']['mean'])}，标准差为 {fmt_number(infrared['global_pixels']['std'])}。",
        "",
        "| 通道对 | 平均绝对差 mean | median | P95 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("B/G", "B_G"), ("B/R", "B_R"), ("G/R", "G_R")):
        item = infrared["channel_mean_absolute_difference"][key]
        lines.append(
            f"| {label} | {fmt_number(item['mean'])} | {fmt_number(item['median'])} | "
            f"{fmt_number(item['p95'])} | {fmt_number(item['max'])} |"
        )
    lines.extend(
        [
            "",
            f"**结论：** Infrared 三通道高度相似，但并非逐像素完全相同。正式统计记录了 {infrared['channel_difference_outlier_count']} 个 Tukey 通道差异异常样本；这些样本是后续人工复核对象，不能据此直接判定为错误数据或自动删除。",
            "",
            "## 4. Infrared 黑边与有效视场",
            "",
            f"- 黑边出现率：{fmt_percent(ir_basic['border_occurrence_ratio'])}。",
            f"- 有效视场比例：均值 {fmt_percent(ir_basic['effective_fov_ratio']['mean'])}，中位数 {fmt_percent(ir_basic['effective_fov_ratio']['median'])}。",
            f"- 右侧连续低值边带宽度：中位数 {fmt_number(ir_basic['border_width_px']['right']['median'], 0)} px。",
            "",
            "黑边测量基于图像边缘连续灰度低值区域，普通 uint8 图像的低值阈值为灰度不高于 8；该指标用于描述边缘带，不代表所有暗像素均无效。较高的黑边出现率和有效视场变化会影响直接像素级融合。原图不应被独立裁切或覆盖；后续可将边缘 mask 作为受控实验输入，并保证涉及几何变换时三模态参数完全同步。",
            "",
            "## 5. PNG Depth 数据特性",
            "",
            f"{summary['depth']['png_file_count']} 张 PNG Depth 均以 `cv2.IMREAD_UNCHANGED` 读取，实测为 shape `1080×1920`、单通道 uint16，数值范围 {fmt_number(png['all_pixels']['min'], 0)}～{fmt_number(png['all_pixels']['max'], 0)}。全像素均值为 {fmt_number(png['all_pixels']['mean'])}，中位数为 {fmt_number(png['all_pixels']['median'], 0)}。",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 全局 zero ratio | {fmt_percent(png['global_zero_ratio'], 4)} |",
            f"| 每图 zero ratio median | {fmt_percent(png['per_image_zero_ratio']['median'], 4)} |",
            f"| 每图 zero ratio P90 | {fmt_percent(png['per_image_zero_ratio']['p90'], 4)} |",
            f"| 每图 zero ratio P95 | {fmt_percent(png['per_image_zero_ratio']['p95'], 4)} |",
            f"| 每图 zero ratio max | {fmt_percent(png['per_image_zero_ratio']['max'], 4)} |",
            f"| 全像素 `<300`，包含零值 | {fmt_percent(png['all_pixel_range_ratios']['below_300_including_zero'], 4)} |",
            f"| 非零有效像素 `<300` | {fmt_percent(png['nonzero_below_300_ratio'], 4)} |",
            f"| 全像素 `300～20000`，含边界 | {fmt_percent(png['all_pixel_range_ratios']['range_300_to_20000_inclusive'], 4)} |",
            f"| 全像素 `>20000` | {fmt_percent(png['all_pixel_range_ratios']['above_20000'], 4)} |",
            "",
            "Depth 中的零值是**无效深度候选**，不能解释为真实 0 mm，也不能把包含大量零值的 `<300` 全像素比例解释为真实近距离比例。物理范围分析必须将 zero ratio 与非零有效像素中的 `<300` 比例分开报告。",
            "",
            "## 6. JPG Depth 数据特性",
            "",
            f"{summary['depth']['jpg_file_count']} 张 JPG Depth 均为 shape `360×640×3`、uint8 三通道表示，全局范围 {fmt_number(jpg['global_pixels']['min'], 0)}～{fmt_number(jpg['global_pixels']['max'], 0)}，全局均值 {fmt_number(jpg['global_pixels']['mean'])}，标准差 {fmt_number(jpg['global_pixels']['std'])}。",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 每图 zero ratio mean | {fmt_percent(jpg['per_image_zero_ratio']['mean'], 4)} |",
            f"| 每图 zero ratio median | {fmt_percent(jpg['per_image_zero_ratio']['median'], 4)} |",
            f"| 动态范围 median | {fmt_number(jpg['per_image_dynamic_range']['median'], 0)} |",
            f"| 灰度熵 mean | {fmt_number(jpg['per_image_grayscale_entropy_bits']['mean'])} bits |",
            f"| 灰度熵 median | {fmt_number(jpg['per_image_grayscale_entropy_bits']['median'])} bits |",
            "",
            "JPG Depth 的物理深度映射无法仅从当前数据确认。JPG 的 0～255 不得解释为毫米，不适用 `<300 mm`、`300～20000 mm` 或 `>20000 mm` 阈值。PNG uint16 与 JPG uint8 是两种显著不同的数据表示，不能直接共享同一套物理深度解释或归一化流程。",
            "",
            "## 7. C4 Depth 专项可视化",
            "",
            "C4 已对代表性 PNG uint16 Depth 生成 2×3 分析图，输出位于 `outputs/visualization/depth_analysis/`。布局包括 Visible 参考图、Raw/Normalized Depth、Valid Mask、Percentile-Clipped Depth、Log Depth 与 Inverse Depth。",
            "",
            "| 可视化 | 处理方式与用途 |",
            "|---|---|",
            "| Valid Mask | 使用 `depth > 0` 标记有效区域，直接检查无效区域分布和 zero ratio。 |",
            "| Percentile-Clipped | 仅用非零像素计算 P2～P98，裁剪后归一化；用于改善显示对比度，并作为候选归一化方案。 |",
            "| Log Depth | 仅对有效像素计算 `log1p(depth)` 并独立归一化；用于压缩远距离动态范围。 |",
            "| Inverse Depth | 仅对有效像素计算 `1/depth`，避免除零后独立归一化；用于突出近距离结构变化。 |",
            "",
            "四类派生结果均保持无效零值区域为黑色，仅用于分析或候选预处理，不改变原始 Depth，也不是已经验证的最优训练方案。代表性输出覆盖指定高 zero ratio 样本及按 zero ratio 分位确定性选取的 val 样本。",
            "",
            "## 8. 三模态空间配准分析",
            "",
            f"配准统计采用 seed {scope['seed']} 的 {scope['registration_sample_count']} 个确定性、格式平衡样本。方法为受约束的局部边缘相关，仅估计有限搜索窗口内的图像级平移；低置信、峰值模糊或边界结果不进入可靠位移汇总。",
            "",
            "| 配准对 | 可靠结果 | 可靠率 | 位移 mean | median | P90 | P95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Visible ↔ IR | {registration_ir['reliable']}/{registration_ir['attempted']} | {fmt_percent(registration_ir['reliable_ratio'])} | {fmt_number(registration_ir['magnitude_px']['mean'])} px | {fmt_number(registration_ir['magnitude_px']['median'])} px | {fmt_number(registration_ir['magnitude_px']['p90'])} px | {fmt_number(registration_ir['magnitude_px']['p95'])} px |",
            f"| Visible ↔ Depth | {registration_depth['reliable']}/{registration_depth['attempted']} | {fmt_percent(registration_depth['reliable_ratio'])} | {fmt_number(registration_depth['magnitude_px']['mean'])} px | {fmt_number(registration_depth['magnitude_px']['median'])} px | {fmt_number(registration_depth['magnitude_px']['p90'])} px | {fmt_number(registration_depth['magnitude_px']['p95'])} px |",
            "",
            "自动配准可靠率有限，这些结果只能作为残余偏移线索，不能外推为全部样本都存在某个固定平移，也不能证明逐像素严格对齐。可靠率本身还受到跨模态外观差异、黑边和有效视场的影响。任何估计偏移均不得用于移动或修改 Visible GT bbox、重写 `labels_clean` 或重新生成官方标注。",
            "",
            "## 9. Near/Far 目标分析",
            "",
            f"Bbox 和目标级统计使用 `labels_clean`。clean-label 目标共 {targets['bbox_area_ratio']['count']} 个，其中 {targets['physical_target_depth']['count']} 个获得可用的 PNG 框内物理深度中位数，覆盖率为 {fmt_percent(targets['physical_depth_coverage'])}。Near/Far 阈值来自有效 PNG 目标深度的 33.3% 与 66.7% 分位数：Near 不高于 {fmt_number(thresholds['near_max'], 2)} mm，Far 不低于 {fmt_number(thresholds['far_min'], 2)} mm，两组各 {near_group['target_count']} 个目标；JPG 样本不参与物理 Near/Far 分组。",
            "",
            "| 分组 | 目标数 | IR 位移 median / P90 | Depth 位移 median / P90 |",
            "|---|---:|---:|---:|",
            f"| Near | {near_group['target_count']} | {fmt_number(near_group['reg_ir']['median'], 2)} / {fmt_number(near_group['reg_ir']['p90'], 2)} px | {fmt_number(near_group['reg_depth']['median'], 2)} / {fmt_number(near_group['reg_depth']['p90'], 2)} px |",
            f"| Far | {far_group['target_count']} | {fmt_number(far_group['reg_ir']['median'], 2)} / {fmt_number(far_group['reg_ir']['p90'], 2)} px | {fmt_number(far_group['reg_depth']['median'], 2)} / {fmt_number(far_group['reg_depth']['p90'], 2)} px |",
            "",
            "这些位移是目标继承的**可靠图像级平移估计**，不是 bbox 内的局部配准或目标视差测量。当前结果不足以断言距离越近或越远必然导致更大的配准误差。",
            "",
            "## 10. 代表性异常样本",
            "",
            f"- PNG near-all-zero：{primary_near_zero}",
            f"- 其他高 zero ratio 代表样本：{other_high_zero}。",
            f"- 最小动态范围 PNG：{minimum_dynamic}",
            f"- 全黑 PNG：{'、'.join(f'`{stem}`' for stem in all_black) if all_black else '未发现'}。",
            f"- 非 uint16、非单通道或 shape 异常 PNG：{'、'.join(item['stem'] for item in unexpected) if unexpected else '未发现'}。",
            f"- 最大值低于 300 或高于 20000 的 PNG：{'、'.join(unexpected_max) if unexpected_max else '未发现'}。",
            f"- Infrared 通道差异 Tukey 异常样本：{infrared['channel_difference_outlier_count']} 个。",
            "",
            "这些记录用于后续人工复核和鲁棒性实验，不构成自动删除、填充、转换或修复数据的依据。",
            "",
            "## 11. IR-only 后续实验假设",
            "",
            "E002 建议在固定 split 上分别验证：原始三通道 IR、灰度单通道 IR、灰度复制三通道、IR 独立归一化，以及 CLAHE 受控消融。三通道高度相似使单通道方案具有验证价值，但仍需与保留三通道的兼容基线直接比较。",
            "",
            "**实验假设：** 在低照度、Visible 对比度不足、阴影或强光干扰等场景中，IR 可能提供额外的轮廓和目标响应信息。该判断仅用于提出分场景实验，不代表灰度化、CLAHE 或 IR-only 一定提升 mAP。",
            "",
            "## 12. Depth-only 后续实验假设",
            "",
            "E003 对 PNG 建议比较 percentile normalization、log-depth、inverse-depth 与 Depth + valid mask。输入形式可分别验证单通道、复制三通道、附加 valid-mask 通道、归一化到 `[0,1]` 以及固定无效值处理。",
            "",
            "JPG Depth 应作为独立的 uint8 representation，采用独立 normalization，不按毫米解释。如果训练代码把 PNG uint16 和 JPG uint8 直接纳入同一套物理尺度归一化，会造成明确的数据表示不一致风险。",
            "",
            "**实验假设：** 在前后景外观相似、尺度变化、遮挡或视觉纹理不足的场景中，有效 Depth 可能提供几何和距离线索；大面积无效区域则可能削弱这一作用，因此必须同时验证 valid mask。上述方案均需通过 E003 实验评估，不能预先认定会提升 mAP。",
            "",
            "## 13. Fusion 实验前安全约束",
            "",
            "后续 E004～E006 的 resize、crop、flip、affine 和 perspective 等几何增强必须在 Visible、IR 和 Depth 上共享完全相同的参数。光度增强与数值归一化可以按模态独立设计，但不得破坏三模态的几何对应关系。",
            "",
            "融合实验应显式考虑 IR black-border mask、Depth valid mask、残余错位和模态专用归一化。IR/Depth 相对 Visible 的残余偏移只可用于数据质量分析、有效区域 mask、鲁棒融合设计、显式配准实验和模态不确定性处理；不得用于移动或修改 Visible bbox、重写 `labels_clean`、依据 IR/Depth 偏移重新生成标注或擅自修正官方 Visible 标注。",
            "",
            "## 14. E002～E006 建议实验顺序",
            "",
            "```text",
            "E001 RGB baseline",
            "→ E002 IR-only",
            "→ E003 Depth-only",
            "→ E004 RGB + IR",
            "→ E005 RGB + Depth",
            "→ E006 RGB + IR + Depth",
            "```",
            "",
            "该顺序先建立单模态基线，再评估双模态与三模态增益，有利于区分额外信息来自哪一模态。成员 C 的报告只提供可验证假设和数据边界，不实现 Fusion。",
            "",
            "## 15. 当前不能下结论的事项",
            "",
            "当前数据分析不能证明：",
            "",
            "- JPG Depth 的物理深度映射；",
            "- 三模态已达到逐像素严格配准；",
            "- 低置信或不可靠样本具有精确可信的 dx/dy；",
            "- 图像级平移等价于目标级局部视差；",
            "- Near 或 Far 必然对应更大的配准误差；",
            "- CLAHE、percentile、log、inverse 或 valid mask 一定提升 mAP；",
            "- Fusion 一定优于 RGB baseline。",
            "",
            "## 16. 成员 C 最终结论",
            "",
            f"1. Infrared 三通道高度相似但并非完全相同，存在 {infrared['channel_difference_outlier_count']} 个值得人工复核的通道差异 Tukey 异常样本。",
            f"2. PNG Depth 为单通道 uint16，范围 {fmt_number(png['all_pixels']['min'], 0)}～{fmt_number(png['all_pixels']['max'], 0)}；JPG Depth 为三通道 uint8，范围 {fmt_number(jpg['global_pixels']['min'], 0)}～{fmt_number(jpg['global_pixels']['max'], 0)}，二者的数据表示和物理解释不可混用。",
            f"3. PNG Depth 全局 zero ratio 为 {fmt_percent(png['global_zero_ratio'], 4)}，零值应作为无效深度候选单独处理；JPG Depth 的物理映射仍无法确认。",
            "4. 三模态尺寸和 stem 完全对应，但现有证据不支持逐像素严格配准；IR 黑边、Depth 无效区及残余偏移需要在后续管线中显式处理。",
            "5. IR 在低照度或 Visible 对比度不足场景、Depth 在可能受益于几何与距离线索的场景中，具有提供额外信息的实验价值；这仍是待验证假设。",
            "6. 后续正式实验必须沿用固定 split，通过 E002～E006 逐项验证模态专用预处理和融合方案，不得根据 IR/Depth 偏移修改 Visible GT。",
        ]
    )
    report = "\n".join(lines) + "\n"
    return report.replace("\r\n", "\n").replace("\r", "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AIC2026 modalities and residual alignment.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--stem", action="append", help="Specific stem; repeat for multiple stems.")
    parser.add_argument("--max-samples", type=int, default=0, help="Deterministic analysis sample; 0 means all.")
    parser.add_argument("--registration-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--depth-visualizations",
        action="store_true",
        help="Write representative uint16 PNG Depth analysis figures.",
    )
    parser.add_argument(
        "--depth-visualization-dir", type=Path, default=DEFAULT_DEPTH_VISUALIZATION_DIR
    )
    parser.add_argument(
        "--depth-visualization-val-samples",
        type=int,
        default=5,
        help="Deterministic val zero-ratio quantile samples in addition to requested stems.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-write", action="store_true", help="Run analysis without writing reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    messages: list[str] = []
    errors: list[str] = []
    visible_map, visible_duplicates = build_stem_map(args.data_root / "visible", IMAGE_SUFFIXES, messages)
    infrared_map, infrared_duplicates = build_stem_map(args.data_root / "infrared", IMAGE_SUFFIXES, messages)
    depth_map, depth_duplicates = build_stem_map(args.data_root / "depth", IMAGE_SUFFIXES, messages)
    label_map, label_duplicates = build_stem_map(args.label_dir, {".txt"}, messages)
    raw_label_map, raw_label_duplicates = build_stem_map(
        args.data_root / "labels", {".txt"}, messages
    )
    maps = {
        "visible": visible_map,
        "infrared": infrared_map,
        "depth": depth_map,
        "labels_clean": label_map,
    }
    common = set.intersection(*(set(mapping) for mapping in maps.values()))
    union = set.union(*(set(mapping) for mapping in maps.values()))
    train_stems = load_fixed_split("train")
    val_stems = load_fixed_split("val")
    train_set = set(train_stems)
    val_set = set(val_stems)

    if args.split == "all":
        allowed = sorted(common)
    else:
        allowed = load_fixed_split(args.split)
    allowed_set = set(allowed)
    if args.stem:
        selected = list(dict.fromkeys(args.stem))
        outside = [stem for stem in selected if stem not in allowed_set]
        if outside:
            raise ValueError(
                f"Requested stem(s) do not belong to split {args.split!r}: {', '.join(outside)}"
            )
    else:
        selected = [stem for stem in allowed if stem in common]
        if args.max_samples > 0 and args.max_samples < len(selected):
            selected = random.Random(args.seed).sample(sorted(selected), args.max_samples)
    incomplete = [stem for stem in selected if stem not in common]
    if incomplete:
        report_warning(messages, f"Skipping {len(incomplete)} incomplete or ambiguous selected stems")
        selected = [stem for stem in selected if stem in common]
    if not selected:
        raise RuntimeError("No complete samples selected")

    registration_count = len(selected) if args.stem else args.registration_samples
    registration_stems = balanced_registration_sample(
        selected, depth_map, registration_count, args.seed
    )
    basics = {name: BasicStats() for name in ("visible", "infrared", "depth")}
    depth_border_basics = {
        "png_uint16_single": BasicStats(),
        "jpg_uint8": BasicStats(),
    }
    infrared_stats = InfraredStats()
    png_depth_stats = PngDepthStats()
    jpg_depth_stats = JpgDepthStats()
    rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    width_matches = 0
    height_matches = 0
    png_count = 0
    jpg_count = 0

    for index, stem in enumerate(selected, start=1):
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(selected)):
            print(f"Analyzing {index}/{len(selected)}: {stem}", flush=True)
        try:
            visible_path = visible_map[stem]
            infrared_path = infrared_map[stem]
            depth_path = depth_map[stem]
            label_path = label_map[stem]
            visible = read_image(visible_path)
            infrared = read_image(infrared_path)
            depth = read_image(depth_path)
        except (OSError, ValueError, cv2.error) as error:
            message = f"{stem}: {error}"
            errors.append(message)
            warnings.warn(message, stacklevel=2)
            continue

        depth_suffix = depth_path.suffix.casefold()
        if depth_suffix == ".png" and depth.ndim == 2 and depth.dtype == np.uint16:
            depth_kind = "png_uint16_single"
            png_count += 1
        elif depth_suffix in {".jpg", ".jpeg"} and depth.dtype == np.uint8:
            depth_kind = "jpg_uint8"
            jpg_count += 1
        else:
            depth_kind = f"other_{depth.dtype}_{shape_text(depth)}"
            report_warning(messages, f"Unexpected Depth representation for {stem}: {depth_kind}")

        borders = {
            "visible": border_analysis(visible, "visible", visible_path.suffix.casefold()),
            "infrared": border_analysis(infrared, "infrared", infrared_path.suffix.casefold()),
            "depth": border_analysis(depth, "depth", depth_suffix),
        }
        basics["visible"].update(visible_path, visible, borders["visible"])
        basics["infrared"].update(infrared_path, infrared, borders["infrared"])
        basics["depth"].update(depth_path, depth, borders["depth"])
        if depth_kind in depth_border_basics:
            depth_border_basics[depth_kind].update(depth_path, depth, borders["depth"])
        ir_values = infrared_stats.update(stem, infrared)
        depth_values: dict[str, float | None]
        if depth_kind == "png_uint16_single":
            depth_values = png_depth_stats.update(stem, depth)
        elif depth_kind == "jpg_uint8":
            depth_values = jpg_depth_stats.update(depth)
        else:
            values = depth.astype(np.float64)
            depth_values = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "zero_ratio": float(np.count_nonzero(depth == 0) / depth.size),
            }

        heights = (visible.shape[0], infrared.shape[0], depth.shape[0])
        widths = (visible.shape[1], infrared.shape[1], depth.shape[1])
        width_equal = len(set(widths)) == 1
        height_equal = len(set(heights)) == 1
        width_matches += int(width_equal)
        height_matches += int(height_equal)
        registrations: dict[str, dict[str, Any]] = {}
        if stem in registration_stems:
            registrations["reg_ir"] = local_edge_registration(visible, infrared, "visible_ir")
            registrations["reg_depth"] = local_edge_registration(
                visible, depth, "visible_depth", depth_kind == "png_uint16_single"
            )

        row: dict[str, Any] = {
            "stem": stem,
            "split": "train" if stem in train_set else "val" if stem in val_set else "unknown",
            "visible_filename": visible_path.name,
            "visible_shape": shape_text(visible),
            "visible_dtype": str(visible.dtype),
            "visible_channels": channel_count(visible),
            "infrared_filename": infrared_path.name,
            "infrared_shape": shape_text(infrared),
            "infrared_dtype": str(infrared.dtype),
            "infrared_channels": channel_count(infrared),
            "depth_filename": depth_path.name,
            "depth_shape": shape_text(depth),
            "depth_dtype": str(depth.dtype),
            "depth_channels": channel_count(depth),
            "depth_kind": depth_kind,
            "width_equal": width_equal,
            "height_equal": height_equal,
            "ir_mean": ir_values["mean"],
            "ir_std": ir_values["std"],
            "ir_diff_bg": ir_values["diff_bg"],
            "ir_diff_br": ir_values["diff_br"],
            "ir_diff_gr": ir_values["diff_gr"],
            "ir_border_left_px": borders["infrared"]["left_px"],
            "ir_border_right_px": borders["infrared"]["right_px"],
            "ir_fov_ratio": borders["infrared"]["fov_ratio"],
            "depth_min": depth_values.get("min"),
            "depth_max": depth_values.get("max"),
            "depth_mean": depth_values.get("mean"),
            "depth_median": depth_values.get("median"),
            "depth_std": depth_values.get("std"),
            "depth_zero_ratio": depth_values.get("zero_ratio"),
            "depth_dynamic_range": depth_values.get("dynamic_range"),
            "depth_entropy_bits": depth_values.get("entropy_bits"),
            "depth_diff_bg": depth_values.get("diff_bg"),
            "depth_diff_br": depth_values.get("diff_br"),
            "depth_diff_gr": depth_values.get("diff_gr"),
            "depth_border_left_px": borders["depth"]["left_px"],
            "depth_border_right_px": borders["depth"]["right_px"],
            "depth_fov_ratio": borders["depth"]["fov_ratio"],
        }
        for prefix, result in registrations.items():
            serialized_result = sanitize_registration_result(result)
            for key in (
                "dx",
                "dy",
                "magnitude",
                "confidence",
                "edge_correlation",
                "peak_margin",
                "peak_ratio",
                "reliable",
                "reason",
            ):
                row[f"{prefix}_{key}"] = serialized_result.get(key)
        for prefix in ("reg_ir", "reg_depth"):
            if prefix not in registrations:
                for key in (
                    "dx",
                    "dy",
                    "magnitude",
                    "confidence",
                    "edge_correlation",
                    "peak_margin",
                    "peak_ratio",
                    "reliable",
                    "reason",
                ):
                    row[f"{prefix}_{key}"] = None
        rows.append(row)
        labels = parse_labels(label_path, messages)
        targets.extend(target_records(stem, labels, depth, depth_kind, registrations))

    decoded_count = len(rows)
    basic_results = {name: stats.result() for name, stats in basics.items()}
    infrared_result = infrared_stats.result(
        int(basics["infrared"].channels.get("3", 0)),
        int(basics["infrared"].channels.get("1", 0)),
        basics["infrared"].decoded,
    )
    png_result = png_depth_stats.result()
    jpg_result = jpg_depth_stats.result()
    matching = {
        "file_counts": {
            "visible": len(visible_map),
            "infrared": len(infrared_map),
            "depth": len(depth_map),
            "labels_clean": len(label_map),
            "raw_labels": len(raw_label_map),
        },
        "common_stems": len(common),
        "union_stems": len(union),
        "missing_by_source": {
            name: sorted(union - set(mapping)) for name, mapping in maps.items()
        },
        "duplicates": {
            "visible": visible_duplicates,
            "infrared": infrared_duplicates,
            "depth": depth_duplicates,
            "labels_clean": label_duplicates,
            "raw_labels": raw_label_duplicates,
        },
        "width_agreement_ratio": width_matches / decoded_count if decoded_count else None,
        "height_agreement_ratio": height_matches / decoded_count if decoded_count else None,
    }
    summary = {
        "scope": {
            "split": args.split,
            "analyzed_stems": len(selected),
            "decoded_stems": decoded_count,
            "fixed_train_count": len(train_stems),
            "fixed_val_count": len(val_stems),
            "fixed_split_overlap": len(train_set & val_set),
            "registration_sample_count": len(registration_stems),
            "seed": args.seed,
            "label_dir": portable_project_path(args.label_dir),
        },
        "matching": matching,
        "basic": basic_results,
        "infrared": infrared_result,
        "depth": {
            "png_file_count": png_count,
            "jpg_file_count": jpg_count,
            "png_uint16_single": png_result,
            "jpg_uint8": jpg_result,
            "border_by_encoding": {
                name: stats.result() for name, stats in depth_border_basics.items()
            },
        },
        "registration": {
            "method": {
                "name": "constrained local edge correlation",
                "max_resized_dimension": 480,
                "search_radius_resized_px": 8,
                "pilot_conclusion": (
                    "ORB-RANSAC produced implausible transforms; global phase correlation was "
                    "stable only for some JPG Visible/IR samples."
                ),
            },
            "visible_ir": summarize_registration(rows, "reg_ir"),
            "visible_depth": summarize_registration(rows, "reg_depth"),
        },
        "near_far": summarize_targets(targets),
        "warnings": messages,
        "errors": errors,
    }

    c4_review: dict[str, Any] = {
        "visualizations": [],
        "missing_preferred": [],
        "anomalies": c4_depth_anomaly_summary(rows),
    }
    if args.depth_visualizations:
        visualization_stems, missing_preferred = select_depth_visualization_stems(
            rows, args.depth_visualization_val_samples
        )
        c4_review["missing_preferred"] = missing_preferred
        for stem in visualization_stems:
            try:
                c4_review["visualizations"].append(
                    write_png_depth_visualization(
                        stem,
                        visible_map[stem],
                        depth_map[stem],
                        args.depth_visualization_dir,
                    )
                )
            except (OSError, ValueError, cv2.error) as error:
                message = f"{stem}: failed to write C4 Depth visualization: {error}"
                errors.append(message)
                warnings.warn(message, stacklevel=2)

    if not args.no_write:
        for path in (args.output_json, args.output_csv, args.output_report):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fieldnames = list(rows[0]) if rows else []
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        args.output_report.write_text(
            generate_report(summary, c4_review), encoding="utf-8", newline="\n"
        )
        print(f"Wrote: {args.output_json}")
        print(f"Wrote: {args.output_csv}")
        print(f"Wrote: {args.output_report}")
    print(
        json.dumps(
            {
                "analyzed_stems": len(selected),
                "decoded_stems": decoded_count,
                "png_depth": png_count,
                "jpg_depth": jpg_count,
                "registration_attempts": len(registration_stems),
                "registration_reliable_ir": summary["registration"]["visible_ir"]["reliable"],
                "registration_reliable_depth": summary["registration"]["visible_depth"]["reliable"],
                "depth_visualizations": len(c4_review["visualizations"]),
                "warnings": len(messages),
                "errors": len(errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
