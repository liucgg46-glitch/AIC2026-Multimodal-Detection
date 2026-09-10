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
SPLIT_DIR = REPO_ROOT / "data" / "splits"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
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


def generate_report(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    matching = summary["matching"]
    basic = summary["basic"]
    infrared = summary["infrared"]
    png = summary["depth"]["png_uint16_single"]
    jpg = summary["depth"]["jpg_uint8"]
    registration = summary["registration"]
    targets = summary["near_far"]
    lines = [
        "# AIC2026 Multimodal Modalities Summary",
        "",
        "> Measurement results describe the selected local dataset. Engineering suggestions are hypotheses for later experiments and are not claims of mAP improvement.",
        "",
        "Official raw labels: `data/raw/train/labels/`",
        "",
        "Clean labels used by this analysis: `data/processed/train/labels_clean/`",
        "",
        "Old path `data/raw/train/labels/labels/` is not used.",
        "",
        "## 1. Data scope and fixed split",
        "",
        f"- Split: `{scope['split']}`; analyzed stems: {scope['analyzed_stems']}.",
        f"- Fixed train/val counts: {scope['fixed_train_count']}/{scope['fixed_val_count']}; overlap: {scope['fixed_split_overlap']}.",
        f"- Registration is a deterministic format-balanced sample: {scope['registration_sample_count']} stems (seed {scope['seed']}).",
        "- No new train/val partition is created.",
        "",
        "## 2. Label paths",
        "",
        f"- Raw label files: {matching['file_counts']['raw_labels']}.",
        f"- Clean label files: {matching['file_counts']['labels_clean']}.",
        "- Bbox and target-depth analysis use only `labels_clean`.",
        "",
        "## 3. Basic modality statistics",
        "",
        "| Modality | Files | Suffixes | Dtypes | Shapes | Channels |",
        "|---|---:|---|---|---|---|",
    ]
    for name in ("visible", "infrared", "depth"):
        item = basic[name]
        lines.append(
            f"| {name} | {item['files']} | {markdown_counter(item['suffix_distribution'])} | "
            f"{markdown_counter(item['dtype_distribution'])} | {markdown_counter(item['shape_distribution'])} | "
            f"{markdown_counter(item['channel_distribution'])} |"
        )
    lines.extend(
        [
            "",
            f"- Common Visible/Infrared/Depth/clean-label stems: {matching['common_stems']}.",
            f"- Three-modality width agreement: {fmt_percent(matching['width_agreement_ratio'])}.",
            f"- Three-modality height agreement: {fmt_percent(matching['height_agreement_ratio'])}.",
            "",
            "## 4. Infrared dtype, channels and distribution",
            "",
            f"- Three-channel ratio: {fmt_percent(infrared['three_channel_ratio'])}; single-channel ratio: {fmt_percent(infrared['single_channel_ratio'])}.",
            f"- Global pixel range/mean/std: {fmt_number(infrared['global_pixels']['min'])} / {fmt_number(infrared['global_pixels']['max'])} / {fmt_number(infrared['global_pixels']['mean'])} / {fmt_number(infrared['global_pixels']['std'])}.",
            f"- B/G mean absolute difference: {compact_distribution(infrared['channel_mean_absolute_difference']['B_G'])}.",
            f"- B/R mean absolute difference: {compact_distribution(infrared['channel_mean_absolute_difference']['B_R'])}.",
            f"- G/R mean absolute difference: {compact_distribution(infrared['channel_mean_absolute_difference']['G_R'])}.",
            f"- Tukey channel-difference outliers: {infrared['channel_difference_outlier_count']} total; up to 20 are stored in JSON.",
            "",
            "## 5. Black borders and effective field of view",
            "",
            "Black-border measurement uses median continuous low-value runs from each image edge; fully low scan lines are excluded from the perpendicular side estimate. For ordinary uint8 imagery, low means grayscale <= 8. For PNG uint16 Depth, it means zero. This is an engineering measurement of edge bands, not proof that every dark pixel is invalid.",
            "",
            "| Modality | Border occurrence | Effective FOV ratio | Low/zero pixel ratio |",
            "|---|---:|---|---|",
        ]
    )
    border_rows = (
        ("visible", basic["visible"]),
        ("infrared", basic["infrared"]),
        ("depth PNG", summary["depth"]["border_by_encoding"]["png_uint16_single"]),
        ("depth JPG", summary["depth"]["border_by_encoding"]["jpg_uint8"]),
    )
    for name, item in border_rows:
        lines.append(
            f"| {name} | {fmt_percent(item['border_occurrence_ratio'])} | "
            f"{compact_distribution(item['effective_fov_ratio'], percent=True)} | "
            f"{compact_distribution(item['low_pixel_ratio'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "## 6. PNG Depth statistics",
            "",
            f"- Files: {summary['depth']['png_file_count']}; representation: single-channel uint16.",
            f"- All pixels: min={fmt_number(png['all_pixels']['min'])}, max={fmt_number(png['all_pixels']['max'])}, mean={fmt_number(png['all_pixels']['mean'])}, median={fmt_number(png['all_pixels']['median'])}.",
            f"- Nonzero pixels: count={png['nonzero_pixels']['count']}, mean={fmt_number(png['nonzero_pixels']['mean'])}, median={fmt_number(png['nonzero_pixels']['median'])}.",
            f"- Global zero ratio: {fmt_percent(png['global_zero_ratio'])}.",
            f"- Per-image zero ratio: {compact_distribution(png['per_image_zero_ratio'], percent=True)}.",
            f"- All-pixel `<300` ratio (includes zero): {fmt_percent(png['all_pixel_range_ratios']['below_300_including_zero'])}.",
            f"- All-pixel `300..20000` inclusive ratio: {fmt_percent(png['all_pixel_range_ratios']['range_300_to_20000_inclusive'])}.",
            f"- All-pixel `>20000` ratio: {fmt_percent(png['all_pixel_range_ratios']['above_20000'])}.",
            f"- Nonzero `<300` ratio: {fmt_percent(png['nonzero_below_300_ratio'])}.",
            "",
            "## 7. JPG Depth statistics",
            "",
            f"- Files: {summary['depth']['jpg_file_count']}; observed dtype/channel distributions are shown in the basic table.",
            f"- Global pixel range/mean/std: {fmt_number(jpg['global_pixels']['min'])} / {fmt_number(jpg['global_pixels']['max'])} / {fmt_number(jpg['global_pixels']['mean'])} / {fmt_number(jpg['global_pixels']['std'])}.",
            f"- Per-image zero ratio: {compact_distribution(jpg['per_image_zero_ratio'], percent=True)}.",
            f"- Per-image dynamic range: {compact_distribution(jpg['per_image_dynamic_range'])}.",
            f"- Grayscale entropy: {compact_distribution(jpg['per_image_grayscale_entropy_bits'])} bits.",
            f"- B/G difference: {compact_distribution(jpg['channel_mean_absolute_difference']['B_G'])}.",
            f"- B/R difference: {compact_distribution(jpg['channel_mean_absolute_difference']['B_R'])}.",
            f"- G/R difference: {compact_distribution(jpg['channel_mean_absolute_difference']['G_R'])}.",
            "- JPG Depth physical mapping cannot be confirmed directly from the current data.",
            "",
            "## 8. PNG/JPG encoding differences",
            "",
            "PNG Depth and JPG Depth are separate representations. PNG files are evaluated as uint16 single-channel depth under the project specification. JPG files are compressed uint8 imagery with unknown physical mapping; their values are not merged with PNG statistics and are never subjected to millimeter thresholds.",
            "",
            "## 9. Spatial alignment analysis",
            "",
            "Pilot comparison found ORB-RANSAC unstable across modalities (implausible scale/rotation/translation) and global phase correlation reliable only for some JPG Visible/IR samples. The selected method is constrained local edge correlation. It searches only a small translation window and rejects weak, ambiguous, or boundary peaks.",
            "",
            "`dx,dy` describe detected modality-content translation relative to Visible in original-image pixels. Reliable flags are mandatory; unreliable estimates are excluded from displacement summaries.",
            "",
        ]
    )
    lines.extend(registration_markdown("Visible ↔ Infrared", registration["visible_ir"]))
    lines.extend(registration_markdown("Visible ↔ Depth", registration["visible_depth"]))
    lines.extend(
        [
            "## 10. Near/far and bbox-scale alignment",
            "",
            f"- Targets with usable PNG physical-depth medians: {targets['physical_target_depth']['count']} ({fmt_percent(targets['physical_depth_coverage'])} of all labels).",
            f"- Physical target-depth distribution: {compact_distribution(targets['physical_target_depth'])}.",
            f"- Bbox area-ratio distribution: {compact_distribution(targets['bbox_area_ratio'])}.",
            "- Near/far thresholds are the 33.3% and 66.7% quantiles of valid PNG target-depth medians. JPG targets are excluded from physical near/far grouping.",
            "- Small/large bbox groups use bbox area-ratio quantiles and represent apparent target scale, not physical distance.",
            f"- Limitation: {targets['method_note']}",
            "",
        ]
    )
    if targets.get("physical_depth_thresholds"):
        thresholds = targets["physical_depth_thresholds"]
        lines.append(
            f"Physical thresholds: near <= {fmt_number(thresholds['near_max'])}, far >= {fmt_number(thresholds['far_min'])}."
        )
        lines.append("")
    for section_name, key in (("Physical depth groups", "physical_depth_groups"), ("BBox scale groups", "bbox_scale_groups")):
        groups = targets.get(key)
        if not groups:
            continue
        lines.extend([f"### {section_name}", "", "| Group | Targets | Visible↔IR magnitude | Visible↔Depth magnitude |", "|---|---:|---|---|"])
        for group_name, item in groups.items():
            lines.append(
                f"| {group_name} | {item['target_count']} | {compact_distribution_with_count(item['reg_ir'])} px | {compact_distribution_with_count(item['reg_depth'])} px |"
            )
        lines.append("")
    lines.extend(
        [
            "## 11. Representative anomalies",
            "",
            f"- IR largest channel differences: {', '.join(item['stem'] for item in infrared['largest_channel_difference_stems']) or 'none'}.",
            f"- PNG Depth highest zero ratios: {', '.join(item['stem'] for item in png['highest_zero_ratio_stems']) or 'none'}.",
            f"- Lowest IR effective FOV: {', '.join(item['stem'] for item in basic['infrared']['lowest_fov_stems']) or 'none'}.",
            f"- Lowest PNG Depth effective FOV: {', '.join(item['stem'] for item in summary['depth']['border_by_encoding']['png_uint16_single']['lowest_fov_stems']) or 'none'}.",
            f"- Lowest JPG Depth visual FOV: {', '.join(item['stem'] for item in summary['depth']['border_by_encoding']['jpg_uint8']['lowest_fov_stems']) or 'none'}.",
            "",
            "## 12. IR-only preprocessing suggestions",
            "",
            "- Treat grayscale conversion or single-channel training as an ablation only if channel-difference statistics confirm strong redundancy; retaining the stored three channels is the compatibility baseline.",
            "- Normalize IR independently from RGB. Consider CLAHE or contrast enhancement only as controlled experiments.",
            "- Preserve or explicitly mask measured edge bands; do not crop them independently from other modalities in fusion training.",
            "",
            "## 13. Depth-only preprocessing suggestions",
            "",
            "- PNG: preserve uint16 on read, maintain a valid mask, and compare percentile, log-depth, or inverse-depth display/input transforms without modifying source files.",
            "- PNG zero regions should remain distinguishable; an additional mask channel is an experiment candidate.",
            "- JPG: treat as an independent uint8 encoded representation with its own normalization. Do not interpret values as millimeters.",
            "- A training pipeline that silently mixes PNG uint16 and JPG uint8 Depth under one normalization is an engineering risk.",
            "",
            "## 14. Fusion preprocessing suggestions",
            "",
            "- Resize, crop, flip, affine and perspective parameters must be shared exactly across Visible, IR and Depth.",
            "- Modality-specific photometric normalization may differ, but geometry must stay synchronized.",
            "- Carry IR edge-band masks and PNG invalid-depth masks where useful, and evaluate robustness to residual misalignment.",
            "",
            "## 15. Current limitations and unsupported conclusions",
            "",
            "- Automatic registration is a translation-only estimate on a deterministic sample, not a dense calibration or proof of pixel-perfect alignment.",
            "- Unreliable matches are not converted into precise offsets.",
            "- Target-group registration inherits image-level shifts and cannot establish target-local parallax.",
            "- JPG Depth physical units and mapping remain unknown.",
            "- These statistics motivate experiments; they do not establish that any preprocessing choice improves mAP.",
            "",
        ]
    )
    if summary["warnings"]:
        lines.extend(["## Warnings", ""] + [f"- {message}" for message in summary["warnings"]] + [""])
    if summary["errors"]:
        lines.extend(["## Errors", ""] + [f"- {message}" for message in summary["errors"]] + [""])
    return "\n".join(lines)


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
                row[f"{prefix}_{key}"] = result.get(key)
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
            "label_dir": str(args.label_dir),
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
        args.output_report.write_text(generate_report(summary), encoding="utf-8", newline="\n")
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
                "warnings": len(messages),
                "errors": len(errors),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
