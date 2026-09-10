"""Create read-only 2x2 visualizations for aligned AIC2026 modalities.

Samples are selected from the team's fixed train/val split, or from all common
stems. Clean labels are used by default. Source images and labels are only read;
the script writes rendered figures under ``outputs/visualization``.
"""

from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw" / "train"
DEFAULT_LABEL_DIR = REPO_ROOT / "data" / "processed" / "train" / "labels_clean"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "visualization"
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


@dataclass(frozen=True)
class SampleFiles:
    stem: str
    visible: Path
    infrared: Path
    depth: Path
    label: Path


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float


def warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def build_stem_map(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    """Index files by case-sensitive stem and exclude ambiguous duplicates."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    candidates: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.casefold() in suffixes:
            candidates.setdefault(path.stem, []).append(path)

    result: dict[str, Path] = {}
    for stem, paths in candidates.items():
        if len(paths) > 1:
            names = ", ".join(path.name for path in paths)
            warn(f"Ambiguous stem {stem!r} in {directory}: {names}; skipping it")
            continue
        result[stem] = paths[0]
    return result


def load_split_stems(split: str) -> list[str]:
    """Load one fixed split without creating or reshuffling it."""
    if split not in {"train", "val"}:
        raise ValueError(f"A fixed split is required here, got {split!r}")
    split_path = SPLIT_DIR / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Fixed split file does not exist: {split_path}")

    stems: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        split_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        stem = Path(value).stem
        if stem in seen:
            raise ValueError(f"Duplicate stem {stem!r} in {split_path} line {line_number}")
        seen.add(stem)
        stems.append(stem)
    if not stems:
        raise ValueError(f"Fixed split is empty: {split_path}")
    return stems


def find_sample_files(
    stem: str,
    visible_map: dict[str, Path],
    infrared_map: dict[str, Path],
    depth_map: dict[str, Path],
    label_map: dict[str, Path],
) -> SampleFiles | None:
    missing = [
        name
        for name, mapping in (
            ("visible", visible_map),
            ("infrared", infrared_map),
            ("depth", depth_map),
            ("label", label_map),
        )
        if stem not in mapping
    ]
    if missing:
        warn(f"Stem {stem!r} is missing or ambiguous in: {', '.join(missing)}; skipping it")
        return None
    return SampleFiles(
        stem=stem,
        visible=visible_map[stem],
        infrared=infrared_map[stem],
        depth=depth_map[stem],
        label=label_map[stem],
    )


def read_image(path: Path, modality: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode {modality} image: {path}")
    return image


def load_visible(path: Path) -> np.ndarray:
    image = read_image(path, "Visible")
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        warn(f"Visible image has four channels; converting BGRA to RGBA: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if image.ndim == 2:
        warn(f"Visible image is single-channel: {path}")
        return image
    raise ValueError(f"Unsupported Visible shape {image.shape}: {path}")


def load_infrared(path: Path) -> np.ndarray:
    image = read_image(path, "Infrared")
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        warn(f"Infrared image has four channels; converting BGRA to RGBA: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    if image.ndim == 2:
        return image
    raise ValueError(f"Unsupported Infrared shape {image.shape}: {path}")


def load_depth(path: Path) -> np.ndarray:
    """Read Depth without changing its stored dtype or channel count."""
    return read_image(path, "Depth")


def normalize_for_display(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    numeric = values.astype(np.float32, copy=True)
    finite_mask = np.isfinite(numeric)
    display_mask = finite_mask if mask is None else finite_mask & mask
    sample = numeric[display_mask]
    normalized = np.zeros(numeric.shape, dtype=np.float32)
    if sample.size == 0:
        return normalized
    low, high = (float(value) for value in np.percentile(sample, (2.0, 98.0)))
    if high <= low:
        normalized[display_mask] = 1.0 if high > 0 else 0.0
        return normalized
    normalized[display_mask] = np.clip(
        (numeric[display_mask] - low) / (high - low), 0.0, 1.0
    )
    return normalized


def make_depth_visual(depth: np.ndarray, path: Path) -> tuple[np.ndarray, str]:
    suffix = path.suffix.casefold()
    if depth.ndim == 2 and depth.dtype == np.uint16:
        valid_mask = depth > 0
        normalized = normalize_for_display(depth, valid_mask)
        visual = colormaps["viridis"](normalized)[..., :3]
        visual[~valid_mask] = 0.0
        return visual, "Depth (uint16)"

    if suffix in {".jpg", ".jpeg"} and depth.ndim == 3 and depth.dtype == np.uint8:
        if depth.shape[2] == 3:
            visual = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        elif depth.shape[2] == 4:
            visual = cv2.cvtColor(depth, cv2.COLOR_BGRA2RGBA)
        else:
            warn(f"Unexpected channel count for JPG Depth {path}: {depth.shape}")
            visual = normalize_for_display(depth)
        return visual, "Depth (uint8 JPG, physical unit unknown)"

    warn(f"Unexpected Depth representation for {path}: dtype={depth.dtype}, shape={depth.shape}")
    if depth.ndim == 2:
        return normalize_for_display(depth), f"Depth (dtype={depth.dtype}, shape={depth.shape})"
    if depth.ndim == 3 and depth.shape[2] in {3, 4}:
        channels = depth[..., :3]
        if channels.dtype == np.uint8:
            visual = cv2.cvtColor(channels, cv2.COLOR_BGR2RGB)
        else:
            visual = normalize_for_display(channels)
        return visual, f"Depth (dtype={depth.dtype}, shape={depth.shape})"
    safe_plane = depth.reshape(depth.shape[0], depth.shape[1], -1)[..., 0]
    return normalize_for_display(safe_plane), f"Depth (dtype={depth.dtype}, shape={depth.shape})"


def load_yolo_labels(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            warn(f"Invalid YOLO field count in {path} line {line_number}; skipping row")
            continue
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError:
            warn(f"Invalid YOLO values in {path} line {line_number}; skipping row")
            continue
        if not 0 <= class_id < len(CLASS_NAMES):
            warn(f"Invalid class_id {class_id} in {path} line {line_number}; skipping row")
            continue
        if not np.isfinite((cx, cy, width, height)).all() or width <= 0 or height <= 0:
            warn(f"Invalid bbox in {path} line {line_number}; skipping row")
            continue
        boxes.append(YoloBox(class_id, cx, cy, width, height))
    return boxes


def draw_gt(visible_rgb: np.ndarray, boxes: list[YoloBox]) -> np.ndarray:
    if visible_rgb.ndim == 2:
        canvas = cv2.cvtColor(visible_rgb, cv2.COLOR_GRAY2RGB)
    else:
        canvas = visible_rgb[..., :3].copy()
    height, width = canvas.shape[:2]
    thickness = max(2, round(min(width, height) / 360))
    font_scale = max(0.45, min(width, height) / 900)

    for box in boxes:
        x1 = int(round((box.cx - box.width / 2.0) * width))
        y1 = int(round((box.cy - box.height / 2.0) * height))
        x2 = int(round((box.cx + box.width / 2.0) * width))
        y2 = int(round((box.cy + box.height / 2.0) * height))
        x1, x2 = sorted((min(max(x1, 0), width - 1), min(max(x2, 0), width - 1)))
        y1, y2 = sorted((min(max(y1, 0), height - 1), min(max(y2, 0), height - 1)))
        if x2 <= x1 or y2 <= y1:
            warn(f"BBox becomes empty after clipping: class_id={box.class_id}")
            continue
        color = (255, 64, 64)
        label = CLASS_NAMES[box.class_id]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_top = max(0, y1 - text_height - baseline - 4)
        text_right = min(width - 1, x1 + text_width + 6)
        cv2.rectangle(canvas, (x1, text_top), (text_right, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def show_image(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    if image.ndim == 2:
        axis.imshow(image, cmap="gray")
    else:
        axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")


def visualize_sample(files: SampleFiles, split: str, output_dir: Path) -> Path:
    visible = load_visible(files.visible)
    infrared = load_infrared(files.infrared)
    depth = load_depth(files.depth)
    depth_visual, depth_title = make_depth_visual(depth, files.depth)
    boxes = load_yolo_labels(files.label)
    visible_gt = draw_gt(visible, boxes)

    shapes = {
        "Visible": visible.shape[:2],
        "Infrared": infrared.shape[:2],
        "Depth": depth.shape[:2],
    }
    if len(set(shapes.values())) != 1:
        warn(f"Spatial size mismatch for {files.stem}: {shapes}")

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    show_image(axes[0, 0], visible, "Visible")
    show_image(axes[0, 1], infrared, "Infrared")
    show_image(axes[1, 0], depth_visual, depth_title)
    label_source = files.label.parent.name
    show_image(
        axes[1, 1],
        visible_gt,
        f"Visible + GT ({len(boxes)} boxes, {label_source})",
    )
    figure.suptitle(f"Sample: {files.stem} | split: {split}", fontsize=16)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.91, wspace=0.04, hspace=0.16)

    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_output_dir / f"{files.stem}_multimodal.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize aligned AIC2026 modalities and clean GT.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument(
        "--stem",
        action="append",
        help="Specific stem to render; repeat the option to render multiple stems.",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    visible_map = build_stem_map(args.data_root / "visible", IMAGE_SUFFIXES)
    infrared_map = build_stem_map(args.data_root / "infrared", IMAGE_SUFFIXES)
    depth_map = build_stem_map(args.data_root / "depth", IMAGE_SUFFIXES)
    label_map = build_stem_map(args.label_dir, {".txt"})
    common_stems = set(visible_map) & set(infrared_map) & set(depth_map) & set(label_map)

    if args.split == "all":
        allowed_stems = sorted(common_stems)
    else:
        allowed_stems = load_split_stems(args.split)
    allowed_set = set(allowed_stems)

    if args.stem:
        requested_stems = list(dict.fromkeys(args.stem))
        outside = [stem for stem in requested_stems if stem not in allowed_set]
        if outside:
            raise ValueError(
                f"Requested stem(s) do not belong to split {args.split!r}: {', '.join(outside)}"
            )
        selected_stems = requested_stems
    else:
        eligible_stems = [stem for stem in allowed_stems if stem in common_stems]
        missing_count = len(allowed_stems) - len(eligible_stems)
        if missing_count:
            warn(f"{missing_count} stem(s) in split {args.split!r} are incomplete or ambiguous")
        if not eligible_stems:
            raise RuntimeError(f"No complete samples are available for split {args.split!r}")
        sample_count = min(args.num_samples, len(eligible_stems))
        if sample_count < args.num_samples:
            warn(f"Requested {args.num_samples} samples but only {sample_count} are available")
        selected_stems = random.Random(args.seed).sample(eligible_stems, sample_count)

    outputs: list[Path] = []
    for stem in selected_stems:
        files = find_sample_files(stem, visible_map, infrared_map, depth_map, label_map)
        if files is None:
            continue
        try:
            output_path = visualize_sample(files, args.split, args.output_dir)
        except (OSError, ValueError, cv2.error) as error:
            warn(f"Failed to visualize stem {stem!r}: {error}")
            continue
        outputs.append(output_path)
        print(f"Generated: {output_path}")

    if not outputs:
        raise RuntimeError("No visualizations were generated")
    print(f"Completed: {len(outputs)} visualization(s) for split {args.split!r}")


if __name__ == "__main__":
    main()
