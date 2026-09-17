"""Add four overlapping high-resolution tile views to an existing RGB prediction export.

The full-image predictions stay authoritative. Tile detections near internal crop
boundaries are dropped, and only non-duplicate tile boxes are appended. This is
one predeclared inference diagnostic, not a tuned submission pipeline.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.evaluate_submission import ious, read_rows  # noqa: E402
from scripts.inference.predict_rgb import discover_images  # noqa: E402


TILE_WIDTH = 1152
TILE_HEIGHT = 648
TILE_CONF = 0.01
TILE_SCORE_SCALE = 0.8
MERGE_IOU = 0.55
INTERNAL_BORDER = 16
MAX_DET = 100


def tile_windows(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    if width <= TILE_WIDTH or height <= TILE_HEIGHT:
        return []
    return [(x, y, x + TILE_WIDTH, y + TILE_HEIGHT)
            for y in (0, height - TILE_HEIGHT)
            for x in (0, width - TILE_WIDTH)]


def convert_tile_box(
    box: Sequence[float], window: Tuple[int, int, int, int],
    width: int, height: int, cls: int, confidence: float,
) -> Optional[List[float]]:
    x0, y0, x1, y1 = window
    left, top, right, bottom = [float(value) for value in box]
    left, right = max(0.0, left), min(float(x1 - x0), right)
    top, bottom = max(0.0, top), min(float(y1 - y0), bottom)
    if right <= left or bottom <= top:
        return None
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    if ((x0 > 0 and center_x < INTERNAL_BORDER)
            or (x1 < width and center_x > x1 - x0 - INTERNAL_BORDER)
            or (y0 > 0 and center_y < INTERNAL_BORDER)
            or (y1 < height and center_y > y1 - y0 - INTERNAL_BORDER)):
        return None
    global_left = x0 + left
    global_top = y0 + top
    cx = (global_left + (right - left) / 2) / width
    cy = (global_top + (bottom - top) / 2) / height
    bw = (right - left) / width
    bh = (bottom - top) / height
    row = [float(cls), cx, cy, bw, bh, confidence * TILE_SCORE_SCALE]
    if not (0 <= cls < 12 and all(math.isfinite(value) for value in row)
            and all(0 <= value <= 1 for value in row[1:]) and bw > 0 and bh > 0):
        raise ValueError("Invalid projected tile detection: %s" % row)
    return row


def merge_with_baseline(baseline: List[List[float]], tiles: List[List[float]]) -> List[List[float]]:
    selected = list(baseline)
    for candidate in sorted(tiles, key=lambda row: -row[5]):
        same_class = [row[1:5] for row in selected if int(row[0]) == int(candidate[0])]
        if len(same_class) and max(ious(candidate[1:5], same_class)) > MERGE_IOU:
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda row: -row[5])[:MAX_DET]


def format_rows(rows: List[List[float]]) -> bytes:
    content = "".join(
        "%d %.8f %.8f %.8f %.8f %.8f\n" %
        (int(row[0]), row[1], row[2], row[3], row[4], row[5])
        for row in rows
    )
    return content.encode("utf-8")


def run(model_path: Path, images_dir: Path, baseline_dir: Path, output: Path,
        device: str, force: bool) -> dict:
    images = discover_images(images_dir)
    stems = {path.stem for path in images}
    if {path.stem for path in baseline_dir.glob("*.txt")} != stems:
        raise ValueError("Baseline prediction TXT set does not match images")
    if output.exists() and not force:
        raise ValueError("Output exists; pass --force to replace it: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s-" % output.name, dir=output.parent))
    tiled_images = 0
    tile_candidates = 0
    try:
        model = YOLO(str(model_path))
        for path in images:
            baseline_path = baseline_dir / (path.stem + ".txt")
            baseline = read_rows(baseline_path, prediction=True)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Cannot read visible image: %s" % path)
            height, width = image.shape[:2]
            windows = tile_windows(width, height)
            if not windows:
                shutil.copyfile(baseline_path, staging / baseline_path.name)
                continue
            tiled_images += 1
            candidates: List[List[float]] = []
            for window in windows:
                x0, y0, x1, y1 = window
                crop = image[y0:y1, x0:x1]
                result = model.predict(
                    source=crop, device=device, imgsz=1280, conf=TILE_CONF,
                    iou=0.7, max_det=MAX_DET, save=False, verbose=False,
                )[0]
                boxes = result.boxes
                if boxes is None:
                    continue
                for xyxy, cls, score in zip(
                    boxes.xyxy.detach().cpu().tolist(),
                    boxes.cls.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                ):
                    row = convert_tile_box(xyxy, window, width, height, int(cls), float(score))
                    if row is not None:
                        candidates.append(row)
            tile_candidates += len(candidates)
            merged = merge_with_baseline(baseline, candidates)
            (staging / baseline_path.name).write_bytes(format_rows(merged))
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"images": len(images), "tiled_images": tiled_images,
            "tile_candidates_before_merge": tile_candidates,
            "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(args.model, args.images, args.baseline, args.output, args.device, args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
