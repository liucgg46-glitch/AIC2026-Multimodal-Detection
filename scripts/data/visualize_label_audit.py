"""Render every reviewed label issue across Visible, Infrared and Depth.

The script is read-only with respect to ``data/raw`` and ``data/processed``.
It creates local, ignored JPEG panels under ``outputs/visualization`` plus a
small CSV/JSON audit index under ``outputs/analysis``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_DIR = REPO_ROOT / "data" / "raw" / "train"
DEFAULT_ISSUES = REPO_ROOT / "outputs" / "analysis" / "label_issues.csv"
DEFAULT_CHANGES = REPO_ROOT / "outputs" / "analysis" / "clean_label_changes.csv"
DEFAULT_VIS_DIR = REPO_ROOT / "outputs" / "visualization" / "label_audit"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EMPTY_STEM = "shuming_102_00000228"
MODALITIES = ("visible", "infrared", "depth")
PANEL_SIZE = (960, 310)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize all reviewed AIC2026 label issues.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--issues", type=Path, default=DEFAULT_ISSUES)
    parser.add_argument("--changes", type=Path, default=DEFAULT_CHANGES)
    parser.add_argument("--visualization-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser.parse_args()


def index_images(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
            result[key] = path
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_box(text: str) -> tuple[int, float, float, float, float] | None:
    fields = text.split()
    if len(fields) != 5:
        return None
    try:
        return int(fields[0]), *(float(value) for value in fields[1:])
    except ValueError:
        return None


def box_corners(box: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    _, x, y, bw, bh = box
    return (
        round((x - bw / 2.0) * width),
        round((y - bh / 2.0) * height),
        round((x + bw / 2.0) * width),
        round((y + bh / 2.0) * height),
    )


def display_image(image: np.ndarray, modality: str) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3].copy()
    gray = image if image.ndim == 2 else image[:, :, 0]
    values = gray.astype(np.float32)
    finite = values[np.isfinite(values)]
    if modality == "depth":
        nonzero = finite[finite > 0]
        sample = nonzero if nonzero.size else finite
        lo, hi = (np.percentile(sample, (1, 99)) if sample.size else (0.0, 1.0))
    else:
        lo, hi = (np.percentile(finite, (1, 99)) if finite.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((values - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    if modality == "depth":
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def crop_bounds(corners: tuple[int, int, int, int] | None, width: int, height: int) -> tuple[int, int, int, int]:
    if corners is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = corners
    cx = min(max((x1 + x2) / 2.0, 0.0), float(width))
    cy = min(max((y1 + y2) / 2.0, 0.0), float(height))
    crop_w = min(width, max(abs(x2 - x1) * 4.0, width * 0.28))
    crop_h = min(height, max(abs(y2 - y1) * 4.0, height * 0.32))
    left = int(round(min(max(cx - crop_w / 2.0, 0.0), width - crop_w)))
    top = int(round(min(max(cy - crop_h / 2.0, 0.0), height - crop_h)))
    return left, top, int(round(left + crop_w)), int(round(top + crop_h))


def draw_box(image: np.ndarray, corners: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = corners
    x1c, y1c = min(max(x1, 0), width - 1), min(max(y1, 0), height - 1)
    x2c, y2c = min(max(x2, 0), width - 1), min(max(y2, 0), height - 1)
    cv2.rectangle(image, (x1c, y1c), (x2c, y2c), color, 5, cv2.LINE_AA)
    cv2.putText(image, label, (max(2, x1c), max(25, y1c + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def fit_tile(image: np.ndarray, width: int = 320, height: int = 240) -> np.ndarray:
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    x0, y0 = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)


def render_panel(
    record: dict[str, str],
    image_indexes: dict[str, dict[str, Path]],
    change_map: dict[tuple[str, int], dict[str, str]],
) -> np.ndarray:
    stem = record["stem"]
    line_number = int(record["line"] or 0)
    issue_type = record["issue_type"]
    raw_box = parse_box(record.get("content", ""))
    change = change_map.get((stem, line_number))
    clean_box = parse_box(change.get("after", "")) if change else None

    raw_visible = cv2.imread(str(image_indexes["visible"][stem.casefold()]), cv2.IMREAD_UNCHANGED)
    if raw_visible is None:
        raise ValueError(f"Cannot read Visible image for {stem}")
    height, width = raw_visible.shape[:2]
    raw_corners = box_corners(raw_box, width, height) if raw_box else None
    clean_corners = box_corners(clean_box, width, height) if clean_box else None
    focus_corners = clean_corners or raw_corners
    crop = crop_bounds(focus_corners, width, height)

    tiles: list[np.ndarray] = []
    for modality in MODALITIES:
        path = image_indexes[modality].get(stem.casefold())
        if path is None:
            raise ValueError(f"Missing {modality} image for {stem}")
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape[:2] != (height, width):
            raise ValueError(f"Unreadable or size-mismatched {modality} image: {path}")
        shown = display_image(raw, modality)
        if raw_corners:
            draw_box(shown, raw_corners, (0, 0, 255), "RAW")
        if clean_corners:
            draw_box(shown, clean_corners, (255, 255, 0), "CLEAN")
        x1, y1, x2, y2 = crop
        tile = fit_tile(shown[y1:y2, x1:x2])
        cv2.putText(tile, modality.upper(), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)

    panel = np.full((PANEL_SIZE[1], PANEL_SIZE[0], 3), 250, dtype=np.uint8)
    panel[70:310, :] = np.hstack(tiles)
    title = f"{stem}  line={line_number or '-'}  {issue_type}"
    detail = record.get("details", "")[:125]
    cv2.putText(panel, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, detail, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 45, 45), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()
    image_indexes = {modality: index_images(args.train_dir / modality) for modality in MODALITIES}
    if any(len(index) != 2000 for index in image_indexes.values()):
        raise RuntimeError({name: len(index) for name, index in image_indexes.items()})

    changes = read_csv(args.changes)
    change_map = {(row["stem"], int(row["line"])): row for row in changes}
    records = read_csv(args.issues)
    records.append(
        {
            "stem": EMPTY_STEM,
            "line": "",
            "severity": "review",
            "issue_type": "empty_label",
            "details": "empty label retained after visual review",
            "content": "",
        }
    )
    records.sort(key=lambda row: (row["stem"].casefold(), int(row["line"] or 0), row["issue_type"]))

    args.visualization_dir.mkdir(parents=True, exist_ok=True)
    panels: list[np.ndarray] = []
    manifest: list[dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        panel = render_panel(record, image_indexes, change_map)
        filename = f"audit_{index:03d}_{safe_name(record['stem'])}_L{record['line'] or '0'}_{safe_name(record['issue_type'])}.jpg"
        path = args.visualization_dir / filename
        if not cv2.imwrite(str(path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Failed to write {path}")
        panels.append(panel)
        manifest.append(
            {
                "audit_index": index,
                "stem": record["stem"],
                "line": record["line"],
                "severity": record["severity"],
                "issue_type": record["issue_type"],
                "details": record["details"],
                "visualization": str(path.relative_to(REPO_ROOT)),
                "review_status": "pending_human_review",
            }
        )

    page_paths: list[str] = []
    blank = np.full_like(panels[0], 245)
    for start in range(0, len(panels), 6):
        page_panels = panels[start : start + 6]
        page_panels.extend([blank] * (6 - len(page_panels)))
        page = np.vstack([np.hstack(page_panels[i : i + 2]) for i in range(0, 6, 2)])
        page_path = args.visualization_dir / f"contact_sheet_{start // 6 + 1:02d}.jpg"
        if not cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"Failed to write {page_path}")
        page_paths.append(str(page_path.relative_to(REPO_ROOT)))

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.analysis_dir / "label_visual_audit_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "audit_item_count": len(records),
        "individual_panel_count": len(panels),
        "contact_sheet_count": len(page_paths),
        "issue_type_counts": dict(Counter(row["issue_type"] for row in records)),
        "modalities": list(MODALITIES),
        "raw_box_color": "red",
        "clean_box_color": "cyan",
        "review_status": "pending_human_review",
        "visualization_dir": str(args.visualization_dir),
        "contact_sheets": page_paths,
        "raw_data_modified": False,
    }
    (args.analysis_dir / "label_visual_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
