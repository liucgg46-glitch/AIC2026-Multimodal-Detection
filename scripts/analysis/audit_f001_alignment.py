"""Audit RGB/IR alignment on every sample in the fixed 400-image validation split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.analyze_modalities import (  # noqa: E402
    border_analysis,
    distribution,
    local_edge_registration,
)
from src.fusion.diagnostics import fixed_val_records, git_sha, require_new_output_dir, sha256, write_csv, write_json  # noqa: E402
from src.fusion.runtime import git_state, require_reviewed_checkout  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/analysis/F001_alignment_val400"


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode audit image: %s" % path)
    return image


def read_normalized_boxes(path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Invalid labels_clean row: %s" % path)
        values = (int(fields[0]),) + tuple(float(value) for value in fields[1:])
        boxes.append(values)
    return boxes


def draw_normalized_gt(image: np.ndarray, boxes: Sequence[Tuple[int, float, float, float, float]]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(height, width) / 360))
    for class_id, cx, cy, box_width, box_height in boxes:
        x1 = int(round((cx - box_width / 2) * width))
        y1 = int(round((cy - box_height / 2) * height))
        x2 = int(round((cx + box_width / 2) * width))
        y2 = int(round((cy + box_height / 2) * height))
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), thickness, cv2.LINE_AA)
            cv2.putText(canvas, str(class_id), (x1 + 2, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 220, 255), max(1, thickness), cv2.LINE_AA)
    return canvas


def edge_overlay(visible: np.ndarray, infrared: np.ndarray) -> np.ndarray:
    visible_edge = cv2.Canny(cv2.cvtColor(visible, cv2.COLOR_BGR2GRAY), 60, 160)
    infrared_edge = cv2.Canny(cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY), 60, 160)
    overlay = np.zeros_like(visible)
    overlay[..., 1] = visible_edge
    overlay[..., 0] = infrared_edge
    overlay[..., 2] = infrared_edge
    return overlay


def _panel(image: np.ndarray, title: str, target_width: int = 960) -> np.ndarray:
    scale = min(1.0, target_width / image.shape[1])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    header = np.zeros((42, resized.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack((header, resized))


def render_review_image(
    record: Tuple[str, Path, Path, Path], registration: Dict[str, Any], output_path: Path
) -> None:
    stem, rgb_path, ir_path, label_path = record
    label_before = label_path.read_bytes()
    visible, infrared = read_image(rgb_path), read_image(ir_path)
    if visible.shape != infrared.shape:
        raise ValueError("Cannot render shape-mismatched audit pair: %s" % stem)
    boxes = read_normalized_boxes(label_path)
    visible_gt = draw_normalized_gt(visible, boxes)
    ir_gt = draw_normalized_gt(infrared, boxes)
    overlay = edge_overlay(visible, infrared)
    height, width = _panel(visible_gt, "Visible + Visible GT").shape[:2]
    text = np.zeros((height - 42, width, 3), dtype=np.uint8)
    lines = [
        "stem: %s" % stem,
        "reliable: %s" % bool(registration.get("reliable", False)),
        "dx: %s" % registration.get("dx"),
        "dy: %s" % registration.get("dy"),
        "magnitude: %s" % registration.get("magnitude"),
        "confidence: %s" % registration.get("confidence"),
        "reason: %s" % registration.get("reason", "unknown"),
        "Visible normalized GT is overlaid unchanged on IR for audit only.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(text, line, (18, 38 + index * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (235, 235, 235), 1, cv2.LINE_AA)
    panels = [
        _panel(visible_gt, "Visible + Visible GT"),
        _panel(ir_gt, "IR + SAME Visible GT"),
        _panel(overlay, "RGB/IR edge overlay"),
        _panel(text, "Registration result"),
    ]
    target_height = max(panel.shape[0] for panel in panels)
    target_width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        padded.append(cv2.copyMakeBorder(panel, 0, target_height - panel.shape[0], 0,
                                         target_width - panel.shape[1], cv2.BORDER_CONSTANT, value=0))
    composite = np.vstack((np.hstack(padded[:2]), np.hstack(padded[2:])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), composite):
        raise OSError("Failed to write alignment review image: %s" % output_path)
    if label_path.read_bytes() != label_before:
        raise RuntimeError("Visualization modified labels_clean: %s" % label_path)


def _encoding_summary(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    output = {}
    for name in sorted({str(row[field]) for row in rows}):
        subset = [row for row in rows if str(row[field]) == name]
        good = [row for row in subset if row["reliable"]]
        output[name] = {
            "attempted": len(subset),
            "reliable": len(good),
            "reliable_ratio": len(good) / len(subset) if subset else None,
        }
        if field == "ir_encoding":
            output[name].update({
                "fov_ratio": distribution(row["ir_fov_ratio"] for row in subset),
                "border_present_ratio": float(np.mean([row["ir_border_present"] for row in subset])),
            })
    return output


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    reliable = [row for row in rows if row["reliable"]]
    unreliable = [row for row in rows if not row["reliable"]]
    reason_components = Counter()
    for row in unreliable:
        for reason in str(row["reason"]).split(","):
            reason_components[reason or "unknown"] += 1

    rgb_groups = _encoding_summary(rows, "rgb_encoding")
    ir_groups = _encoding_summary(rows, "ir_encoding")
    return {
        "attempted": len(rows),
        "reliable": len(reliable),
        "reliable_ratio": len(reliable) / len(rows) if rows else None,
        "reliable_ratio_by_encoding": {"rgb": rgb_groups, "ir": ir_groups},
        "encoding_grouping_note": "RGB and IR encodings are recorded and summarized by modality; equality is not assumed.",
        "unreliable_reasons": dict(sorted(reason_components.items())),
        "reliable_subset_shift_px": {
            "dx": distribution(row["dx"] for row in reliable),
            "dy": distribution(row["dy"] for row in reliable),
            "magnitude": distribution(row["magnitude"] for row in reliable),
        },
        "reliable_subset_scope_note": (
            "Shift distributions describe only registrations classified reliable; "
            "they are not an empirical error distribution for all val400 samples."
        ),
        "ir_fov_border": {
            "overall_fov_ratio": distribution(row["ir_fov_ratio"] for row in rows),
            "overall_border_present_ratio": float(np.mean([row["ir_border_present"] for row in rows])) if rows else None,
            "left_px": distribution(row["ir_border_left_px"] for row in rows),
            "right_px": distribution(row["ir_border_right_px"] for row in rows),
            "top_px": distribution(row["ir_border_top_px"] for row in rows),
            "bottom_px": distribution(row["ir_border_bottom_px"] for row in rows),
            "by_ir_encoding": ir_groups,
        },
    }


def select_review_rows(rows: Sequence[Dict[str, Any]], priority_stems: Iterable[str] = ()) -> List[Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}

    def add(row: Dict[str, Any], selection: str) -> None:
        stem = str(row["stem"])
        if stem not in selected:
            selected[stem] = {"row": row, "selections": []}
        selected[stem]["selections"].append(selection)

    def add_category(candidates: Sequence[Dict[str, Any]], selection: str, count: int = 3) -> None:
        ordered = list(candidates)
        if not ordered:
            return
        if len(ordered) <= count:
            chosen = ordered
        else:
            indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
            chosen = [ordered[index] for index in indices]
        for row in chosen:
            add(row, selection)

    by_stem = {str(row["stem"]): row for row in rows}
    for stem in list(priority_stems)[:5]:
        if stem in by_stem:
            add(by_stem[stem], "prediction_priority")
    reliable = sorted((row for row in rows if row["reliable"]), key=lambda row: (row["magnitude"], row["stem"]))
    if reliable:
        midpoint = max(1, len(reliable) // 2)
        add_category(reliable[:midpoint], "reliable_small_displacement")
        add_category(reliable[midpoint:] or reliable[-1:], "reliable_large_displacement")
    unreliable = sorted((row for row in rows if not row["reliable"]), key=lambda row: row["stem"])
    add_category(unreliable, "unreliable")
    for reason in ("search_boundary", "low_edge_correlation", "ambiguous_peak"):
        candidates = sorted((row for row in unreliable if reason in str(row["reason"]).split(",")),
                            key=lambda row: row["stem"])
        add_category(candidates, reason)
    if rows:
        add_category(sorted(rows, key=lambda row: (row["ir_fov_ratio"], row["stem"])), "worst_ir_fov")
    return [
        {**value["row"], "selection_reasons": ",".join(value["selections"])}
        for _, value in sorted(selected.items())
    ]


def run_alignment_audit(
    records: Sequence[Tuple[str, Path, Path, Path]],
    output_dir: Path,
    result_provider: Callable[[str, np.ndarray, np.ndarray], Dict[str, Any]] = None,
    priority_stems: Iterable[str] = (),
    render: bool = True,
    current_git_sha: str = None,
) -> Dict[str, Any]:
    if len(records) != 400 or len({record[0] for record in records}) != 400:
        raise RuntimeError("Alignment audit requires exactly 400 unique fixed-val stems")
    output_dir = require_new_output_dir(output_dir)
    label_hashes = {record[0]: sha256(record[3]) for record in records}
    rows = []
    records_by_stem = {record[0]: record for record in records}
    for stem, rgb_path, ir_path, _ in records:
        visible, infrared = read_image(rgb_path), read_image(ir_path)
        registration = (result_provider(stem, visible, infrared) if result_provider is not None
                        else local_edge_registration(visible, infrared, "visible_ir"))
        border = border_analysis(infrared, "infrared", ir_path.suffix.casefold())
        reliable = bool(registration.get("reliable", False))
        rows.append({
            "stem": stem,
            "rgb_encoding": "PNG" if rgb_path.suffix.casefold() == ".png" else "JPG",
            "ir_encoding": "PNG" if ir_path.suffix.casefold() == ".png" else "JPG",
            "height": visible.shape[0],
            "width": visible.shape[1],
            "reliable": reliable,
            "reason": registration.get("reason", "unknown"),
            "dx": registration.get("dx") if reliable else None,
            "dy": registration.get("dy") if reliable else None,
            "magnitude": registration.get("magnitude") if reliable else None,
            "confidence": registration.get("confidence"),
            "edge_correlation": registration.get("edge_correlation"),
            "peak_margin": registration.get("peak_margin"),
            "peak_ratio": registration.get("peak_ratio"),
            "ir_border_present": bool(border["present"]),
            "ir_border_left_px": border["left_px"],
            "ir_border_right_px": border["right_px"],
            "ir_border_top_px": border["top_px"],
            "ir_border_bottom_px": border["bottom_px"],
            "ir_fov_ratio": border["fov_ratio"],
            "ir_low_pixel_ratio": border["low_pixel_ratio"],
        })
    if [row["stem"] for row in rows] != [record[0] for record in records]:
        raise RuntimeError("Alignment audit changed fixed val order")

    fieldnames = list(rows[0])
    write_csv(output_dir / "registration.csv", rows, fieldnames)
    summary = summarize_rows(rows)
    summary.update({
        "fixed_val_count": len(records),
        "fixed_val_stems_match": True,
        "git_sha": current_git_sha or git_sha(),
        "method": "constrained local edge correlation",
        "labels_unchanged": all(sha256(record[3]) == label_hashes[record[0]] for record in records),
    })
    if not summary["labels_unchanged"]:
        raise RuntimeError("Alignment audit modified labels_clean")
    write_json(output_dir / "summary.json", summary)

    review_rows = select_review_rows(rows, priority_stems)
    manifest = []
    for row in review_rows:
        filename = row["stem"] + ".png"
        if render:
            render_review_image(records_by_stem[row["stem"]], row, output_dir / "review_images" / filename)
        manifest.append({
            "stem": row["stem"],
            "selection_reasons": row["selection_reasons"],
            "reliable": row["reliable"],
            "reason": row["reason"],
            "dx": row["dx"],
            "dy": row["dy"],
            "magnitude": row["magnitude"],
            "confidence": row["confidence"],
            "review_image": ("review_images/" + filename) if render else "",
        })
    write_csv(
        output_dir / "review_manifest.csv",
        manifest,
        ("stem", "selection_reasons", "reliable", "reason", "dx", "dy", "magnitude", "confidence", "review_image"),
    )
    if not all(sha256(record[3]) == label_hashes[record[0]] for record in records):
        raise RuntimeError("Review rendering modified labels_clean")
    return summary


def load_priority_stems(path: Path) -> List[str]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for key in ("stem", "image_stem"):
        values = [str(row[key]).strip() for row in rows if row.get(key)]
        if values:
            return values
    raise ValueError("Priority CSV must contain stem or image_stem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--priority-stems-csv", type=Path)
    parser.add_argument("--expected-sha", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_sha = require_reviewed_checkout(args.expected_sha, git_state(ROOT))
    summary = run_alignment_audit(
        fixed_val_records(),
        args.output_dir,
        priority_stems=load_priority_stems(args.priority_stems_csv),
        current_git_sha=current_sha,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
