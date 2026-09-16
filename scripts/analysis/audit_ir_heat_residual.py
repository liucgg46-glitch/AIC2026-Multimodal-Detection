"""Decide whether aligned IR heat residual is strong enough for a formal fusion experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.analyze_modalities import local_edge_registration  # noqa: E402


CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
    "light", "garbage can", "uav", "tricycle",
]
THERMAL_CLASS_IDS = {0, 2}
EXPECTED_LABEL_SHA256 = "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"
PASS_THRESHOLDS = {
    "minimum_alignment_reliable_ratio": 0.20,
    "minimum_evaluable_thermal_boxes": 50,
    "minimum_heat_positive_snr_ratio": 0.60,
    "minimum_heat_median_snr": 0.75,
    "minimum_median_snr_gain_over_raw_ir": 0.10,
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class IrAuditError(RuntimeError):
    """Raised when the read-only IR audit contract is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_labels(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: (item.name.casefold(), item.name)):
        digest.update(("%s\0%d\0%s\n" % (path.name, path.stat().st_size, sha256(path))).encode("utf-8"))
    return digest.hexdigest()


def index_images(path: Path) -> Dict[str, Path]:
    result: Dict[str, List[Path]] = {}
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            result.setdefault(item.stem, []).append(item)
    if any(len(paths) != 1 for paths in result.values()):
        raise IrAuditError("图像目录存在重复 stem: %s" % path)
    return {stem: paths[0] for stem, paths in result.items()}


def fixed_val_records(
    visible_dir: Path, infrared_dir: Path, label_dir: Path, split_path: Path,
) -> List[Tuple[str, Path, Path, Path]]:
    stems = [line.strip() for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(stems) != 400 or len(set(stems)) != 400:
        raise IrAuditError("IR audit 要求固定且唯一的 val=400 split")
    visible, infrared = index_images(visible_dir), index_images(infrared_dir)
    records = []
    for stem in stems:
        label = label_dir / (stem + ".txt")
        if stem not in visible or stem not in infrared or not label.is_file():
            raise IrAuditError("固定 val 样本不完整: %s" % stem)
        records.append((stem, visible[stem], infrared[stem], label))
    all_labels = list(label_dir.glob("*.txt"))
    if len(all_labels) != 2000 or aggregate_labels(all_labels) != EXPECTED_LABEL_SHA256:
        raise IrAuditError("canonical labels_clean 身份不一致")
    return records


def border_valid_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = (gray <= 8).astype(np.uint8)
    count, components = cv2.connectedComponents(low, connectivity=8)
    if count == 1:
        return np.ones(gray.shape, dtype=bool)
    edge_labels = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    edge_labels = edge_labels[edge_labels != 0]
    return ~np.isin(components, edge_labels)


def robust_gray(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    values = gray[valid]
    if values.size < 16:
        raise IrAuditError("有效 IR 视野过小")
    low, high = np.percentile(values, (2, 98))
    result = np.clip((gray - low) / max(float(high - low), 1.0), 0.0, 1.0)
    result[~valid] = 0.0
    return result


def align_ir(image: np.ndarray, valid: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    # local_edge_registration reports modality displacement relative to RGB.
    # Apply its inverse to move IR into the RGB coordinate frame.
    matrix = np.asarray([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    aligned_valid = cv2.warpAffine(valid.astype(np.uint8), matrix, (width, height),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0).astype(bool)
    return aligned, aligned_valid


def parse_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise IrAuditError("标签字段数错误: %s" % path)
        row = (int(fields[0]),) + tuple(float(value) for value in fields[1:])
        if row[0] not in range(12) or any(value < 0 or value > 1 for value in row[1:]) or min(row[3:]) <= 0:
            raise IrAuditError("标签值越界: %s" % path)
        rows.append(row)
    return rows


def box_contrast(
    image: np.ndarray, valid: np.ndarray, box: Tuple[int, float, float, float, float],
) -> Optional[Dict[str, float]]:
    _, cx, cy, box_width, box_height = box
    height, width = image.shape
    x1 = max(0, int(np.floor((cx - box_width / 2) * width)))
    y1 = max(0, int(np.floor((cy - box_height / 2) * height)))
    x2 = min(width, int(np.ceil((cx + box_width / 2) * width)))
    y2 = min(height, int(np.ceil((cy + box_height / 2) * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    margin_x, margin_y = max(4, (x2 - x1) // 2), max(4, (y2 - y1) // 2)
    rx1, ry1, rx2, ry2 = max(0, x1 - margin_x), max(0, y1 - margin_y), min(width, x2 + margin_x), min(height, y2 + margin_y)
    inside_mask = np.zeros((height, width), dtype=bool)
    ring_mask = np.zeros((height, width), dtype=bool)
    inside_mask[y1:y2, x1:x2] = True
    ring_mask[ry1:ry2, rx1:rx2] = True
    ring_mask &= ~inside_mask
    inside, ring = image[inside_mask & valid], image[ring_mask & valid]
    if inside.size < 9 or ring.size < 16:
        return None
    inside_median, ring_median = float(np.median(inside)), float(np.median(ring))
    robust_sigma = max(float(np.median(np.abs(ring - ring_median))) * 1.4826, 0.03)
    return {
        "inside_median": inside_median, "ring_median": ring_median,
        "delta": inside_median - ring_median,
        "snr": (inside_median - ring_median) / robust_sigma,
    }


def summarize(
    rows: Sequence[Dict[str, Any]], image_count: int, reliable_image_count: Optional[int] = None,
) -> Dict[str, Any]:
    reliable = [row for row in rows if row["alignment_reliable"]]
    thermal = [row for row in reliable if row["class_id"] in THERMAL_CLASS_IDS and row["heat_snr"] is not None]
    heat_snrs = [float(row["heat_snr"]) for row in thermal]
    raw_snrs = [float(row["raw_ir_snr"]) for row in thermal]
    reliable_stems = {str(row["stem"]) for row in reliable}
    reliable_count = len(reliable_stems) if reliable_image_count is None else reliable_image_count
    alignment_ratio = reliable_count / image_count if image_count else 0.0
    metrics = {
        "alignment_reliable_images": reliable_count,
        "alignment_reliable_ratio": alignment_ratio,
        "evaluable_thermal_boxes": len(thermal),
        "heat_positive_snr_ratio": (sum(value > 0 for value in heat_snrs) / len(heat_snrs)) if heat_snrs else 0.0,
        "heat_median_snr": median(heat_snrs) if heat_snrs else None,
        "raw_ir_median_snr": median(raw_snrs) if raw_snrs else None,
        "median_snr_gain_over_raw_ir": median([heat - raw for heat, raw in zip(heat_snrs, raw_snrs)]) if heat_snrs else None,
    }
    checks = {
        "alignment_coverage": alignment_ratio >= PASS_THRESHOLDS["minimum_alignment_reliable_ratio"],
        "thermal_sample_size": len(thermal) >= PASS_THRESHOLDS["minimum_evaluable_thermal_boxes"],
        "heat_positive_ratio": metrics["heat_positive_snr_ratio"] >= PASS_THRESHOLDS["minimum_heat_positive_snr_ratio"],
        "heat_absolute_signal": metrics["heat_median_snr"] is not None and metrics["heat_median_snr"] >= PASS_THRESHOLDS["minimum_heat_median_snr"],
        "heat_gain_over_raw_ir": metrics["median_snr_gain_over_raw_ir"] is not None and metrics["median_snr_gain_over_raw_ir"] >= PASS_THRESHOLDS["minimum_median_snr_gain_over_raw_ir"],
    }
    return {
        "metrics": metrics, "thresholds": PASS_THRESHOLDS, "checks": checks,
        "decision": "PASS" if all(checks.values()) else "REJECT",
        "decision_rule": "All five predeclared checks must pass; otherwise do not start heat-residual fusion training.",
    }


def run_audit(records: Sequence[Tuple[str, Path, Path, Path]], output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists():
        raise IrAuditError("拒绝覆盖既有 audit 输出: %s" % output_dir)
    output_dir.mkdir(parents=True)
    label_before = {stem: sha256(label) for stem, _, _, label in records}
    rows = []
    reliable_image_count = 0
    for stem, rgb_path, ir_path, label_path in records:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
        if rgb is None or ir is None or rgb.shape != ir.shape:
            raise IrAuditError("RGB/IR 无法读取或形状不一致: %s" % stem)
        registration = local_edge_registration(rgb, ir, "visible_ir")
        reliable = bool(registration.get("reliable", False))
        reliable_image_count += int(reliable)
        valid = border_valid_mask(ir)
        aligned_ir, aligned_valid = align_ir(ir, valid, float(registration.get("dx", 0.0)),
                                             float(registration.get("dy", 0.0))) if reliable else (ir, valid)
        ir_gray = robust_gray(aligned_ir, aligned_valid)
        rgb_gray = robust_gray(rgb, aligned_valid)
        heat = np.maximum(ir_gray - rgb_gray, 0.0)
        for box_index, box in enumerate(parse_labels(label_path)):
            raw_metric = box_contrast(ir_gray, aligned_valid, box) if reliable else None
            heat_metric = box_contrast(heat, aligned_valid, box) if reliable else None
            rows.append({
                "stem": stem, "box_index": box_index, "class_id": box[0],
                "class_name": CLASS_NAMES[box[0]], "alignment_reliable": reliable,
                "dx": registration.get("dx") if reliable else None,
                "dy": registration.get("dy") if reliable else None,
                "alignment_reason": registration.get("reason"),
                "raw_ir_snr": raw_metric["snr"] if raw_metric else None,
                "heat_snr": heat_metric["snr"] if heat_metric else None,
                "raw_ir_delta": raw_metric["delta"] if raw_metric else None,
                "heat_delta": heat_metric["delta"] if heat_metric else None,
            })
    if any(sha256(label) != label_before[stem] for stem, _, _, label in records):
        raise IrAuditError("只读 audit 修改了 labels_clean")
    with (output_dir / "targets.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows, len(records), reliable_image_count)
    report.update({"fixed_val_count": len(records), "labels_unchanged": True, "thermal_classes": ["person", "animal"]})
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def require_checkout(expected_sha: str) -> str:
    if len(expected_sha) != 40 or any(char not in "0123456789abcdefABCDEF" for char in expected_sha):
        raise IrAuditError("--expected-sha 必须是完整 40 位 Git SHA")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT), text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(PROJECT_ROOT), text=True).strip()
    if actual.lower() != expected_sha.lower() or status or branch:
        raise IrAuditError("audit 要求指定 SHA 的 detached、clean checkout")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--visible-dir", type=Path, default=PROJECT_ROOT / "data/raw/train/visible")
    parser.add_argument("--infrared-dir", type=Path, default=PROJECT_ROOT / "data/raw/train/infrared")
    parser.add_argument("--label-dir", type=Path, default=PROJECT_ROOT / "data/processed/train/labels_clean")
    parser.add_argument("--split", type=Path, default=PROJECT_ROOT / "data/splits/val.txt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/analysis/IR_HEAT_AUDIT_R5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        commit = require_checkout(args.expected_sha)
        records = fixed_val_records(args.visible_dir, args.infrared_dir, args.label_dir, args.split)
        report = run_audit(records, args.output_dir)
        report["git_sha"] = commit
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (IrAuditError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
