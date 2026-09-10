"""Create a minimally corrected AIC2026 label copy without editing data/raw.

Only explicitly reviewed rows are changed: five invalid YOLO boxes are clipped
to the image boundary, one exact duplicate is omitted, and five visually
confirmed class mismatches are reclassified. All other rows, including valid
boxes whose corners extend outside the image, are copied verbatim. The script
is deterministic and verifies that source-file hashes do not change while it
runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "data" / "raw" / "train" / "labels"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "train" / "labels_clean"
ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"

# (stem, one-based line number) values come from the reviewed B2 issue report.
CLIP_LINES = {
    ("000003_080_00000307", 8),
    ("000050", 27),
    ("000050", 37),
    ("003107", 2),
    ("003817", 3),
}
DROP_DUPLICATE_LINES = {("hehe_10_00000044", 14)}
RECLASS_LINES = {
    ("000003_026_00000001", 1): 6,  # truck: person -> car
    ("000005_026_00000001", 1): 0,  # four people: car -> person
    ("000005_026_00000001", 2): 0,
    ("000005_026_00000001", 3): 0,
    ("000005_026_00000001", 4): 0,
    ("002517", 1): 8,  # seven street lights: sign -> light
    ("002517", 2): 8,
    ("002517", 3): 8,
    ("002517", 4): 8,
    ("002517", 5): 8,
    ("002517", 6): 8,
    ("002517", 7): 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_float(value: float) -> str:
    return f"{value:.10g}"


def clip_yolo_row(text: str) -> tuple[str, dict[str, float]]:
    fields = text.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields in reviewed row, got: {text!r}")
    class_id = int(fields[0])
    x, y, width, height = map(float, fields[1:])
    x1, y1 = x - width / 2.0, y - height / 2.0
    x2, y2 = x + width / 2.0, y + height / 2.0
    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(1.0, x2), min(1.0, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"Reviewed box has no area after clipping: {text!r}")
    new_x = (cx1 + cx2) / 2.0
    new_y = (cy1 + cy2) / 2.0
    new_width = cx2 - cx1
    new_height = cy2 - cy1
    result = " ".join(
        [str(class_id), *(format_float(v) for v in (new_x, new_y, new_width, new_height))]
    )
    return result, {
        "x": new_x,
        "y": new_y,
        "width": new_width,
        "height": new_height,
    }


def reclassify_yolo_row(text: str, new_class_id: int) -> str:
    fields = text.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields in reviewed row, got: {text!r}")
    old_class_id = int(fields[0])
    if old_class_id == new_class_id:
        raise ValueError(f"Reviewed row already has class {new_class_id}: {text!r}")
    return " ".join([str(new_class_id), *fields[1:]])


def main() -> None:
    source_paths = sorted(SOURCE_DIR.glob("*.txt"), key=lambda p: p.name.casefold())
    if len(source_paths) != 2000:
        raise RuntimeError(f"Expected 2000 source labels, found {len(source_paths)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in OUTPUT_DIR.glob("*.txt")}
    expected = {p.name for p in source_paths}
    unexpected = sorted(existing - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected existing clean-label files: {unexpected[:10]}")

    before_hashes = {path.name: sha256(path) for path in source_paths}
    changes: list[dict[str, object]] = []
    applied_clip_lines: set[tuple[str, int]] = set()
    applied_drop_lines: set[tuple[str, int]] = set()
    applied_reclass_lines: set[tuple[str, int]] = set()

    for source_path in source_paths:
        source_text = source_path.read_text(encoding="utf-8-sig")
        source_lines = source_text.splitlines()
        output_lines: list[str] = []
        for line_number, line in enumerate(source_lines, start=1):
            key = (source_path.stem, line_number)
            if key in DROP_DUPLICATE_LINES:
                applied_drop_lines.add(key)
                changes.append(
                    {
                        "stem": source_path.stem,
                        "line": line_number,
                        "action": "drop_exact_duplicate",
                        "before": line,
                        "after": "",
                    }
                )
                continue
            if key in CLIP_LINES:
                clipped, _ = clip_yolo_row(line)
                applied_clip_lines.add(key)
                changes.append(
                    {
                        "stem": source_path.stem,
                        "line": line_number,
                        "action": "clip_reviewed_invalid_box",
                        "before": line,
                        "after": clipped,
                    }
                )
                output_lines.append(clipped)
            elif key in RECLASS_LINES:
                reclassified = reclassify_yolo_row(line, RECLASS_LINES[key])
                applied_reclass_lines.add(key)
                changes.append(
                    {
                        "stem": source_path.stem,
                        "line": line_number,
                        "action": "reclassify_visually_confirmed_box",
                        "before": line,
                        "after": reclassified,
                    }
                )
                output_lines.append(reclassified)
            else:
                output_lines.append(line)

        suffix = "\n" if source_text.endswith(("\n", "\r")) or output_lines else ""
        (OUTPUT_DIR / source_path.name).write_text(
            "\n".join(output_lines) + suffix, encoding="utf-8", newline="\n"
        )

    if applied_clip_lines != CLIP_LINES:
        raise RuntimeError(f"Missing clip rows: {sorted(CLIP_LINES - applied_clip_lines)}")
    if applied_drop_lines != DROP_DUPLICATE_LINES:
        raise RuntimeError(f"Missing duplicate rows: {sorted(DROP_DUPLICATE_LINES - applied_drop_lines)}")
    if applied_reclass_lines != set(RECLASS_LINES):
        raise RuntimeError(f"Missing reclass rows: {sorted(set(RECLASS_LINES) - applied_reclass_lines)}")

    after_hashes = {path.name: sha256(path) for path in source_paths}
    if before_hashes != after_hashes:
        raise RuntimeError("A source label changed while clean labels were generated")

    output_paths = sorted(OUTPUT_DIR.glob("*.txt"), key=lambda p: p.name.casefold())
    if len(output_paths) != len(source_paths):
        raise RuntimeError(f"Expected 2000 clean labels, found {len(output_paths)}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv = ANALYSIS_DIR / "clean_label_changes.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stem", "line", "action", "before", "after"))
        writer.writeheader()
        writer.writerows(changes)

    summary = {
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "source_label_files": len(source_paths),
        "output_label_files": len(output_paths),
        "clipped_invalid_boxes": len(applied_clip_lines),
        "removed_exact_duplicates": len(applied_drop_lines),
        "reclassified_boxes": len(applied_reclass_lines),
        "source_files_unchanged": True,
        "empty_label_preserved": (OUTPUT_DIR / "shuming_102_00000228.txt").read_text(
            encoding="utf-8-sig"
        ).strip() == "",
        "unreviewed_rows_changed": 0,
    }
    (ANALYSIS_DIR / "clean_label_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
