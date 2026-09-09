"""Validate competition-format detection TXT files."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NUM_CLASSES = 12


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def image_stems(image_dir: Path) -> tuple[set[str], list[str]]:
    errors: list[str] = []
    if not image_dir.is_dir():
        return set(), [f"测试图目录不存在: {image_dir}"]
    by_stem: dict[str, list[str]] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem.setdefault(path.stem, []).append(path.name)
    if not by_stem:
        errors.append(f"测试图目录中没有受支持的图像: {image_dir}")
    for stem, names in sorted(by_stem.items()):
        if len(names) > 1:
            errors.append(f"测试图 stem 重复: {stem} -> {names}")
    return set(by_stem), errors


def validate_prediction_file(path: Path, max_det: int) -> list[str]:
    errors: list[str] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) > max_det:
        errors.append(f"{path.name}: 预测框数量 {len(lines)} 超过 {max_det}")
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 6:
            errors.append(f"{path.name}:{line_number}: 应有 6 个字段，实际 {len(fields)}")
            continue
        try:
            class_id = int(fields[0])
        except ValueError:
            errors.append(f"{path.name}:{line_number}: class_id 不是整数")
            continue
        try:
            x, y, width, height, confidence = map(float, fields[1:])
        except ValueError:
            errors.append(f"{path.name}:{line_number}: 坐标或 confidence 不是数值")
            continue
        if not 0 <= class_id < NUM_CLASSES:
            errors.append(f"{path.name}:{line_number}: class_id 越界: {class_id}")
        values = (x, y, width, height, confidence)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path.name}:{line_number}: 包含 NaN 或 Inf")
            continue
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            errors.append(f"{path.name}:{line_number}: 归一化坐标越界")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path.name}:{line_number}: width/height 必须大于 0")
        if not 0.0 <= confidence <= 1.0:
            errors.append(f"{path.name}:{line_number}: confidence 越界")
    return errors


def validate_submission(image_dir: Path, prediction_dir: Path, max_det: int = 100) -> list[str]:
    expected, errors = image_stems(image_dir)
    if not prediction_dir.is_dir():
        return errors + [f"预测目录不存在: {prediction_dir}"]
    prediction_files = sorted(prediction_dir.glob("*.txt"))
    actual = {path.stem for path in prediction_files}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"缺少预测 TXT: {missing[:10]}（共 {len(missing)} 个）")
    if extra:
        errors.append(f"存在多余预测 TXT: {extra[:10]}（共 {len(extra)} 个）")
    for path in prediction_files:
        errors.extend(validate_prediction_file(path, max_det))
    return errors


def test_accepts_valid_and_empty_predictions(tmp_path: Path) -> None:
    images = tmp_path / "images"
    predictions = tmp_path / "predictions"
    images.mkdir()
    predictions.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    (images / "b.png").write_bytes(b"image")
    (predictions / "a.txt").write_text("6 0.5 0.5 0.2 0.3 0.9\n", encoding="utf-8")
    (predictions / "b.txt").write_text("", encoding="utf-8")

    assert validate_submission(images, predictions) == []


@pytest.mark.parametrize(
    "content, expected",
    [
        ("12 0.5 0.5 0.2 0.3 0.9\n", "class_id 越界"),
        ("0 1.1 0.5 0.2 0.3 0.9\n", "归一化坐标越界"),
        ("0 0.5 0.5 0.0 0.3 0.9\n", "width/height 必须大于 0"),
        ("0 0.5 0.5 0.2 0.3 1.1\n", "confidence 越界"),
        ("0 0.5 0.5 0.2 0.3\n", "应有 6 个字段"),
    ],
)
def test_rejects_invalid_rows(tmp_path: Path, content: str, expected: str) -> None:
    prediction = tmp_path / "sample.txt"
    prediction.write_text(content, encoding="utf-8")

    assert any(expected in error for error in validate_prediction_file(prediction, 100))


def test_detects_missing_and_extra_txt(tmp_path: Path) -> None:
    images = tmp_path / "images"
    predictions = tmp_path / "predictions"
    images.mkdir()
    predictions.mkdir()
    (images / "expected.jpg").write_bytes(b"image")
    (predictions / "extra.txt").write_text("", encoding="utf-8")

    errors = validate_submission(images, predictions)

    assert any("缺少预测 TXT" in error for error in errors)
    assert any("多余预测 TXT" in error for error in errors)


def test_rejects_more_than_100_predictions(tmp_path: Path) -> None:
    prediction = tmp_path / "sample.txt"
    prediction.write_text(
        "0 0.5 0.5 0.2 0.3 0.9\n" * 101,
        encoding="utf-8",
    )

    assert any("预测框数量 101 超过 100" in error for error in validate_prediction_file(prediction, 100))


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    if not 1 <= args.max_det <= 100:
        print("错误: --max-det 必须位于 [1, 100]", file=sys.stderr)
        return 2
    errors = validate_submission(
        project_path(args.images), project_path(args.predictions), args.max_det
    )
    if errors:
        print(f"提交格式检查失败，共 {len(errors)} 个问题:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    image_count = len(image_stems(project_path(args.images))[0])
    print(f"提交格式检查通过: {image_count} 张测试图，TXT 一一对应")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
