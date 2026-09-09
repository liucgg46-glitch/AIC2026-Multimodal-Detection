"""Run RGB detection and export competition-format prediction TXT files."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NUM_CLASSES = 12


class InferenceError(ValueError):
    """Raised when inference inputs or predictions violate the submission contract."""


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def discover_images(source: Path) -> list[Path]:
    if not source.is_dir():
        raise InferenceError(f"RGB 测试图目录不存在: {source}")
    images = sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise InferenceError(f"RGB 测试图目录中没有受支持的图像: {source}")

    by_stem: dict[str, list[str]] = {}
    for image in images:
        by_stem.setdefault(image.stem, []).append(image.name)
    duplicate = next(((stem, names) for stem, names in by_stem.items() if len(names) > 1), None)
    if duplicate:
        raise InferenceError(f"同一个 stem 匹配到多个 RGB 测试图: {duplicate[0]} -> {duplicate[1]}")
    return images


def prediction_lines(result: object, max_det: int) -> list[str]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    if len(boxes) > max_det:
        raise InferenceError(f"单图预测框数量 {len(boxes)} 超过限制 {max_det}: {result.path}")

    xywhn = boxes.xywhn.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    lines: list[str] = []
    for class_value, coordinates, confidence in zip(classes, xywhn, confidences):
        class_id = int(class_value)
        values = [*coordinates, confidence]
        if not 0 <= class_id < NUM_CLASSES:
            raise InferenceError(f"预测 class_id 越界: {class_id} ({result.path})")
        if not all(math.isfinite(value) for value in values):
            raise InferenceError(f"预测包含 NaN 或 Inf: {result.path}")
        x, y, width, height = coordinates
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise InferenceError(f"预测归一化坐标越界: {coordinates} ({result.path})")
        if width <= 0.0 or height <= 0.0:
            raise InferenceError(f"预测框宽高必须大于 0: {coordinates} ({result.path})")
        if not 0.0 <= confidence <= 1.0:
            raise InferenceError(f"预测 confidence 越界: {confidence} ({result.path})")
        lines.append(
            f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f} {confidence:.8f}"
        )
    return lines


def run_inference(
    model_source: str,
    source: Path,
    output: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    force: bool,
) -> tuple[int, int, float]:
    if not 0.0 <= conf <= 1.0:
        raise InferenceError("--conf 必须位于 [0, 1]")
    if not 0.0 <= iou <= 1.0:
        raise InferenceError("--iou 必须位于 [0, 1]")
    if not 1 <= max_det <= 100:
        raise InferenceError("--max-det 必须位于 [1, 100]")
    if imgsz <= 0:
        raise InferenceError("--imgsz 必须大于 0")

    images = discover_images(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise InferenceError(f"输出目录已存在；如需重建请显式使用 --force: {output}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    started = time.perf_counter()
    try:
        model = YOLO(model_source)
        results = model.predict(
            source=[str(path) for path in images],
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            stream=True,
            save=False,
            save_txt=False,
            verbose=False,
        )
        seen: set[str] = set()
        prediction_count = 0
        for result in results:
            stem = Path(result.path).stem
            if stem in seen:
                raise InferenceError(f"推理结果出现重复 stem: {stem}")
            seen.add(stem)
            lines = prediction_lines(result, max_det)
            prediction_count += len(lines)
            content = "\n".join(lines) + ("\n" if lines else "")
            (staging / f"{stem}.txt").write_text(content, encoding="utf-8")

        expected = {path.stem for path in images}
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise InferenceError(f"推理结果与测试图不对应；缺失={missing[:5]}，多余={extra[:5]}")

        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return len(images), prediction_count, time.perf_counter() - started


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Weight/model path or Ultralytics model name")
    parser.add_argument("--source", default="data/raw/test/visible")
    parser.add_argument("--output", default="outputs/submissions/predictions")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    model_path = project_path(args.model)
    model_source = str(model_path) if model_path.is_file() else args.model
    try:
        image_count, prediction_count, elapsed = run_inference(
            model_source,
            project_path(args.source),
            project_path(args.output),
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            force=args.force,
        )
    except (InferenceError, OSError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    print(f"推理完成: {image_count} 张图像，{prediction_count} 个预测框")
    print(f"输出目录: {project_path(args.output)}")
    print(f"耗时: {elapsed:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
