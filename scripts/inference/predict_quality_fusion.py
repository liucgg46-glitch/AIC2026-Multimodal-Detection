"""Offline three-modality inference and AIC result ZIP for one quality-fusion checkpoint."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes, xyxy2xywh

from src.fusion.paired_dataset import read_raw3
from src.fusion.quality_dataset import decode_depth, encode_transport
from src.fusion.quality_model import QualityTriModalModel
from src.fusion.quality_initialization import file_sha256


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
COLOR_ORDER = [2, 1, 0, 5, 4, 3, 6, 7, 8, 9, 10]


def resolve_device(value):
    value = str(value).lower()
    if value == "cpu":
        return torch.device("cpu")
    if value.isdecimal():
        value = "cuda:" + value
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def index_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Image directory is missing: " + str(directory))
    indexed = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError("Duplicate image stem: " + path.stem)
        indexed[key] = path
    return indexed


def paired_test_records(rgb_dir, ir_dir, depth_dir, expected_count=1000):
    images = {"RGB": index_images(rgb_dir), "IR": index_images(ir_dir), "Depth": index_images(depth_dir)}
    stems = set(images["RGB"])
    if len(stems) != expected_count or any(set(index) != stems for index in images.values()):
        raise ValueError("Three modalities do not have the expected same %d stems" % expected_count)
    records = []
    for key in sorted(stems):
        rgb, ir, depth = (images[name][key] for name in ("RGB", "IR", "Depth"))
        if not (rgb.suffix.lower() == ir.suffix.lower() == depth.suffix.lower()):
            raise ValueError("Image extension differs: " + rgb.stem)
        records.append((rgb.stem, rgb, ir, depth))
    return records


def prepare_image(rgb_path, ir_path, depth_path, imgsz):
    rgb, ir = read_raw3(rgb_path), read_raw3(ir_path)
    if rgb.shape != ir.shape:
        raise ValueError("RGB/IR image shapes differ: " + rgb_path.stem)
    encoded, jpg = decode_depth(depth_path, rgb.shape[:2])
    raw = encode_transport(rgb, ir, encoded, jpg)
    gain = min(imgsz / rgb.shape[0], imgsz / rgb.shape[1])
    resized_h = round(rgb.shape[0] * gain)
    resized_w = round(rgb.shape[1] * gain)
    pad_x = round((imgsz - resized_w) / 2 - 0.1)
    pad_y = round((imgsz - resized_h) / 2 - 0.1)
    padded = LetterBox((imgsz, imgsz), auto=False, scaleup=True)(image=raw)
    if padded.shape != (imgsz, imgsz, 11):
        raise RuntimeError("LetterBox changed tri-modal transport shape")
    padded[..., 10] = 255 if jpg else 0
    padded = np.ascontiguousarray(padded[..., COLOR_ORDER].transpose(2, 0, 1))
    ratio_pad = ((gain, gain), (pad_x, pad_y))
    return padded, rgb.shape[:2], ratio_pad


def load_model(checkpoint_path, expected_sha256, device):
    checkpoint_path = Path(checkpoint_path)
    digest = file_sha256(checkpoint_path)
    if digest.lower() != expected_sha256.lower():
        raise ValueError("Fusion checkpoint SHA256 differs: " + digest)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, QualityTriModalModel):
        raise ValueError("Checkpoint does not contain the single QualityTriModalModel")
    return model.float().to(device).eval(), digest


def detection_lines(detections, original_shape):
    height, width = original_shape
    if len(detections) > 100:
        raise ValueError("More than 100 detections survived NMS")
    xywh = xyxy2xywh(detections[:, :4]).detach().cpu().numpy()
    xywh[:, [0, 2]] /= width
    xywh[:, [1, 3]] /= height
    scores = detections[:, 4].detach().cpu().numpy()
    classes = detections[:, 5].detach().cpu().numpy()
    lines = []
    for coords, confidence, class_value in zip(xywh, scores, classes):
        class_id = int(class_value)
        if (class_id < 0 or class_id >= 12 or not np.isfinite(coords).all()
                or not math.isfinite(float(confidence)) or confidence < 0 or confidence > 1):
            raise ValueError("Invalid class, coordinate or confidence")
        if coords[2] <= 0 or coords[3] <= 0:
            continue
        if not ((coords >= 0).all() and (coords <= 1).all()):
            raise ValueError("Normalized bbox is outside [0,1]")
        lines.append("%d %.8f %.8f %.8f %.8f %.8f" %
                     (class_id, *coords.tolist(), float(confidence)))
    return lines


def run_prediction(records, model, output_zip, device, imgsz=960, conf=0.001, iou=0.7):
    output_zip = Path(output_zip)
    if output_zip.exists():
        raise FileExistsError("Prediction ZIP already exists: " + str(output_zip))
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    handle, staged_name = tempfile.mkstemp(prefix="." + output_zip.stem + "-", suffix=".zip",
                                           dir=output_zip.parent)
    os.close(handle)
    staging = Path(staged_name)
    total_boxes = 0
    try:
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for stem, rgb_path, ir_path, depth_path in records:
                image, original_shape, ratio_pad = prepare_image(rgb_path, ir_path, depth_path, imgsz)
                tensor = torch.from_numpy(image).unsqueeze(0).to(device).float() / 255.0
                with torch.inference_mode():
                    prediction, _ = model(tensor)
                    detections = non_max_suppression(
                        prediction, conf_thres=conf, iou_thres=iou, max_det=100, nc=12,
                        max_time_img=2.0,
                    )[0]
                if len(detections):
                    detections[:, :4] = scale_boxes(
                        tensor.shape[2:], detections[:, :4], original_shape, ratio_pad=ratio_pad
                    )
                lines = detection_lines(detections, original_shape)
                total_boxes += len(lines)
                archive.writestr(stem + ".txt", "\n".join(lines) + ("\n" if lines else ""))
        with zipfile.ZipFile(staging) as archive:
            expected = {row[0] + ".txt" for row in records}
            if set(archive.namelist()) != expected or archive.testzip() is not None:
                raise RuntimeError("Prediction ZIP failed integrity/stem verification")
        if output_zip.exists():
            raise FileExistsError("Prediction ZIP appeared during generation: " + str(output_zip))
        staging.replace(output_zip)
    finally:
        if staging.exists():
            staging.unlink()
    return total_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    if (args.imgsz <= 0 or args.imgsz % 32 or not 0 <= args.conf <= 1
            or not 0 <= args.iou <= 1 or args.expected_count <= 0):
        parser.error("Invalid image size, confidence, IoU or expected image count")
    records = paired_test_records(args.rgb_dir, args.ir_dir, args.depth_dir, args.expected_count)
    device = resolve_device(args.device)
    model, digest = load_model(args.model, args.model_sha256, device)
    boxes = run_prediction(records, model, args.output_zip, device,
                           imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    print("QUALITY_FUSION_PREDICTION_PASS images=%d boxes=%d checkpoint_sha256=%s zip=%s" %
          (len(records), boxes, digest, args.output_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
