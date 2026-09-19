from types import SimpleNamespace

import cv2
import numpy as np
import torch

from scripts.data.prepare_depth_candidate_yolo import render_candidate_gray
from src.fusion.pretrained_dataset import PretrainedTriModalDataset, decode_pretrained_depth


def hyp():
    return SimpleNamespace(
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        degrees=0.0, shear=0.0, perspective=0.0, multi_scale=False,
        bgr=0.0, flipud=0.0, fliplr=0.0, translate=0.0, scale=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    )


def write_record(tmp_path, extension):
    rgb_path = tmp_path / ("sample_visible" + extension)
    ir_path = tmp_path / ("sample_infrared" + extension)
    depth_path = tmp_path / ("sample_depth" + extension)
    label_path = tmp_path / "sample.txt"
    rgb = np.full((64, 64, 3), (20, 30, 40), dtype=np.uint8)
    ir = np.full((64, 64, 3), (50, 60, 70), dtype=np.uint8)
    ir[:, :8] = 0
    if extension == ".png":
        depth = np.arange(4096, dtype=np.uint16).reshape(64, 64) * 4
        depth[:, :8] = 0
    else:
        depth = np.full((64, 64, 3), (80, 90, 100), dtype=np.uint8)
        depth[:, :8] = 0
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(ir_path), ir)
    assert cv2.imwrite(str(depth_path), depth)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    return ("sample", rgb_path, ir_path, depth_path, label_path), depth


def test_png_depth_exactly_matches_r4_log_candidate(tmp_path):
    record, source = write_record(tmp_path, ".png")
    encoded, valid = decode_pretrained_depth(record[3], source.shape)
    expected = render_candidate_gray(source, "log")
    assert np.array_equal(encoded[..., 0], expected)
    assert np.array_equal(encoded[..., 0], encoded[..., 1])
    assert np.array_equal(encoded[..., 1], encoded[..., 2])
    assert np.array_equal(valid > 0, source > 0)


def test_transport_preserves_pretrained_triplets_and_both_masks(tmp_path):
    record, _ = write_record(tmp_path, ".jpg")
    dataset = PretrainedTriModalDataset(
        [record], 64, hyp(), augment=False, dropout_probability=0.0,
        rgb_protocol_resize=True,
    )
    sample = dataset[0]["img"]
    assert sample.shape == (11, 64, 64)
    assert sample.dtype == torch.uint8
    # BGR source triplets are independently converted to RGB for all encoders.
    assert sample[:3, 32, 32].tolist() == [40, 30, 20]
    assert sample[3:6, 32, 32].tolist() == [70, 60, 50]
    assert sample[6:9, 32, 32].tolist() == [100, 90, 80]
    assert sample[9, 32, 4] == 0 and sample[9, 32, 32] == 255
    assert sample[10, 32, 4] == 0 and sample[10, 32, 32] == 255
