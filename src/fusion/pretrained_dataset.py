"""Transport that exactly reproduces the inputs of the trained IR and Depth detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from scripts.data.depth_preprocessing import read_image_unchanged
from scripts.data.prepare_depth_candidate_yolo import D_FAR_MM, L_MAX, L_MIN
from .quality_dataset import QualityTriModalDataset, TriModalFormat, edge_dark_valid_region


TRANSPORT_CHANNELS = 11


class PretrainedTriModalFormat(TriModalFormat):
    def __call__(self, labels):
        image = labels["img"]
        labels["img"] = np.ascontiguousarray(
            image[..., [2, 1, 0, 5, 4, 3, 8, 7, 6, 9, 10]]
        )
        # Skip TriModalFormat's V2 channel permutation.
        return super(TriModalFormat, self).__call__(labels)


def decode_pretrained_depth(path: Path, expected_shape: Tuple[int, int]):
    """Reproduce the frozen ``depth_trainable/log`` pixels without a disk view."""
    depth = read_image_unchanged(path)
    if depth is None or depth.shape[:2] != expected_shape:
        raise ValueError("Depth decode or shape mismatch: " + str(path))
    if path.suffix.lower() == ".png":
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError("PNG Depth must be 2D uint16: " + str(path))
        if int(depth.max()) > D_FAR_MM:
            raise ValueError("PNG Depth exceeds the frozen R4 log range: " + str(path))
        valid = depth > 0
        values = np.zeros(expected_shape, dtype=np.float32)
        values[valid] = (np.log1p(depth[valid].astype(np.float32)) - L_MIN) / (L_MAX - L_MIN)
        gray = np.floor(np.clip(values, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
        encoded = np.repeat(gray[..., None], 3, axis=2)
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        if depth.dtype != np.uint8 or depth.ndim != 3 or depth.shape[2] != 3:
            raise ValueError("JPG Depth must be 3-channel uint8: " + str(path))
        encoded = depth  # candidate builder copied these bytes unchanged
        gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        valid = (gray > 0) & edge_dark_valid_region(depth).astype(bool)
    else:
        raise ValueError("Unsupported Depth format: " + str(path))
    return np.ascontiguousarray(encoded), valid.astype(np.uint8) * 255


class PretrainedTriModalDataset(QualityTriModalDataset):
    """V3 dataset: RGB raw3 + IR raw3 + the exact R4 Depth log/raw3 input."""

    def _encode_record(self, rgb, ir, depth_path):
        depth, depth_valid = decode_pretrained_depth(depth_path, rgb.shape[:2])
        ir_valid = edge_dark_valid_region(ir)
        image = np.concatenate((rgb, ir, depth, ir_valid[..., None], depth_valid[..., None]), axis=2)
        if image.shape[2] != TRANSPORT_CHANNELS:
            raise RuntimeError("Pretrained transport channel count changed")
        # False keeps the parent from rewriting channel 10 as a format token.
        return np.ascontiguousarray(image), False

    def _make_format(self):
        return PretrainedTriModalFormat(bbox_format="xywh", normalize=True, batch_idx=True)

    def _finalize_transport(self, image, jpg):
        return None

    def _drop_depth(self, image):
        image[6:9].zero_()
        image[10].zero_()
