"""Canonical 11-channel RGB/IR/Depth transport with shared geometry and validity masks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset
from ultralytics.data.augment import Compose, Format, RandomFlip, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances

from scripts.data.depth_preprocessing import read_image_unchanged
from scripts.data.prepare_ir_yolo import index_ir_images
from .paired_dataset import PairedLetterBox, RGBOnlyHSV, ROOT, canonical_records, read_raw3
from .quality_model import TRANSPORT_CHANNELS


DEPTH_DIR = ROOT / "data/raw/train/depth"
PNG_MAX_DEPTH_MM = 20000


def canonical_tri_records():
    paired = canonical_records()
    depth = index_ir_images(DEPTH_DIR)
    stems = {row[0] for mode in ("train", "val") for row in paired[mode]}
    if set(depth) != stems:
        raise ValueError("Depth stems differ from canonical 1600/400 RGB/IR pairs")
    records = {}
    for mode, rows in paired.items():
        records[mode] = []
        for stem, rgb, ir, label in rows:
            dep = depth[stem]
            if not (rgb.suffix.lower() == ir.suffix.lower() == dep.suffix.lower()):
                raise ValueError("Tri-modal source format differs: " + stem)
            records[mode].append((stem, rgb, ir, dep, label))
    return records


def decode_depth(path: Path, expected_shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    depth = read_image_unchanged(path)
    if depth is None or depth.shape[:2] != expected_shape:
        raise ValueError("Depth decode or shape mismatch: " + str(path))
    if path.suffix.lower() == ".png":
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise ValueError("PNG Depth must be 2D uint16: " + str(path))
        valid = (depth > 0) & (depth <= PNG_MAX_DEPTH_MM)
        linear = np.zeros(expected_shape, dtype=np.uint8)
        inverse = np.zeros(expected_shape, dtype=np.uint8)
        values = depth[valid].astype(np.float32)
        linear[valid] = np.rint(values * (255.0 / PNG_MAX_DEPTH_MM)).astype(np.uint8)
        inverse[valid] = np.rint(255000.0 / (1000.0 + values)).astype(np.uint8)
        jpg = False
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        if depth.dtype != np.uint8 or depth.ndim != 3 or depth.shape[2] != 3:
            raise ValueError("JPG Depth must be 3-channel uint8: " + str(path))
        gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        valid = (gray > 0) & edge_dark_valid_region(depth).astype(bool)
        # Encoded JPG values are not interpreted as physical millimeters. Near-black
        # compression noise along the no-data border is excluded from the valid mask.
        linear = gray
        inverse = 255 - gray
        jpg = True
    else:
        raise ValueError("Unsupported Depth format: " + str(path))
    encoded = np.stack((linear, inverse, valid.astype(np.uint8) * 255), axis=-1)
    return encoded, jpg


def encode_transport(rgb: np.ndarray, ir: np.ndarray, depth: np.ndarray, jpg: bool) -> np.ndarray:
    if rgb.dtype != np.uint8 or ir.dtype != np.uint8 or rgb.shape != ir.shape:
        raise ValueError("RGB/IR must be aligned uint8 triplets")
    if depth.shape != (*rgb.shape[:2], 3) or depth.dtype != np.uint8:
        raise ValueError("Encoded Depth must be HxWx3 uint8")
    ir_valid = edge_dark_valid_region(ir)
    fmt = np.full(rgb.shape[:2], 255 if jpg else 0, dtype=np.uint8)
    image = np.concatenate((rgb, ir, depth, ir_valid[..., None], fmt[..., None]), axis=2)
    if image.shape[2] != TRANSPORT_CHANNELS:
        raise RuntimeError("Tri-modal transport channel count changed")
    return np.ascontiguousarray(image)


def edge_dark_valid_region(image: np.ndarray) -> np.ndarray:
    """Exclude an edge-connected dark border, including JPEG compression noise."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Border validity expects an HxWx3 uint8 image")
    low = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) <= 8).astype(np.uint8)
    count, components = cv2.connectedComponents(low, connectivity=8)
    if count == 1:
        return np.full(image.shape[:2], 255, dtype=np.uint8)
    edge_labels = np.unique(np.concatenate((components[0], components[-1],
                                            components[:, 0], components[:, -1])))
    edge_labels = edge_labels[edge_labels != 0]
    invalid = np.isin(components, edge_labels)
    return (~invalid).astype(np.uint8) * 255


class TriModalFormat(Format):
    def __call__(self, labels):
        image = labels["img"]
        labels["img"] = np.ascontiguousarray(image[..., [2, 1, 0, 5, 4, 3, 6, 7, 8, 9, 10]])
        return super().__call__(labels)


class QualityTriModalDataset(Dataset):
    collate_fn = staticmethod(YOLODataset.collate_fn)
    rect = False
    mosaic = False

    def __init__(self, records, imgsz, hyp, augment=False, dropout_probability=0.2):
        if imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a multiple of 32")
        if not 0 <= dropout_probability < 1:
            raise ValueError("invalid modality dropout probability")
        if any(getattr(hyp, key, 0) for key in (
            "mosaic", "mixup", "cutmix", "copy_paste", "degrees", "shear", "perspective",
            "multi_scale", "bgr", "flipud",
        )):
            raise ValueError("V2 allows shared translation/scale/fliplr, not unsynchronized mixing")
        self.records = list(records)
        self.imgsz = imgsz
        self.augment = augment
        self.dropout_probability = dropout_probability
        self.im_files = [str(row[1]) for row in records]
        self.labels: List[Dict[str, object]] = []
        for stem, rgb_path, ir_path, depth_path, label_path in records:
            rgb, ir = read_raw3(rgb_path), read_raw3(ir_path)
            encoded, _ = decode_depth(depth_path, rgb.shape[:2])
            if ir.shape != rgb.shape or encoded.shape[:2] != rgb.shape[:2]:
                raise ValueError("Tri-modal image shapes differ: " + stem)
            rows = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if any(len(row) != 5 for row in rows):
                raise ValueError("Invalid clean label columns: " + stem)
            boxes = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if (not np.isfinite(boxes).all() or (boxes[:, 0] != boxes[:, 0].astype(int)).any()
                    or (boxes[:, 0] < 0).any() or (boxes[:, 0] > 11).any()
                    or (boxes[:, 1:] < 0).any() or (boxes[:, 1:] > 1).any()
                    or (boxes[:, 3:] <= 0).any()):
                raise ValueError("Invalid clean labels: " + stem)
            self.labels.append(dict(im_file=str(rgb_path), shape=rgb.shape[:2],
                                    cls=boxes[:, :1], bboxes=boxes[:, 1:]))

        transforms = [PairedLetterBox((imgsz, imgsz), scaleup=augment)]
        if augment:
            transforms += [
                RandomPerspective(translate=hyp.translate, scale=hyp.scale),
                RandomFlip(hyp.fliplr, "horizontal"),
                RGBOnlyHSV(hyp),
            ]
        transforms.append(TriModalFormat(bbox_format="xywh", normalize=True, batch_idx=True))
        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        stem, rgb_path, ir_path, depth_path, _ = self.records[index]
        rgb, ir = read_raw3(rgb_path), read_raw3(ir_path)
        encoded, jpg = decode_depth(depth_path, rgb.shape[:2])
        image = encode_transport(rgb, ir, encoded, jpg)
        label = deepcopy(self.labels[index])
        label.pop("shape")
        boxes = label.pop("bboxes")
        label.update(img=image, ori_shape=rgb.shape[:2], ratio_pad=(1.0, 1.0),
                     instances=Instances(boxes, np.zeros((0, 1000, 2), dtype=np.float32),
                                         bbox_format="xywh", normalized=True))
        sample = self.transforms(label)
        sample["img"][10].fill_(255 if jpg else 0)  # format is an image-level token
        if self.augment:
            if np.random.random() < self.dropout_probability:
                sample["img"][3:6].zero_()
                sample["img"][9].zero_()
            if np.random.random() < self.dropout_probability:
                sample["img"][6:9].zero_()
        return sample
