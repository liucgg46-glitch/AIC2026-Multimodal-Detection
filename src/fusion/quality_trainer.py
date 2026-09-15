"""Ultralytics trainer for the single-model M960 RGB+IR+Depth quality fusion."""

from __future__ import annotations

from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.models.yolo.detect.train import DetectionTrainer

from scripts.data.prepare_ir_yolo import CLASS_NAMES
from .paired_dataset import ROOT
from .quality_dataset import QualityTriModalDataset, canonical_tri_records
from .quality_initialization import load_m960_rgb
from .quality_model import QualityTriModalModel, TRANSPORT_CHANNELS


class QualityFusionTrainer(DetectionTrainer):
    dropout_probability = 0.2

    def get_dataset(self):
        if (self.args.model != "yolo11m.yaml" or self.args.pretrained is not False
                or self.args.seed != 2026 or self.args.resume or self.args.rect
                or self.args.cache or self.args.augment or self.args.multi_scale):
            raise ValueError("Quality fusion V2 requires frozen M960 graph/data/seed policy")
        if not isinstance(self.args.freeze, int) or self.args.freeze != 24:
            raise ValueError("Stage 1 must freeze all 24 RGB model layers")
        path = Path(self.args.data)
        path = path if path.is_absolute() else ROOT / path
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {
            "train": "data/splits/train.txt",
            "val": "data/splits/val.txt",
            "rgb": "data/raw/train/visible",
            "ir": "data/raw/train/infrared",
            "depth": "data/raw/train/depth",
            "labels": "data/processed/train/labels_clean",
        }
        for key, value in expected.items():
            if (path.parent / data[key]).resolve() != (ROOT / value).resolve():
                raise ValueError("Quality fusion requires canonical " + key)
        if (data["representation"] != "quality11_v1"
                or data["channels"] != TRANSPORT_CHANNELS
                or data["names"] != list(CLASS_NAMES)):
            raise ValueError("Quality fusion data YAML differs from canonical 11-channel contract")
        self.args.data = str(path.resolve())
        self.tri_records = canonical_tri_records()
        print("QUALITY_FUSION_PAIRING train=%d val=%d" % (
            len(self.tri_records["train"]), len(self.tri_records["val"])
        ))
        return {
            "train": str((ROOT / expected["train"]).resolve()),
            "val": str((ROOT / expected["val"]).resolve()),
            "nc": 12,
            "channels": TRANSPORT_CHANNELS,
            "names": dict(enumerate(CLASS_NAMES)),
        }

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode not in ("train", "val") or img_path != self.data[mode]:
            raise ValueError("Quality fusion dataset split differs from trainer split")
        records = self.tri_records[mode]
        if self.args.name.endswith("_SMOKE"):
            records = records[:32 if mode == "train" else 16]
        return QualityTriModalDataset(
            records, self.args.imgsz, self.args, augment=mode == "train",
            dropout_probability=self.dropout_probability,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg not in (None, "yolo11m.yaml") and not isinstance(cfg, dict):
            raise ValueError("Quality fusion V2 supports YOLO11m only")
        if weights is not None:
            raise ValueError("Quality fusion V2 disallows an implicit trainer checkpoint")
        model = QualityTriModalModel(nc=self.data["nc"], verbose=verbose)
        model.names = self.data["names"]
        load_m960_rgb(model)
        return model

    def get_validator(self):
        if ultralytics.__version__ == "8.3.253":
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return super().get_validator()

    def preprocess_batch(self, batch):
        image = batch["img"]
        if (image.ndim != 4 or image.shape[1] != TRANSPORT_CHANNELS
                or image.dtype != torch.uint8):
            raise ValueError("Expected uint8 [B,11,H,W] tri-modal transport")
        return super().preprocess_batch(batch)
