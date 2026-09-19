"""Trainer for checkpoint-initialized P4/P5 multimodal residual fusion."""

from __future__ import annotations

from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics.models.yolo.detect.train import DetectionTrainer

from scripts.data.prepare_ir_yolo import CLASS_NAMES
from .paired_dataset import ROOT
from .pretrained_dataset import PretrainedTriModalDataset, TRANSPORT_CHANNELS
from .pretrained_initialization import initialize_pretrained_fusion
from .pretrained_model import PretrainedTriModalModel
from .quality_dataset import canonical_tri_records


class PretrainedFusionTrainer(DetectionTrainer):
    use_depth = True
    dropout_probability = 0.1

    def get_dataset(self):
        if (self.args.model != "yolo11m.yaml" or self.args.pretrained is not False
                or self.args.seed != 2026 or self.args.resume or self.args.rect
                or self.args.cache or self.args.augment or self.args.multi_scale):
            raise ValueError("Pretrained fusion requires frozen graph/data/seed policy")
        if self.args.freeze != 24:
            raise ValueError("All 24 RGB layers must stay frozen")
        path = Path(self.args.data)
        path = path if path.is_absolute() else ROOT / path
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {
            "train": "data/splits/train.txt", "val": "data/splits/val.txt",
            "rgb": "data/raw/train/visible", "ir": "data/raw/train/infrared",
            "depth": "data/raw/train/depth", "labels": "data/processed/train/labels_clean",
        }
        for key, value in expected.items():
            if (path.parent / data[key]).resolve() != (ROOT / value).resolve():
                raise ValueError("Pretrained fusion requires canonical " + key)
        if (data["representation"] != "pretrained11_v1"
                or data["channels"] != TRANSPORT_CHANNELS
                or data["names"] != list(CLASS_NAMES)):
            raise ValueError("Pretrained fusion data contract differs")
        self.args.data = str(path.resolve())
        self.tri_records = canonical_tri_records()
        return {
            "train": str((ROOT / expected["train"]).resolve()),
            "val": str((ROOT / expected["val"]).resolve()),
            "nc": 12, "channels": TRANSPORT_CHANNELS,
            "names": dict(enumerate(CLASS_NAMES)),
        }

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode not in ("train", "val") or img_path != self.data[mode]:
            raise ValueError("Pretrained fusion dataset split differs")
        records = self.tri_records[mode]
        if self.args.name.endswith("_SMOKE"):
            records = records[:32 if mode == "train" else 16]
        return PretrainedTriModalDataset(
            records, self.args.imgsz, self.args, augment=mode == "train",
            dropout_probability=self.dropout_probability,
            rect_batch_size=16 if mode == "val" else None,
            rgb_protocol_resize=True,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg not in (None, "yolo11m.yaml") and not isinstance(cfg, dict):
            raise ValueError("Pretrained fusion supports YOLO11m only")
        if weights is not None:
            raise ValueError("Implicit trainer weights are disallowed")
        model = PretrainedTriModalModel(
            nc=self.data["nc"], use_depth=self.use_depth, verbose=verbose
        )
        model.names = self.data["names"]
        initialize_pretrained_fusion(model)
        return model

    def build_optimizer(self, model, *args, **kwargs):
        # Ultralytics may re-enable custom parameters before optimizer creation.
        for parameter in model.model.parameters():
            parameter.requires_grad_(False)
        for backbone in filter(None, (model.ir_backbone, model.depth_backbone)):
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        allowed = ("ir_fusion.", "depth_fusion.") if model.use_depth else ("ir_fusion.",)
        unexpected = [name for name, parameter in model.named_parameters()
                      if parameter.requires_grad and not name.startswith(allowed)]
        if unexpected:
            raise RuntimeError("Unexpected trainable fusion parameters: " + str(unexpected[:5]))
        return super().build_optimizer(model, *args, **kwargs)

    def get_validator(self):
        if ultralytics.__version__ == "8.3.253":
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return super().get_validator()

    def preprocess_batch(self, batch):
        image = batch["img"]
        if (image.ndim != 4 or image.shape[1] != TRANSPORT_CHANNELS
                or image.dtype != torch.uint8):
            raise ValueError("Expected uint8 [B,11,H,W] pretrained transport")
        return super().preprocess_batch(batch)
