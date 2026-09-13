"""Minimal DetectionTrainer extensions; no custom training loop."""
import torch
from pathlib import Path
import yaml
from ultralytics.models.yolo.detect.train import DetectionTrainer

from scripts.data.prepare_ir_yolo import CLASS_NAMES
from . import require_ultralytics
from .dual_stream_model import DualStreamModel
from .initialization import initialize_canonical, verify_checkpoint
from .paired_dataset import ROOT, PairedDataset, canonical_records, check_augmentation


class FusionTrainer(DetectionTrainer):
    def get_dataset(self):
        require_ultralytics()
        check_augmentation(self.args)
        if self.args.seed != 2026:
            raise ValueError("Fusion requires seed=2026")
        if self.args.model != "yolo11n.yaml" or self.args.close_mosaic or self.args.augment:
            raise ValueError("Fusion v1 requires yolo11n.yaml, close_mosaic=0, TTA disabled")
        if (self.args.rect or self.args.cache or self.args.single_cls or self.args.classes
                or self.args.fraction != 1.0 or self.args.plots
                or self.args.batch < 1 or self.args.compile or self.args.resume):
            raise ValueError("v1 requires full split, fixed batch, plots/compile/resume/cache/rect disabled")
        initial_path = Path(self.args.pretrained)
        initial_path = initial_path if initial_path.is_absolute() else ROOT / initial_path
        verify_checkpoint(initial_path)
        path = Path(self.args.data)
        path = path if path.is_absolute() else ROOT / path
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = dict(train="data/splits/train.txt", val="data/splits/val.txt",
                        rgb="data/raw/train/visible", ir="data/raw/train/infrared",
                        labels="data/processed/train/labels_clean")
        for key, value in expected.items():
            if (path.parent / data[key]).resolve() != (ROOT / value).resolve():
                raise ValueError(f"Fusion requires canonical {key}")
        if data["representation"] != "raw3" or data["channels"] != 6 or data["names"] != list(CLASS_NAMES):
            raise ValueError("Fusion representation/channels/classes mismatch")
        self.args.data = str(path.resolve())  # upstream final validator can read this metadata
        self.paired_records = canonical_records()
        print("Pairing validated: train=%d, val=%d" %
              (len(self.paired_records["train"]), len(self.paired_records["val"])))
        return dict(train=str((ROOT / expected["train"]).resolve()),
                    val=str((ROOT / expected["val"]).resolve()), nc=12, channels=6,
                    names=dict(enumerate(CLASS_NAMES)))

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode not in ("train", "val") or img_path != self.data[mode]:
            raise ValueError("Fusion dataset split mismatch")
        return PairedDataset(self.paired_records[mode], self.args.imgsz, self.args, augment=mode == "train")

    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg not in (None, "yolo11n.yaml") and not isinstance(cfg, dict):
            raise ValueError("Fusion v1 supports YOLO11n only")
        model = DualStreamModel(nc=self.data["nc"], verbose=verbose)
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)  # introduced after 8.3.253
        else:
            model.names = self.data["names"]
        if weights is not None:
            initial_path = Path(self.args.pretrained)
            initial_path = initial_path if initial_path.is_absolute() else ROOT / initial_path
            initialize_canonical(model, initial_path, verbose=verbose)
        return model

    def get_validator(self):
        # 8.3.253 needs fixed tuple; 8.4.144 derives dict loss names at first batch.
        import ultralytics
        if ultralytics.__version__ == "8.3.253":
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return super().get_validator()

    def preprocess_batch(self, batch):
        image = batch["img"]
        if image.ndim != 4 or image.shape[1] != 6 or image.dtype != torch.uint8:
            raise ValueError("Expected uint8 [B,6,H,W] transport batch")
        check_augmentation(self.args)
        return super().preprocess_batch(batch)
