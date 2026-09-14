"""RGB-only CTRL001 built from the reviewed F001 data and initialization contracts."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from scripts.data.prepare_ir_yolo import CLASS_NAMES
from . import require_ultralytics
from .initialization import load_matching_state, verify_checkpoint
from .paired_dataset import ROOT, PairedDataset, canonical_records, check_augmentation


class RGBControlDataset(Dataset):
    """Run the paired F001 transform graph, then expose only its RGB triplet."""

    collate_fn = staticmethod(PairedDataset.collate_fn)
    rect = False
    mosaic = False

    def __init__(self, paired_dataset):
        if not isinstance(paired_dataset, PairedDataset):
            raise TypeError("CTRL001 must wrap the repository PairedDataset")
        self.paired_dataset = paired_dataset
        self.records = paired_dataset.records
        self.labels = paired_dataset.labels
        self.im_files = paired_dataset.im_files

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, index):
        sample = dict(self.paired_dataset[index])
        image = sample["img"]
        if image.ndim != 3 or image.shape[0] != 6:
            raise ValueError("Paired transform must produce [6,H,W] before RGB drop")
        sample["img"] = image[:3].contiguous()
        return sample


def canonical_control_records():
    """Validate real canonical bytes/stems through the single F001 source of truth."""
    records = canonical_records()
    if set(records) != {"train", "val"}:
        raise RuntimeError("Canonical records must contain train and val only")
    if len(records["train"]) != 1600 or len(records["val"]) != 400:
        raise RuntimeError("CTRL001 requires canonical train=1600 and val=400")
    train_stems = [row[0] for row in records["train"]]
    val_stems = [row[0] for row in records["val"]]
    if len(set(train_stems + val_stems)) != 2000:
        raise RuntimeError("CTRL001 canonical stems are duplicated or split-overlapping")
    return records


def initialize_rgb_control(model, path=None, verbose=True):
    """Apply the exact F001 authenticated shape-match policy to an RGB YOLO graph."""
    path = Path(path or ROOT / "weights/yolo11n.pt")
    digest = verify_checkpoint(path)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

    class RGBPolicyAdapter:
        """Present the F001 loader with its expected groups without adding model parameters."""

        def state_dict(self):
            state = model.state_dict()
            adapted = OrderedDict(state)
            for key, tensor in state.items():
                if key.startswith("model.") and int(key.split(".")[1]) <= 10:
                    adapted["ir_encoder." + key[len("model."):]] = tensor
            return adapted

        def load_state_dict(self, selected, strict=False):
            rgb = {key: value for key, value in selected.items() if key.startswith("model.")}
            return model.load_state_dict(rgb, strict=strict)

    report = load_matching_state(RGBPolicyAdapter(), checkpoint["model"].float().state_dict(), verbose=False)
    report.update(checkpoint="weights/yolo11n.pt", checkpoint_sha256=digest,
                  policy_adapter="IR group is audited as a mirror only; CTRL001 stores RGB model.* keys only")
    model.load_report = report
    if verbose:
        print("PRETRAINED_LOAD_REPORT=" + json.dumps(report, sort_keys=True))
    return report


class RGBControlTrainer(DetectionTrainer):
    """DetectionTrainer with F001 pairing/transforms and a three-channel model."""

    def get_dataset(self):
        require_ultralytics()
        check_augmentation(self.args)
        if self.args.seed != 2026:
            raise ValueError("CTRL001 requires seed=2026")
        if self.args.model != "yolo11n.yaml" or self.args.close_mosaic or self.args.augment:
            raise ValueError("CTRL001 requires yolo11n.yaml, close_mosaic=0, TTA disabled")
        if (self.args.rect or self.args.cache or self.args.single_cls or self.args.classes
                or self.args.fraction != 1.0 or self.args.plots or self.args.batch < 1
                or self.args.compile or self.args.resume):
            raise ValueError("CTRL001 requires the complete reviewed split and runtime policy")

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
                raise ValueError("CTRL001 requires canonical %s" % key)
        if (data.get("representation") != "paired_transform_then_rgb_drop"
                or data.get("channels") != 3 or data.get("names") != list(CLASS_NAMES)):
            raise ValueError("CTRL001 representation/channels/classes mismatch")
        self.args.data = str(path.resolve())
        self.paired_records = canonical_control_records()
        return dict(train=str((ROOT / expected["train"]).resolve()),
                    val=str((ROOT / expected["val"]).resolve()), nc=12, channels=3,
                    names=dict(enumerate(CLASS_NAMES)))

    def build_dataset(self, img_path, mode="train", batch=None):
        del batch
        if mode not in ("train", "val") or img_path != self.data[mode]:
            raise ValueError("CTRL001 dataset split mismatch")
        paired = PairedDataset(self.paired_records[mode], self.args.imgsz, self.args, augment=mode == "train")
        return RGBControlDataset(paired)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if cfg not in (None, "yolo11n.yaml") and not isinstance(cfg, dict):
            raise ValueError("CTRL001 supports YOLO11n only")
        model = DetectionModel("yolo11n.yaml", ch=3, nc=self.data["nc"], verbose=verbose)
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)
        else:
            model.names = self.data["names"]
        if weights is not None:
            initial_path = Path(self.args.pretrained)
            initial_path = initial_path if initial_path.is_absolute() else ROOT / initial_path
            initialize_rgb_control(model, initial_path, verbose=verbose)
        return model

    def get_validator(self):
        import ultralytics
        if ultralytics.__version__ == "8.3.253":
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return super().get_validator()

    def preprocess_batch(self, batch):
        image = batch["img"]
        if image.ndim != 4 or image.shape[1] != 3 or image.dtype != torch.uint8:
            raise ValueError("Expected uint8 [B,3,H,W] CTRL001 batch")
        check_augmentation(self.args)
        return super().preprocess_batch(batch)


assert RGBControlTrainer.train is DetectionTrainer.train
