"""Read-only API probe for the audited local/server Ultralytics environments."""
import argparse
import inspect
import json
import os
import platform
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_PROBE_CONFIG = tempfile.TemporaryDirectory(prefix="f001-ultralytics-probe-")
os.environ["YOLO_CONFIG_DIR"] = _PROBE_CONFIG.name  # do not touch persistent user settings

import numpy as np
import torch
import torchvision
import ultralytics
from ultralytics.data.augment import Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import BaseModel, DetectionModel, load_checkpoint

from src.fusion.dual_stream_model import DualStreamModel


AUDITED = {
    "8.3.253": {"python": "3.8.10", "torch": "2.4.1+cu118", "torchvision": "0.19.1+cu118"},
    "8.4.144": {"python": "3.10.21", "torch": "2.14.0+cpu", "torchvision": "0.29.0+cpu"},
}


def parameters(owner, name):
    return list(inspect.signature(getattr(owner, name)).parameters)


def require_prefix(actual, expected, label):
    if actual[:len(expected)] != expected:
        raise RuntimeError("%s signature mismatch: %s" % (label, actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true", help="Require the exact declared server versions and CUDA")
    args = parser.parse_args()
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if ultralytics.__version__ not in AUDITED:
        raise RuntimeError("Ultralytics version has not been source-audited: " + ultralytics.__version__)
    expected = AUDITED[ultralytics.__version__]
    if args.server:
        if ultralytics.__version__ != "8.3.253" or any(versions[k] != v for k, v in expected.items()):
            raise RuntimeError("Server version mismatch: " + json.dumps(versions))
        if not torch.cuda.is_available() or torch.version.cuda != "11.8":
            raise RuntimeError("Server CUDA 11.8 device is required")
    require_prefix(parameters(DetectionTrainer, "build_dataset"), ["self", "img_path", "mode", "batch"], "build_dataset")
    require_prefix(parameters(DetectionTrainer, "get_dataloader"),
                   ["self", "dataset_path", "batch_size", "rank", "mode"], "get_dataloader")
    require_prefix(parameters(DetectionTrainer, "get_model"), ["self", "cfg", "weights", "verbose"], "get_model")
    require_prefix(parameters(DetectionTrainer, "get_validator"), ["self"], "get_validator")
    require_prefix(parameters(DetectionTrainer, "preprocess_batch"), ["self", "batch"], "preprocess_batch")
    require_prefix(parameters(BaseTrainer, "get_dataset"), ["self"], "get_dataset")
    require_prefix(parameters(BaseTrainer, "setup_model"), ["self"], "setup_model")
    require_prefix(parameters(BaseTrainer, "_setup_train"), ["self"], "BaseTrainer._setup_train")
    require_prefix(parameters(BaseTrainer, "build_optimizer"), ["self", "model", "name", "lr", "momentum",
                                                               "decay", "iterations"], "build_optimizer")
    require_prefix(parameters(BaseTrainer, "optimizer_step"), ["self"], "optimizer_step")
    require_prefix(parameters(BaseTrainer, "save_model"), ["self"], "save_model")
    require_prefix(parameters(BaseTrainer, "train"), ["self"], "BaseTrainer.train")
    require_prefix(parameters(DetectionModel, "__init__"), ["self", "cfg", "ch", "nc", "verbose"], "DetectionModel")
    require_prefix(parameters(BaseModel, "forward"), ["self", "x"], "BaseModel.forward")
    require_prefix(parameters(BaseModel, "loss"), ["self", "batch", "preds"], "BaseModel.loss")
    require_prefix(parameters(BaseModel, "predict"), ["self", "x", "profile"], "BaseModel.predict")
    require_prefix(parameters(DetectionValidator, "preprocess"), ["self", "batch"], "validator.preprocess")
    require_prefix(parameters(DetectionValidator, "__call__"), ["self", "trainer", "model"], "validator.__call__")
    require_prefix(list(inspect.signature(load_checkpoint).parameters),
                   ["weight", "device", "inplace", "fuse"], "load_checkpoint")
    if not callable(YOLODataset.collate_fn):
        raise RuntimeError("YOLODataset.collate_fn is unavailable")
    image = np.zeros((19, 31, 6), dtype=np.uint8)
    if LetterBox((32, 32))(image=image).shape != (32, 32, 6):
        raise RuntimeError("LetterBox does not preserve six-channel transport")
    model = DualStreamModel(nc=12, verbose=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 6, 64, 64))[0]
    if tuple(output.shape) != (1, 16, 84):
        raise RuntimeError("Dual-stream forward contract mismatch: %s" % (tuple(output.shape),))
    api = {
        "public_stable": ["DetectionTrainer", "DetectionTrainer.train", "YOLODataset"],
        "semi_internal": ["DetectionModel", "DetectionValidator", "YOLODataset.collate_fn", "load_checkpoint",
                           "BaseTrainer.get_dataset/setup_model", "DetectionTrainer.get_dataloader/get_validator",
                          "DetectionTrainer.build_dataset/get_model/preprocess_batch",
                          "BaseModel.forward/loss/predict", "DetectionValidator.preprocess",
                          "Compose", "LetterBox", "RandomFlip", "RandomHSV", "Format", "Instances",
                          "get_cfg", "init_seeds"],
        "private_internal": ["BaseTrainer._setup_train (server compatibility smoke call only)",
                              "Ultralytics module graph fields model/save and module i/f",
                             "checkpoint['model'].state_dict key layout"],
        "private_overrides": [],
        "audited_but_not_used": ["BaseDataset", "build_yolo_dataset", "v8_transforms"],
    }
    print(json.dumps({"status": "PASS", "versions": versions, "signatures": {
        "build_dataset": parameters(DetectionTrainer, "build_dataset"),
        "get_model": parameters(DetectionTrainer, "get_model"),
        "get_dataloader": parameters(DetectionTrainer, "get_dataloader"),
        "get_validator": parameters(DetectionTrainer, "get_validator"),
        "preprocess_batch": parameters(DetectionTrainer, "preprocess_batch"),
        "setup_train": parameters(BaseTrainer, "_setup_train"),
        "build_optimizer": parameters(BaseTrainer, "build_optimizer"),
        "save_model": parameters(BaseTrainer, "save_model"),
        "load_checkpoint": list(inspect.signature(load_checkpoint).parameters),
        "base_predict": parameters(BaseModel, "predict"),
        "base_predict_once_reviewed_not_overridden": parameters(BaseModel, "_predict_once")},
        "api_risk": api}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
